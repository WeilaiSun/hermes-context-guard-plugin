"""context-guard--plugin v2.3 — PACE predictive context compression.

Registers three hooks (pre_llm_call, on_session_start,
on_session_end) and one tool (pace_glimpse) to implement the PACE
algorithm for context management.

Lifecycle:
  on_session_start → disable built-in compression, load memory, warm BGE-M3
  pre_llm_call     → Algorithm 1 steps 1-15 (process prev turn 13-15, compress 1-12)
  on_session_end   → restore compression, persist memory, shutdown summarizer

Note: post_llm_call exists in VALID_HOOKS but is never invoked by Hermes
(no invoke_hook call in source). Steps 13-15 are merged into pre_llm_call.

References:
  - PACE design §5.1 Algorithm 1 (all 15 steps)
  - PACE design §3.4 replace_messages format
  - PACE design §6.4 degradation strategy
  - Hermes plugin API: hermes_cli/plugins.py register_hook/register_tool
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

# ── Module-level state (initialized in register()) ──────────────────────
_config: dict = {}
_scorer: Any = None       # PACEScorer
_compressor: Any = None   # PACECompressor
_memory_store: Any = None # PACEMemoryStore
_summarizer: Any = None   # PaceSummarizer
_glimpse_tool: Any = None # GlimpseTool
_turn_counter: int = 0
_last_pace_stats: dict | None = None  # cached for footnote in transform_llm_output
_session_active: bool = False


def _load_config() -> dict:
    """Load config.yaml from the plugin directory."""
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:
        logger.warning("PACE: failed to load config.yaml: %s, using defaults", exc)
        return {}


def _get_agent() -> Any:
    """Look up the active agent via the cli module global.

    The agent is NOT passed as a kwarg to on_session_start/on_session_end.
    Instead, the CLI sets ``cli._active_agent_ref`` when a session starts
    (cli_agent_setup_mixin.py:415). We import the module and read the attr.
    """
    for mod_path in ["hermes_cli.cli", "cli"]:
        try:
            mod = __import__(mod_path, fromlist=["_active_agent_ref"])
            agent = getattr(mod, "_active_agent_ref", None)
            if agent is not None:
                return agent
        except ImportError:
            continue
    return None


# ── Hook callbacks ──────────────────────────────────────────────────────

def _on_pre_llm_call(**kwargs: Any) -> dict | None:
    """PACE pre_llm_call hook — Algorithm 1 steps 1-15.

    Returns ``{"replace_messages": C_t}`` to replace the full conversation
    history with the PACE-compressed context, or ``None`` to skip.
    
    Steps 13-15 (previous turn processing) are prefixed because
    post_llm_call is never invoked by Hermes.
    """
    global _turn_counter

    if not _session_active:
        return None

    session_id = kwargs.get("session_id", "")
    user_message = kwargs.get("user_message", "")
    conversation_history = kwargs.get("conversation_history", [])
    is_first_turn = kwargs.get("is_first_turn", True)

    # ── Skip conditions ──
    if is_first_turn:
        logger.debug("PACE: skipping first turn (no history to compress)")
        return None

    _turn_counter += 1  # increment on every non-first turn (v2.3: post_llm_call removal)

    # ── Algorithm 1 steps 13-15: Process previous turn (replaces post_llm_call) ──
    # post_llm_call exists in VALID_HOOKS but is never invoked by Hermes
    # (no invoke_hook("post_llm_call",...) in source). We piggyback on
    # pre_llm_call which fires every turn instead.
    if _turn_counter > 0 and conversation_history:
        prev_assistant = ""
        for msg in reversed(conversation_history[:-2]):  # skip current user+recent
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                prev_assistant = msg.get("content", "")
                break
        if prev_assistant:
            r_full = f"User: {user_message}\n\nAssistant: {prev_assistant[:500]}"
            key_emb = _scorer.encode_key(r_full)
            chunk = _memory_store.add_chunk(
                session_id=session_id,
                turn_index=_turn_counter - 1,
                r_full=r_full,
                key_embedding=key_emb,
            )
            _summarizer.enqueue(
                chunk_id=chunk.chunk_id,
                chunk_text=r_full,
                writeback=_memory_store.set_summary,
                levels=[2, 1],
            )
            _memory_store.clear_glimpse_flags(session_id)
            _glimpse_tool.reset_step(turn_id=str(kwargs.get("turn_id", "")))
            logger.debug("PACE: processed previous turn (chunk %s)", chunk.chunk_id)

    min_history = _config.get("plugin", {}).get("min_history_length", 5)
    if len(conversation_history) < min_history:
        logger.debug("PACE: history too short (%d < %d), skipping", len(conversation_history), min_history)
        return None

    # ── Compact trigger (v2.3: word-boundary regex, M3 fix) ──
    compact_regex = _config.get("plugin", {}).get("compact_regex", r"\bcompact\b")
    force_compact = bool(re.search(compact_regex, user_message, re.IGNORECASE))

    # ── Step 1: Encode query q_t ← BGE-M3-Encode(Q ⊕ R_full^{(t-N:t-1)}) ──
    N = _compressor.N
    recent_count = N * 2  # user+assistant pairs
    recent_2_full = conversation_history[-recent_count:] if recent_count > 0 else []
    query_emb = _scorer.encode_query(user_message, recent_2_full)

    # ── Steps 2-3: Get chunks M and cached keys {k_i} ──
    chunks = _memory_store.get_chunks(session_id)
    cached_keys = _memory_store.get_keys(session_id)
    M = len(chunks)

    if M == 0:
        logger.debug("PACE: no chunks in memory store, skipping")
        return None

    # BGE-M3 unavailable → skip PACE (all R_full via passthrough) — design §6.4
    if query_emb is None:
        logger.info("PACE: BGE-M3 unavailable, skipping compression (all R_full)")
        return None

    # Ensure all chunks have key embeddings (encoding failures → skip this turn)
    if len(cached_keys) < M:
        logger.warning(
            "PACE: %d/%d chunks missing key embeddings, skipping compression",
            M - len(cached_keys), M,
        )
        return None

    # ── Step 4: Score s_i ← cosine(q_t, k_i) ──
    scores = _scorer.score(query_emb, cached_keys)

    # ── Steps 5-6: w_i ← softmax(s_i/τ), w̃_i ← M·w_i ──
    w_tilde = _compressor.compute_weights(scores)

    # ── Step 7: P_t ← max(t/T_max, |C_{t-1}|/B_max) ──
    t = _turn_counter
    last_ctx_tokens = _memory_store.last_context_tokens
    P_t = _compressor.compute_pressure(
        t, last_ctx_tokens, fallback_messages=conversation_history,
    )

    # ── Step 8: α_t, β_t, γ_t ← adapt(P_t) ──
    alpha_t, beta_t, gamma_t = _compressor.adapt_thresholds(P_t)

    # ── Steps 9-10: Select granularity and assemble C_t ──
    C_t = _compressor.assemble_messages(
        conversation_history=conversation_history,
        chunks=chunks,
        w_tilde=w_tilde,
        thresholds=(alpha_t, beta_t, gamma_t),
        user_message=user_message,
        n_recent=N,
        force_compact=force_compact,
    )

    # ── Step 11: Cache |C_t| ──
    token_count = _compressor.count_tokens(C_t)
    _memory_store.last_context_tokens = token_count

    # ── Step 12: Return ──
    logger.info(
        "PACE: compressed %d chunks → %d tokens (P_t=%.3f, α=%.2f, β=%.2f, γ=%.2f, compact=%s)",
        M, token_count, P_t, alpha_t, beta_t, gamma_t, force_compact,
    )
    # Cache stats for footnote in transform_llm_output
    global _last_pace_stats
    _last_pace_stats = {"M": M, "tokens": token_count, "P_t": P_t, "compact": force_compact}
    return {"replace_messages": C_t}


def _on_session_start(**kwargs: Any) -> None:
    """Disable built-in compression and initialize PACE state."""
    global _session_active, _turn_counter

    _session_active = True
    _turn_counter = 0

    # Load memory store from disk (session recovery)
    _memory_store.load()

    # Disable built-in compression (threshold → 0.99)
    if _config.get("plugin", {}).get("disable_builtin_compression", True):
        agent = _get_agent()
        if agent and hasattr(agent, "context_compressor"):
            saved = agent.context_compressor.threshold_percent
            agent._pace_saved_threshold = saved
            agent.context_compressor.threshold_percent = 0.99
            logger.info("PACE: disabled built-in compression (threshold %.2f → 0.99)", saved)

    # Pre-load BGE-M3 model (non-blocking — encode_* will retry if not ready)
    _scorer.load_model()

    logger.info("PACE: session started (session_id=%s)", kwargs.get("session_id", ""))
    return None


def _on_session_end(**kwargs: Any) -> None:
    """Restore built-in compression and persist PACE state."""
    global _session_active

    _session_active = False

    # Persist memory store
    _memory_store.save()

    # Shutdown summarizer thread pool
    _summarizer.shutdown()

    # Restore built-in compression threshold
    agent = _get_agent()
    if agent and hasattr(agent, "context_compressor"):
        saved = getattr(agent, "_pace_saved_threshold", None)
        if saved is not None:
            agent.context_compressor.threshold_percent = saved
            logger.info("PACE: restored built-in compression (threshold → %.2f)", saved)
        else:
            agent.context_compressor.threshold_percent = 0.50
            logger.info("PACE: restored built-in compression to default (0.50)")

    logger.info("PACE: session ended (completed=%s)", kwargs.get("completed", False))
    return None


def _on_transform_llm_output(output: str, **kwargs: Any) -> str | None:
    """Append a PACE compression footnote to the LLM response.

    Reads stats cached by _on_pre_llm_call. Returns modified output
    with a compact one-line footnote, or None if no compression happened.
    """
    global _last_pace_stats
    stats = _last_pace_stats
    _last_pace_stats = None
    if stats is None:
        return None
    compact_str = " · 强制精简" if stats["compact"] else ""
    footnote = (
        f"\n\n---\n"
        f"📦 上下文压缩: {stats['M']} 个片段 → {stats['tokens']:,} tokens "
        f"(压力指数 {stats['P_t']:.2f}){compact_str}"
    )
    return output + footnote


# ── Plugin registration ────────────────────────────────────────────────

def register(ctx) -> None:
    """Register PACE context-guard plugin hooks and tools."""
    global _config, _scorer, _compressor, _memory_store, _summarizer, _glimpse_tool

    _config = _load_config()

    pace_cfg = _config.get("pace", {})
    ms_cfg = _config.get("memory_store", {})
    sc_cfg = _config.get("scorer", {})
    sm_cfg = _config.get("summarizer", {})
    gl_cfg = _config.get("glimpse", {})

    # Initialize components
    from .scorer import PACEScorer
    from .compressor import PACECompressor
    from .memory_store import PACEMemoryStore
    from .summarizer import PaceSummarizer
    from .glimpse import (
        GlimpseTool,
        GLIMPSE_TOOL_SCHEMA,
        GLIMPSE_TOOL_DESCRIPTION,
        GLIMPSE_TOOL_EMOJI,
    )

    _scorer = PACEScorer(
        model_name=sc_cfg.get("model_name", "BAAI/bge-m3"),
        device=sc_cfg.get("device", "cpu"),
        normalize_embeddings=sc_cfg.get("normalize_embeddings", True),
    )

    _compressor = PACECompressor(
        tau=pace_cfg.get("tau", 0.3),
        N=pace_cfg.get("N", 2),
        alpha_0=pace_cfg.get("alpha_0", 0.4),
        beta_0=pace_cfg.get("beta_0", 0.8),
        gamma_0=pace_cfg.get("gamma_0", 1.5),
        lambda_adapt=pace_cfg.get("lambda_adapt", 0.5),
        B_max=pace_cfg.get("B_max", 128000),
        T_max=pace_cfg.get("T_max", 200),
    )

    persist_path = ms_cfg.get("persist_path")
    if persist_path is None:
        persist_path = os.path.join(
            os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")),
            "context-guard-memory.json",
        )

    _memory_store = PACEMemoryStore(
        max_chunks=ms_cfg.get("max_chunks", 200),
        persist_path=persist_path,
    )

    _summarizer = PaceSummarizer(
        provider=sm_cfg.get("provider", "deepseek"),
        model=sm_cfg.get("model", "deepseek-chat"),
        fallback_provider=sm_cfg.get("fallback_provider", "zai"),
        fallback_model=sm_cfg.get("fallback_model", "qwen-plus"),
        timeout=sm_cfg.get("timeout", 30),
        max_workers=sm_cfg.get("max_workers", 2),
        skip_threshold=sm_cfg.get("skip_threshold", 0.1),
        detailed_target_words=sm_cfg.get("detailed_target_words", 100),
        brief_target_sentences=sm_cfg.get("brief_target_sentences", 2),
    )

    _glimpse_tool = GlimpseTool(
        memory_store=_memory_store,
        max_per_step=gl_cfg.get("max_per_step", 3),
    )

    # Register hooks
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    ctx.register_hook("on_session_start", _on_session_start)
    ctx.register_hook("on_session_end", _on_session_end)
    ctx.register_hook("transform_llm_output", _on_transform_llm_output)

    # Register Glimpse tool
    ctx.register_tool(
        name="pace_glimpse",
        toolset="context-guard",
        schema=GLIMPSE_TOOL_SCHEMA,
        handler=_glimpse_tool,
        description=GLIMPSE_TOOL_DESCRIPTION,
        emoji=GLIMPSE_TOOL_EMOJI,
    )

    logger.info(
        "context-guard--plugin v2.3 registered (PACE: τ=%.2f, N=%d, B_max=%d)",
        _compressor.tau, _compressor.N, _compressor.B_max,
    )
