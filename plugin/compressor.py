"""PACE compression core — pressure, thresholds, granularity, assembly.

Implements the heart of the PACE algorithm: computing compression pressure
from context token usage, adapting granularity thresholds, selecting the
representation level for each historical chunk, and assembling the final
compressed message list (C_t) that replaces the full history (H_t).

PACE Algorithm 1 steps 7-12:
  7.  P_t ← max(t / T_max, |C_{t-1}| / B_max)
  8.  α_t, β_t, γ_t ← adapt(P_t)
  9.  R*_i ← Select(Chunk_i, w̃_i, α_t, β_t, γ_t)
  10. C_t ← assemble_messages(Q, {R*_i}, R_full^{(t-N:t-1)})
  11. Cache |C_t| → memory_store.last_context_tokens
  12. Return {"replace_messages": C_t}

References:
  - PACE design §5.1 Algorithm 1 (all 15 steps)
  - PACE design §5.2 hyperparameters (Table 2 ablation)
  - PACE design §3.4 replace_messages format convention
  - PACE design §6.4 degradation strategy
"""

from __future__ import annotations

import logging
import math
from typing import Any, Sequence

try:
    from .watermark import Watermark
except ImportError:
    from watermark import Watermark

logger = logging.getLogger(__name__)

# Granularity levels
LEVEL_PH = 0       # placeholder
LEVEL_BRIEF = 1    # 1-2 sentence summary
LEVEL_DETAILED = 2 # ~100 word summary
LEVEL_FULL = 3     # full text


class PACECompressor:
    """PACE compression engine.

    Computes pressure, adapts thresholds, selects granularity, and assembles
    the compressed context C_t that replaces the full conversation history.
    """

    def __init__(
        self,
        tau: float = 0.3,
        N: int = 2,
        alpha_0: float = 0.4,
        beta_0: float = 0.8,
        gamma_0: float = 1.5,
        lambda_adapt: float = 0.5,
        B_max: int = 128000,
        T_max: int = 200,
    ) -> None:
        self.tau = tau
        self.N = N
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        self.gamma_0 = gamma_0
        self.lambda_adapt = lambda_adapt
        self.B_max = B_max
        self.T_max = T_max
        self.watermark = Watermark(B_max=B_max)

    # ── Step 7: Pressure ──────────────────────────────────────────────

    def compute_pressure(
        self,
        t: int,
        last_context_tokens: int | None,
        T_max: int | None = None,
        B_max: int | None = None,
        fallback_messages: Sequence[dict] | None = None,
    ) -> float:
        """Compute compression pressure P_t ∈ [0, 1].

        PACE step 7:
          P_t ← max(t / T_max, |C_{t-1}| / B_max)

        v2.3: uses the exact cached |C_t| from the previous turn's
        ``assemble_messages``. On the first turn (no cache), falls back
        to watermark token estimation of the initial context (design §5.1
        step 7 note + M2 fix).
        """
        t_max = T_max if T_max is not None else self.T_max
        b_max = B_max if B_max is not None else self.B_max

        turn_pressure = t / t_max if t_max > 0 else 0.0

        if last_context_tokens is not None and last_context_tokens > 0:
            context_pressure = last_context_tokens / b_max if b_max > 0 else 0.0
        elif fallback_messages is not None:
            # First-turn fallback: estimate |C_0| via watermark
            context_pressure = (
                self.watermark.estimate_tokens(fallback_messages) / b_max
                if b_max > 0
                else 0.0
            )
        else:
            context_pressure = 0.0

        P_t = max(turn_pressure, context_pressure)
        return min(P_t, 1.0)

    def compute_C0(
        self,
        system_prompt: str,
        user_query: str,
        tokenizer: Any = None,
    ) -> int:
        """Compute |C_0| — initial context token count.

        PACE step 7 fallback (M2 fix): uses the exact formula
        ``len(tokenizer.encode(system_prompt + query))``.
        Falls back to watermark char-based estimation if no tokenizer.
        """
        combined = system_prompt + user_query
        if tokenizer is not None:
            try:
                tokens = tokenizer.encode(combined)
                return len(tokens)
            except Exception:
                pass
        # Fallback: char-based estimate
        return max(1, int(len(combined) / self.watermark.chars_per_token))

    # ── Step 8: Threshold adaptation ──────────────────────────────────

    def adapt_thresholds(
        self,
        P_t: float,
        alpha_0: float | None = None,
        beta_0: float | None = None,
        gamma_0: float | None = None,
        lambda_adapt: float | None = None,
    ) -> tuple[float, float, float]:
        """Adapt granularity thresholds based on pressure.

        PACE step 8:
          α_t ← α_0 × (1 + λ·P_t)
          β_t ← β_0 × (1 + λ·P_t)
          γ_t ← γ_0 × (1 + λ·P_t)

        Higher pressure → higher thresholds → more aggressive compression
        (fewer chunks qualify for R_full/R_detailed).
        """
        a0 = alpha_0 if alpha_0 is not None else self.alpha_0
        b0 = beta_0 if beta_0 is not None else self.beta_0
        g0 = gamma_0 if gamma_0 is not None else self.gamma_0
        lam = lambda_adapt if lambda_adapt is not None else self.lambda_adapt

        factor = 1.0 + lam * P_t
        alpha_t = a0 * factor
        beta_t = b0 * factor
        gamma_t = g0 * factor
        return alpha_t, beta_t, gamma_t

    # ── Steps 5-6: Attention weights ──────────────────────────────────

    def compute_weights(self, scores: Any) -> Any:
        """Compute relative attention weights w̃_i.

        PACE steps 5-6:
          w_i ← softmax(s_i / τ)
          w̃_i ← M · w_i

        where M = number of chunks. This normalizes scores so the
        thresholds α, β, γ are comparable across different M values.
        """
        try:
            import numpy as np
        except ImportError:
            logger.warning("PACE compressor: numpy not available for weight computation")
            return []

        if len(scores) == 0:
            return np.array([], dtype=np.float32)

        # softmax(s_i / τ)
        scaled = scores / self.tau
        shifted = scaled - np.max(scaled)  # numerical stability
        exp_scores = np.exp(shifted)
        w = exp_scores / np.sum(exp_scores)

        # w̃_i = M · w_i
        M = len(scores)
        w_tilde = M * w
        return w_tilde

    # ── Step 9: Granularity selection ─────────────────────────────────

    def select_granularity(
        self,
        w_tilde: Any,
        alpha_t: float,
        beta_t: float,
        gamma_t: float,
    ) -> list[int]:
        """Select granularity level for each chunk based on attention weight.

        PACE step 9:
          w̃_i > γ_t → R_full    (level 3)
          β_t < w̃_i ≤ γ_t → R_detailed (level 2)
          α_t < w̃_i ≤ β_t → R_brief     (level 1)
          w̃_i ≤ α_t → R_ph      (level 0)

        Returns a list of integer levels, one per chunk.
        """
        try:
            import numpy as np
        except ImportError:
            return []

        if len(w_tilde) == 0:
            return []

        levels = np.where(
            w_tilde > gamma_t, LEVEL_FULL,
            np.where(
                w_tilde > beta_t, LEVEL_DETAILED,
                np.where(w_tilde > alpha_t, LEVEL_BRIEF, LEVEL_PH),
            ),
        )
        return levels.tolist()

    # ── Step 10: Message assembly ─────────────────────────────────────

    def assemble_messages(
        self,
        conversation_history: Sequence[dict],
        chunks: Sequence[Any],
        w_tilde: Any,
        thresholds: tuple[float, float, float],
        user_message: str,
        n_recent: int | None = None,
        force_compact: bool = False,
    ) -> list[dict]:
        """Assemble the compressed context C_t.

        PACE step 10:
          C_t ← assemble_messages(Q, {R*_i}, R_full^{(t-N:t-1)})
          Format: [system, ...compressed_history..., user_msg]

        - Preserves the system message (first message with role="system")
        - Keeps the last ``n_recent`` (default N=2) rounds as R_full
        - Selects granularity for all other chunks
        - Appends the current user message as the final message
        - ``force_compact=True``: downgrades every chunk by one level
          (R_full→R_detailed→R_brief→R_ph), but still protects N=2 recent

        All returned messages are dicts with a "role" key (format
        validation requirement, design §3.4).
        """
        n = n_recent if n_recent is not None else self.N
        alpha_t, beta_t, gamma_t = thresholds

        if not conversation_history and not chunks:
            return [{"role": "user", "content": user_message}]

        # 1. Extract system message (preserve verbatim)
        system_msg: dict | None = None
        system_idx = -1
        for i, msg in enumerate(conversation_history):
            if isinstance(msg, dict) and msg.get("role") == "system":
                system_msg = dict(msg)
                system_idx = i
                break

        # 2. Select granularity for all chunks via PACE scoring.
        #    The last n chunks are then forced to R_full — this is the
        #    L1 safety net (N=2 recent rounds always full-text, design §2.2).
        #    Using chunk.get_content(LEVEL_FULL) avoids double-counting with
        #    the raw conversation_history messages.
        levels = self.select_granularity(w_tilde, alpha_t, beta_t, gamma_t)

        if force_compact:
            # Downgrade historical levels by 1 (but not below LEVEL_PH=0).
            # Recent N=2 chunks are still protected (forced to R_full below).
            levels = [max(l - 1, LEVEL_PH) for l in levels]

        num_chunks = len(chunks)
        recent_start = max(0, num_chunks - n)  # index where recent window begins

        # 3. Build compressed history messages from chunks
        compressed_msgs: list[dict] = []
        for i, chunk in enumerate(chunks):
            if i < len(levels):
                level = levels[i]
            else:
                level = LEVEL_PH  # default to placeholder if no score

            # Recent N=2 safety net: force last n chunks to R_full
            if i >= recent_start:
                level = LEVEL_FULL

            # Glimpse safety net: if this chunk was glimpse-retrieved last
            # turn, keep it at R_full for one turn
            if hasattr(chunk, "glimpse_requested") and chunk.glimpse_requested:
                level = LEVEL_FULL

            content = chunk.get_content(level) if hasattr(chunk, "get_content") else str(chunk)

            # Alternate role for compressed history (keeps message structure valid)
            role = "user" if i % 2 == 0 else "assistant"
            # Prefix with granularity tag for debugging/transparency
            tag = {0: "R_ph", 1: "R_brief", 2: "R_detailed", 3: "R_full"}.get(level, "R_ph")
            compressed_msgs.append({
                "role": role,
                "content": f"[{tag}] {content}" if level < LEVEL_FULL else content,
            })

        # 4. Assemble final message list: [system, ...compressed_history..., user_msg]
        result: list[dict] = []
        if system_msg is not None:
            result.append(system_msg)
        result.extend(compressed_msgs)
        result.append({"role": "user", "content": user_message})

        # 6. Format validation: every element must be a dict with "role"
        validated = [
            m for m in result
            if isinstance(m, dict) and "role" in m
        ]
        return validated

    # ── Step 11: Token counting ───────────────────────────────────────

    def count_tokens(
        self,
        messages: Sequence[dict],
        tokenizer: Any = None,
    ) -> int:
        """Count tokens in the assembled message list.

        PACE step 11: compute |C_t| and cache to memory_store.
        Uses the tokenizer if available, otherwise falls back to
        watermark estimation.
        """
        if tokenizer is not None:
            try:
                total = 0
                for msg in messages:
                    content = msg.get("content", "") if isinstance(msg, dict) else ""
                    if isinstance(content, str):
                        total += len(tokenizer.encode(content))
                    elif isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict):
                                total += len(tokenizer.encode(str(part.get("text", ""))))
                    total += 4  # role overhead
                return total
            except Exception:
                pass
        return self.watermark.estimate_tokens(messages)
