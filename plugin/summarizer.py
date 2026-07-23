"""Async LLM summarization for PACE chunk granularity levels.

Generates R_detailed (~100 words) and R_brief (1-2 sentences) summaries
for conversation chunks using a cheap, fast LLM (DeepSeek Flash). Summaries
are generated asynchronously via ThreadPoolExecutor and written back to
the memory store when ready.

PACE Algorithm 1 step 14:
  Async trigger R_detailed, R_brief summary generation

Degradation (design §6.4):
  - Summary not ready → R_full (chunk returns full text)
  - API fail/timeout(30s) → R_ph (placeholder)
  - skip_threshold (w̃ < 0.1) → skip summary generation entirely

References:
  - PACE design §5.1 step 14
  - PACE design §6.3 async summarization pipeline
  - PACE design §6.4 degradation strategy
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable

logger = logging.getLogger(__name__)


class PaceSummarizer:
    """Async LLM summarizer for R_detailed and R_brief chunk levels.

    Uses a ThreadPoolExecutor to generate summaries without blocking the
    main PACE pipeline. Summaries are written back to the memory store
    via a done-callback on each future.

    The LLM client is lazily initialized on first use to avoid importing
    openai at module load time (keeps plugin load fast when summarization
    isn't needed).
    """

    MIN_TEXT_LENGTH = 20

    def __init__(
        self,
        provider: str = "deepseek",
        model: str = "deepseek-chat",
        fallback_provider: str = "zai",
        fallback_model: str = "qwen-plus",
        timeout: int = 30,
        max_workers: int = 2,
        skip_threshold: float = 0.1,
        detailed_target_words: int = 100,
        brief_target_sentences: int = 2,
    ) -> None:
        self.provider = provider
        self.model = model
        self.fallback_provider = fallback_provider
        self.fallback_model = fallback_model
        self.timeout = timeout
        self.max_workers = max_workers
        self.skip_threshold = skip_threshold
        self.detailed_target_words = detailed_target_words
        self.brief_target_sentences = brief_target_sentences
        self._executor: ThreadPoolExecutor | None = None
        self._client: Any = None
        self._client_initialized: bool = False
        self._pending: dict[str, list[Future]] = {}

    @property
    def executor(self) -> ThreadPoolExecutor:
        """Lazily initialize the thread pool."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix="pace-summarizer",
            )
        return self._executor

    def _get_client(self) -> Any:
        """Lazily initialize the OpenAI-compatible LLM client.

        Supports DeepSeek (primary) and ZAI (fallback) via their
        OpenAI-compatible API interfaces.
        """
        if self._client_initialized:
            return self._client
        self._client_initialized = True
        try:
            from openai import OpenAI

            if self.provider == "deepseek":
                api_key = os.environ.get("DEEPSEEK_API_KEY", "")
                base_url = "https://api.deepseek.com/v1"
            else:
                api_key = os.environ.get("ZAI_API_KEY", os.environ.get("DASHSCOPE_API_KEY", ""))
                base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

            if not api_key:
                logger.warning(
                    "PACE summarizer: no API key for provider '%s', summaries will be skipped",
                    self.provider,
                )
                return None

            self._client = OpenAI(api_key=api_key, base_url=base_url)
            logger.info("PACE summarizer: LLM client initialized (provider=%s, model=%s)", self.provider, self.model)
        except Exception as exc:
            logger.warning("PACE summarizer: client init failed: %s", exc)
            self._client = None
        return self._client

    def _call_llm(self, prompt: str, system: str = "") -> str | None:
        """Synchronous LLM call with timeout and fallback.

        Returns the generated text, or None on failure.
        """
        client = self._get_client()
        if client is None:
            return None

        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        for attempt_model in [self.model, self.fallback_model]:
            try:
                resp = client.chat.completions.create(
                    model=attempt_model,
                    messages=messages,
                    max_tokens=300,
                    temperature=0.3,
                    timeout=self.timeout,
                )
                text = resp.choices[0].message.content
                if text:
                    return text.strip()
            except Exception as exc:
                logger.warning(
                    "PACE summarizer: LLM call failed (model=%s): %s",
                    attempt_model,
                    exc,
                )
                if attempt_model == self.model:
                    # Try fallback model
                    try:
                        from openai import OpenAI

                        if self.fallback_provider == "zai":
                            api_key = os.environ.get("ZAI_API_KEY", os.environ.get("DASHSCOPE_API_KEY", ""))
                            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
                            if api_key:
                                fallback_client = OpenAI(api_key=api_key, base_url=base_url)
                                resp = fallback_client.chat.completions.create(
                                    model=self.fallback_model,
                                    messages=messages,
                                    max_tokens=300,
                                    temperature=0.3,
                                    timeout=self.timeout,
                                )
                                text = resp.choices[0].message.content
                                if text:
                                    return text.strip()
                    except Exception as exc2:
                        logger.warning("PACE summarizer: fallback also failed: %s", exc2)
        return None

    def generate_detailed(self, chunk_text: str) -> str | None:
        """Generate R_detailed (~100 words) summary.

        Called in a worker thread. Returns None on failure (chunk stays
        at R_full fallback per degradation rules). Returns None for very
        short texts (not worth summarizing).
        """
        if len(chunk_text) < self.MIN_TEXT_LENGTH:
            return None
        prompt = (
            f"Summarize the following conversation turn in approximately "
            f"{self.detailed_target_words} words. Preserve key decisions, "
            f"code changes, and action items.\n\n---\n{chunk_text[:4000]}\n---"
        )
        return self._call_llm(prompt, system="You are a precise conversation summarizer.")

    def generate_brief(self, chunk_text: str) -> str | None:
        """Generate R_brief (1-2 sentences) summary.

        Called in a worker thread. Returns None on failure. Returns None
        for very short texts.
        """
        if len(chunk_text) < self.MIN_TEXT_LENGTH:
            return None
        prompt = (
            f"In {self.brief_target_sentences} sentence(s), summarize the key "
            f"outcome of this conversation turn:\n\n---\n{chunk_text[:2000]}\n---"
        )
        return self._call_llm(prompt, system="You are a concise conversation summarizer.")

    def enqueue(
        self,
        chunk_id: str,
        chunk_text: str,
        writeback: Callable[[str, int, str], bool],
        levels: list[int] | None = None,
    ) -> Future | None:
        """Enqueue async summary generation for a chunk.

        PACE step 14: async trigger R_detailed and R_brief generation.

        The ``levels`` parameter specifies which granularity levels were
        selected for this chunk. Summary generation is skipped when all
        levels are PH (0) or FULL (3) — no summary needed.

        - levels containing 1 (BRIEF): generates both R_detailed and R_brief
          (R_detailed is a prerequisite for R_brief)
        - levels containing 2 (DETAILED): generates R_detailed only
        - levels containing only 0 or 3: skipped

        The ``writeback`` callback signature is:
            writeback(chunk_id: str, level: int, summary: str) -> bool
        where level 2 = R_detailed, level 1 = R_brief.

        Returns the Future for the combined task, or None if skipped.
        """
        if levels is None:
            levels = [2, 1]

        need_detailed = any(l in (1, 2) for l in levels)
        need_brief = 1 in levels

        if not need_detailed:
            logger.debug(
                "PACE summarizer: skipping chunk %s (levels=%s, no summary needed)",
                chunk_id, levels,
            )
            return None

        def _worker() -> None:
            if need_detailed:
                detailed = self.generate_detailed(chunk_text)
                if detailed:
                    writeback(chunk_id, 2, detailed)
            if need_brief:
                brief = self.generate_brief(chunk_text)
                if brief:
                    writeback(chunk_id, 1, brief)

        future = self.executor.submit(_worker)
        future.add_done_callback(self._done_callback)
        self._pending.setdefault(chunk_id, []).append(future)
        return future

    def _done_callback(self, future: Future) -> None:
        """Log exceptions from async summary tasks."""
        exc = future.exception()
        if exc:
            logger.warning("PACE summarizer: async summary task failed: %s", exc)

    def shutdown(self, wait: bool = True) -> None:
        """Clean shutdown of the thread pool."""
        if self._executor is not None:
            self._executor.shutdown(wait=wait, cancel_futures=not wait)
            self._executor = None


SummaryGenerator = PaceSummarizer
