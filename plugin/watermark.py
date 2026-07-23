"""Context water-level monitoring and boundary guard.

Provides token estimation and boundary checking for the PACE compression
pipeline. Used as a fallback for |C_0| initialization (first turn, before
any compressed context exists) and for ongoing context-growth tracking.

References:
  - PACE design §5.1 step 7: |C_0| fallback via watermark
  - PACE design §8.1: boundary guard (>95% critical, >97% hard limit)
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Sequence

logger = logging.getLogger(__name__)


class Watermark:
    """Track context token growth across turns and enforce boundary guards.

    The watermark is an auxiliary monitor — PACE's primary pressure signal
    is ``memory_store.last_context_tokens`` (the exact |C_t| cached after
    each ``assemble_messages``). This class provides the fallback estimate
    for the first turn and a boundary checker for safety warnings.
    """

    def __init__(
        self,
        chars_per_token: float = 3.5,
        warn_ratio: float = 0.8,
        critical_ratio: float = 0.95,
        hard_limit_ratio: float = 0.97,
        B_max: int = 128000,
    ) -> None:
        self.chars_per_token = chars_per_token
        self.warn_ratio = warn_ratio
        self.critical_ratio = critical_ratio
        self.hard_limit_ratio = hard_limit_ratio
        self.B_max = B_max
        self._history: deque[int] = deque(maxlen=200)

    def estimate_tokens(self, messages: Sequence[dict]) -> int:
        """Approximate token count for a message list.

        Uses a simple chars/chars_per_token heuristic. Good enough for
        pressure estimation fallback; the exact count comes from the
        tokenizer in ``compressor.compute_C0``.
        """
        total_chars = 0
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        total_chars += len(str(part.get("text", "")))
            # role label overhead
            total_chars += len(str(msg.get("role", ""))) + 4
        return max(1, int(total_chars / self.chars_per_token))

    def check_boundary(self, token_count: int, B_max: int | None = None) -> str:
        """Classify context pressure against the boundary limits.

        Returns one of: ``"normal"``, ``"warn"``, ``"critical"``,
        ``"hard_limit"``.
        """
        limit = B_max or self.B_max
        if limit <= 0:
            return "normal"
        ratio = token_count / limit
        if ratio >= self.hard_limit_ratio:
            return "hard_limit"
        if ratio >= self.critical_ratio:
            return "critical"
        if ratio >= self.warn_ratio:
            return "warn"
        return "normal"

    def update(self, token_count: int) -> None:
        """Record the context token count for this turn."""
        self._history.append(token_count)

    @property
    def history(self) -> list[int]:
        return list(self._history)

    @property
    def last_token_count(self) -> int:
        return self._history[-1] if self._history else 0
