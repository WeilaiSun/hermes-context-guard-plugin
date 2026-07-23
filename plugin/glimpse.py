"""Glimpse tool — on-demand full-text chunk retrieval for PACE.

Provides the agent with a tool to "glimpse" (read the full content of)
chunks that were compressed to R_brief or R_ph during context compression.
This is the recovery mechanism described in PACE design §3.5: when the
agent needs detail from a compressed chunk, it calls pace_glimpse with
the chunk IDs visible in the R_ph placeholders.

Constraints (design §5.2 + §7):
  - Maximum 3 glimpses per agent step (prevent context re-inflation)
  - After glimpsing a chunk, the next turn keeps it at R_full (safety net)
  - Returns full R_full content for the requested chunks

References:
  - PACE design §3.5 Glimpse mechanism
  - PACE design §5.2: Glimpse ≤ 3/step
  - PACE design §7 Phase 1: max_per_step=3
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Schema for the pace_glimpse tool (registered via ctx.register_tool)
GLIMPSE_TOOL_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "chunk_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Chunk IDs to retrieve full text for. These IDs appear in "
                "R_ph placeholders like [R_ph: chunk_42, turn 5, ...]. "
                "Maximum 3 chunks per call."
            ),
        }
    },
    "required": ["chunk_ids"],
}

GLIMPSE_TOOL_DESCRIPTION = (
    "Retrieve the full (uncompressed) content of conversation chunks that "
    "were compressed to brief or placeholder level by PACE. Use when you "
    "need detail from a previous turn that was summarized. Maximum 3 chunks "
    "per step."
)

GLIMPSE_TOOL_EMOJI = "🔍"


class GlimpseTool:
    """Stateful Glimpse tool handler with per-step rate limiting.

    The tool handler is a callable that accepts the chunk_ids parameter
    and returns a JSON string with the full chunk contents. Rate limiting
    is enforced per step (resets between agent turns).
    """

    def __init__(
        self,
        memory_store: Any,
        max_per_step: int = 3,
    ) -> None:
        self.memory_store = memory_store
        self.max_per_step = max_per_step
        self._step_count: int = 0
        self._current_turn: str | None = None

    def reset_step(self, turn_id: str | None = None) -> None:
        """Reset the per-step glimpse counter.

        Called by the plugin's pre_llm_call hook at the end of each turn.
        """
        self._step_count = 0
        self._current_turn = turn_id

    def __call__(self, chunk_ids: list[str] | None = None, **kwargs: Any) -> str:
        """Handle a pace_glimpse tool call.

        Returns a JSON string with chunk contents or an error message.
        Marks glimpsed chunks so the next turn keeps them at R_full.
        """
        if chunk_ids is None:
            chunk_ids = []

        if not isinstance(chunk_ids, list):
            return json.dumps({"error": "chunk_ids must be a list of strings"})

        # Rate limit: max_per_step per agent step
        remaining = self.max_per_step - self._step_count
        if remaining <= 0:
            return json.dumps({
                "error": f"Rate limit: maximum {self.max_per_step} glimpses per step. "
                         f"Wait for the next turn.",
                "glimpsed_this_step": self._step_count,
            })

        # Enforce limit
        requested = len(chunk_ids)
        allowed_ids = chunk_ids[:remaining]
        if len(allowed_ids) < requested:
            logger.info(
                "PACE glimpse: requested %d chunks but only %d remaining this step",
                requested,
                remaining,
            )

        results: list[dict] = []
        for chunk_id in allowed_ids:
            chunk = self.memory_store.get_chunk_obj(chunk_id)
            if chunk is None:
                results.append({
                    "chunk_id": chunk_id,
                    "error": "chunk not found",
                })
                continue

            # Mark for next-turn R_full safety net
            self.memory_store.mark_glimpse(chunk_id)

            results.append({
                "chunk_id": chunk_id,
                "turn_index": chunk.turn_index,
                "content": chunk.r_full,
            })
            self._step_count += 1

        response = {
            "chunks": results,
            "glimpsed_this_step": self._step_count,
            "remaining_this_step": max(0, self.max_per_step - self._step_count),
        }
        return json.dumps(response, ensure_ascii=False, indent=2)
