"""Chunk persistence and embedding cache for PACE compression.

Stores conversation chunks (one per turn) with their BGE-M3 key embeddings.
Provides multi-granularity retrieval (R_full, R_detailed, R_brief, R_ph)
and caches the last compressed context token count (|C_t|) for pressure
calculation in the next turn.

PACE Algorithm 1 steps 2-3, 7, 11, 13, 15:
  2.  M ← len(memory_store.chunks)
  3.  Read cached {k_i}_{i=1}^M
  7.  |C_{t-1}| ← memory_store.last_context_tokens
  11. Cache |C_t| → memory_store.last_context_tokens
  13. Create Chunk_t, k_t ← BGE-M3-Encode(Trunc(R_full^{(t)}, L_max))
  15. M_t ← M_{t-1} ∪ {Chunk_t}

References:
  - PACE design §5.1, §6.2
  - LRU eviction: max_chunks=200 (design §7 Phase 1)
  - JSON persistence for session recovery
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import OrderedDict
from typing import Any

logger = logging.getLogger(__name__)


class Chunk:
    """A single conversation turn stored in the PACE memory store.

    Each chunk stores:
      - ``r_full``: the original full conversation text for this turn
      - ``key_embedding``: BGE-M3 encoding of the truncated R_full
      - ``r_detailed``: ~100-word summary (lazy, async-generated)
      - ``r_brief``: 1-2 sentence summary (lazy, async-generated)
      - ``glimpse_requested``: flag set when Glimpse retrieves this chunk
        (next turn keeps it at R_full as a safety net)

    Granularity levels:
      0 = R_ph (placeholder), 1 = R_brief, 2 = R_detailed, 3 = R_full
    """

    __slots__ = (
        "chunk_id",
        "session_id",
        "turn_index",
        "r_full",
        "key_embedding",
        "r_detailed",
        "r_brief",
        "glimpse_requested",
        "created_at",
    )

    def __init__(
        self,
        chunk_id: str,
        session_id: str,
        turn_index: int,
        r_full: str,
        key_embedding: Any = None,
    ) -> None:
        self.chunk_id = chunk_id
        self.session_id = session_id
        self.turn_index = turn_index
        self.r_full = r_full
        self.key_embedding = key_embedding
        self.r_detailed: str | None = None
        self.r_brief: str | None = None
        self.glimpse_requested: bool = False
        self.created_at = time.time()

    def get_content(self, level: int) -> str:
        """Return chunk content at the requested granularity level.

        Level 3 → R_full, 2 → R_detailed (fallback R_full if not ready),
        1 → R_brief (fallback R_full if not ready), 0 → R_ph placeholder.

        Degradation rule (design §6.4): if a summary is not yet generated,
        fall back to R_full rather than losing information.
        """
        if level >= 3:
            return self.r_full
        if level == 2:
            return self.r_detailed if self.r_detailed else self.r_full
        if level == 1:
            return self.r_brief if self.r_brief else self.r_full
        # level 0: placeholder
        return self._placeholder()

    def _placeholder(self) -> str:
        """Generate a compact placeholder for this chunk (R_ph).

        Preserves a pointer to the chunk so it can be recovered via Glimpse.
        """
        snippet = self.r_full[:60].replace("\n", " ").strip()
        return f"[R_ph: chunk {self.chunk_id}, turn {self.turn_index}, preview: {snippet}...]"

    def to_dict(self) -> dict:
        """Serialize chunk for JSON persistence (excludes embedding)."""
        return {
            "chunk_id": self.chunk_id,
            "session_id": self.session_id,
            "turn_index": self.turn_index,
            "r_full": self.r_full,
            "r_detailed": self.r_detailed,
            "r_brief": self.r_brief,
            "glimpse_requested": self.glimpse_requested,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        chunk = cls(
            chunk_id=data["chunk_id"],
            session_id=data.get("session_id", ""),
            turn_index=data.get("turn_index", 0),
            r_full=data.get("r_full", ""),
        )
        chunk.r_detailed = data.get("r_detailed")
        chunk.r_brief = data.get("r_brief")
        chunk.glimpse_requested = data.get("glimpse_requested", False)
        chunk.created_at = data.get("created_at", time.time())
        return chunk


class PACEMemoryStore:
    """PACE chunk store with LRU eviction and |C_t| token caching.

    Chunks are stored in an OrderedDict keyed by chunk_id. Access (get)
    moves the chunk to the end (most recently used). When ``max_chunks``
    is exceeded, the oldest (least recently used) chunk is evicted.

    Per-session isolation: chunks are filtered by ``session_id`` so
    concurrent sessions don't mix data (design risk #9).
    """

    def __init__(
        self,
        max_chunks: int = 200,
        persist_path: str | None = None,
    ) -> None:
        self.max_chunks = max_chunks
        self.persist_path = persist_path or os.path.join(
            os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")),
            "context-guard-memory.json",
        )
        self._chunks: OrderedDict[str, Chunk] = OrderedDict()
        self.last_context_tokens: int | None = None
        self._counter: int = 0

    @property
    def chunks(self) -> OrderedDict[str, Chunk]:
        """Return the chunk OrderedDict (LRU order)."""
        return self._chunks

    def get_keys(self, session_id: str | None = None) -> list[Any]:
        """Return cached key embeddings for all chunks.

        PACE step 3: read {k_i}_{i=1}^M from memory store.
        Optionally filter by session_id for multi-session isolation.
        """
        keys: list[Any] = []
        for chunk in self._chunks.values():
            if session_id and chunk.session_id != session_id:
                continue
            if chunk.key_embedding is not None:
                keys.append(chunk.key_embedding)
        return keys

    def get_chunks(self, session_id: str | None = None) -> list[Chunk]:
        """Return all chunks, optionally filtered by session_id."""
        if session_id is None:
            return list(self._chunks.values())
        return [c for c in self._chunks.values() if c.session_id == session_id]

    def get_chunk(self, chunk_id: str, level: int = 3) -> str | None:
        """Get chunk content at the specified granularity.

        PACE step 9: R*_i ← Select(Chunk_i, w̃_i, α_t, β_t, γ_t).
        Also marks the chunk as glimpse_requested if level < 3 (compressed).

        Returns None if chunk_id is not found.
        """
        chunk = self._chunks.get(chunk_id)
        if chunk is None:
            return None
        # LRU: move to end (most recently used)
        self._chunks.move_to_end(chunk_id)
        return chunk.get_content(level)

    def get_chunk_obj(self, chunk_id: str) -> Chunk | None:
        """Return the raw Chunk object (for Glimpse full-text retrieval)."""
        chunk = self._chunks.get(chunk_id)
        if chunk is not None:
            self._chunks.move_to_end(chunk_id)
        return chunk

    def add_chunk(
        self,
        session_id: str,
        turn_index: int,
        r_full: str,
        key_embedding: Any = None,
    ) -> Chunk:
        """Create and store a new chunk.

        PACE steps 13, 15: create Chunk_t and add to M_t.
        Triggers LRU eviction if over max_chunks.

        Returns the created Chunk.
        """
        self._counter += 1
        chunk_id = f"chunk_{self._counter}"
        chunk = Chunk(
            chunk_id=chunk_id,
            session_id=session_id,
            turn_index=turn_index,
            r_full=r_full,
            key_embedding=key_embedding,
        )
        self._chunks[chunk_id] = chunk

        # LRU eviction
        while len(self._chunks) > self.max_chunks:
            evicted_id, evicted = self._chunks.popitem(last=False)
            logger.debug(
                "PACE memory_store: LRU evicted chunk %s (turn %d)",
                evicted_id,
                evicted.turn_index,
            )

        logger.debug(
            "PACE memory_store: added chunk %s (turn %d, total=%d)",
            chunk_id,
            turn_index,
            len(self._chunks),
        )
        return chunk

    def set_summary(self, chunk_id: str, level: int, summary: str) -> bool:
        """Write back an async-generated summary to a chunk.

        Called by the summarizer's future.done_callback.
        level 2 → r_detailed, level 1 → r_brief.
        """
        chunk = self._chunks.get(chunk_id)
        if chunk is None:
            return False
        if level == 2:
            chunk.r_detailed = summary
        elif level == 1:
            chunk.r_brief = summary
        logger.debug(
            "PACE memory_store: summary written back for chunk %s (level=%d)",
            chunk_id,
            level,
        )
        return True

    def mark_glimpse(self, chunk_id: str) -> None:
        """Mark a chunk as glimpse-retrieved (next turn keeps R_full)."""
        chunk = self._chunks.get(chunk_id)
        if chunk is not None:
            chunk.glimpse_requested = True

    def clear_glimpse_flags(self, session_id: str | None = None) -> None:
        """Reset glimpse flags after a turn (so R_full only lasts one turn)."""
        for chunk in self._chunks.values():
            if session_id is None or chunk.session_id == session_id:
                chunk.glimpse_requested = False

    def save(self, path: str | None = None) -> bool:
        """Persist chunks and metadata to JSON.

        Key embeddings are not serialized (they're regenerated from r_full
        on load). Only stores r_full, summaries, and metadata.
        """
        save_path = path or self.persist_path
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            data = {
                "version": "2.3",
                "last_context_tokens": self.last_context_tokens,
                "counter": self._counter,
                "chunks": [c.to_dict() for c in self._chunks.values()],
            }
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.debug("PACE memory_store: saved %d chunks to %s", len(self._chunks), save_path)
            return True
        except Exception as exc:
            logger.warning("PACE memory_store: save failed: %s", exc)
            return False

    def load(self, path: str | None = None) -> bool:
        """Load chunks and metadata from JSON.

        Key embeddings are NOT loaded (they need re-encoding). The scorer
        will re-encode keys lazily when get_keys() is called. For sessions
        that don't need history recovery, this is a no-op.
        """
        load_path = path or self.persist_path
        if not os.path.exists(load_path):
            return False
        try:
            with open(load_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._chunks.clear()
            self._counter = data.get("counter", 0)
            self.last_context_tokens = data.get("last_context_tokens")
            for chunk_data in data.get("chunks", []):
                chunk = Chunk.from_dict(chunk_data)
                self._chunks[chunk.chunk_id] = chunk
            logger.info(
                "PACE memory_store: loaded %d chunks from %s",
                len(self._chunks),
                load_path,
            )
            return True
        except Exception as exc:
            logger.warning("PACE memory_store: load failed: %s", exc)
            return False

    def reset(self) -> None:
        """Clear all chunks and reset state (for session reset)."""
        self._chunks.clear()
        self.last_context_tokens = None
        self._counter = 0
