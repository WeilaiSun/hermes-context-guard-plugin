"""Tests for PACEMemoryStore — chunk storage, LRU eviction, persistence.

Tests cover:
  - add_chunk / get_chunk basic operations
  - LRU eviction at max_chunks
  - Multi-granularity content retrieval (R_full, R_detailed, R_brief, R_ph)
  - Key embedding caching
  - Summary writeback
  - Glimpse flag management
  - JSON save/load round-trip
  - Per-session isolation
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from memory_store import Chunk, PACEMemoryStore


@pytest.fixture
def store():
    return PACEMemoryStore(max_chunks=5, persist_path=tempfile.mktemp(suffix=".json"))


@pytest.fixture
def filled_store(store):
    """Store with 3 chunks for session 's1'."""
    for i in range(3):
        store.add_chunk(
            session_id="s1",
            turn_index=i,
            r_full=f"Turn {i} full text content",
            key_embedding=[float(i)] * 4,
        )
    return store


class TestAddChunk:
    def test_basic_add(self, store):
        chunk = store.add_chunk("s1", 0, "hello", [1.0, 0.0])
        assert chunk.chunk_id == "chunk_1"
        assert chunk.session_id == "s1"
        assert chunk.turn_index == 0
        assert chunk.r_full == "hello"
        assert chunk.key_embedding == [1.0, 0.0]
        assert len(store._chunks) == 1

    def test_incremental_ids(self, store):
        c1 = store.add_chunk("s1", 0, "a")
        c2 = store.add_chunk("s1", 1, "b")
        c3 = store.add_chunk("s1", 2, "c")
        assert c1.chunk_id == "chunk_1"
        assert c2.chunk_id == "chunk_2"
        assert c3.chunk_id == "chunk_3"


class TestLRUEviction:
    def test_eviction_at_max(self, store):
        # max_chunks=5
        for i in range(5):
            store.add_chunk("s1", i, f"turn {i}")
        assert len(store._chunks) == 5

        # Adding 6th should evict the oldest
        store.add_chunk("s1", 5, "turn 5")
        assert len(store._chunks) == 5
        # chunk_1 should be evicted
        assert "chunk_1" not in store._chunks
        assert "chunk_6" in store._chunks

    def test_lru_order_on_get(self):
        store = PACEMemoryStore(max_chunks=3, persist_path=tempfile.mktemp(suffix=".json"))
        c1 = store.add_chunk("s1", 0, "first")
        c2 = store.add_chunk("s1", 1, "second")
        c3 = store.add_chunk("s1", 2, "third")

        # Access c1 → moves to end (most recently used)
        store.get_chunk("chunk_1")

        # Add c4 → should evict c2 (least recently used now)
        store.add_chunk("s1", 3, "fourth")
        assert "chunk_1" in store._chunks  # c1 was accessed, not evicted
        assert "chunk_2" not in store._chunks  # c2 is evicted
        assert "chunk_3" in store._chunks


class TestGetChunk:
    def test_get_full(self, filled_store):
        content = filled_store.get_chunk("chunk_1", level=3)
        assert "Turn 0 full text content" in content

    def test_get_detailed_fallback(self, filled_store):
        # R_detailed not generated yet → fallback to R_full
        content = filled_store.get_chunk("chunk_1", level=2)
        assert "Turn 0 full text content" in content

    def test_get_brief_fallback(self, filled_store):
        content = filled_store.get_chunk("chunk_1", level=1)
        assert "Turn 0 full text content" in content

    def test_get_placeholder(self, filled_store):
        content = filled_store.get_chunk("chunk_1", level=0)
        assert "[R_ph:" in content
        assert "chunk_1" in content

    def test_get_nonexistent(self, filled_store):
        content = filled_store.get_chunk("nonexistent", level=3)
        assert content is None


class TestGetKeys:
    def test_get_all_keys(self, filled_store):
        keys = filled_store.get_keys()
        assert len(keys) == 3
        assert all(len(k) == 4 for k in keys)

    def test_get_keys_filtered_by_session(self, filled_store):
        filled_store.add_chunk("s2", 0, "other session", [0.0, 0.0])
        keys = filled_store.get_keys(session_id="s1")
        assert len(keys) == 3  # only s1 chunks

    def test_get_keys_excludes_none(self, store):
        store.add_chunk("s1", 0, "no embedding", key_embedding=None)
        store.add_chunk("s1", 1, "has embedding", key_embedding=[1.0])
        keys = store.get_keys()
        assert len(keys) == 1


class TestSetSummary:
    def test_set_detailed(self, filled_store):
        result = filled_store.set_summary("chunk_1", 2, "Detailed summary text")
        assert result is True
        chunk = filled_store.get_chunk_obj("chunk_1")
        assert chunk.r_detailed == "Detailed summary text"

    def test_set_brief(self, filled_store):
        result = filled_store.set_summary("chunk_2", 1, "Brief summary")
        assert result is True
        chunk = filled_store.get_chunk_obj("chunk_2")
        assert chunk.r_brief == "Brief summary"

    def test_set_nonexistent_chunk(self, filled_store):
        result = filled_store.set_summary("nonexistent", 2, "text")
        assert result is False

    def test_summary_used_in_get_content(self, filled_store):
        filled_store.set_summary("chunk_1", 2, "Detailed version")
        filled_store.set_summary("chunk_1", 1, "Brief version")
        assert filled_store.get_chunk("chunk_1", level=2) == "Detailed version"
        assert filled_store.get_chunk("chunk_1", level=1) == "Brief version"


class TestGlimpseFlags:
    def test_mark_glimpse(self, filled_store):
        filled_store.mark_glimpse("chunk_1")
        chunk = filled_store.get_chunk_obj("chunk_1")
        assert chunk.glimpse_requested is True

    def test_clear_glimpse_all(self, filled_store):
        filled_store.mark_glimpse("chunk_1")
        filled_store.mark_glimpse("chunk_2")
        filled_store.clear_glimpse_flags()
        for chunk in filled_store._chunks.values():
            assert chunk.glimpse_requested is False

    def test_clear_glimpse_by_session(self, filled_store):
        filled_store.add_chunk("s2", 0, "other", [0.0])
        filled_store.mark_glimpse("chunk_1")
        filled_store.mark_glimpse("chunk_4")

        filled_store.clear_glimpse_flags(session_id="s1")
        assert filled_store.get_chunk_obj("chunk_1").glimpse_requested is False
        assert filled_store.get_chunk_obj("chunk_4").glimpse_requested is True


class TestPersistence:
    def test_save_load_roundtrip(self, filled_store):
        # Add summaries
        filled_store.set_summary("chunk_1", 2, "Detailed")
        filled_store.set_summary("chunk_1", 1, "Brief")
        filled_store.last_context_tokens = 5000

        # Save
        assert filled_store.save() is True

        # Load into new store
        new_store = PACEMemoryStore(persist_path=filled_store.persist_path)
        assert new_store.load() is True
        assert len(new_store._chunks) == 3
        assert new_store.last_context_tokens == 5000

        chunk = new_store.get_chunk_obj("chunk_1")
        assert chunk is not None
        assert chunk.r_full == "Turn 0 full text content"
        assert chunk.r_detailed == "Detailed"
        assert chunk.r_brief == "Brief"

    def test_load_nonexistent_file(self, store):
        assert store.load("/nonexistent/path.json") is False

    def test_save_creates_directory(self, store):
        path = os.path.join(tempfile.mkdtemp(), "subdir", "memory.json")
        store.persist_path = path
        assert store.save() is True
        assert os.path.exists(path)


class TestReset:
    def test_reset_clears_all(self, filled_store):
        filled_store.reset()
        assert len(filled_store._chunks) == 0
        assert filled_store.last_context_tokens is None
        assert filled_store._counter == 0


class TestSessionIsolation:
    def test_get_chunks_filtered(self, store):
        store.add_chunk("s1", 0, "a")
        store.add_chunk("s2", 0, "b")
        store.add_chunk("s1", 1, "c")

        s1_chunks = store.get_chunks(session_id="s1")
        s2_chunks = store.get_chunks(session_id="s2")
        assert len(s1_chunks) == 2
        assert len(s2_chunks) == 1
