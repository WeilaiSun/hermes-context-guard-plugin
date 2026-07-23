"""Tests for PACECompressor — granularity selection and message assembly."""

from __future__ import annotations

from context_guard__plugin.compressor import PACECompressor, LEVEL_PH, LEVEL_BRIEF, LEVEL_DETAILED, LEVEL_FULL
from context_guard__plugin.memory_store import Chunk


def make_chunk(cid: str, turn: int, text: str) -> Chunk:
    return Chunk(chunk_id=cid, session_id="s1", turn_index=turn, r_full=text)


class TestSelectGranularity:
    def test_empty_weights_returns_empty(self):
        comp = PACECompressor()
        levels = comp.select_granularity([], 0.4, 0.8, 1.5)
        assert levels == []

    def test_high_weight_gets_full(self):
        """w̃_i > γ → LEVEL_FULL"""
        comp = PACECompressor()
        try:
            import numpy as np
        except ImportError:
            return
        w = np.array([2.0])  # > γ=1.5
        levels = comp.select_granularity(w, 0.4, 0.8, 1.5)
        assert levels[0] == LEVEL_FULL

    def test_medium_weight_gets_detailed(self):
        """β < w̃_i ≤ γ → LEVEL_DETAILED"""
        comp = PACECompressor()
        try:
            import numpy as np
        except ImportError:
            return
        w = np.array([1.0])  # 0.8 < 1.0 ≤ 1.5
        levels = comp.select_granularity(w, 0.4, 0.8, 1.5)
        assert levels[0] == LEVEL_DETAILED

    def test_low_weight_gets_brief(self):
        """α < w̃_i ≤ β → LEVEL_BRIEF"""
        comp = PACECompressor()
        try:
            import numpy as np
        except ImportError:
            return
        w = np.array([0.5])  # 0.4 < 0.5 ≤ 0.8
        levels = comp.select_granularity(w, 0.4, 0.8, 1.5)
        assert levels[0] == LEVEL_BRIEF

    def test_very_low_weight_gets_placeholder(self):
        """w̃_i ≤ α → LEVEL_PH"""
        comp = PACECompressor()
        try:
            import numpy as np
        except ImportError:
            return
        w = np.array([0.1])  # ≤ 0.4
        levels = comp.select_granularity(w, 0.4, 0.8, 1.5)
        assert levels[0] == LEVEL_PH


class TestComputeWeights:
    def test_empty_scores_returns_empty(self):
        comp = PACECompressor()
        try:
            import numpy as np
        except ImportError:
            return
        w = comp.compute_weights(np.array([]))
        assert len(w) == 0

    def test_weights_sum_to_M(self):
        """w̃_i = M · w_i, and sum(w_i) = 1, so sum(w̃_i) = M"""
        comp = PACECompressor()
        try:
            import numpy as np
        except ImportError:
            return
        scores = np.array([0.9, 0.5, 0.1], dtype=np.float32)
        w = comp.compute_weights(scores)
        M = len(scores)
        assert abs(np.sum(w) - M) < 0.01

    def test_higher_score_gets_higher_weight(self):
        comp = PACECompressor()
        try:
            import numpy as np
        except ImportError:
            return
        scores = np.array([0.9, 0.1], dtype=np.float32)
        w = comp.compute_weights(scores)
        assert w[0] > w[1]


class TestAssembleMessages:
    def test_empty_history_returns_user_only(self):
        comp = PACECompressor()
        result = comp.assemble_messages(
            conversation_history=[],
            chunks=[],
            w_tilde=[],
            thresholds=(0.4, 0.8, 1.5),
            user_message="Hello",
        )
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello"

    def test_system_message_preserved(self):
        comp = PACECompressor()
        history = [{"role": "system", "content": "You are helpful."}]
        result = comp.assemble_messages(
            conversation_history=history,
            chunks=[],
            w_tilde=[],
            thresholds=(0.4, 0.8, 1.5),
            user_message="Hi",
        )
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are helpful."

    def test_all_messages_are_dicts_with_role(self):
        """Format validation: every element must be a dict with 'role' key."""
        comp = PACECompressor()
        chunks = [make_chunk("c1", 0, "Turn 0"), make_chunk("c2", 1, "Turn 1")]
        try:
            import numpy as np
            w = np.array([0.5, 0.5])
        except ImportError:
            w = [0.5, 0.5]
        result = comp.assemble_messages(
            conversation_history=[{"role": "system", "content": "sys"}],
            chunks=chunks,
            w_tilde=w,
            thresholds=(0.4, 0.8, 1.5),
            user_message="Question",
        )
        for msg in result:
            assert isinstance(msg, dict)
            assert "role" in msg

    def test_recent_n_chunks_forced_to_full(self):
        """Last N=2 chunks should be at R_full level (no [R_xxx] tag)."""
        comp = PACECompressor(N=2)
        chunks = [
            make_chunk("c1", 0, "Turn 0"),
            make_chunk("c2", 1, "Turn 1"),
            make_chunk("c3", 2, "Turn 2"),
            make_chunk("c4", 3, "Turn 3"),
        ]
        try:
            import numpy as np
            w = np.array([0.1, 0.1, 0.1, 0.1])  # all low → PH
        except ImportError:
            return
        result = comp.assemble_messages(
            conversation_history=[],
            chunks=chunks,
            w_tilde=w,
            thresholds=(0.4, 0.8, 1.5),
            user_message="Q",
        )
        # Last 2 chunks should NOT have [R_ph] prefix
        chunk_msgs = [m for m in result if m["role"] in ("user", "assistant") and m["content"] != "Q"]
        # The last 2 chunk messages should not have [R_ph] tag
        non_ph_count = sum(1 for m in chunk_msgs if not m["content"].startswith("[R_ph"))
        assert non_ph_count >= 2

    def test_force_compact_downgrades_levels(self):
        comp = PACECompressor(N=0)  # no recent protection for this test
        chunks = [make_chunk("c1", 0, "Turn 0")]
        try:
            import numpy as np
            w = np.array([2.0])  # would be FULL
        except ImportError:
            return
        result_normal = comp.assemble_messages(
            conversation_history=[], chunks=chunks, w_tilde=w,
            thresholds=(0.4, 0.8, 1.5), user_message="Q", force_compact=False,
        )
        result_compact = comp.assemble_messages(
            conversation_history=[], chunks=chunks, w_tilde=w,
            thresholds=(0.4, 0.8, 1.5), user_message="Q", force_compact=True,
        )
        # In compact mode, FULL→DETAILED, so content should have [R_detailed] tag
        normal_chunk = [m for m in result_normal if m["content"] != "Q"][0]
        compact_chunk = [m for m in result_compact if m["content"] != "Q"][0]
        assert not normal_chunk["content"].startswith("[R_")  # FULL has no tag
        assert compact_chunk["content"].startswith("[R_detailed]")  # downgraded


class TestCountTokens:
    def test_count_tokens_returns_positive(self):
        comp = PACECompressor()
        msgs = [{"role": "user", "content": "Hello world"}]
        count = comp.count_tokens(msgs)
        assert count > 0

    def test_count_tokens_proportional_to_content(self):
        comp = PACECompressor()
        short = [{"role": "user", "content": "Hi"}]
        long = [{"role": "user", "content": "x" * 1000}]
        assert comp.count_tokens(long) > comp.count_tokens(short)
