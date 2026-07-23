"""Tests for SummaryGenerator async summary generation."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from summarizer import PaceSummarizer


class TestPaceSummarizerInit:
    def test_default_config(self):
        sg = PaceSummarizer()
        assert sg.provider == "deepseek"
        assert sg.model == "deepseek-chat"
        assert sg.timeout == 30
        assert sg.max_workers == 2
        assert sg.skip_threshold == 0.1

    def test_custom_config(self):
        sg = PaceSummarizer(provider="openai", model="gpt-4o", timeout=60, max_workers=4)
        assert sg.provider == "openai"
        assert sg.model == "gpt-4o"
        assert sg.timeout == 60


class TestGenerateDetailed:
    def test_short_text_returns_none(self):
        sg = PaceSummarizer()
        result = sg.generate_detailed("Hi")
        assert result is None

    def test_empty_text_returns_none(self):
        sg = PaceSummarizer()
        result = sg.generate_detailed("")
        assert result is None

    def test_calls_llm(self):
        sg = PaceSummarizer()
        with patch.object(sg, "_call_llm", return_value="Summary text"):
            result = sg.generate_detailed("This is a long enough chunk of text to summarize properly.")
            assert result == "Summary text"


class TestGenerateBrief:
    def test_short_text_returns_none(self):
        sg = PaceSummarizer()
        result = sg.generate_brief("Hi")
        assert result is None

    def test_calls_llm(self):
        sg = PaceSummarizer()
        with patch.object(sg, "_call_llm", return_value="Brief summary."):
            result = sg.generate_brief("This is a long enough chunk of text to summarize properly.")
            assert result == "Brief summary."


class TestEnqueue:
    def test_enqueue_skips_when_all_levels_ph(self):
        """skip_threshold: chunks with max_level < 1 are skipped."""
        sg = PaceSummarizer()
        writeback = MagicMock(return_value=True)
        sg.enqueue("chunk_1", "text", levels=[0], writeback=writeback)
        # No futures should be created
        assert "chunk_1" not in sg._pending or len(sg._pending.get("chunk_1", [])) == 0

    def test_enqueue_triggers_detailed_for_level_2(self):
        sg = PaceSummarizer()
        writeback = MagicMock(return_value=True)
        with patch.object(sg, "generate_detailed", return_value="Detailed"):
            with patch.object(sg, "generate_brief", return_value="Brief"):
                sg.enqueue("chunk_1", "A" * 100, levels=[2], writeback=writeback)
                # Wait for async completion
                if "chunk_1" in sg._pending:
                    for f in sg._pending["chunk_1"]:
                        f.result(timeout=5)
                time.sleep(0.5)  # allow callbacks to run
        # Detailed writeback should have been called
        assert writeback.called

    def test_enqueue_triggers_both_for_level_1(self):
        sg = PaceSummarizer()
        writeback = MagicMock(return_value=True)
        with patch.object(sg, "generate_detailed", return_value="Detailed"):
            with patch.object(sg, "generate_brief", return_value="Brief"):
                sg.enqueue("chunk_1", "A" * 100, levels=[1], writeback=writeback)
                if "chunk_1" in sg._pending:
                    for f in sg._pending["chunk_1"]:
                        f.result(timeout=5)
                time.sleep(0.5)
        # Both detailed and brief should be triggered
        assert writeback.call_count >= 2


class TestShutdown:
    def test_shutdown_doesnt_crash(self):
        sg = PaceSummarizer()
        sg.shutdown(wait=False)
        # Should not raise
