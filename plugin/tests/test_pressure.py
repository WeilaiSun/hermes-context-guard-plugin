"""Tests for pressure computation — focused unit tests for compute_pressure.

These tests isolate the pressure formula P_t = max(t/T_max, |C_{t-1}|/B_max)
to verify edge cases and boundary behavior.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from compressor import PACECompressor


@pytest.fixture
def comp():
    return PACECompressor(T_max=200, B_max=128000)


class TestPressureBoundaries:
    def test_zero_turn_zero_context(self, comp):
        P = comp.compute_pressure(t=0, last_context_tokens=0)
        assert P == 0.0

    def test_zero_turn_none_context(self, comp):
        P = comp.compute_pressure(t=0, last_context_tokens=None)
        assert P == 0.0

    def test_first_turn_with_fallback(self, comp):
        msgs = [{"role": "system", "content": "x" * 3500}]  # ~1000 tokens
        P = comp.compute_pressure(t=0, last_context_tokens=None, fallback_messages=msgs)
        assert 0 < P < 0.1  # ~1000/128000 ≈ 0.0078

    def test_pressure_never_exceeds_1(self, comp):
        P = comp.compute_pressure(t=500, last_context_tokens=500000)
        assert P == 1.0

    def test_pressure_is_in_0_1_range(self, comp):
        for t in range(0, 250, 10):
            for ctx in [0, 1000, 50000, 100000, 128000, 200000]:
                P = comp.compute_pressure(t=t, last_context_tokens=ctx)
                assert 0.0 <= P <= 1.0


class TestPressureFormula:
    def test_turn_dominates(self, comp):
        # t/T_max = 100/200 = 0.5, |C|/B_max = 1000/128000 ≈ 0.008
        P = comp.compute_pressure(t=100, last_context_tokens=1000)
        assert P == pytest.approx(0.5, abs=0.001)

    def test_context_dominates(self, comp):
        # t/T_max = 1/200 = 0.005, |C|/B_max = 100000/128000 ≈ 0.781
        P = comp.compute_pressure(t=1, last_context_tokens=100000)
        assert P == pytest.approx(100000 / 128000, abs=0.001)

    def test_equal_contributions(self, comp):
        # Both contribute equally: t/T_max = 0.5, |C|/B_max = 0.5
        P = comp.compute_pressure(t=100, last_context_tokens=64000)
        assert P == pytest.approx(0.5, abs=0.001)


class TestPressureWithCustomParams:
    def test_custom_T_max(self, comp):
        P = comp.compute_pressure(t=5, last_context_tokens=0, T_max=10)
        assert P == pytest.approx(0.5)

    def test_custom_B_max(self, comp):
        P = comp.compute_pressure(t=0, last_context_tokens=500, B_max=1000)
        assert P == pytest.approx(0.5)

    def test_zero_T_max(self, comp):
        P = comp.compute_pressure(t=100, last_context_tokens=0, T_max=0)
        assert P == 0.0

    def test_zero_B_max(self, comp):
        P = comp.compute_pressure(t=0, last_context_tokens=1000, B_max=0)
        assert P == 0.0
