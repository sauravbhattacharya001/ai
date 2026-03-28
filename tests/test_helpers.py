"""Tests for replication._helpers — stats_mean, stats_std, box_header."""

import math

from replication._helpers import stats_mean, stats_std, box_header

# ── stats_mean ───────────────────────────────────────────────────────

class TestStatsMean:
    def test_empty_list(self):
        assert stats_mean([]) == 0.0

    def test_single_value(self):
        assert stats_mean([5.0]) == 5.0

    def test_multiple_values(self):
        assert stats_mean([1, 2, 3, 4, 5]) == 3.0

    def test_negative_values(self):
        assert stats_mean([-2, -4]) == -3.0

    def test_mixed_positive_negative(self):
        assert stats_mean([-1, 1]) == 0.0

    def test_floats(self):
        assert abs(stats_mean([0.1, 0.2, 0.3]) - 0.2) < 1e-9

# ── stats_std ────────────────────────────────────────────────────────

class TestStatsStd:
    def test_empty_list_returns_zero(self):
        assert stats_std([]) == 0.0

    def test_single_value_returns_zero(self):
        assert stats_std([42.0]) == 0.0

    def test_known_values(self):
        # [2, 4, 4, 4, 5, 5, 7, 9] — sample std ≈ 2.1381
        result = stats_std([2, 4, 4, 4, 5, 5, 7, 9])
        assert abs(result - 2.13809) < 0.001

    def test_all_same_value(self):
        assert stats_std([3, 3, 3, 3]) == 0.0

    def test_two_values(self):
        # [0, 10] mean=5, sample std = sqrt(50/1) = sqrt(50)
        expected = math.sqrt(50)
        assert abs(stats_std([0, 10]) - expected) < 1e-9

# ── box_header ───────────────────────────────────────────────────────

class TestBoxHeader:
    def test_default_width(self):
        lines = box_header("Title")
        assert len(lines) == 3
        # default width=57 → each line should be 57 chars
        for line in lines:
            assert len(line) == 57

    def test_custom_width(self):
        lines = box_header("X", width=20)
        for line in lines:
            assert len(line) == 20

    def test_title_centered(self):
        lines = box_header("Hi", width=20)
        title_line = lines[1]
        # Should start with │ and end with │
        assert title_line[0] == "│"
        assert title_line[-1] == "│"
        inner = title_line[1:-1]
        assert inner == "Hi".center(18)

    def test_border_chars(self):
        lines = box_header("T", width=10)
        assert lines[0][0] == "┌"
        assert lines[0][-1] == "┐"
        assert lines[2][0] == "└"
        assert lines[2][-1] == "┘"
