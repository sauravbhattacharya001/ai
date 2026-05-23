"""Tests for replication._helpers — stats_mean, stats_std, box_header, jaccard,
linear_regression, pearson_correlation."""

import math
import random

from replication._helpers import (
    Severity,
    jaccard,
    severity_rank,
    sparkline,
    stats_mean,
    stats_std,
    box_header,
    linear_regression,
    pearson_correlation,
)

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


# ── jaccard ─────────────────────────────────────────────


class TestJaccard:
    def test_both_empty_returns_zero(self):
        # Avoid undefined 0/0; matches the convention of the three
        # legacy local copies this helper replaced.
        assert jaccard(set(), set()) == 0.0

    def test_one_empty_returns_zero(self):
        assert jaccard({"a"}, set()) == 0.0
        assert jaccard(set(), {"a"}) == 0.0

    def test_identical_sets(self):
        assert jaccard({1, 2, 3}, {1, 2, 3}) == 1.0

    def test_disjoint_sets(self):
        assert jaccard({1, 2}, {3, 4}) == 0.0

    def test_partial_overlap(self):
        # |A ∩ B| = 2 ({2,3}), |A ∪ B| = 4 ({1,2,3,4}) → 0.5
        assert jaccard({1, 2, 3}, {2, 3, 4}) == 0.5

    def test_accepts_lists_and_tuples(self):
        assert jaccard([1, 1, 2], (2, 3)) == 1 / 3

    def test_accepts_generators(self):
        assert jaccard((x for x in [1, 2]), (x for x in [2, 3])) == 1 / 3

    def test_string_tokens(self):
        assert jaccard({"sql", "auth"}, {"auth", "xss"}) == 1 / 3


# ── linear_regression ────────────────────────────────────────────────
#
# These tests pin down the shared :func:`linear_regression` helper used by
# *drift*, *hoarding*, and several detectors. They double as a regression
# suite for the closed-form ``ss_xx = n*(n^2-1)/12`` optimisation: if that
# identity is ever broken, the perfect-fit tests below catch it instantly.


class TestLinearRegressionHelper:
    def test_empty(self):
        slope, intercept, r2 = linear_regression([])
        assert slope == 0.0 and intercept == 0.0 and r2 == 0.0

    def test_single(self):
        slope, intercept, r2 = linear_regression([7.5])
        assert slope == 0.0
        assert intercept == 7.5
        assert r2 == 0.0

    def test_flat_returns_zero_slope(self):
        slope, intercept, r2 = linear_regression([4.0, 4.0, 4.0, 4.0])
        assert slope == 0.0
        assert intercept == 4.0
        # No y-variance → r² defined as 0 in this helper.
        assert r2 == 0.0

    def test_perfect_positive_line(self):
        # y = 2*x + 3 for x = 0..4
        ys = [3.0, 5.0, 7.0, 9.0, 11.0]
        slope, intercept, r2 = linear_regression(ys)
        assert abs(slope - 2.0) < 1e-12
        assert abs(intercept - 3.0) < 1e-12
        assert abs(r2 - 1.0) < 1e-12

    def test_perfect_negative_line(self):
        # y = -3*x + 10
        ys = [10.0, 7.0, 4.0, 1.0, -2.0]
        slope, intercept, r2 = linear_regression(ys)
        assert abs(slope - (-3.0)) < 1e-12
        assert abs(intercept - 10.0) < 1e-12
        assert abs(r2 - 1.0) < 1e-12

    def test_two_points_is_perfect_fit(self):
        slope, intercept, r2 = linear_regression([0.0, 10.0])
        assert abs(slope - 10.0) < 1e-12
        assert abs(intercept - 0.0) < 1e-12
        assert abs(r2 - 1.0) < 1e-12

    def test_noisy_trend_is_imperfect(self):
        slope, _, r2 = linear_regression([1.0, 3.0, 2.0, 4.0, 3.0])
        assert slope > 0.0
        assert 0.0 < r2 < 1.0

    def test_large_inputs_match_naive_three_pass(self):
        # Regression guard for the closed-form ss_xx + two-pass
        # optimisation. Compare against a deliberately naive reference
        # implementation on a 1k-sample sequence.
        random.seed(20260521)
        ys = [random.gauss(0.0, 1.0) + 0.01 * i for i in range(1000)]

        def reference(ys):
            n = len(ys)
            x_mean = (n - 1) / 2.0
            y_mean = sum(ys) / n
            ss_xy = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(ys))
            ss_xx = sum((i - x_mean) ** 2 for i in range(n))
            ss_yy = sum((y - y_mean) ** 2 for y in ys)
            slope = ss_xy / ss_xx
            return slope, y_mean - slope * x_mean, (ss_xy ** 2) / (ss_xx * ss_yy)

        got = linear_regression(ys)
        ref = reference(ys)
        for a, b in zip(got, ref):
            assert abs(a - b) < 1e-9, (a, b)

    def test_constant_input_does_not_divide_by_zero(self):
        # Pathological: ss_yy = 0; must not raise and must return r²=0.
        slope, intercept, r2 = linear_regression([2.5] * 50)
        assert slope == 0.0
        assert intercept == 2.5
        assert r2 == 0.0

    def test_closed_form_ss_xx_identity(self):
        # The optimisation relies on sum_{i=0..n-1} (i - (n-1)/2)^2 =
        # n*(n^2 - 1) / 12. If that is wrong, the fitted slope of an
        # exactly-linear sequence drifts. Sweep several n.
        for n in (3, 4, 5, 17, 64, 257):
            ys = [3.0 * i - 1.0 for i in range(n)]
            slope, intercept, r2 = linear_regression(ys)
            assert abs(slope - 3.0) < 1e-9, n
            assert abs(intercept - (-1.0)) < 1e-9, n
            assert abs(r2 - 1.0) < 1e-9, n


# ── pearson_correlation ──────────────────────────────────────────────


class TestPearsonCorrelationHelper:
    def test_perfect_positive(self):
        assert abs(pearson_correlation([1, 2, 3, 4], [2, 4, 6, 8]) - 1.0) < 1e-12

    def test_perfect_negative(self):
        assert abs(pearson_correlation([1, 2, 3, 4], [8, 6, 4, 2]) - (-1.0)) < 1e-12

    def test_zero_for_short_inputs(self):
        assert pearson_correlation([], []) == 0.0
        assert pearson_correlation([1], [1]) == 0.0
        assert pearson_correlation([1, 2], [3]) == 0.0  # len(y) < 2
        assert pearson_correlation([1], [3, 4]) == 0.0  # len(x) < 2

    def test_zero_variance_returns_zero(self):
        assert pearson_correlation([1, 1, 1, 1], [1, 2, 3, 4]) == 0.0
        assert pearson_correlation([1, 2, 3, 4], [5, 5, 5, 5]) == 0.0

    def test_unequal_length_truncates_to_shorter(self):
        # zip() semantics in the prior implementation truncated to the
        # shorter sequence; the optimised version preserves that.
        # corr([1,2,3], [2,4,6]) = +1.
        r = pearson_correlation([1, 2, 3, 99, 99], [2, 4, 6])
        assert abs(r - 1.0) < 1e-12

    def test_known_value(self):
        # Manually computed reference: anscombe-ish small sample.
        x = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
        y = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]
        # Anscombe Quartet I correlation is famously ≈ 0.8164.
        r = pearson_correlation(x, y)
        assert abs(r - 0.8164) < 1e-3

    def test_symmetric(self):
        random.seed(42)
        x = [random.random() for _ in range(200)]
        y = [random.random() for _ in range(200)]
        assert abs(pearson_correlation(x, y) - pearson_correlation(y, x)) < 1e-12

    def test_scale_and_shift_invariant(self):
        # Pearson is invariant under positive affine transforms.
        random.seed(7)
        x = [random.random() for _ in range(150)]
        y = [random.random() for _ in range(150)]
        base = pearson_correlation(x, y)
        shifted = pearson_correlation([xi + 100 for xi in x], [3 * yi - 7 for yi in y])
        assert abs(base - shifted) < 1e-9

    def test_bounded_in_unit_interval(self):
        random.seed(99)
        for _ in range(20):
            n = random.randint(5, 100)
            x = [random.gauss(0, 1) for _ in range(n)]
            y = [random.gauss(0, 1) for _ in range(n)]
            r = pearson_correlation(x, y)
            assert -1.0 - 1e-9 <= r <= 1.0 + 1e-9


class TestSeverityRankFastPath:
    """Regression coverage for the string fast-path added in v3.14.

    The earlier implementation only recognised lowercase enum-value
    strings (``"high"``). Real callers in the codebase pass a mix of
    casings; the optimised helper must keep accepting all of them while
    still returning ``0`` for unknown labels.
    """

    def test_enum_inputs(self):
        assert severity_rank(Severity.INFO) == 0
        assert severity_rank(Severity.LOW) == 1
        assert severity_rank(Severity.MEDIUM) == 2
        assert severity_rank(Severity.HIGH) == 3
        assert severity_rank(Severity.CRITICAL) == 4

    def test_none_returns_zero(self):
        assert severity_rank(None) == 0

    def test_string_inputs_lowercase(self):
        assert severity_rank("info") == 0
        assert severity_rank("low") == 1
        assert severity_rank("medium") == 2
        assert severity_rank("high") == 3
        assert severity_rank("critical") == 4

    def test_string_inputs_uppercase(self):
        # Enum-name spellings are common; pre-baked table should hit fast.
        assert severity_rank("INFO") == 0
        assert severity_rank("LOW") == 1
        assert severity_rank("MEDIUM") == 2
        assert severity_rank("HIGH") == 3
        assert severity_rank("CRITICAL") == 4

    def test_string_inputs_mixed_case_and_whitespace(self):
        assert severity_rank("High") == 3
        assert severity_rank("  Critical  ") == 4
        assert severity_rank("\tlow\n") == 1

    def test_unknown_strings_return_zero(self):
        assert severity_rank("bogus") == 0
        assert severity_rank("") == 0
        assert severity_rank("severe") == 0

    def test_non_string_non_enum_coerces(self):
        # Anything that stringifies to a known severity should still map.
        class _S:
            def __str__(self) -> str:
                return "high"

        assert severity_rank(_S()) == 3


class TestSparkline:
    """Sparkline must stay bit-for-bit identical to the prior impl.

    The v3.14 single-pass min/max refactor preserves the original
    per-element formula precisely; these tests pin that contract so a
    future micro-optimisation can't silently shift the rendered glyph.
    """

    SPARK = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"

    @classmethod
    def _reference(cls, values):
        if not values:
            return ""
        lo, hi = min(values), max(values)
        spread = hi - lo if hi != lo else 1.0
        n = len(cls.SPARK)
        return "".join(
            cls.SPARK[min(int((v - lo) / spread * (n - 1)), n - 1)]
            for v in values
        )

    def test_empty(self):
        assert sparkline([]) == ""

    def test_single_value(self):
        # spread degenerates to 1.0; single char should land on the
        # lowest block.
        assert sparkline([42.0]) == "\u2581"

    def test_constant_values(self):
        # All identical inputs are well-defined and should produce
        # only the lowest sparkline block, never crash on zero spread.
        out = sparkline([7.0, 7.0, 7.0, 7.0])
        assert out == "\u2581" * 4

    def test_monotonic_spread(self):
        out = sparkline([0, 1, 2, 3, 4, 5, 6, 7])
        assert out == self.SPARK  # one of each block, in order

    def test_matches_reference_random(self):
        rng = random.Random(2026)
        for _ in range(50):
            n = rng.randint(1, 200)
            vs = [rng.uniform(-100, 100) for _ in range(n)]
            assert sparkline(vs) == self._reference(vs)

    def test_handles_negative_only(self):
        # min/max must work with all-negative inputs (previously the
        # three-pass version did fine, single-pass must too).
        vs = [-5.0, -1.0, -3.0, -2.0]
        assert sparkline(vs) == self._reference(vs)
