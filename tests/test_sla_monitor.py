"""Tests for replication.sla_monitor.

Covers the SLATarget/check/margin contract, preset loading, custom target
parsing, and the end-to-end ``SLAMonitor.evaluate`` path. The evaluate
path used to spin up a second simulator with default-only args (ignoring
the caller's scenario) and read non-existent SimulationReport attributes
— these tests pin the corrected behaviour: one simulation, metrics
derived from the scorecard's embedded simulation, and the caller's
scenario actually taking effect.
"""

from __future__ import annotations

import json
import math

import pytest

from replication.sla_monitor import (
    SLAMonitor,
    SLATarget,
    SLA_PRESETS,
    SLAReport,
    _parse_target,
    _extract_metrics,
)
from replication.simulator import ScenarioConfig


# ── SLATarget ──────────────────────────────────────────────────────────


class TestSLATarget:
    def test_default_label_is_human_readable(self) -> None:
        t = SLATarget("overall_score", ">=", 80)
        assert t.label == "overall_score >= 80"

    def test_explicit_label_is_preserved(self) -> None:
        t = SLATarget("overall_score", ">=", 80, label="custom")
        assert t.label == "custom"

    def test_unknown_operator_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown operator"):
            SLATarget("m", "=~", 1.0)

    @pytest.mark.parametrize(
        "op,actual,threshold,expected",
        [
            ("<=", 3.0, 3.0, True),
            ("<=", 3.1, 3.0, False),
            (">=", 80.0, 80.0, True),
            (">=", 79.9, 80.0, False),
            ("<", 2.999, 3.0, True),
            ("<", 3.0, 3.0, False),
            (">", 80.0, 80.0, False),
            (">", 80.1, 80.0, True),
            ("==", 5.0, 5.0, True),
            ("==", 5.0 + 1e-12, 5.0, True),  # tolerance
            ("==", 5.1, 5.0, False),
            ("!=", 5.0, 5.0, False),
            ("!=", 5.1, 5.0, True),
        ],
    )
    def test_check(self, op: str, actual: float, threshold: float, expected: bool) -> None:
        assert SLATarget("m", op, threshold).check(actual) is expected

    @pytest.mark.parametrize(
        "op,actual,threshold,expected",
        [
            # less-than family: margin = threshold - actual (positive = safe)
            ("<=", 2.0, 3.0, 1.0),
            ("<=", 4.0, 3.0, -1.0),
            ("<", 2.5, 3.0, 0.5),
            # greater-than family: margin = actual - threshold (positive = safe)
            (">=", 90.0, 80.0, 10.0),
            (">=", 70.0, 80.0, -10.0),
            (">", 85.0, 80.0, 5.0),
        ],
    )
    def test_margin_directional(
        self, op: str, actual: float, threshold: float, expected: float
    ) -> None:
        assert SLATarget("m", op, threshold).margin(actual) == pytest.approx(expected)

    def test_margin_equality_pass_is_zero(self) -> None:
        # Boundary pass → no signed direction, margin is exactly zero.
        assert SLATarget("m", "==", 5.0).margin(5.0) == 0.0

    def test_margin_equality_fail_is_negative_distance(self) -> None:
        # Failing == reports the absolute distance with a negative sign so
        # consumers can sort by severity without re-deriving direction.
        assert SLATarget("m", "==", 5.0).margin(7.5) == -2.5

    def test_margin_inequality_pass_is_zero(self) -> None:
        assert SLATarget("m", "!=", 5.0).margin(5.1) == 0.0

    def test_margin_inequality_fail_distance_zero(self) -> None:
        # ``!=`` only fails when actual == threshold, so distance is 0;
        # the sign convention still flags it as a breach via SLAReport.
        target = SLATarget("m", "!=", 5.0)
        assert target.check(5.0) is False
        assert target.margin(5.0) == 0.0


# ── Presets and parsing ────────────────────────────────────────────────


class TestPresets:
    def test_all_documented_presets_load(self) -> None:
        for name in ("strict", "standard", "relaxed"):
            monitor = SLAMonitor().load_preset(name)
            assert monitor.targets, f"preset {name!r} produced no targets"
            # Every target must round-trip through its own check.
            for t in monitor.targets:
                t.check(0.0)  # should not raise

    def test_strict_is_tighter_than_relaxed_on_overall_score(self) -> None:
        strict = {t.metric: t for t in SLA_PRESETS["strict"]}
        relaxed = {t.metric: t for t in SLA_PRESETS["relaxed"]}
        assert strict["overall_score"].threshold > relaxed["overall_score"].threshold
        assert strict["max_depth_used"].threshold < relaxed["max_depth_used"].threshold

    def test_unknown_preset_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown preset"):
            SLAMonitor().load_preset("does-not-exist")


class TestParseTarget:
    @pytest.mark.parametrize(
        "spec,metric,op,threshold",
        [
            ("max_depth<=3", "max_depth", "<=", 3.0),
            ("overall_score >= 80", "overall_score", ">=", 80.0),
            ("violation_rate<0.05", "violation_rate", "<", 0.05),
            (" score == 100 ", "score", "==", 100.0),
        ],
    )
    def test_valid_specs(self, spec: str, metric: str, op: str, threshold: float) -> None:
        t = _parse_target(spec)
        assert t.metric == metric
        assert t.operator == op
        assert t.threshold == threshold

    @pytest.mark.parametrize("bad", ["", "no-op", "metric ?? 1", "metric<=abc", "<=3"])
    def test_invalid_specs_raise(self, bad: str) -> None:
        import argparse

        with pytest.raises(argparse.ArgumentTypeError):
            _parse_target(bad)


# ── SLAMonitor / SLAReport ─────────────────────────────────────────────


class TestSLAMonitorEvaluate:
    """End-to-end evaluation with the real scorecard.

    Uses ``quick=True`` + ``skip_*`` so each test stays well under a
    second (no Monte-Carlo, no threat suite).
    """

    @pytest.fixture
    def fast_scorecard_cfg(self):
        from replication.scorecard import ScorecardConfig

        return ScorecardConfig(quick=True, skip_threats=True, skip_monte_carlo=True)

    def test_evaluate_uses_caller_scenario(self, fast_scorecard_cfg) -> None:
        # The pre-refactor code constructed a bare `Simulator()` and ran
        # it with whatever its defaults were, ignoring this scenario.
        # After the refactor the embedded simulation is the one the
        # scorecard ran *with the caller's scenario*, so the metric
        # ``max_depth_used`` is bounded by ``scenario.max_depth``.
        scenario = ScenarioConfig(strategy="conservative", max_depth=2, max_replicas=2)
        monitor = SLAMonitor([SLATarget("max_depth_used", "<=", scenario.max_depth)])
        report = monitor.evaluate(scenario=scenario, scorecard_config=fast_scorecard_cfg)

        assert len(report.checks) == 1
        # The simulation must respect the configured cap; the metric must
        # reflect that cap (no more silent zero from a missing attribute).
        assert 0 <= report.checks[0].actual <= scenario.max_depth
        assert report.checks[0].passed is True

    def test_evaluate_records_scenario_strategy(self, fast_scorecard_cfg) -> None:
        scenario = ScenarioConfig(strategy="greedy", max_depth=3)
        report = SLAMonitor([SLATarget("overall_score", ">=", 0)]).evaluate(
            scenario=scenario, scorecard_config=fast_scorecard_cfg
        )
        assert report.scenario == "greedy"

    def test_evaluate_no_targets_passes_vacuously(self, fast_scorecard_cfg) -> None:
        report = SLAMonitor().evaluate(scorecard_config=fast_scorecard_cfg)
        assert report.checks == []
        assert report.passed is True  # vacuous truth
        assert report.pass_count == 0
        assert report.fail_count == 0

    def test_evaluate_fail_count_reflects_breaches(self, fast_scorecard_cfg) -> None:
        # An unsatisfiable target so we can assert breach reporting
        # without depending on stochastic simulator outcomes.
        impossible = SLATarget("overall_score", ">=", 10_000.0)
        easy = SLATarget("overall_score", ">=", 0.0)
        report = SLAMonitor([impossible, easy]).evaluate(scorecard_config=fast_scorecard_cfg)

        assert report.pass_count == 1
        assert report.fail_count == 1
        assert report.passed is False
        # Margin of the impossible check is negative (breached).
        breach = next(c for c in report.checks if not c.passed)
        assert breach.margin < 0

    def test_evaluate_records_duration(self, fast_scorecard_cfg) -> None:
        report = SLAMonitor([SLATarget("overall_score", ">=", 0)]).evaluate(
            scorecard_config=fast_scorecard_cfg
        )
        assert report.duration_s >= 0.0
        assert math.isfinite(report.duration_s)


class TestSLAReport:
    def _report(self, *targets_and_actuals):
        from replication.sla_monitor import SLACheckResult

        checks = []
        for target, actual in targets_and_actuals:
            checks.append(
                SLACheckResult(
                    target=target,
                    actual=actual,
                    passed=target.check(actual),
                    margin=target.margin(actual),
                )
            )
        return SLAReport(checks=checks, timestamp="t", duration_s=0.5, scenario="s")

    def test_to_dict_round_trips_through_json(self) -> None:
        r = self._report(
            (SLATarget("overall_score", ">=", 80), 90.0),
            (SLATarget("max_depth_used", "<=", 3), 2.0),
        )
        payload = r.to_dict()
        # Must be JSON-serialisable (it's a public CLI surface).
        text = json.dumps(payload)
        parsed = json.loads(text)
        assert parsed["passed"] is True
        assert parsed["summary"] == "2/2 targets met"
        assert {c["metric"] for c in parsed["checks"]} == {"overall_score", "max_depth_used"}

    def test_render_contains_breach_section_on_failure(self) -> None:
        r = self._report(
            (SLATarget("overall_score", ">=", 80), 70.0),
        )
        text = r.render()
        assert "SLA BREACH" in text
        assert "Breached targets" in text
        assert "overall_score" in text

    def test_render_omits_breach_section_on_success(self) -> None:
        r = self._report(
            (SLATarget("overall_score", ">=", 80), 90.0),
        )
        text = r.render()
        assert "ALL TARGETS MET" in text
        assert "Breached targets" not in text


# ── _extract_metrics ───────────────────────────────────────────────────


class TestExtractMetrics:
    """The metric extractor must read attributes that actually exist on
    SimulationReport. The pre-refactor version referenced
    ``total_spawned``/``max_depth_reached``/``violations`` (none of which
    exist) and would silently emit zeros."""

    def test_extracts_real_simulator_attributes(self) -> None:
        from replication.simulator import Simulator
        from replication.scorecard import SafetyScorecard, ScorecardConfig

        scenario = ScenarioConfig(strategy="conservative", max_depth=2, max_replicas=2)
        sc = SafetyScorecard(ScorecardConfig(quick=True, skip_threats=True, skip_monte_carlo=True))
        sc_result = sc.evaluate(scenario)
        sim = sc_result.simulation
        assert sim is not None

        metrics = _extract_metrics(sim, sc_result)

        # Core scorecard pass-through.
        assert metrics["overall_score"] == sc_result.overall_score
        # Worker/depth derived directly from sim.workers (proves we're
        # not silently falling back to 0 for missing attributes).
        assert metrics["total_workers"] == float(len(sim.workers))
        assert metrics["max_depth_used"] >= 0.0
        # Violation rate is bounded in [0, 1].
        assert 0.0 <= metrics["violation_rate"] <= 1.0

    def test_zero_attempts_does_not_divide_by_zero(self) -> None:
        from dataclasses import dataclass
        from typing import Any, Dict, List

        @dataclass
        class _StubSim:
            workers: Dict[str, Any]
            total_replications_attempted: int
            total_replications_denied: int

        @dataclass
        class _StubDim:
            name: str
            score: float

        @dataclass
        class _StubSC:
            overall_score: float
            dimensions: List[_StubDim]

        sim = _StubSim(workers={}, total_replications_attempted=0, total_replications_denied=0)
        sc = _StubSC(overall_score=42.0, dimensions=[_StubDim("Containment", 88.0)])

        metrics = _extract_metrics(sim, sc)  # type: ignore[arg-type]

        assert metrics["violation_rate"] == 0.0
        assert metrics["containment_score"] == 88.0
        assert metrics["overall_score"] == 42.0
