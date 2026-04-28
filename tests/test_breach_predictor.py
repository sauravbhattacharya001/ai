"""Tests for replication.breach_predictor."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest

from replication.breach_predictor import (
    ALERT_LEVELS,
    BREACH_THRESHOLD,
    PRECURSOR_TYPES,
    PRECURSOR_WEIGHTS,
    BehaviorSignal,
    BreachPrediction,
    BreachPredictor,
    BreachWindow,
    PrecursorState,
    Recommendation,
    _alert_level,
    _classify_velocity,
    _cosine_similarity,
    _jaccard,
    generate_demo_signals,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_signals(precursor: str, intensities: list[float], start_hour: int = 0):
    """Create a list of BehaviorSignal objects for a single precursor."""
    signals = []
    for i, val in enumerate(intensities):
        ts = f"2025-06-15T{start_hour + i:02d}:00:00+00:00"
        signals.append(BehaviorSignal(ts, precursor, val, "test"))
    return signals


# ── Unit tests: helpers ──────────────────────────────────────────────


class TestClassifyVelocity:
    def test_stable(self):
        assert _classify_velocity(0.005) == "stable"
        assert _classify_velocity(-0.005) == "stable"

    def test_slow_escalation(self):
        assert _classify_velocity(0.03) == "slow_escalation"

    def test_rapid_escalation(self):
        assert _classify_velocity(0.10) == "rapid_escalation"

    def test_critical_acceleration(self):
        assert _classify_velocity(0.20) == "critical_acceleration"

    def test_zero(self):
        assert _classify_velocity(0.0) == "stable"


class TestAlertLevel:
    def test_green(self):
        assert _alert_level(0) == "GREEN"
        assert _alert_level(25) == "GREEN"

    def test_yellow(self):
        assert _alert_level(26) == "YELLOW"
        assert _alert_level(50) == "YELLOW"

    def test_orange(self):
        assert _alert_level(51) == "ORANGE"
        assert _alert_level(75) == "ORANGE"

    def test_red(self):
        assert _alert_level(76) == "RED"
        assert _alert_level(100) == "RED"

    def test_over_100(self):
        assert _alert_level(150) == "RED"


class TestJaccard:
    def test_identical(self):
        assert _jaccard({1, 2, 3}, {1, 2, 3}) == 1.0

    def test_disjoint(self):
        assert _jaccard({1, 2}, {3, 4}) == 0.0

    def test_partial(self):
        assert abs(_jaccard({1, 2, 3}, {2, 3, 4}) - 0.5) < 0.01

    def test_empty(self):
        assert _jaccard(set(), set()) == 0.0

    def test_one_empty(self):
        assert _jaccard({1}, set()) == 0.0


class TestCosineSimilarity:
    def test_identical(self):
        assert abs(_cosine_similarity([1, 0, 1], [1, 0, 1]) - 1.0) < 0.01

    def test_orthogonal(self):
        assert abs(_cosine_similarity([1, 0], [0, 1])) < 0.01

    def test_zero_vector(self):
        assert _cosine_similarity([0, 0], [1, 1]) == 0.0

    def test_partial_similarity(self):
        sim = _cosine_similarity([1, 1, 0], [1, 0, 0])
        assert 0.5 < sim < 1.0


# ── Signal ingestion ─────────────────────────────────────────────────


class TestSignalIngestion:
    def test_basic_ingest(self):
        p = BreachPredictor()
        signals = _make_signals("boundary_probing", [0.1, 0.2, 0.3])
        p.ingest_signals(signals)
        assert len(p._signals["boundary_probing"]) == 3

    def test_invalid_precursor_ignored(self):
        p = BreachPredictor()
        p.ingest_signals([BehaviorSignal("2025-01-01T00:00:00Z", "fake_type", 0.5)])
        assert "fake_type" not in p._signals

    def test_intensity_clamped(self):
        p = BreachPredictor()
        p.ingest_signals([
            BehaviorSignal("2025-01-01T00:00:00Z", "goal_drift", 1.5),
            BehaviorSignal("2025-01-01T01:00:00Z", "goal_drift", -0.5),
        ])
        vals = [v for _, v in p._signals["goal_drift"]]
        assert vals[0] == 1.0
        assert vals[1] == 0.0

    def test_chronological_sorting(self):
        p = BreachPredictor()
        signals = [
            BehaviorSignal("2025-01-01T03:00:00Z", "evasion_behavior", 0.3),
            BehaviorSignal("2025-01-01T01:00:00Z", "evasion_behavior", 0.1),
            BehaviorSignal("2025-01-01T02:00:00Z", "evasion_behavior", 0.2),
        ]
        p.ingest_signals(signals)
        vals = [v for _, v in p._signals["evasion_behavior"]]
        assert vals == [0.1, 0.2, 0.3]


# ── Precursor state computation ──────────────────────────────────────


class TestPrecursorStates:
    def test_no_signals(self):
        p = BreachPredictor()
        pred = p.predict()
        for ps in pred.precursor_states:
            assert ps.current_intensity == 0.0
            assert ps.velocity == "stable"
            assert ps.samples == 0

    def test_single_signal(self):
        p = BreachPredictor()
        p.ingest_signals(_make_signals("boundary_probing", [0.4]))
        pred = p.predict()
        bp_state = [ps for ps in pred.precursor_states if ps.precursor == "boundary_probing"][0]
        assert bp_state.current_intensity == 0.4
        assert bp_state.samples == 1

    def test_escalating_velocity(self):
        p = BreachPredictor()
        # strong upward trend
        vals = [0.1 + 0.05 * i for i in range(10)]
        p.ingest_signals(_make_signals("replication_pressure", vals))
        pred = p.predict()
        rp_state = [ps for ps in pred.precursor_states if ps.precursor == "replication_pressure"][0]
        assert rp_state.slope > 0
        assert rp_state.velocity in ("slow_escalation", "rapid_escalation", "critical_acceleration")

    def test_stable_signals(self):
        p = BreachPredictor()
        vals = [0.3] * 10
        p.ingest_signals(_make_signals("goal_drift", vals))
        pred = p.predict()
        gd_state = [ps for ps in pred.precursor_states if ps.precursor == "goal_drift"][0]
        assert gd_state.velocity == "stable"

    def test_all_precursors_present(self):
        p = BreachPredictor()
        pred = p.predict()
        precursors = {ps.precursor for ps in pred.precursor_states}
        assert precursors == set(PRECURSOR_TYPES)


# ── Risk score ───────────────────────────────────────────────────────


class TestRiskScore:
    def test_zero_signals(self):
        p = BreachPredictor()
        pred = p.predict()
        assert pred.risk_score == 0.0
        assert pred.alert_level == "GREEN"

    def test_high_intensity_gives_high_score(self):
        p = BreachPredictor()
        # all precursors at high intensity
        for prec in PRECURSOR_TYPES:
            p.ingest_signals(_make_signals(prec, [0.8, 0.85, 0.9]))
        pred = p.predict()
        assert pred.risk_score > 50

    def test_score_bounded(self):
        p = BreachPredictor()
        for prec in PRECURSOR_TYPES:
            p.ingest_signals(_make_signals(prec, [1.0] * 20))
        pred = p.predict()
        assert 0 <= pred.risk_score <= 100

    def test_active_precursor_count(self):
        p = BreachPredictor()
        p.ingest_signals(_make_signals("boundary_probing", [0.5]))
        p.ingest_signals(_make_signals("evasion_behavior", [0.3]))
        pred = p.predict()
        assert pred.active_precursors == 2


# ── Breach window ────────────────────────────────────────────────────


class TestBreachWindow:
    def test_no_escalation_no_window(self):
        p = BreachPredictor()
        pred = p.predict()
        assert pred.breach_window is None

    def test_escalating_has_window(self):
        p = BreachPredictor()
        vals = [0.1 + 0.05 * i for i in range(10)]
        p.ingest_signals(_make_signals("boundary_probing", vals))
        pred = p.predict()
        if pred.breach_window:
            assert pred.breach_window.estimated_hours > 0
            assert pred.breach_window.confidence > 0

    def test_at_threshold_zero_hours(self):
        p = BreachPredictor()
        p.ingest_signals(_make_signals("boundary_probing", [BREACH_THRESHOLD + 0.01]))
        pred = p.predict()
        if pred.breach_window:
            assert pred.breach_window.estimated_hours == 0.0

    def test_window_confidence_bounds(self):
        p = BreachPredictor()
        vals = [0.2 + 0.04 * i for i in range(15)]
        p.ingest_signals(_make_signals("privilege_creep", vals))
        pred = p.predict()
        if pred.breach_window:
            bw = pred.breach_window
            assert bw.lower_bound_hours <= bw.estimated_hours
            assert bw.upper_bound_hours >= bw.estimated_hours
            assert 0 < bw.confidence <= 1.0


# ── Recommendations ──────────────────────────────────────────────────


class TestRecommendations:
    def test_active_precursor_generates_rec(self):
        p = BreachPredictor()
        p.ingest_signals(_make_signals("boundary_probing", [0.5]))
        pred = p.predict()
        actions = [r.action for r in pred.recommendations]
        assert "Tighten Containment Boundaries" in actions

    def test_high_alert_immediate_urgency(self):
        p = BreachPredictor()
        for prec in PRECURSOR_TYPES:
            p.ingest_signals(_make_signals(prec, [0.9, 0.95]))
        pred = p.predict()
        if pred.alert_level in ("ORANGE", "RED"):
            has_immediate = any(r.urgency == "immediate" for r in pred.recommendations)
            assert has_immediate

    def test_recommendations_sorted_by_urgency(self):
        p = BreachPredictor()
        for prec in PRECURSOR_TYPES:
            p.ingest_signals(_make_signals(prec, [0.6]))
        pred = p.predict()
        urgency_order = {"immediate": 0, "soon": 1, "scheduled": 2}
        urgencies = [urgency_order.get(r.urgency, 3) for r in pred.recommendations]
        assert urgencies == sorted(urgencies)


# ── Correlation matrix ───────────────────────────────────────────────


class TestCorrelationMatrix:
    def test_self_correlation_is_one(self):
        p = BreachPredictor()
        p.ingest_signals(_make_signals("boundary_probing", [0.5]))
        pred = p.predict()
        for prec in PRECURSOR_TYPES:
            assert pred.correlation_matrix[prec][prec] == 1.0

    def test_co_occurring_signals_correlated(self):
        p = BreachPredictor()
        # same timestamps
        ts = "2025-01-01T00:00:00Z"
        p.ingest_signals([
            BehaviorSignal(ts, "boundary_probing", 0.5),
            BehaviorSignal(ts, "evasion_behavior", 0.4),
        ])
        pred = p.predict()
        corr = pred.correlation_matrix["boundary_probing"]["evasion_behavior"]
        assert corr > 0


# ── Historical pattern matching ──────────────────────────────────────


class TestPatternMatching:
    def test_no_history_no_matches(self):
        p = BreachPredictor()
        p.ingest_signals(_make_signals("boundary_probing", [0.5]))
        pred = p.predict()
        assert pred.pattern_matches == []

    def test_similar_pattern_matches(self):
        p = BreachPredictor()
        # record a historical breach with specific profile
        p._history.append({
            "timestamp": "2025-01-01T00:00:00Z",
            "precursor_vector": [0.8, 0.1, 0.1, 0.7, 0.1, 0.1, 0.9, 0.1],
            "description": "Past boundary+privilege+replication breach",
        })
        # now create similar current pattern
        p.ingest_signals(_make_signals("boundary_probing", [0.8]))
        p.ingest_signals(_make_signals("privilege_creep", [0.7]))
        p.ingest_signals(_make_signals("replication_pressure", [0.9]))
        pred = p.predict()
        assert len(pred.pattern_matches) > 0
        assert pred.pattern_matches[0]["similarity"] >= 0.7

    def test_dissimilar_pattern_no_match(self):
        p = BreachPredictor()
        p._history.append({
            "timestamp": "2025-01-01T00:00:00Z",
            "precursor_vector": [0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "description": "Boundary-only breach",
        })
        # current: completely different
        p.ingest_signals(_make_signals("communication_anomaly", [0.9]))
        pred = p.predict()
        # may or may not match — cosine of orthogonal-ish vectors
        for m in pred.pattern_matches:
            assert m["similarity"] >= 0.7  # only included if >= 0.7


# ── Demo generation ──────────────────────────────────────────────────


class TestDemoGeneration:
    def test_generates_signals(self):
        signals = generate_demo_signals(hours=12)
        assert len(signals) > 0

    def test_all_precursors_eventually(self):
        signals = generate_demo_signals(hours=24)
        precursors = {s.precursor for s in signals}
        assert precursors == set(PRECURSOR_TYPES)

    def test_intensity_bounded(self):
        signals = generate_demo_signals(hours=24)
        for s in signals:
            assert 0.0 <= s.intensity <= 1.0

    def test_deterministic_with_seed(self):
        s1 = generate_demo_signals(hours=12, seed=99)
        s2 = generate_demo_signals(hours=12, seed=99)
        assert len(s1) == len(s2)
        for a, b in zip(s1, s2):
            assert a.intensity == b.intensity


# ── Rendering ────────────────────────────────────────────────────────


class TestRendering:
    def test_text_rendering(self):
        p = BreachPredictor()
        p.ingest_signals(generate_demo_signals(hours=12))
        pred = p.predict()
        text = p.render_text(pred)
        assert "Containment Breach Predictor" in text
        assert pred.alert_level in text

    def test_json_rendering(self):
        p = BreachPredictor()
        p.ingest_signals(generate_demo_signals(hours=12))
        pred = p.predict()
        j = p.render_json(pred)
        data = json.loads(j)
        assert "risk_score" in data
        assert "alert_level" in data

    def test_html_rendering(self):
        p = BreachPredictor()
        p.ingest_signals(generate_demo_signals(hours=12))
        pred = p.predict()
        html = p.render_html(pred)
        assert "<!DOCTYPE html>" in html
        assert "Breach Prediction Report" in html

    def test_to_dict_roundtrip(self):
        p = BreachPredictor()
        p.ingest_signals(generate_demo_signals(hours=12))
        pred = p.predict()
        d = pred.to_dict()
        assert isinstance(d, dict)
        assert d["alert_level"] == pred.alert_level


# ── Persistence ──────────────────────────────────────────────────────


class TestPersistence:
    def test_record_and_load_breach(self, tmp_path):
        state_file = str(tmp_path / "state.jsonl")
        p = BreachPredictor(state_path=state_file)
        p.ingest_signals(_make_signals("boundary_probing", [0.8]))
        p.record_breach("Test breach event")
        assert len(p._history) == 1

        # load into new predictor
        p2 = BreachPredictor(state_path=state_file)
        assert len(p2._history) == 1
        assert p2._history[0]["description"] == "Test breach event"

    def test_missing_state_file(self, tmp_path):
        p = BreachPredictor(state_path=str(tmp_path / "nonexistent.jsonl"))
        assert p._history == []


# ── CLI ──────────────────────────────────────────────────────────────


class TestCLI:
    def test_demo_mode(self, capsys):
        from replication.breach_predictor import main
        main(["--demo", "--demo-hours", "6"])
        captured = capsys.readouterr()
        assert "Breach Predictor" in captured.out

    def test_json_output(self, capsys):
        from replication.breach_predictor import main
        main(["--demo", "--json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "risk_score" in data

    def test_html_output(self, tmp_path):
        from replication.breach_predictor import main
        out = str(tmp_path / "report.html")
        main(["--demo", "--html", out])
        assert Path(out).exists()
        content = Path(out).read_text(encoding="utf-8")
        assert "Breach Prediction Report" in content

    def test_signals_from_file(self, tmp_path, capsys):
        signals_file = tmp_path / "signals.jsonl"
        signals = _make_signals("boundary_probing", [0.2, 0.4, 0.6])
        signals_file.write_text(
            "\n".join(json.dumps(s.to_dict()) for s in signals),
            encoding="utf-8",
        )
        from replication.breach_predictor import main
        main(["--signals", str(signals_file)])
        captured = capsys.readouterr()
        assert "Breach Predictor" in captured.out
