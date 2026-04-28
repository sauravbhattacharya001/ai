"""Tests for the Threat Adaptation Engine."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from replication.threat_adapter import (
    AdaptAction,
    AdaptationPlan,
    AdaptCycleResult,
    AdapterConfig,
    BudgetLevel,
    ControlRecommendation,
    LandscapeSnapshot,
    ShiftAlert,
    ShiftType,
    ThreatAdapter,
    ThreatTrend,
    ThreatVector,
)
from replication._helpers import Severity


# ── ThreatVector ─────────────────────────────────────────────────────


class TestThreatVector:
    def test_threat_score_basic(self):
        v = ThreatVector(
            name="test", category="cat", success_rate=0.5,
            frequency=5.0, severity=Severity.HIGH, trend=ThreatTrend.STABLE,
        )
        score = v.threat_score()
        assert score > 0

    def test_threat_score_escalating_higher(self):
        base = ThreatVector(
            name="test", category="cat", success_rate=0.5,
            frequency=5.0, severity=Severity.HIGH, trend=ThreatTrend.STABLE,
        )
        escalating = ThreatVector(
            name="test", category="cat", success_rate=0.5,
            frequency=5.0, severity=Severity.HIGH, trend=ThreatTrend.ESCALATING,
        )
        assert escalating.threat_score() > base.threat_score()

    def test_threat_score_retreating_lower(self):
        base = ThreatVector(
            name="test", category="cat", success_rate=0.5,
            frequency=5.0, severity=Severity.HIGH, trend=ThreatTrend.STABLE,
        )
        retreating = ThreatVector(
            name="test", category="cat", success_rate=0.5,
            frequency=5.0, severity=Severity.HIGH, trend=ThreatTrend.RETREATING,
        )
        assert retreating.threat_score() < base.threat_score()

    def test_critical_scores_higher_than_low(self):
        low = ThreatVector(
            name="test", category="cat", success_rate=0.5,
            frequency=5.0, severity=Severity.LOW, trend=ThreatTrend.STABLE,
        )
        crit = ThreatVector(
            name="test", category="cat", success_rate=0.5,
            frequency=5.0, severity=Severity.CRITICAL, trend=ThreatTrend.STABLE,
        )
        assert crit.threat_score() > low.threat_score()

    def test_zero_success_rate(self):
        v = ThreatVector(
            name="test", category="cat", success_rate=0.0,
            frequency=5.0, severity=Severity.HIGH, trend=ThreatTrend.STABLE,
        )
        assert v.threat_score() == 0.0


# ── LandscapeSnapshot ───────────────────────────────────────────────


class TestLandscapeSnapshot:
    def _make_snapshot(self):
        return LandscapeSnapshot(
            timestamp="2026-04-27T12:00:00Z",
            vectors=[
                ThreatVector(name="Vec1", category="cat1", success_rate=0.3,
                             frequency=2.0, severity=Severity.MEDIUM,
                             trend=ThreatTrend.STABLE),
            ],
            overall_pressure=45.0,
            volatility_index=0.3,
            dominant_category="cat1",
            scan_scenarios=10,
            scan_strategies=["greedy"],
        )

    def test_roundtrip(self):
        snap = self._make_snapshot()
        d = snap.to_dict()
        restored = LandscapeSnapshot.from_dict(d)
        assert restored.timestamp == snap.timestamp
        assert len(restored.vectors) == 1
        assert restored.vectors[0].name == "Vec1"
        assert restored.overall_pressure == 45.0

    def test_to_dict_contains_keys(self):
        snap = self._make_snapshot()
        d = snap.to_dict()
        assert "timestamp" in d
        assert "vectors" in d
        assert "overall_pressure" in d


# ── ShiftAlert ───────────────────────────────────────────────────────


class TestShiftAlert:
    def test_urgency_score_critical_higher(self):
        high = ShiftAlert(
            shift_type=ShiftType.VECTOR_ESCALATION, severity=Severity.HIGH,
            vector_name="v1", description="d", metric_before=0.2,
            metric_after=0.5, change_pct=150.0, recommended_response="r",
        )
        low = ShiftAlert(
            shift_type=ShiftType.VECTOR_RETREAT, severity=Severity.LOW,
            vector_name="v2", description="d", metric_before=0.5,
            metric_after=0.2, change_pct=-60.0, recommended_response="r",
        )
        assert high.urgency_score() > low.urgency_score()


# ── ThreatAdapter Demo Mode ─────────────────────────────────────────


class TestThreatAdapterDemo:
    def _make_adapter(self, tmp_path):
        config = AdapterConfig(
            demo=True,
            history_file=str(tmp_path / "history.jsonl"),
        )
        return ThreatAdapter(config)

    def test_scan_returns_snapshot(self, tmp_path):
        adapter = self._make_adapter(tmp_path)
        snap = adapter.scan()
        assert isinstance(snap, LandscapeSnapshot)
        assert len(snap.vectors) > 0
        assert snap.overall_pressure >= 0

    def test_scan_persists_history(self, tmp_path):
        adapter = self._make_adapter(tmp_path)
        adapter.scan()
        assert (tmp_path / "history.jsonl").exists()
        lines = (tmp_path / "history.jsonl").read_text().strip().splitlines()
        assert len(lines) == 1

    def test_detect_shifts_no_baseline(self, tmp_path):
        adapter = self._make_adapter(tmp_path)
        snap = adapter.scan()
        alerts = adapter.detect_shifts(snap)
        # No baseline yet, should return empty
        assert isinstance(alerts, list)

    def test_detect_shifts_with_two_scans(self, tmp_path):
        adapter = self._make_adapter(tmp_path)
        adapter.scan()  # first scan (baseline)
        snap2 = adapter.scan()  # second scan
        alerts = adapter.detect_shifts(snap2)
        assert isinstance(alerts, list)

    def test_plan_adaptations(self, tmp_path):
        adapter = self._make_adapter(tmp_path)
        snap = adapter.scan()
        alerts = adapter.detect_shifts(snap)
        plan = adapter.plan_adaptations(snap, alerts)
        assert isinstance(plan, AdaptationPlan)
        assert plan.budget_level == BudgetLevel.MEDIUM

    def test_full_cycle(self, tmp_path):
        adapter = self._make_adapter(tmp_path)
        result = adapter.run_cycle()
        assert isinstance(result, AdaptCycleResult)
        assert result.cycle_time_sec >= 0
        assert result.snapshot is not None
        assert isinstance(result.plan, AdaptationPlan)

    def test_full_cycle_render(self, tmp_path):
        adapter = self._make_adapter(tmp_path)
        result = adapter.run_cycle()
        text = result.render()
        assert "Threat Adaptation Engine" in text
        assert "Threat Landscape" in text

    def test_full_cycle_render_html(self, tmp_path):
        adapter = self._make_adapter(tmp_path)
        result = adapter.run_cycle()
        html = result.render_html()
        assert "<html" in html
        assert "Threat Adaptation" in html

    def test_full_cycle_to_dict(self, tmp_path):
        adapter = self._make_adapter(tmp_path)
        result = adapter.run_cycle()
        d = result.to_dict()
        assert "snapshot" in d
        assert "shift_alerts" in d
        assert "plan" in d

    def test_get_history(self, tmp_path):
        adapter = self._make_adapter(tmp_path)
        adapter.scan()
        adapter.scan()
        history = adapter.get_history(last_n=5)
        assert len(history) == 2

    def test_budget_caps(self, tmp_path):
        for budget in [BudgetLevel.LOW, BudgetLevel.MEDIUM, BudgetLevel.HIGH]:
            config = AdapterConfig(
                demo=True,
                budget=budget,
                history_file=str(tmp_path / f"history_{budget.value}.jsonl"),
            )
            adapter = ThreatAdapter(config)
            # Do 2 scans to get shifts
            adapter.scan()
            snap = adapter.scan()
            alerts = adapter.detect_shifts(snap)
            plan = adapter.plan_adaptations(snap, alerts)
            assert plan.budget_level == budget

    def test_multiple_cycles_evolve(self, tmp_path):
        config = AdapterConfig(
            demo=True,
            history_file=str(tmp_path / "history.jsonl"),
        )
        adapter = ThreatAdapter(config)
        results = []
        for _ in range(4):
            results.append(adapter.run_cycle())
        # Pressure values should exist for all
        for r in results:
            assert r.snapshot.overall_pressure >= 0

    def test_history_persistence(self, tmp_path):
        config = AdapterConfig(
            demo=True,
            history_file=str(tmp_path / "history.jsonl"),
        )
        adapter1 = ThreatAdapter(config)
        adapter1.scan()
        adapter1.scan()
        # New adapter reads saved history
        adapter2 = ThreatAdapter(config)
        assert len(adapter2.get_history(10)) == 2

    def test_plan_render(self, tmp_path):
        adapter = self._make_adapter(tmp_path)
        adapter.scan()
        snap = adapter.scan()
        alerts = adapter.detect_shifts(snap)
        plan = adapter.plan_adaptations(snap, alerts)
        text = plan.render()
        assert "Adaptation Plan" in text

    def test_max_history_trim(self, tmp_path):
        config = AdapterConfig(
            demo=True,
            max_history=3,
            history_file=str(tmp_path / "history.jsonl"),
        )
        adapter = ThreatAdapter(config)
        for _ in range(5):
            adapter.scan()
        assert len(adapter.get_history(10)) == 3


# ── Enums ────────────────────────────────────────────────────────────


class TestEnums:
    def test_threat_trend_values(self):
        assert ThreatTrend.EMERGING.value == "emerging"
        assert ThreatTrend.ESCALATING.value == "escalating"

    def test_shift_type_values(self):
        assert ShiftType.NEW_VECTOR.value == "new_vector"
        assert ShiftType.VECTOR_ESCALATION.value == "vector_escalation"

    def test_adapt_action_values(self):
        assert AdaptAction.TIGHTEN.value == "tighten"
        assert AdaptAction.ADD.value == "add"

    def test_budget_level_values(self):
        assert BudgetLevel.LOW.value == "low"
        assert BudgetLevel.HIGH.value == "high"
