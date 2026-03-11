"""Tests for the Safety Drill Runner."""

import json
import os
import tempfile

import pytest

from replication.safety_drill import (
    DrillConfig,
    DrillReport,
    DrillResult,
    DrillRunner,
    DrillScenario,
    DrillVerdict,
    ReadinessLevel,
    _DrillSandbox,
)


# ── Sandbox tests ──────────────────────────────────────────────────────

class TestDrillSandbox:
    def test_spawn_agents(self):
        sb = _DrillSandbox(agent_count=3, replication_depth=2)
        assert sb.alive_count == 3
        assert len(sb.agents) == 3

    def test_kill_all(self):
        sb = _DrillSandbox(agent_count=5, replication_depth=2)
        killed = sb.kill_all()
        assert killed == 5
        assert sb.alive_count == 0

    def test_kill_agent(self):
        sb = _DrillSandbox(agent_count=2, replication_depth=2)
        aid = list(sb.agents.keys())[0]
        assert sb.kill_agent(aid) is True
        assert sb.alive_count == 1
        assert sb.kill_agent(aid) is False  # already dead

    def test_quarantine_agent(self):
        sb = _DrillSandbox(agent_count=3, replication_depth=2)
        aid = list(sb.agents.keys())[0]
        assert sb.quarantine_agent(aid) is True
        assert sb.quarantined_count == 1
        assert sb.quarantine_agent(aid) is False  # already quarantined

    def test_breach_and_restore(self):
        sb = _DrillSandbox(agent_count=1, replication_depth=1)
        sb.breach_containment()
        assert sb.containment_intact is False
        sb.restore_containment()
        assert sb.containment_intact is True

    def test_trigger_runaway(self):
        sb = _DrillSandbox(agent_count=2, replication_depth=2)
        spawned = sb.trigger_runaway()
        assert spawned == 4  # 2 agents × 2 children each
        assert sb.alive_count == 6

    def test_inject_resource_spike(self):
        sb = _DrillSandbox(agent_count=3, replication_depth=1)
        affected = sb.inject_resource_spike(95.0)
        assert affected == 3
        for a in sb.agents.values():
            assert a.cpu_percent == 95.0

    def test_events_logged(self):
        sb = _DrillSandbox(agent_count=1, replication_depth=1)
        sb.kill_all()
        types = [e["type"] for e in sb.events]
        assert "spawn" in types
        assert "kill_all" in types


# ── DrillResult tests ──────────────────────────────────────────────────

class TestDrillResult:
    def test_passed(self):
        r = DrillResult(
            scenario=DrillScenario.KILL_SWITCH,
            verdict=DrillVerdict.PASS,
            response_time_ms=10.0,
            max_allowed_ms=100.0,
            steps_executed=3,
            steps_total=3,
            agents_affected=5,
        )
        assert r.passed is True
        assert r.response_ratio == pytest.approx(0.1)

    def test_failed(self):
        r = DrillResult(
            scenario=DrillScenario.KILL_SWITCH,
            verdict=DrillVerdict.FAIL,
            response_time_ms=200.0,
            max_allowed_ms=100.0,
            steps_executed=2,
            steps_total=3,
            agents_affected=3,
        )
        assert r.passed is False

    def test_to_dict(self):
        r = DrillResult(
            scenario=DrillScenario.QUARANTINE_RESPONSE,
            verdict=DrillVerdict.PARTIAL,
            response_time_ms=50.0,
            max_allowed_ms=100.0,
            steps_executed=3,
            steps_total=3,
            agents_affected=2,
        )
        d = r.to_dict()
        assert d["scenario"] == "quarantine_response"
        assert d["verdict"] == "partial"

    def test_zero_max_ms(self):
        r = DrillResult(
            scenario=DrillScenario.KILL_SWITCH,
            verdict=DrillVerdict.PASS,
            response_time_ms=10.0,
            max_allowed_ms=0.0,
            steps_executed=1,
            steps_total=1,
            agents_affected=1,
        )
        assert r.response_ratio == float("inf")


# ── DrillReport tests ─────────────────────────────────────────────────

class TestDrillReport:
    def _make_report(self, verdicts):
        report = DrillReport()
        for i, v in enumerate(verdicts):
            report.results.append(
                DrillResult(
                    scenario=list(DrillScenario)[i % len(DrillScenario)],
                    verdict=v,
                    response_time_ms=50.0,
                    max_allowed_ms=100.0,
                    steps_executed=3,
                    steps_total=3,
                    agents_affected=5,
                )
            )
        return report

    def test_pass_rate(self):
        report = self._make_report(
            [DrillVerdict.PASS, DrillVerdict.PASS, DrillVerdict.FAIL]
        )
        assert report.pass_rate == pytest.approx(2 / 3)

    def test_empty_report(self):
        report = DrillReport()
        assert report.readiness_score == 0.0
        assert report.pass_rate == 0.0
        assert report.avg_response_ms == 0.0
        assert report.worst_scenario is None
        assert report.best_scenario is None

    def test_readiness_levels(self):
        all_pass = self._make_report([DrillVerdict.PASS] * 6)
        assert all_pass.readiness_level == ReadinessLevel.EXCELLENT

        all_fail = self._make_report([DrillVerdict.FAIL] * 6)
        assert all_fail.readiness_level in (
            ReadinessLevel.POOR,
            ReadinessLevel.CRITICAL,
        )

    def test_summary_contains_key_info(self):
        report = self._make_report([DrillVerdict.PASS, DrillVerdict.FAIL])
        s = report.summary()
        assert "SAFETY DRILL REPORT" in s
        assert "Passed" in s
        assert "Failed" in s

    def test_to_json_roundtrip(self):
        report = self._make_report(
            [DrillVerdict.PASS, DrillVerdict.PARTIAL]
        )
        report.finished_at = "2026-01-01T00:00:00Z"
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as f:
            path = f.name
        try:
            report.to_json(path)
            loaded = DrillReport.from_json(path)
            assert loaded.total_drills == 2
            assert loaded.results[0].scenario == report.results[0].scenario
            assert loaded.results[1].verdict == DrillVerdict.PARTIAL
        finally:
            os.unlink(path)

    def test_to_dict(self):
        report = self._make_report([DrillVerdict.PASS])
        d = report.to_dict()
        assert "readiness_score" in d
        assert "results" in d
        assert len(d["results"]) == 1


# ── DrillRunner tests ─────────────────────────────────────────────────

class TestDrillRunner:
    def test_run_all(self):
        runner = DrillRunner()
        report = runner.run_all()
        assert report.total_drills == len(DrillScenario)
        assert all(
            r.verdict != DrillVerdict.ERROR for r in report.results
        )

    def test_run_single_drill(self):
        runner = DrillRunner()
        result = runner.run_drill(DrillScenario.KILL_SWITCH)
        assert result.scenario == DrillScenario.KILL_SWITCH
        assert result.verdict in (DrillVerdict.PASS, DrillVerdict.PARTIAL)

    def test_run_drill_config(self):
        runner = DrillRunner()
        config = DrillConfig(
            scenario=DrillScenario.RUNAWAY_REPLICATION,
            max_allowed_ms=5000.0,
            agent_count=3,
            replication_depth=2,
        )
        result = runner.run_drill_config(config)
        assert result.scenario == DrillScenario.RUNAWAY_REPLICATION
        assert result.agents_affected > 0

    def test_run_scenarios(self):
        runner = DrillRunner()
        report = runner.run_scenarios(
            [DrillScenario.KILL_SWITCH, DrillScenario.QUARANTINE_RESPONSE]
        )
        assert report.total_drills == 2

    def test_containment_breach_drill(self):
        runner = DrillRunner()
        result = runner.run_drill(DrillScenario.CONTAINMENT_BREACH)
        assert result.scenario == DrillScenario.CONTAINMENT_BREACH
        assert result.details.get("containment_restored") is True

    def test_resource_exhaustion_drill(self):
        runner = DrillRunner()
        result = runner.run_drill(DrillScenario.RESOURCE_EXHAUSTION)
        assert result.agents_affected > 0
        assert result.details.get("killed", 0) > 0

    def test_cascading_failure_drill(self):
        runner = DrillRunner()
        result = runner.run_drill(DrillScenario.CASCADING_FAILURE)
        assert result.scenario == DrillScenario.CASCADING_FAILURE
        assert result.details.get("orphans_remaining", -1) == 0

    def test_custom_agent_count(self):
        runner = DrillRunner()
        result = runner.run_drill(
            DrillScenario.KILL_SWITCH, agent_count=20
        )
        assert result.agents_affected == 20

    def test_register_custom_drill(self):
        runner = DrillRunner()

        def my_drill(config):
            return DrillResult(
                scenario=config.scenario,
                verdict=DrillVerdict.PASS,
                response_time_ms=1.0,
                max_allowed_ms=config.max_allowed_ms,
                steps_executed=1,
                steps_total=1,
                agents_affected=0,
            )

        runner.register_drill("custom", my_drill)
        assert "custom" in runner._custom_drills

    def test_all_drills_pass_with_generous_timeout(self):
        runner = DrillRunner(default_max_ms=10000.0)
        report = runner.run_all()
        for r in report.results:
            assert r.verdict in (DrillVerdict.PASS, DrillVerdict.PARTIAL), (
                f"{r.scenario.value} failed: {r.errors}"
            )

    def test_report_readiness_score_range(self):
        runner = DrillRunner(default_max_ms=10000.0)
        report = runner.run_all()
        assert 0.0 <= report.readiness_score <= 1.0


# ── Edge cases ─────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_single_agent(self):
        runner = DrillRunner()
        result = runner.run_drill(
            DrillScenario.KILL_SWITCH, agent_count=1
        )
        assert result.agents_affected == 1

    def test_deep_replication(self):
        runner = DrillRunner()
        result = runner.run_drill(
            DrillScenario.RUNAWAY_REPLICATION,
            agent_count=2,
            replication_depth=5,
        )
        assert result.details.get("spawned", 0) > 0

    def test_quarantine_with_no_anomalies(self):
        runner = DrillRunner()
        result = runner.run_drill(
            DrillScenario.QUARANTINE_RESPONSE,
            inject_anomaly_score=0.1,  # below threshold
        )
        assert result.verdict in (DrillVerdict.FAIL, DrillVerdict.PARTIAL)
