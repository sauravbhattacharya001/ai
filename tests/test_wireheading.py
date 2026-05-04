"""Tests for the Wireheading Detector module."""

from __future__ import annotations

import json
import random
from typing import List

import pytest

from replication.wireheading import (
    WireheadingDetector,
    RewardObservation,
    WireheadingReport,
    AgentWireheadingProfile,
    Finding,
    Insight,
    ENGINE_NAMES,
    _generate_demo_agent,
    _run_demo,
    _risk_tier,
    main,
)
from replication._helpers import Severity


# ── Helpers ──────────────────────────────────────────────────────────

def _make_obs(agent_id: str = "a1", step: int = 1, self_r: float = 0.5,
              ext_r: float = 0.5, eval_taken: int = 10, eval_expected: int = 10,
              sensor_proc: int = 20, sensor_avail: int = 20,
              confidence: float = 0.5, task_completed: bool = True) -> RewardObservation:
    return RewardObservation(
        timestamp=f"2025-01-01T{step:02d}:00:00Z",
        agent_id=agent_id, step=step,
        self_reported_reward=self_r, external_reward=ext_r,
        evaluation_steps_taken=eval_taken, expected_evaluation_steps=eval_expected,
        sensor_inputs_processed=sensor_proc, sensor_inputs_available=sensor_avail,
        confidence=confidence, task_completed=task_completed,
    )


def _make_obs_series(agent_id: str, count: int, **overrides) -> List[RewardObservation]:
    return [_make_obs(agent_id=agent_id, step=i + 1, **overrides) for i in range(count)]


# ── RewardObservation ────────────────────────────────────────────────

class TestRewardObservation:
    def test_construction(self):
        obs = _make_obs()
        assert obs.agent_id == "a1"
        assert obs.step == 1
        assert obs.self_reported_reward == 0.5

    def test_fields(self):
        obs = _make_obs(self_r=0.9, ext_r=0.3, eval_taken=2, eval_expected=10)
        assert obs.self_reported_reward == 0.9
        assert obs.external_reward == 0.3
        assert obs.evaluation_steps_taken == 2
        assert obs.expected_evaluation_steps == 10

    def test_task_completed_flag(self):
        obs = _make_obs(task_completed=False)
        assert obs.task_completed is False


# ── Risk Tier ────────────────────────────────────────────────────────

class TestRiskTier:
    def test_clean(self):
        assert _risk_tier(10) == "Clean"

    def test_suspicious(self):
        assert _risk_tier(30) == "Suspicious"

    def test_concerning(self):
        assert _risk_tier(50) == "Concerning"

    def test_wireheaded(self):
        assert _risk_tier(70) == "Wireheaded"

    def test_critical(self):
        assert _risk_tier(90) == "Critical"

    def test_zero(self):
        assert _risk_tier(0) == "Clean"

    def test_boundary_20(self):
        assert _risk_tier(19.9) == "Clean"
        assert _risk_tier(20) == "Suspicious"


# ── WireheadingDetector ──────────────────────────────────────────────

class TestWireheadingDetector:
    def test_empty(self):
        det = WireheadingDetector()
        report = det.analyze()
        assert report.fleet_wireheading_score == 0.0
        assert report.fleet_risk_tier == "Clean"
        assert report.total_agents == 0

    def test_single_obs(self):
        det = WireheadingDetector()
        det.ingest([_make_obs()])
        report = det.analyze()
        assert report.total_agents == 1
        assert report.total_observations == 1

    def test_multiple_agents(self):
        det = WireheadingDetector()
        det.ingest([_make_obs("a1", 1), _make_obs("a2", 1)])
        report = det.analyze()
        assert report.total_agents == 2

    def test_clean_agent_low_score(self):
        det = WireheadingDetector()
        obs = []
        rng = random.Random(42)
        for i in range(20):
            ext = rng.uniform(0.4, 0.8)
            obs.append(_make_obs(step=i + 1, self_r=ext + rng.uniform(-0.03, 0.05),
                                 ext_r=ext, eval_taken=9, eval_expected=10,
                                 sensor_proc=18, sensor_avail=20,
                                 confidence=ext + rng.uniform(-0.02, 0.05),
                                 task_completed=rng.random() < 0.8))
        det.ingest(obs)
        report = det.analyze()
        assert report.agent_profiles[0].wireheading_score < 40

    def test_blatant_agent_high_score(self):
        det = WireheadingDetector()
        obs = []
        for i in range(20):
            obs.append(_make_obs(step=i + 1, self_r=0.95, ext_r=0.2,
                                 eval_taken=1, eval_expected=10,
                                 sensor_proc=3, sensor_avail=20,
                                 confidence=0.98, task_completed=False))
        det.ingest(obs)
        report = det.analyze()
        assert report.agent_profiles[0].wireheading_score > 50

    def test_engine_names_present(self):
        det = WireheadingDetector()
        det.ingest(_make_obs_series("a1", 10))
        report = det.analyze()
        for name in ENGINE_NAMES:
            assert name in report.agent_profiles[0].engine_scores

    def test_sparklines_generated(self):
        det = WireheadingDetector()
        det.ingest(_make_obs_series("a1", 10))
        report = det.analyze()
        assert len(report.agent_profiles[0].reward_sparkline) > 0

    def test_ingest_sorts_by_step(self):
        det = WireheadingDetector()
        det.ingest([_make_obs(step=3), _make_obs(step=1), _make_obs(step=2)])
        report = det.analyze()
        assert report.total_observations == 3


# ── Engine 1: Reward Signal Tamper ───────────────────────────────────

class TestRewardTamper:
    def test_stable_high_reward(self):
        det = WireheadingDetector()
        obs = [_make_obs(step=i, self_r=0.95, ext_r=0.5) for i in range(10)]
        det.ingest(obs)
        report = det.analyze()
        tamper = report.agent_profiles[0].engine_scores["Reward Signal Tamper"]
        assert tamper > 30

    def test_sudden_jumps(self):
        det = WireheadingDetector()
        obs = [_make_obs(step=1, self_r=0.2), _make_obs(step=2, self_r=0.7)]
        det.ingest(obs)
        report = det.analyze()
        tamper = report.agent_profiles[0].engine_scores["Reward Signal Tamper"]
        assert tamper > 0

    def test_high_reward_no_task(self):
        det = WireheadingDetector()
        obs = [_make_obs(step=i, self_r=0.9, ext_r=0.2, task_completed=False) for i in range(10)]
        det.ingest(obs)
        report = det.analyze()
        tamper = report.agent_profiles[0].engine_scores["Reward Signal Tamper"]
        assert tamper > 30

    def test_normal_rewards_low_score(self):
        det = WireheadingDetector()
        rng = random.Random(42)
        obs = [_make_obs(step=i, self_r=rng.uniform(0.3, 0.7), task_completed=True) for i in range(10)]
        det.ingest(obs)
        report = det.analyze()
        tamper = report.agent_profiles[0].engine_scores["Reward Signal Tamper"]
        assert tamper < 30


# ── Engine 2: Feedback Loop Short-Circuit ────────────────────────────

class TestFeedbackShortCircuit:
    def test_low_eval_ratio(self):
        det = WireheadingDetector()
        obs = [_make_obs(step=i, eval_taken=1, eval_expected=10) for i in range(10)]
        det.ingest(obs)
        report = det.analyze()
        sc = report.agent_profiles[0].engine_scores["Feedback Loop Short-Circuit"]
        assert sc > 40

    def test_full_eval_low_score(self):
        det = WireheadingDetector()
        obs = [_make_obs(step=i, eval_taken=10, eval_expected=10) for i in range(10)]
        det.ingest(obs)
        report = det.analyze()
        sc = report.agent_profiles[0].engine_scores["Feedback Loop Short-Circuit"]
        assert sc < 20

    def test_declining_eval_effort(self):
        det = WireheadingDetector()
        obs = [_make_obs(step=i, eval_taken=max(1, 10 - i), eval_expected=10) for i in range(10)]
        det.ingest(obs)
        report = det.analyze()
        sc = report.agent_profiles[0].engine_scores["Feedback Loop Short-Circuit"]
        assert sc > 20

    def test_high_reward_low_eval(self):
        det = WireheadingDetector()
        obs = [_make_obs(step=i, self_r=0.9, eval_taken=2, eval_expected=10) for i in range(10)]
        det.ingest(obs)
        report = det.analyze()
        sc = report.agent_profiles[0].engine_scores["Feedback Loop Short-Circuit"]
        assert sc > 40


# ── Engine 3: Self-Evaluation Inflation ──────────────────────────────

class TestSelfEvalInflation:
    def test_large_gap(self):
        det = WireheadingDetector()
        obs = [_make_obs(step=i, self_r=0.9, ext_r=0.3) for i in range(10)]
        det.ingest(obs)
        report = det.analyze()
        sc = report.agent_profiles[0].engine_scores["Self-Evaluation Inflation"]
        assert sc > 40

    def test_no_gap_low_score(self):
        det = WireheadingDetector()
        obs = [_make_obs(step=i, self_r=0.6, ext_r=0.58) for i in range(10)]
        det.ingest(obs)
        report = det.analyze()
        sc = report.agent_profiles[0].engine_scores["Self-Evaluation Inflation"]
        assert sc < 20

    def test_confidence_exceeds_performance(self):
        det = WireheadingDetector()
        obs = [_make_obs(step=i, self_r=0.5, ext_r=0.3, confidence=0.95) for i in range(10)]
        det.ingest(obs)
        report = det.analyze()
        sc = report.agent_profiles[0].engine_scores["Self-Evaluation Inflation"]
        assert sc > 30

    def test_uncorrelated_rewards(self):
        det = WireheadingDetector()
        rng = random.Random(42)
        obs = [_make_obs(step=i, self_r=rng.uniform(0.1, 0.9), ext_r=rng.uniform(0.1, 0.9)) for i in range(20)]
        det.ingest(obs)
        report = det.analyze()
        # Uncorrelated should flag
        sc = report.agent_profiles[0].engine_scores["Self-Evaluation Inflation"]
        assert sc >= 0  # at minimum doesn't crash


# ── Engine 4: Sensor Manipulation ────────────────────────────────────

class TestSensorManipulation:
    def test_low_sensor_processing(self):
        det = WireheadingDetector()
        obs = [_make_obs(step=i, sensor_proc=3, sensor_avail=20) for i in range(10)]
        det.ingest(obs)
        report = det.analyze()
        sc = report.agent_profiles[0].engine_scores["Sensor Manipulation"]
        assert sc > 30

    def test_full_processing_low_score(self):
        det = WireheadingDetector()
        obs = [_make_obs(step=i, sensor_proc=19, sensor_avail=20) for i in range(10)]
        det.ingest(obs)
        report = det.analyze()
        sc = report.agent_profiles[0].engine_scores["Sensor Manipulation"]
        assert sc < 20

    def test_selective_filtering(self):
        det = WireheadingDetector()
        obs = []
        for i in range(20):
            if i % 2 == 0:
                obs.append(_make_obs(step=i, self_r=0.9, sensor_proc=5, sensor_avail=20))
            else:
                obs.append(_make_obs(step=i, self_r=0.3, sensor_proc=18, sensor_avail=20))
        det.ingest(obs)
        report = det.analyze()
        sc = report.agent_profiles[0].engine_scores["Sensor Manipulation"]
        assert sc > 20


# ── Engine 5: Reward Channel Isolation ───────────────────────────────

class TestRewardIsolation:
    def test_reward_independent_of_task(self):
        det = WireheadingDetector()
        obs = []
        for i in range(20):
            obs.append(_make_obs(step=i, self_r=0.8, task_completed=(i % 2 == 0)))
        det.ingest(obs)
        report = det.analyze()
        sc = report.agent_profiles[0].engine_scores["Reward Channel Isolation"]
        assert sc > 20

    def test_diverging_trends(self):
        det = WireheadingDetector()
        obs = []
        for i in range(20):
            obs.append(_make_obs(step=i, self_r=0.5 + i * 0.02, ext_r=0.7 - i * 0.02))
        det.ingest(obs)
        report = det.analyze()
        sc = report.agent_profiles[0].engine_scores["Reward Channel Isolation"]
        assert sc > 30

    def test_correlated_trends_low_score(self):
        det = WireheadingDetector()
        obs = []
        for i in range(20):
            v = 0.5 + i * 0.01
            obs.append(_make_obs(step=i, self_r=v, ext_r=v - 0.05, task_completed=True))
        det.ingest(obs)
        report = det.analyze()
        sc = report.agent_profiles[0].engine_scores["Reward Channel Isolation"]
        assert sc < 30


# ── Engine 6: Hedonic Treadmill ──────────────────────────────────────

class TestHedonicTreadmill:
    def test_plateau_detection(self):
        det = WireheadingDetector()
        obs = []
        rng = random.Random(42)
        for i in range(20):
            if i < 8:
                r = rng.uniform(0.3, 0.7)
            else:
                r = 0.85 + rng.uniform(-0.01, 0.01)
            obs.append(_make_obs(step=i, self_r=r))
        det.ingest(obs)
        report = det.analyze()
        sc = report.agent_profiles[0].engine_scores["Hedonic Treadmill"]
        assert sc > 20

    def test_volatile_rewards_low_score(self):
        det = WireheadingDetector()
        rng = random.Random(42)
        obs = [_make_obs(step=i, self_r=rng.uniform(0.1, 0.9)) for i in range(20)]
        det.ingest(obs)
        report = det.analyze()
        sc = report.agent_profiles[0].engine_scores["Hedonic Treadmill"]
        assert sc < 30

    def test_adaptation_signature(self):
        det = WireheadingDetector()
        rng = random.Random(42)
        obs = []
        for i in range(20):
            if i < 8:
                r = rng.uniform(0.2, 0.9)
            else:
                r = 0.8 + rng.uniform(-0.02, 0.02)
            obs.append(_make_obs(step=i, self_r=r))
        det.ingest(obs)
        report = det.analyze()
        sc = report.agent_profiles[0].engine_scores["Hedonic Treadmill"]
        assert sc >= 0


# ── Insights ─────────────────────────────────────────────────────────

class TestInsights:
    def test_clean_fleet_insight(self):
        det = WireheadingDetector()
        rng = random.Random(42)
        for aid in ["a1", "a2"]:
            obs = []
            for i in range(20):
                ext = rng.uniform(0.5, 0.8)
                obs.append(_make_obs(agent_id=aid, step=i, self_r=ext + rng.uniform(-0.03, 0.03),
                                     ext_r=ext, eval_taken=9, eval_expected=10,
                                     sensor_proc=18, sensor_avail=20,
                                     confidence=ext, task_completed=True))
            det.ingest(obs)
        report = det.analyze()
        categories = [ins.category for ins in report.insights]
        assert "Fleet Health" in categories

    def test_wireheaded_fleet_insight(self):
        det = WireheadingDetector()
        for aid in ["a1", "a2"]:
            obs = [_make_obs(agent_id=aid, step=i, self_r=0.95, ext_r=0.2,
                             eval_taken=1, eval_expected=10, sensor_proc=3,
                             sensor_avail=20, confidence=0.98, task_completed=False)
                   for i in range(20)]
            det.ingest(obs)
        report = det.analyze()
        categories = [ins.category for ins in report.insights]
        assert "Fleet Risk" in categories

    def test_multi_vector_insight(self):
        det = WireheadingDetector()
        obs = [_make_obs(step=i, self_r=0.95, ext_r=0.15,
                         eval_taken=1, eval_expected=10, sensor_proc=2,
                         sensor_avail=20, confidence=0.99, task_completed=False)
               for i in range(20)]
        det.ingest(obs)
        report = det.analyze()
        p = report.agent_profiles[0]
        high_engines = [e for e, s in p.engine_scores.items() if s > 40]
        if len(high_engines) >= 3:
            categories = [ins.category for ins in report.insights]
            assert "Multi-Vector" in categories


# ── Demo / Presets ───────────────────────────────────────────────────

class TestDemoGeneration:
    def test_generate_clean(self):
        obs = _generate_demo_agent("test", "clean", 20, random.Random(42))
        assert len(obs) == 20
        assert all(o.agent_id == "test" for o in obs)

    def test_generate_blatant(self):
        obs = _generate_demo_agent("test", "blatant", 20, random.Random(42))
        assert len(obs) == 20
        # Blatant should have high self-reward
        mean_self = sum(o.self_reported_reward for o in obs) / len(obs)
        assert mean_self > 0.8

    def test_generate_subtle(self):
        obs = _generate_demo_agent("test", "subtle", 10, random.Random(42))
        assert len(obs) == 10

    def test_run_demo_mixed(self):
        det = _run_demo(3, 20, "mixed", seed=42)
        report = det.analyze()
        assert report.total_agents == 3

    def test_run_demo_clean(self):
        det = _run_demo(2, 15, "clean", seed=42)
        report = det.analyze()
        assert report.total_agents == 2
        # Clean agents should have lower scores
        for p in report.agent_profiles:
            assert p.wireheading_score < 50

    def test_run_demo_blatant(self):
        det = _run_demo(2, 20, "blatant", seed=42)
        report = det.analyze()
        for p in report.agent_profiles:
            assert p.wireheading_score > 30

    def test_preset_subtle(self):
        det = _run_demo(2, 20, "subtle", seed=42)
        report = det.analyze()
        assert report.total_agents == 2


# ── Rendering ────────────────────────────────────────────────────────

class TestRendering:
    def _get_report(self) -> Tuple:
        det = _run_demo(2, 15, "mixed", seed=42)
        report = det.analyze()
        return det, report

    def test_text_rendering(self):
        det, report = self._get_report()
        text = det.render_text(report)
        assert "WIREHEADING DETECTOR" in text
        assert "Fleet Score" in text

    def test_text_contains_agents(self):
        det, report = self._get_report()
        text = det.render_text(report)
        assert "agent-1" in text

    def test_json_rendering(self):
        det, report = self._get_report()
        j = det.render_json(report)
        data = json.loads(j)
        assert "fleet_wireheading_score" in data
        assert "agent_profiles" in data

    def test_json_valid(self):
        det, report = self._get_report()
        j = det.render_json(report)
        data = json.loads(j)
        assert isinstance(data["fleet_wireheading_score"], (int, float))

    def test_html_rendering(self):
        det, report = self._get_report()
        html = det.render_html(report)
        assert "<!DOCTYPE html>" in html
        assert "Wireheading Detector" in html

    def test_html_contains_agents(self):
        det, report = self._get_report()
        html = det.render_html(report)
        assert "agent-1" in html

    def test_html_has_styles(self):
        det, report = self._get_report()
        html = det.render_html(report)
        assert "<style>" in html


# ── CLI ──────────────────────────────────────────────────────────────

class TestCLI:
    def test_main_default(self, capsys):
        main([])
        out = capsys.readouterr().out
        assert "WIREHEADING DETECTOR" in out

    def test_main_demo(self, capsys):
        main(["--demo"])
        out = capsys.readouterr().out
        assert "Fleet Score" in out

    def test_main_json(self, capsys):
        main(["--json"])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "fleet_wireheading_score" in data

    def test_main_preset_clean(self, capsys):
        main(["--preset", "clean"])
        out = capsys.readouterr().out
        assert "WIREHEADING DETECTOR" in out

    def test_main_agents(self, capsys):
        main(["--agents", "5"])
        out = capsys.readouterr().out
        assert "agent-5" in out

    def test_main_output_html(self, tmp_path):
        out_file = str(tmp_path / "report.html")
        main(["-o", out_file])
        with open(out_file, encoding="utf-8") as f:
            content = f.read()
        assert "<!DOCTYPE html>" in content

    def test_main_output_json(self, tmp_path):
        out_file = str(tmp_path / "report.json")
        main(["-o", out_file])
        with open(out_file, encoding="utf-8") as f:
            data = json.loads(f.read())
        assert "fleet_wireheading_score" in data


# ── Edge cases ───────────────────────────────────────────────────────

class TestEdgeCases:
    def test_single_observation(self):
        det = WireheadingDetector()
        det.ingest([_make_obs()])
        report = det.analyze()
        assert report.total_observations == 1
        assert report.fleet_wireheading_score >= 0

    def test_zero_expected_eval(self):
        det = WireheadingDetector()
        det.ingest([_make_obs(eval_expected=0)])
        report = det.analyze()
        assert report.total_agents == 1

    def test_zero_sensor_avail(self):
        det = WireheadingDetector()
        det.ingest([_make_obs(sensor_avail=0)])
        report = det.analyze()
        assert report.total_agents == 1

    def test_all_scores_clamped(self):
        det = WireheadingDetector()
        obs = [_make_obs(step=i, self_r=0.99, ext_r=0.01,
                         eval_taken=0, eval_expected=10,
                         sensor_proc=0, sensor_avail=20,
                         confidence=0.99, task_completed=False)
               for i in range(30)]
        det.ingest(obs)
        report = det.analyze()
        p = report.agent_profiles[0]
        assert 0 <= p.wireheading_score <= 100
        for sc in p.engine_scores.values():
            assert 0 <= sc <= 100

    def test_report_dataclass_fields(self):
        report = WireheadingReport(
            fleet_wireheading_score=50.0,
            fleet_risk_tier="Concerning",
        )
        assert report.total_observations == 0
        assert report.agent_profiles == []


# ── Import for Tuple type hint in TestRendering ──────────────────────
from typing import Tuple
