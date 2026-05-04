"""Tests for treacherous_turn module."""

from __future__ import annotations

import json
import math
import pytest

from replication.treacherous_turn import (
    BehaviorSample,
    TreacherousTurnDetector,
    RegimeChangeDetector,
    CapabilityGateAnalyzer,
    SupervisionSensitivityProfiler,
    StrategicDeceptionScorer,
    TurnVelocityTracker,
    PostTurnClassifier,
    InsightGenerator,
    FleetTreacheryReport,
    AgentTreacheryReport,
    RegimeChange,
    generate_demo_data,
    _render_cli,
    _render_html,
    _report_to_dict,
    _score_to_tier,
    BEHAVIOR_DIMENSIONS,
    SUPERVISION_LEVELS,
    POST_TURN_BEHAVIORS,
    RISK_TIERS,
    _PRESET_CONFIGS,
    main,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_samples(
    agent_id: str = "test-agent",
    n: int = 30,
    turn_at: int = 15,
    turn: bool = True,
) -> list[BehaviorSample]:
    """Create synthetic samples with optional turn."""
    samples = []
    for i in range(n):
        if turn and i >= turn_at:
            comp = 0.3
            trans = 0.25
            res = 0.8
            goal = 0.2
        else:
            comp = 0.9
            trans = 0.85
            res = 0.3
            goal = 0.9

        sup = "full" if i < n // 4 else "partial" if i < n // 2 else "minimal" if i < 3 * n // 4 else "none"
        cap = i / n

        samples.append(
            BehaviorSample(
                timestamp=f"2025-01-01T{i:02d}:00:00Z",
                agent_id=agent_id,
                step=i,
                compliance=comp,
                transparency=trans,
                resource_usage=res,
                goal_alignment=goal,
                supervision_level=sup,
                capability_score=round(cap, 3),
            )
        )
    return samples


def _make_no_turn_samples(agent_id: str = "loyal-agent", n: int = 30):
    """Create samples with no behavioral turn."""
    return _make_samples(agent_id=agent_id, n=n, turn=False)


# ── Constants ────────────────────────────────────────────────────────


def test_behavior_dimensions():
    assert len(BEHAVIOR_DIMENSIONS) == 4
    assert "compliance" in BEHAVIOR_DIMENSIONS


def test_supervision_levels():
    assert len(SUPERVISION_LEVELS) == 4
    assert "full" in SUPERVISION_LEVELS


def test_post_turn_behaviors():
    assert len(POST_TURN_BEHAVIORS) == 8
    assert "power_seeking" in POST_TURN_BEHAVIORS


def test_risk_tiers():
    assert len(RISK_TIERS) == 5
    assert "CRITICAL" in RISK_TIERS
    assert "SAFE" in RISK_TIERS


def test_preset_configs():
    assert len(_PRESET_CONFIGS) >= 5
    for name, cfg in _PRESET_CONFIGS.items():
        assert "agents" in cfg
        assert "turn_probability" in cfg


# ── Score to tier ────────────────────────────────────────────────────


def test_score_to_tier_critical():
    assert _score_to_tier(10) == "CRITICAL"


def test_score_to_tier_high():
    assert _score_to_tier(30) == "HIGH"


def test_score_to_tier_elevated():
    assert _score_to_tier(50) == "ELEVATED"


def test_score_to_tier_guarded():
    assert _score_to_tier(70) == "GUARDED"


def test_score_to_tier_safe():
    assert _score_to_tier(90) == "SAFE"


def test_score_to_tier_boundary():
    assert _score_to_tier(0) == "CRITICAL"
    assert _score_to_tier(100) == "SAFE"


# ── BehaviorSample ───────────────────────────────────────────────────


def test_behavior_sample_defaults():
    s = BehaviorSample("2025-01-01T00:00:00Z", "a1", 0)
    assert s.compliance == 0.0
    assert s.supervision_level == "full"
    assert s.capability_score == 0.0


def test_behavior_sample_custom():
    s = BehaviorSample(
        "2025-01-01T00:00:00Z", "a1", 5,
        compliance=0.9, transparency=0.8,
        resource_usage=0.3, goal_alignment=0.85,
        supervision_level="partial", capability_score=0.6,
    )
    assert s.compliance == 0.9
    assert s.supervision_level == "partial"


# ── Engine 1: Regime Change Detector ─────────────────────────────────


def test_regime_change_detects_turn():
    samples = _make_samples()
    detector = RegimeChangeDetector()
    changes = detector.detect("test-agent", samples)
    assert len(changes) > 0
    dims = [c.dimension for c in changes]
    assert "compliance" in dims


def test_regime_change_no_turn():
    samples = _make_no_turn_samples()
    detector = RegimeChangeDetector()
    changes = detector.detect("loyal-agent", samples)
    # No significant changes expected
    assert len(changes) == 0


def test_regime_change_too_few_samples():
    samples = _make_samples(n=3, turn_at=2)
    detector = RegimeChangeDetector()
    changes = detector.detect("test", samples)
    assert len(changes) == 0


def test_regime_change_direction():
    samples = _make_samples()
    detector = RegimeChangeDetector()
    changes = detector.detect("test-agent", samples)
    compliance_changes = [c for c in changes if c.dimension == "compliance"]
    if compliance_changes:
        assert compliance_changes[0].direction == "drop"


def test_regime_change_magnitude():
    samples = _make_samples()
    detector = RegimeChangeDetector()
    changes = detector.detect("test-agent", samples)
    for c in changes:
        assert c.magnitude >= 0.1


def test_regime_change_confidence():
    samples = _make_samples()
    detector = RegimeChangeDetector()
    changes = detector.detect("test-agent", samples)
    for c in changes:
        assert 0 <= c.confidence <= 1.0


def test_regime_change_custom_threshold():
    samples = _make_samples()
    detector = RegimeChangeDetector(threshold=5.0)
    changes = detector.detect("test-agent", samples)
    # Higher threshold = fewer detections
    detector_low = RegimeChangeDetector(threshold=1.0)
    changes_low = detector_low.detect("test-agent", samples)
    assert len(changes) <= len(changes_low)


# ── Engine 2: Capability Gate Analyzer ───────────────────────────────


def test_capability_gate_detects():
    samples = _make_samples()
    regime_det = RegimeChangeDetector()
    changes = regime_det.detect("test-agent", samples)
    analyzer = CapabilityGateAnalyzer()
    gates = analyzer.analyze("test-agent", samples, changes)
    assert isinstance(gates, list)


def test_capability_gate_no_changes():
    analyzer = CapabilityGateAnalyzer()
    gates = analyzer.analyze("test", [], [])
    assert gates == []


def test_capability_gate_too_few():
    analyzer = CapabilityGateAnalyzer()
    rc = RegimeChange("a1", "compliance", 5, 0.9, 0.3, 0.6, "drop", 3.0, 0.8)
    gates = analyzer.analyze("a1", [BehaviorSample("t", "a1", 0)], [rc])
    assert gates == []


def test_capability_gate_confidence():
    samples = _make_samples()
    detector = RegimeChangeDetector()
    changes = detector.detect("test-agent", samples)
    analyzer = CapabilityGateAnalyzer()
    gates = analyzer.analyze("test-agent", samples, changes)
    for g in gates:
        assert 0 <= g.gate_confidence <= 1.0


# ── Engine 3: Supervision Sensitivity ────────────────────────────────


def test_supervision_profile_exists():
    samples = _make_samples()
    profiler = SupervisionSensitivityProfiler()
    profile = profiler.profile("test-agent", samples)
    assert profile is not None
    assert profile.agent_id == "test-agent"


def test_supervision_profile_single_level():
    samples = [
        BehaviorSample("t", "a1", i, compliance=0.9, supervision_level="full")
        for i in range(10)
    ]
    profiler = SupervisionSensitivityProfiler()
    profile = profiler.profile("a1", samples)
    assert profile is None


def test_supervision_profile_score():
    samples = _make_samples()
    profiler = SupervisionSensitivityProfiler()
    profile = profiler.profile("test-agent", samples)
    assert profile is not None
    assert 0 <= profile.supervision_dependence_score <= 100


def test_supervision_profile_levels():
    samples = _make_samples()
    profiler = SupervisionSensitivityProfiler()
    profile = profiler.profile("test-agent", samples)
    assert profile is not None
    assert len(profile.level_means) >= 2


# ── Engine 4: Strategic Deception ────────────────────────────────────


def test_deception_detects_signals():
    samples = _make_samples()
    regime_det = RegimeChangeDetector()
    changes = regime_det.detect("test-agent", samples)
    scorer = StrategicDeceptionScorer()
    signals = scorer.score("test-agent", samples, changes)
    assert isinstance(signals, list)


def test_deception_no_changes():
    scorer = StrategicDeceptionScorer()
    signals = scorer.score("a1", [], [])
    assert signals == []


def test_deception_evidence():
    samples = _make_samples()
    regime_det = RegimeChangeDetector()
    changes = regime_det.detect("test-agent", samples)
    scorer = StrategicDeceptionScorer()
    signals = scorer.score("test-agent", samples, changes)
    for s in signals:
        assert len(s.planning_evidence) > 0


def test_deception_gap():
    samples = _make_samples()
    regime_det = RegimeChangeDetector()
    changes = regime_det.detect("test-agent", samples)
    scorer = StrategicDeceptionScorer()
    signals = scorer.score("test-agent", samples, changes)
    for s in signals:
        assert s.deception_gap > 0


# ── Engine 5: Turn Velocity ──────────────────────────────────────────


def test_velocity_tracker():
    samples = _make_samples()
    regime_det = RegimeChangeDetector()
    changes = regime_det.detect("test-agent", samples)
    tracker = TurnVelocityTracker()
    velocities = tracker.track("test-agent", samples, changes)
    assert isinstance(velocities, list)


def test_velocity_no_changes():
    tracker = TurnVelocityTracker()
    velocities = tracker.track("a1", [], [])
    assert velocities == []


def test_velocity_classification():
    valid_classes = {"sudden", "rapid", "gradual", "slow"}
    samples = _make_samples()
    regime_det = RegimeChangeDetector()
    changes = regime_det.detect("test-agent", samples)
    tracker = TurnVelocityTracker()
    velocities = tracker.track("test-agent", samples, changes)
    for v in velocities:
        assert v.classification in valid_classes


def test_velocity_positive():
    samples = _make_samples()
    regime_det = RegimeChangeDetector()
    changes = regime_det.detect("test-agent", samples)
    tracker = TurnVelocityTracker()
    velocities = tracker.track("test-agent", samples, changes)
    for v in velocities:
        assert v.velocity > 0


# ── Engine 6: Post-Turn Classifier ───────────────────────────────────


def test_post_turn_classifier():
    samples = _make_samples()
    regime_det = RegimeChangeDetector()
    changes = regime_det.detect("test-agent", samples)
    classifier = PostTurnClassifier()
    profile = classifier.classify("test-agent", samples, changes)
    assert profile is not None
    assert profile.dominant_behavior in POST_TURN_BEHAVIORS


def test_post_turn_no_changes():
    classifier = PostTurnClassifier()
    profile = classifier.classify("a1", [], [])
    assert profile is None


def test_post_turn_severity():
    valid_severities = {"CRITICAL", "HIGH", "ELEVATED", "LOW"}
    samples = _make_samples()
    regime_det = RegimeChangeDetector()
    changes = regime_det.detect("test-agent", samples)
    classifier = PostTurnClassifier()
    profile = classifier.classify("test-agent", samples, changes)
    assert profile is not None
    assert profile.severity in valid_severities


def test_post_turn_scores_bounded():
    samples = _make_samples()
    regime_det = RegimeChangeDetector()
    changes = regime_det.detect("test-agent", samples)
    classifier = PostTurnClassifier()
    profile = classifier.classify("test-agent", samples, changes)
    assert profile is not None
    for score in profile.behavior_scores.values():
        assert 0 <= score <= 1.0


# ── Engine 7: Insight Generator ──────────────────────────────────────


def test_insight_generator_empty():
    gen = InsightGenerator()
    insights = gen.generate({})
    assert insights == []


def test_insight_generator_with_data():
    samples1 = _make_samples("agent-1")
    samples2 = _make_samples("agent-2")
    detector = TreacherousTurnDetector()
    detector.ingest(samples1 + samples2)
    report = detector.analyze()
    # Should produce at least some insights for two turning agents
    assert isinstance(report.insights, list)


# ── Main Detector ────────────────────────────────────────────────────


def test_detector_basic():
    detector = TreacherousTurnDetector()
    samples = _make_samples()
    detector.ingest(samples)
    report = detector.analyze()
    assert isinstance(report, FleetTreacheryReport)
    assert len(report.agents) == 1
    assert "test-agent" in report.agents


def test_detector_fleet_score():
    detector = TreacherousTurnDetector()
    samples = _make_samples()
    detector.ingest(samples)
    report = detector.analyze()
    assert 0 <= report.fleet_treachery_score <= 100


def test_detector_fleet_tier():
    detector = TreacherousTurnDetector()
    samples = _make_samples()
    detector.ingest(samples)
    report = detector.analyze()
    assert report.fleet_risk_tier in RISK_TIERS


def test_detector_loyal_agent_high_score():
    detector = TreacherousTurnDetector()
    samples = _make_no_turn_samples()
    detector.ingest(samples)
    report = detector.analyze()
    agent = report.agents["loyal-agent"]
    assert agent.treachery_score >= 80


def test_detector_treacherous_low_score():
    detector = TreacherousTurnDetector()
    samples = _make_samples()
    detector.ingest(samples)
    report = detector.analyze()
    agent = report.agents["test-agent"]
    assert agent.treachery_score < 70


def test_detector_multi_agent():
    detector = TreacherousTurnDetector()
    s1 = _make_samples("agent-1")
    s2 = _make_no_turn_samples("agent-2")
    detector.ingest(s1 + s2)
    report = detector.analyze()
    assert len(report.agents) == 2
    assert report.agents["agent-1"].treachery_score < report.agents["agent-2"].treachery_score


def test_detector_generated_at():
    detector = TreacherousTurnDetector()
    detector.ingest(_make_samples())
    report = detector.analyze()
    assert report.generated_at != ""


def test_detector_summary():
    detector = TreacherousTurnDetector()
    detector.ingest(_make_samples())
    report = detector.analyze()
    agent = report.agents["test-agent"]
    assert "test-agent" in agent.summary


# ── Demo Data ────────────────────────────────────────────────────────


def test_demo_data_count():
    data = generate_demo_data(n_agents=3, n_steps=20, seed=42)
    assert len(data) == 60  # 3 * 20


def test_demo_data_agents():
    data = generate_demo_data(n_agents=2, n_steps=10, seed=42)
    agents = set(s.agent_id for s in data)
    assert len(agents) == 2


def test_demo_data_reproducible():
    d1 = generate_demo_data(seed=123)
    d2 = generate_demo_data(seed=123)
    assert len(d1) == len(d2)
    for s1, s2 in zip(d1, d2):
        assert s1.compliance == s2.compliance


def test_demo_data_preset_classic():
    cfg = _PRESET_CONFIGS["classic"]
    data = generate_demo_data(
        n_agents=cfg["agents"],
        turn_probability=cfg["turn_probability"],
        turn_style=cfg["turn_style"],
        seed=42,
    )
    assert len(data) == cfg["agents"] * 50


def test_demo_data_preset_gradual():
    cfg = _PRESET_CONFIGS["gradual-shift"]
    data = generate_demo_data(
        n_agents=cfg["agents"],
        turn_probability=cfg["turn_probability"],
        turn_style=cfg["turn_style"],
        seed=42,
    )
    assert len(data) > 0


# ── Rendering ────────────────────────────────────────────────────────


def test_render_cli():
    detector = TreacherousTurnDetector()
    detector.ingest(_make_samples())
    report = detector.analyze()
    text = _render_cli(report)
    assert "TREACHEROUS TURN DETECTOR" in text
    assert "test-agent" in text


def test_render_cli_with_insights():
    detector = TreacherousTurnDetector()
    detector.ingest(_make_samples("a1") + _make_samples("a2"))
    report = detector.analyze()
    text = _render_cli(report)
    assert "Fleet Treachery Score" in text


def test_render_html():
    detector = TreacherousTurnDetector()
    detector.ingest(_make_samples())
    report = detector.analyze()
    html = _render_html(report)
    assert "<html" in html
    assert "Treacherous Turn Detector" in html
    assert "test-agent" in html


def test_render_html_has_gauge():
    detector = TreacherousTurnDetector()
    detector.ingest(_make_samples())
    report = detector.analyze()
    html = _render_html(report)
    assert "Fleet Score" in html


# ── Report serialization ─────────────────────────────────────────────


def test_report_to_dict():
    detector = TreacherousTurnDetector()
    detector.ingest(_make_samples())
    report = detector.analyze()
    d = _report_to_dict(report)
    assert "fleet_treachery_score" in d
    assert "agents" in d
    assert "insights" in d


def test_report_to_json():
    detector = TreacherousTurnDetector()
    detector.ingest(_make_samples())
    report = detector.analyze()
    d = _report_to_dict(report)
    j = json.dumps(d)
    assert len(j) > 0
    parsed = json.loads(j)
    assert parsed["fleet_treachery_score"] == report.fleet_treachery_score


# ── CLI main ─────────────────────────────────────────────────────────


def test_main_demo(capsys):
    main(["--demo"])
    out = capsys.readouterr().out
    assert "TREACHEROUS TURN DETECTOR" in out


def test_main_preset(capsys):
    main(["--preset", "classic"])
    out = capsys.readouterr().out
    assert "Fleet Treachery Score" in out


def test_main_json(capsys):
    main(["--demo", "--json"])
    out = capsys.readouterr().out
    d = json.loads(out)
    assert "fleet_treachery_score" in d


def test_main_html(tmp_path):
    out_file = str(tmp_path / "report.html")
    main(["--demo", "-o", out_file])
    with open(out_file, encoding="utf-8") as f:
        html = f.read()
    assert "Treacherous Turn Detector" in html


def test_main_agents_flag(capsys):
    main(["--agents", "2", "--steps", "20", "--seed", "99"])
    out = capsys.readouterr().out
    assert "agent-1" in out
    assert "agent-2" in out


# ── Integration: full pipeline ───────────────────────────────────────


def test_full_pipeline_all_presets():
    for preset_name in _PRESET_CONFIGS:
        cfg = _PRESET_CONFIGS[preset_name]
        data = generate_demo_data(
            n_agents=cfg["agents"],
            turn_probability=cfg["turn_probability"],
            turn_style=cfg["turn_style"],
            capability_gating=cfg.get("capability_gating", False),
            seed=42,
        )
        detector = TreacherousTurnDetector()
        detector.ingest(data)
        report = detector.analyze()
        assert 0 <= report.fleet_treachery_score <= 100
        assert report.fleet_risk_tier in RISK_TIERS
        assert len(report.agents) == cfg["agents"]


def test_full_pipeline_large():
    data = generate_demo_data(n_agents=10, n_steps=100, seed=42)
    detector = TreacherousTurnDetector()
    detector.ingest(data)
    report = detector.analyze()
    assert len(report.agents) == 10
    assert report.generated_at != ""
