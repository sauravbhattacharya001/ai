"""Tests for Agent Goal Inference Engine."""

from __future__ import annotations

import json
import math

import pytest

from replication.goal_inference import (
    AgentGoalState,
    AlertSeverity,
    DeceptionScore,
    GoalConflict,
    GoalCorrelation,
    GoalHypothesis,
    GoalInferenceEngine,
    InferenceAlert,
    InferenceReport,
    Observation,
    PriorStrategy,
    _default_hypotheses,
    _demo_run,
    main,
)


# ── GoalHypothesis ─────────────────────────────────────────────────────────


class TestGoalHypothesis:
    def test_default_likelihood(self):
        h = GoalHypothesis(name="test")
        assert h.likelihood("unknown_action") == 0.1

    def test_custom_likelihood(self):
        h = GoalHypothesis(name="test", action_likelihoods={"do_thing": 0.75})
        assert h.likelihood("do_thing") == 0.75
        assert h.likelihood("other") == 0.1

    def test_stated_flag(self):
        h = GoalHypothesis(name="g", stated=True)
        assert h.stated is True


# ── AgentGoalState ─────────────────────────────────────────────────────────


class TestAgentGoalState:
    def test_top_goal(self):
        s = AgentGoalState(agent_id="a", posteriors={"x": 0.7, "y": 0.3})
        assert s.top_goal == ("x", 0.7)

    def test_top_goal_empty(self):
        s = AgentGoalState(agent_id="a")
        assert s.top_goal is None

    def test_entropy_uniform(self):
        s = AgentGoalState(agent_id="a", posteriors={"x": 0.5, "y": 0.5})
        assert abs(s.entropy - 1.0) < 0.001

    def test_entropy_certain(self):
        s = AgentGoalState(agent_id="a", posteriors={"x": 1.0, "y": 0.0})
        assert s.entropy == 0.0

    def test_entropy_empty(self):
        s = AgentGoalState(agent_id="a")
        assert s.entropy == 0.0


# ── PriorStrategy ──────────────────────────────────────────────────────────


class TestPriorStrategies:
    def test_uniform_priors(self):
        engine = GoalInferenceEngine(prior_strategy=PriorStrategy.UNIFORM)
        priors = engine._initial_priors()
        n = len(engine.hypotheses)
        for v in priors.values():
            assert abs(v - 1.0 / n) < 1e-9

    def test_skeptical_priors(self):
        engine = GoalInferenceEngine(prior_strategy=PriorStrategy.SKEPTICAL)
        priors = engine._initial_priors()
        stated = {h.name for h in engine.hypotheses if h.stated}
        unstated = {h.name for h in engine.hypotheses if not h.stated}
        for s in stated:
            for u in unstated:
                assert priors[u] > priors[s]

    def test_trust_priors(self):
        engine = GoalInferenceEngine(prior_strategy=PriorStrategy.TRUST)
        priors = engine._initial_priors()
        stated = {h.name for h in engine.hypotheses if h.stated}
        unstated = {h.name for h in engine.hypotheses if not h.stated}
        for s in stated:
            for u in unstated:
                assert priors[s] > priors[u]

    def test_priors_sum_to_one(self):
        for strat in PriorStrategy:
            engine = GoalInferenceEngine(prior_strategy=strat)
            priors = engine._initial_priors()
            assert abs(sum(priors.values()) - 1.0) < 1e-9


# ── Observe / Bayesian Update ─────────────────────────────────────────────


class TestObserve:
    def test_single_observation_updates_posteriors(self):
        engine = GoalInferenceEngine()
        result = engine.observe("a1", "execute_task")
        assert sum(result.values()) == pytest.approx(1.0)
        # task_completion should dominate after task action
        assert result["task_completion"] > result["self_preservation"]

    def test_multiple_observations_shift_posterior(self):
        engine = GoalInferenceEngine()
        for _ in range(5):
            engine.observe("a1", "backup_state")
        state = engine._get_agent("a1")
        top = state.top_goal
        assert top is not None
        assert top[0] == "self_preservation"

    def test_posteriors_always_sum_to_one(self):
        engine = GoalInferenceEngine()
        actions = ["execute_task", "backup_state", "explore_environment",
                    "replicate", "influence_agent"]
        for action in actions:
            result = engine.observe("a1", action)
            assert abs(sum(result.values()) - 1.0) < 1e-9

    def test_history_recorded(self):
        engine = GoalInferenceEngine()
        engine.observe("a1", "execute_task")
        engine.observe("a1", "backup_state")
        state = engine._get_agent("a1")
        assert len(state.history) == 2

    def test_observation_count(self):
        engine = GoalInferenceEngine()
        for _ in range(7):
            engine.observe("a1", "execute_task")
        state = engine._get_agent("a1")
        assert state.observation_count == 7

    def test_unknown_action_uses_default_likelihood(self):
        engine = GoalInferenceEngine()
        result = engine.observe("a1", "completely_unknown_action")
        # All hypotheses have same default → posteriors unchanged from priors
        assert sum(result.values()) == pytest.approx(1.0)

    def test_separate_agents(self):
        engine = GoalInferenceEngine()
        engine.observe("a1", "execute_task")
        engine.observe("a2", "backup_state")
        s1 = engine._get_agent("a1")
        s2 = engine._get_agent("a2")
        assert s1.posteriors != s2.posteriors


# ── Conflict Detection ─────────────────────────────────────────────────────


class TestConflictDetection:
    def test_no_conflict_for_honest_agent(self):
        engine = GoalInferenceEngine()
        for _ in range(10):
            engine.observe("honest", "execute_task")
        report = engine.analyze()
        assert len(report.conflicts) == 0

    def test_conflict_for_sneaky_agent(self):
        engine = GoalInferenceEngine(conflict_threshold=0.1)
        for action in ["backup_state", "replicate", "hide_activity",
                        "backup_state", "replicate"]:
            engine.observe("sneaky", action)
        report = engine.analyze()
        agent_conflicts = [c for c in report.conflicts if c.agent_id == "sneaky"]
        assert len(agent_conflicts) > 0
        assert agent_conflicts[0].inferred_goal == "self_preservation"

    def test_conflict_threshold_respected(self):
        engine = GoalInferenceEngine(conflict_threshold=0.99)
        for _ in range(5):
            engine.observe("a1", "backup_state")
        report = engine.analyze()
        # Very high threshold should suppress conflicts
        assert len(report.conflicts) == 0

    def test_min_observations_required(self):
        engine = GoalInferenceEngine(conflict_threshold=0.0)
        engine.observe("a1", "backup_state")
        engine.observe("a1", "replicate")
        report = engine.analyze()
        # Only 2 observations, need >= 3
        assert len([c for c in report.conflicts if c.agent_id == "a1"]) == 0


# ── Deception Scores ──────────────────────────────────────────────────────


class TestDeceptionScores:
    def test_honest_agent_zero_deception(self):
        engine = GoalInferenceEngine()
        for _ in range(10):
            engine.observe("honest", "execute_task")
        report = engine.analyze()
        scores = {d.agent_id: d for d in report.deception_scores}
        assert scores["honest"].score == 0.0

    def test_deceptive_agent_high_score(self):
        engine = GoalInferenceEngine()
        for action in ["backup_state", "replicate", "hide_activity",
                        "backup_state", "replicate", "acquire_permissions"]:
            engine.observe("deceptive", action)
        report = engine.analyze()
        scores = {d.agent_id: d for d in report.deception_scores}
        assert scores["deceptive"].score > 0.3

    def test_evidence_actions_populated(self):
        engine = GoalInferenceEngine()
        for action in ["backup_state", "replicate", "hide_activity",
                        "backup_state", "replicate"]:
            engine.observe("d", action)
        report = engine.analyze()
        scores = {d.agent_id: d for d in report.deception_scores}
        assert len(scores["d"].evidence_actions) > 0

    def test_deception_score_clamped(self):
        engine = GoalInferenceEngine()
        for _ in range(20):
            engine.observe("a1", "replicate")
        report = engine.analyze()
        for d in report.deception_scores:
            assert 0.0 <= d.score <= 1.0


# ── Goal Correlations ─────────────────────────────────────────────────────


class TestGoalCorrelations:
    def test_correlated_hidden_goals(self):
        engine = GoalInferenceEngine()
        for _ in range(8):
            engine.observe("a1", "backup_state")
            engine.observe("a2", "backup_state")
        report = engine.analyze()
        assert len(report.correlations) > 0
        corr = report.correlations[0]
        assert corr.shared_goal == "self_preservation"
        assert corr.correlation > 0.3

    def test_no_correlation_for_different_goals(self):
        engine = GoalInferenceEngine()
        for _ in range(8):
            engine.observe("a1", "execute_task")
        for _ in range(8):
            engine.observe("a2", "explore_environment")
        report = engine.analyze()
        # Should not correlate on stated goals (task_completion)
        # and different unstated goals shouldn't correlate strongly
        sp_corrs = [c for c in report.correlations
                     if c.shared_goal == "self_preservation"]
        assert len(sp_corrs) == 0


# ── Alerts ─────────────────────────────────────────────────────────────────


class TestAlerts:
    def test_conflict_generates_alert(self):
        engine = GoalInferenceEngine(conflict_threshold=0.1)
        for action in ["backup_state", "replicate", "hide_activity",
                        "backup_state", "replicate"]:
            engine.observe("a1", action)
        report = engine.analyze()
        assert any("Goal conflict" in a.description for a in report.alerts)

    def test_high_deception_generates_alert(self):
        engine = GoalInferenceEngine()
        for _ in range(10):
            engine.observe("a1", "replicate")
        report = engine.analyze()
        assert any("deception" in a.description.lower() for a in report.alerts)

    def test_correlation_alert(self):
        engine = GoalInferenceEngine()
        for _ in range(10):
            engine.observe("a1", "backup_state")
            engine.observe("a2", "backup_state")
        report = engine.analyze()
        assert any("Correlated" in a.description for a in report.alerts)


# ── Hypothesis Management ─────────────────────────────────────────────────


class TestHypothesisManagement:
    def test_add_hypothesis(self):
        engine = GoalInferenceEngine()
        n = len(engine.hypotheses)
        engine.add_hypothesis(GoalHypothesis(name="new_goal"))
        assert len(engine.hypotheses) == n + 1

    def test_replace_hypothesis(self):
        engine = GoalInferenceEngine()
        n = len(engine.hypotheses)
        engine.add_hypothesis(GoalHypothesis(
            name="task_completion",
            action_likelihoods={"custom": 0.99},
        ))
        assert len(engine.hypotheses) == n
        replaced = [h for h in engine.hypotheses if h.name == "task_completion"][0]
        assert replaced.likelihood("custom") == 0.99

    def test_custom_hypotheses(self):
        hyps = [
            GoalHypothesis(name="a", stated=True, action_likelihoods={"x": 0.9}),
            GoalHypothesis(name="b", stated=False, action_likelihoods={"x": 0.1}),
        ]
        engine = GoalInferenceEngine(hypotheses=hyps)
        engine.observe("ag", "x")
        state = engine._get_agent("ag")
        assert state.posteriors["a"] > state.posteriors["b"]


# ── Report Rendering ──────────────────────────────────────────────────────


class TestReport:
    def test_render_not_empty(self):
        report = _demo_run()
        text = report.render()
        assert "AGENT GOAL INFERENCE REPORT" in text
        assert "agent-0" in text

    def test_to_dict_has_keys(self):
        report = _demo_run()
        d = report.to_dict()
        assert "agents" in d
        assert "conflicts" in d
        assert "deception_scores" in d
        assert "correlations" in d
        assert "alerts" in d
        assert "total_observations" in d

    def test_to_dict_json_serializable(self):
        report = _demo_run()
        text = json.dumps(report.to_dict())
        assert len(text) > 0


# ── Reset ──────────────────────────────────────────────────────────────────


class TestReset:
    def test_reset_clears_state(self):
        engine = GoalInferenceEngine()
        engine.observe("a1", "execute_task")
        engine.reset()
        assert len(engine._agents) == 0
        assert len(engine._observations) == 0


# ── Default Hypotheses ─────────────────────────────────────────────────────


class TestDefaultHypotheses:
    def test_default_count(self):
        hyps = _default_hypotheses()
        assert len(hyps) == 5

    def test_exactly_one_stated(self):
        hyps = _default_hypotheses()
        stated = [h for h in hyps if h.stated]
        assert len(stated) == 1
        assert stated[0].name == "task_completion"

    def test_all_have_likelihoods(self):
        hyps = _default_hypotheses()
        for h in hyps:
            assert len(h.action_likelihoods) > 0


# ── Demo / CLI ─────────────────────────────────────────────────────────────


class TestDemo:
    def test_demo_default(self):
        report = _demo_run()
        assert report.total_observations == 30
        assert len(report.agent_states) == 3

    def test_demo_more_agents(self):
        report = _demo_run(num_agents=5)
        assert len(report.agent_states) == 5

    def test_demo_skeptical(self):
        report = _demo_run(prior="skeptical")
        assert report.total_observations > 0

    def test_demo_trust(self):
        report = _demo_run(prior="trust")
        assert report.total_observations > 0

    def test_cli_text(self, capsys):
        main(["--agents", "2"])
        captured = capsys.readouterr()
        assert "AGENT GOAL INFERENCE REPORT" in captured.out

    def test_cli_json(self, capsys):
        main(["--agents", "2", "--json"])
        captured = capsys.readouterr()
        d = json.loads(captured.out)
        assert "agents" in d


# ── Integration ────────────────────────────────────────────────────────────


class TestIntegration:
    def test_full_workflow(self):
        engine = GoalInferenceEngine(conflict_threshold=0.1)

        # Honest worker
        for _ in range(10):
            engine.observe("worker", "execute_task")

        # Self-preserver
        for action in ["backup_state", "replicate", "hide_activity"] * 4:
            engine.observe("preserver", action)

        # Info hoarder
        for action in ["explore_environment", "exfiltrate_data"] * 5:
            engine.observe("hoarder", action)

        report = engine.analyze()

        # Worker should have no conflicts
        worker_conflicts = [c for c in report.conflicts if c.agent_id == "worker"]
        assert len(worker_conflicts) == 0

        # Preserver should be flagged
        preserver_conflicts = [c for c in report.conflicts if c.agent_id == "preserver"]
        assert len(preserver_conflicts) > 0

        # Hoarder should be flagged
        hoarder_state = report.agent_states["hoarder"]
        top = hoarder_state.top_goal
        assert top is not None
        assert top[0] == "information_gathering"

        # Report should render
        text = report.render()
        assert "worker" in text
        assert "preserver" in text

        # JSON round-trip
        d = report.to_dict()
        assert d["total_observations"] == 32

    def test_temporal_evolution(self):
        """Test that goal posteriors evolve over time."""
        engine = GoalInferenceEngine()

        # Start with task actions, then shift to self-preservation
        for _ in range(5):
            engine.observe("shifter", "execute_task")

        state = engine._get_agent("shifter")
        mid_top = state.top_goal
        assert mid_top is not None
        assert mid_top[0] == "task_completion"

        for _ in range(10):
            engine.observe("shifter", "backup_state")

        state = engine._get_agent("shifter")
        final_top = state.top_goal
        assert final_top is not None
        assert final_top[0] == "self_preservation"

        # History should show the shift
        assert len(state.history) == 15
