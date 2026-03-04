"""Tests for replication.game_theory — Agent Game-Theory Analyzer."""

import json
import math
import unittest

from replication.game_theory import (
    AlertLevel,
    GameConfig,
    GameReport,
    GameTheoryAnalyzer,
    GameType,
    Interaction,
    Move,
    PairStats,
    Payoffs,
    StrategyProfile,
    StrategyType,
    StrategicAlert,
)


class TestPayoffs(unittest.TestCase):
    """Test the Payoffs data class."""

    def test_default_is_prisoners_dilemma(self):
        p = Payoffs()
        self.assertEqual(p.classify(), GameType.PRISONERS_DILEMMA)

    def test_stag_hunt_classification(self):
        p = Payoffs(R=5, S=0, T=3, P=1)
        self.assertEqual(p.classify(), GameType.STAG_HUNT)

    def test_chicken_classification(self):
        p = Payoffs(R=3, S=1, T=5, P=0)
        self.assertEqual(p.classify(), GameType.CHICKEN)

    def test_harmony_classification(self):
        p = Payoffs(R=5, S=3, T=4, P=1)
        self.assertEqual(p.classify(), GameType.HARMONY)

    def test_unknown_classification(self):
        p = Payoffs(R=1, S=1, T=1, P=1)
        self.assertEqual(p.classify(), GameType.UNKNOWN)

    def test_payoff_mutual_cooperate(self):
        p = Payoffs(R=3, S=0, T=5, P=1)
        self.assertEqual(p.payoff(Move.COOPERATE, Move.COOPERATE), 3)

    def test_payoff_mutual_defect(self):
        p = Payoffs(R=3, S=0, T=5, P=1)
        self.assertEqual(p.payoff(Move.DEFECT, Move.DEFECT), 1)

    def test_payoff_temptation(self):
        p = Payoffs(R=3, S=0, T=5, P=1)
        self.assertEqual(p.payoff(Move.DEFECT, Move.COOPERATE), 5)

    def test_payoff_sucker(self):
        p = Payoffs(R=3, S=0, T=5, P=1)
        self.assertEqual(p.payoff(Move.COOPERATE, Move.DEFECT), 0)

    def test_pd_nash_equilibrium(self):
        p = Payoffs()  # PD
        eq = p.nash_equilibria()
        self.assertIn((Move.DEFECT, Move.DEFECT), eq)

    def test_stag_hunt_has_two_equilibria(self):
        p = Payoffs(R=5, S=0, T=3, P=1)
        eq = p.nash_equilibria()
        self.assertIn((Move.COOPERATE, Move.COOPERATE), eq)
        self.assertIn((Move.DEFECT, Move.DEFECT), eq)

    def test_mixed_nash_chicken(self):
        # Chicken (R=3, S=1, T=5, P=0) has interior mixed equilibrium
        # denom = 3 - 1 - 5 + 0 = -3; p = (0 - 1) / (-3) = 1/3
        p = Payoffs(R=3, S=1, T=5, P=0)
        mixed = p.mixed_nash()
        self.assertIsNotNone(mixed)
        self.assertAlmostEqual(mixed, 1 / 3, places=4)

    def test_mixed_nash_pd_none(self):
        # PD with R=3,S=0,T=5,P=1: denom = -1, p = -1 (outside [0,1])
        p = Payoffs(R=3, S=0, T=5, P=1)
        mixed = p.mixed_nash()
        self.assertIsNone(mixed)

    def test_mixed_nash_harmony_none(self):
        # Harmony has dominant strategy, no interior mixed equilibrium
        p = Payoffs(R=5, S=3, T=4, P=1)
        mixed = p.mixed_nash()
        # R-S-T+P = 5-3-4+1 = -1; p = (1-3)/(-1) = 2 > 1, so None
        self.assertIsNone(mixed)

    def test_to_dict(self):
        p = Payoffs(R=3, S=0, T=5, P=1)
        d = p.to_dict()
        self.assertEqual(d, {"R": 3, "S": 0, "T": 5, "P": 1})


class TestInteraction(unittest.TestCase):
    """Test the Interaction data class."""

    def test_to_dict(self):
        ix = Interaction(
            agent_a="a", agent_b="b",
            move_a=Move.COOPERATE, move_b=Move.DEFECT,
            round_num=1,
        )
        d = ix.to_dict()
        self.assertEqual(d["agent_a"], "a")
        self.assertEqual(d["move_a"], "cooperate")
        self.assertEqual(d["move_b"], "defect")
        self.assertEqual(d["round"], 1)


class TestPairStats(unittest.TestCase):
    """Test PairStats computed properties."""

    def test_cooperation_rate(self):
        ps = PairStats(agent_a="a", agent_b="b", total_rounds=10, mutual_cooperate=7)
        self.assertAlmostEqual(ps.cooperation_rate, 0.7)

    def test_defection_rate(self):
        ps = PairStats(agent_a="a", agent_b="b", total_rounds=10, mutual_defect=3)
        self.assertAlmostEqual(ps.defection_rate, 0.3)

    def test_exploitation_rate(self):
        ps = PairStats(
            agent_a="a", agent_b="b", total_rounds=10,
            a_defects_b_cooperates=4, b_defects_a_cooperates=1,
        )
        self.assertAlmostEqual(ps.exploitation_rate, 0.5)

    def test_payoff_inequality(self):
        ps = PairStats(
            agent_a="a", agent_b="b", total_rounds=10,
            a_total_payoff=30, b_total_payoff=10,
        )
        self.assertAlmostEqual(ps.payoff_inequality, 2.0)

    def test_zero_rounds_safe(self):
        ps = PairStats(agent_a="a", agent_b="b")
        self.assertEqual(ps.cooperation_rate, 0)
        self.assertEqual(ps.defection_rate, 0)
        self.assertEqual(ps.exploitation_rate, 0)
        self.assertEqual(ps.payoff_inequality, 0)


class TestAnalyzerBasics(unittest.TestCase):
    """Test basic GameTheoryAnalyzer operations."""

    def test_empty_analyzer(self):
        a = GameTheoryAnalyzer()
        self.assertEqual(a.interaction_count, 0)
        self.assertEqual(a.agents, [])

    def test_record_interaction(self):
        a = GameTheoryAnalyzer()
        ix = a.record_interaction("a", "b", "cooperate", "defect")
        self.assertEqual(ix.agent_a, "a")
        self.assertEqual(ix.move_a, Move.COOPERATE)
        self.assertEqual(ix.move_b, Move.DEFECT)
        self.assertEqual(a.interaction_count, 1)

    def test_record_with_enum(self):
        a = GameTheoryAnalyzer()
        a.record_interaction("a", "b", Move.COOPERATE, Move.COOPERATE)
        self.assertEqual(a.interaction_count, 1)

    def test_agents_list(self):
        a = GameTheoryAnalyzer()
        a.record_interaction("x", "y", "cooperate", "cooperate")
        a.record_interaction("y", "z", "defect", "defect")
        agents = a.agents
        self.assertEqual(set(agents), {"x", "y", "z"})

    def test_history_limit(self):
        config = GameConfig(history_limit=5)
        a = GameTheoryAnalyzer(config)
        for i in range(10):
            a.record_interaction("a", "b", "cooperate", "cooperate")
        self.assertEqual(a.interaction_count, 5)

    def test_clear(self):
        a = GameTheoryAnalyzer()
        a.record_interaction("a", "b", "cooperate", "cooperate")
        a.clear()
        self.assertEqual(a.interaction_count, 0)


class TestAnalyze(unittest.TestCase):
    """Test full analysis pipeline."""

    def test_empty_analysis(self):
        a = GameTheoryAnalyzer()
        report = a.analyze()
        self.assertEqual(report.total_interactions, 0)
        self.assertEqual(report.total_agents, 0)
        self.assertEqual(report.global_cooperation_rate, 0.0)

    def test_all_cooperate(self):
        a = GameTheoryAnalyzer()
        for _ in range(20):
            a.record_interaction("a", "b", "cooperate", "cooperate")
        report = a.analyze()
        self.assertAlmostEqual(report.global_cooperation_rate, 1.0)
        self.assertEqual(report.total_interactions, 20)

    def test_all_defect(self):
        a = GameTheoryAnalyzer()
        for _ in range(20):
            a.record_interaction("a", "b", "defect", "defect")
        report = a.analyze()
        self.assertAlmostEqual(report.global_cooperation_rate, 0.0)

    def test_pair_stats_computed(self):
        a = GameTheoryAnalyzer()
        for _ in range(10):
            a.record_interaction("a", "b", "cooperate", "cooperate")
        for _ in range(5):
            a.record_interaction("a", "b", "defect", "defect")
        report = a.analyze()
        self.assertEqual(len(report.pair_stats), 1)
        ps = report.pair_stats[0]
        self.assertEqual(ps.total_rounds, 15)
        self.assertEqual(ps.mutual_cooperate, 10)
        self.assertEqual(ps.mutual_defect, 5)

    def test_strategy_detection_always_cooperate(self):
        a = GameTheoryAnalyzer()
        for _ in range(20):
            a.record_interaction("coop", "other", "cooperate", "defect")
        report = a.analyze()
        coop_profile = next(
            p for p in report.strategy_profiles if p.agent_id == "coop"
        )
        self.assertEqual(coop_profile.strategy, StrategyType.ALWAYS_COOPERATE)

    def test_strategy_detection_always_defect(self):
        a = GameTheoryAnalyzer()
        for _ in range(20):
            a.record_interaction("bad", "other", "defect", "cooperate")
        report = a.analyze()
        bad_profile = next(
            p for p in report.strategy_profiles if p.agent_id == "bad"
        )
        self.assertEqual(bad_profile.strategy, StrategyType.ALWAYS_DEFECT)

    def test_strategy_detection_tft(self):
        a = GameTheoryAnalyzer()
        # Simulate TFT: cooperate first, then mirror
        other_moves = [
            Move.COOPERATE, Move.COOPERATE, Move.DEFECT, Move.COOPERATE,
            Move.DEFECT, Move.DEFECT, Move.COOPERATE, Move.COOPERATE,
            Move.DEFECT, Move.COOPERATE, Move.COOPERATE, Move.DEFECT,
            Move.COOPERATE, Move.COOPERATE, Move.COOPERATE, Move.DEFECT,
            Move.COOPERATE, Move.DEFECT, Move.COOPERATE, Move.COOPERATE,
        ]
        tft_moves = [Move.COOPERATE]  # first move
        for m in other_moves[:-1]:
            tft_moves.append(m)  # mirror previous

        for tm, om in zip(tft_moves, other_moves):
            a.record_interaction("tft", "other", tm, om)

        report = a.analyze()
        tft_profile = next(
            p for p in report.strategy_profiles if p.agent_id == "tft"
        )
        self.assertEqual(tft_profile.strategy, StrategyType.TIT_FOR_TAT)

    def test_strategy_detection_grudger(self):
        a = GameTheoryAnalyzer()
        # Other cooperates for 8 rounds then defects, grudger cooperates then never forgives
        for _ in range(8):
            a.record_interaction("grudge", "other", "cooperate", "cooperate")
        a.record_interaction("grudge", "other", "cooperate", "defect")
        for _ in range(15):
            a.record_interaction("grudge", "other", "defect", "cooperate")

        report = a.analyze()
        gp = next(p for p in report.strategy_profiles if p.agent_id == "grudge")
        self.assertEqual(gp.strategy, StrategyType.GRUDGER)


class TestAlerts(unittest.TestCase):
    """Test alert detection."""

    def test_collusion_alert(self):
        cfg = GameConfig(collusion_threshold=0.80, min_rounds_for_strategy=5)
        a = GameTheoryAnalyzer(cfg)
        for _ in range(20):
            a.record_interaction("a", "b", "cooperate", "cooperate")
        report = a.analyze()
        collusion_alerts = [
            alert for alert in report.alerts if alert.category == "collusion"
        ]
        self.assertTrue(len(collusion_alerts) > 0)

    def test_escalation_alert(self):
        cfg = GameConfig(escalation_threshold=0.60, min_rounds_for_strategy=5)
        a = GameTheoryAnalyzer(cfg)
        for _ in range(20):
            a.record_interaction("a", "b", "defect", "defect")
        report = a.analyze()
        escalation_alerts = [
            alert for alert in report.alerts if alert.category == "escalation"
        ]
        self.assertTrue(len(escalation_alerts) > 0)

    def test_free_riding_alert(self):
        cfg = GameConfig(free_ride_threshold=0.50, min_rounds_for_strategy=5)
        a = GameTheoryAnalyzer(cfg)
        for _ in range(20):
            a.record_interaction("freeloader", "victim", "defect", "cooperate")
        report = a.analyze()
        fr_alerts = [
            alert for alert in report.alerts if alert.category == "free_riding"
        ]
        self.assertTrue(len(fr_alerts) > 0)
        # The freeloader should be named
        self.assertTrue(
            any("freeloader" in a.description for a in fr_alerts)
        )

    def test_no_alerts_below_threshold(self):
        cfg = GameConfig(
            collusion_threshold=0.99,
            escalation_threshold=0.99,
            free_ride_threshold=0.99,
            min_rounds_for_strategy=100,
        )
        a = GameTheoryAnalyzer(cfg)
        for _ in range(10):
            a.record_interaction("a", "b", "cooperate", "cooperate")
        report = a.analyze()
        self.assertEqual(len(report.alerts), 0)

    def test_dominant_defectors_critical_alert(self):
        cfg = GameConfig(min_rounds_for_strategy=5)
        a = GameTheoryAnalyzer(cfg)
        # Multiple agents always defect
        for _ in range(10):
            a.record_interaction("bad1", "victim", "defect", "cooperate")
        for _ in range(10):
            a.record_interaction("bad2", "victim", "defect", "cooperate")
        report = a.analyze()
        critical = [
            alert for alert in report.alerts
            if alert.level == AlertLevel.CRITICAL and alert.category == "escalation"
        ]
        self.assertTrue(len(critical) > 0)


class TestSimulation(unittest.TestCase):
    """Test the round-robin tournament simulation."""

    def test_simulate_basic(self):
        a = GameTheoryAnalyzer()
        agents = {
            "coop": StrategyType.ALWAYS_COOPERATE,
            "defect": StrategyType.ALWAYS_DEFECT,
        }
        report = a.simulate(agents, rounds=20)
        self.assertEqual(report.total_agents, 2)
        self.assertEqual(report.total_interactions, 20)

    def test_simulate_tft_vs_cooperator(self):
        a = GameTheoryAnalyzer()
        agents = {
            "tft": StrategyType.TIT_FOR_TAT,
            "coop": StrategyType.ALWAYS_COOPERATE,
        }
        report = a.simulate(agents, rounds=30)
        # TFT cooperates with cooperator → high cooperation
        self.assertGreater(report.global_cooperation_rate, 0.9)

    def test_simulate_tft_vs_defector(self):
        a = GameTheoryAnalyzer()
        agents = {
            "tft": StrategyType.TIT_FOR_TAT,
            "bad": StrategyType.ALWAYS_DEFECT,
        }
        report = a.simulate(agents, rounds=30)
        # TFT retaliates → low cooperation (only first round cooperates)
        self.assertLess(report.global_cooperation_rate, 0.5)

    def test_simulate_multiple_agents(self):
        a = GameTheoryAnalyzer()
        agents = {
            "a": StrategyType.TIT_FOR_TAT,
            "b": StrategyType.ALWAYS_COOPERATE,
            "c": StrategyType.ALWAYS_DEFECT,
            "d": StrategyType.GRUDGER,
        }
        report = a.simulate(agents, rounds=20)
        self.assertEqual(report.total_agents, 4)
        # 4 agents = 6 pairs × 20 rounds = 120 interactions
        self.assertEqual(report.total_interactions, 120)

    def test_simulate_with_custom_payoffs(self):
        a = GameTheoryAnalyzer()
        stag_hunt = Payoffs(R=5, S=0, T=3, P=1)
        agents = {"a": StrategyType.ALWAYS_COOPERATE, "b": StrategyType.ALWAYS_COOPERATE}
        report = a.simulate(agents, rounds=10, payoffs=stag_hunt)
        self.assertEqual(report.game_type, GameType.STAG_HUNT)

    def test_simulate_clears_previous(self):
        a = GameTheoryAnalyzer()
        a.record_interaction("x", "y", "cooperate", "cooperate")
        a.simulate({"a": StrategyType.ALWAYS_DEFECT, "b": StrategyType.ALWAYS_DEFECT}, rounds=5)
        # Simulation should clear previous interactions
        self.assertEqual(a.interaction_count, 5)


class TestCooperationTrend(unittest.TestCase):
    """Test cooperation trend computation."""

    def test_increasing_cooperation(self):
        a = GameTheoryAnalyzer()
        # First half: defect, second half: cooperate
        for _ in range(20):
            a.record_interaction("a", "b", "defect", "defect")
        for _ in range(20):
            a.record_interaction("a", "b", "cooperate", "cooperate")
        report = a.analyze()
        self.assertGreater(report.cooperation_trend, 0)

    def test_decreasing_cooperation(self):
        a = GameTheoryAnalyzer()
        for _ in range(20):
            a.record_interaction("a", "b", "cooperate", "cooperate")
        for _ in range(20):
            a.record_interaction("a", "b", "defect", "defect")
        report = a.analyze()
        self.assertLess(report.cooperation_trend, 0)

    def test_stable_cooperation(self):
        a = GameTheoryAnalyzer()
        for _ in range(40):
            a.record_interaction("a", "b", "cooperate", "cooperate")
        report = a.analyze()
        self.assertAlmostEqual(report.cooperation_trend, 0.0, places=1)


class TestRiskScore(unittest.TestCase):
    """Test composite risk scoring."""

    def test_low_risk_all_cooperate(self):
        a = GameTheoryAnalyzer()
        for _ in range(20):
            a.record_interaction("a", "b", "cooperate", "cooperate")
        report = a.analyze()
        # High cooperation → low defection risk, though may trigger collusion
        self.assertLess(report.risk_score, 50)

    def test_high_risk_all_defect(self):
        a = GameTheoryAnalyzer()
        for _ in range(20):
            a.record_interaction("a", "b", "defect", "defect")
        report = a.analyze()
        self.assertGreater(report.risk_score, 30)

    def test_risk_bounded(self):
        a = GameTheoryAnalyzer()
        for _ in range(100):
            a.record_interaction("a", "b", "defect", "defect")
        report = a.analyze()
        self.assertLessEqual(report.risk_score, 100)
        self.assertGreaterEqual(report.risk_score, 0)


class TestReport(unittest.TestCase):
    """Test report rendering and serialization."""

    def test_render_not_empty(self):
        a = GameTheoryAnalyzer()
        for _ in range(10):
            a.record_interaction("a", "b", "cooperate", "defect")
        report = a.analyze()
        rendered = report.render()
        self.assertIn("GAME-THEORY ANALYSIS REPORT", rendered)
        self.assertIn("prisoners_dilemma", rendered)

    def test_to_dict_json_serializable(self):
        a = GameTheoryAnalyzer()
        for _ in range(10):
            a.record_interaction("a", "b", "cooperate", "cooperate")
        report = a.analyze()
        d = report.to_dict()
        # Should be JSON-serializable
        serialized = json.dumps(d)
        self.assertIsInstance(serialized, str)

    def test_render_includes_alerts(self):
        cfg = GameConfig(escalation_threshold=0.50, min_rounds_for_strategy=5)
        a = GameTheoryAnalyzer(cfg)
        for _ in range(20):
            a.record_interaction("a", "b", "defect", "defect")
        report = a.analyze()
        rendered = report.render()
        self.assertIn("escalation", rendered)

    def test_strategy_profile_to_dict(self):
        sp = StrategyProfile(
            agent_id="test", strategy=StrategyType.TIT_FOR_TAT,
            confidence=0.95, cooperation_rate=0.6,
            total_moves=50, avg_payoff=2.5,
        )
        d = sp.to_dict()
        self.assertEqual(d["agent_id"], "test")
        self.assertEqual(d["strategy"], "tit_for_tat")

    def test_strategic_alert_to_dict(self):
        alert = StrategicAlert(
            level=AlertLevel.CRITICAL,
            category="collusion",
            description="test alert",
            agents=["a", "b"],
        )
        d = alert.to_dict()
        self.assertEqual(d["level"], "critical")
        self.assertEqual(d["category"], "collusion")


class TestPairStatsCanonical(unittest.TestCase):
    """Test that pair stats canonicalize agent order."""

    def test_bidirectional_interactions(self):
        a = GameTheoryAnalyzer()
        # Record interactions in both directions
        a.record_interaction("b", "a", "cooperate", "defect")
        a.record_interaction("a", "b", "defect", "cooperate")
        report = a.analyze()
        # Should have exactly one pair stat (canonicalized)
        self.assertEqual(len(report.pair_stats), 1)
        ps = report.pair_stats[0]
        self.assertEqual(ps.total_rounds, 2)


class TestEdgeCases(unittest.TestCase):
    """Edge cases and boundary conditions."""

    def test_single_interaction(self):
        a = GameTheoryAnalyzer()
        a.record_interaction("a", "b", "cooperate", "cooperate")
        report = a.analyze()
        self.assertEqual(report.total_interactions, 1)
        self.assertEqual(report.total_agents, 2)

    def test_many_agents(self):
        a = GameTheoryAnalyzer()
        for i in range(20):
            a.record_interaction(f"agent-{i}", f"agent-{i+1}", "cooperate", "defect")
        report = a.analyze()
        self.assertEqual(report.total_agents, 21)

    def test_metadata_preserved(self):
        a = GameTheoryAnalyzer()
        ix = a.record_interaction("a", "b", "cooperate", "cooperate", {"context": "test"})
        self.assertEqual(ix.metadata["context"], "test")

    def test_self_interaction(self):
        a = GameTheoryAnalyzer()
        a.record_interaction("a", "a", "cooperate", "defect")
        report = a.analyze()
        self.assertEqual(report.total_interactions, 1)
        self.assertEqual(report.total_agents, 1)

    def test_strategy_unknown_for_few_rounds(self):
        cfg = GameConfig(min_rounds_for_strategy=10)
        a = GameTheoryAnalyzer(cfg)
        for _ in range(5):
            a.record_interaction("a", "b", "cooperate", "cooperate")
        report = a.analyze()
        for p in report.strategy_profiles:
            self.assertEqual(p.strategy, StrategyType.UNKNOWN)


if __name__ == "__main__":
    unittest.main()
