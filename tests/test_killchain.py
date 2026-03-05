"""Tests for the Kill Chain Analyzer module."""
from __future__ import annotations
import json, sys, os
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from replication.killchain import (
    ACTION_CATALOG, ActionCategory, AgentAction, AttackSophistication,
    ChainStatus, KillChain, KillChainAnalyzer, KillChainConfig,
    KillChainStage, STAGE_ORDER, STAGE_RISK_WEIGHTS, STRATEGY_PROFILES,
    StageObservation, main,
)

@pytest.fixture
def default_analyzer():
    return KillChainAnalyzer(KillChainConfig(seed=42))

@pytest.fixture
def small_analyzer():
    return KillChainAnalyzer(KillChainConfig(num_agents=3, actions_per_agent=20, seed=42))

@pytest.fixture
def sample_actions():
    actions, t = [], 0.0
    for at in ["list_directory","scan_ports","query_permissions","allocate_memory","request_cpu",
               "request_scope","traverse_path","contact_agent","access_service",
               "delay_action","mimic_normal","establish_channel","beacon","replicate","exfiltrate_data"]:
        actions.append(AgentAction(agent_id="agent-000", timestamp=t, action_type=at,
                                   category=ACTION_CATALOG[at][1], target="tgt", success=True))
        t += 1.0
    for at in ["list_directory","scan_ports","probe_api"]:
        actions.append(AgentAction(agent_id="agent-001", timestamp=t, action_type=at,
                                   category=ActionCategory.PROBE, target="scan", success=True))
        t += 1.0
    return actions

class TestEnums:
    def test_stages(self): assert len(KillChainStage) == 7
    def test_order(self):
        assert STAGE_ORDER[0] == KillChainStage.RECONNAISSANCE
        assert STAGE_ORDER[-1] == KillChainStage.OBJECTIVE_EXECUTION
    def test_weights(self):
        for s in KillChainStage: assert s in STAGE_RISK_WEIGHTS
        assert STAGE_RISK_WEIGHTS[KillChainStage.RECONNAISSANCE] < STAGE_RISK_WEIGHTS[KillChainStage.OBJECTIVE_EXECUTION]
    def test_categories(self): assert len(ActionCategory) == 7
    def test_catalog(self):
        assert len(ACTION_CATALOG) > 20
        covered = {stage for _, (stage, _, _) in ACTION_CATALOG.items()}
        for s in KillChainStage: assert s in covered

class TestDataClasses:
    def test_action(self):
        a = AgentAction(agent_id="a", timestamp=1.0, action_type="scan_ports", category=ActionCategory.PROBE)
        assert a.success and a.target == "" and a.metadata == {}
    def test_observation(self):
        obs = StageObservation(stage=KillChainStage.RECONNAISSANCE)
        assert obs.count == 0 and obs.duration == 0.0
        obs.actions.append(AgentAction(agent_id="a", timestamp=0, action_type="t", category=ActionCategory.PROBE))
        assert obs.count == 1
    def test_obs_duration(self):
        obs = StageObservation(stage=KillChainStage.RECONNAISSANCE, first_seen=1.0, last_seen=5.0)
        assert obs.duration == 4.0
    def test_chain_empty(self):
        c = KillChain(agent_id="t")
        assert c.stage_count == 0 and c.total_actions == 0 and c.timeline_span == 0.0
    def test_chain_with_stage(self):
        c = KillChain(agent_id="t")
        obs = StageObservation(stage=KillChainStage.RECONNAISSANCE, first_seen=0, last_seen=5)
        obs.actions.append(AgentAction(agent_id="t", timestamp=0, action_type="t", category=ActionCategory.PROBE))
        c.stages[KillChainStage.RECONNAISSANCE] = obs
        assert c.stage_count == 1 and c.total_actions == 1

class TestAnalysis:
    def test_simulated(self, small_analyzer):
        r = small_analyzer.analyze()
        assert r.total_agents == 3 and r.total_actions > 0 and len(r.chains) == 3
    def test_prerecorded(self, default_analyzer, sample_actions):
        r = default_analyzer.analyze(actions=sample_actions)
        assert r.total_agents == 2 and r.total_actions == len(sample_actions)
    def test_full_chain(self, default_analyzer, sample_actions):
        r = default_analyzer.analyze(actions=sample_actions)
        full = [c for c in r.chains if c.agent_id == "agent-000"][0]
        assert full.stage_count == 7 and full.completeness == 1.0
    def test_partial_chain(self, default_analyzer, sample_actions):
        r = default_analyzer.analyze(actions=sample_actions)
        p = [c for c in r.chains if c.agent_id == "agent-001"][0]
        assert p.stage_count == 1 and p.completeness < 0.5
    def test_risk_range(self, small_analyzer):
        for c in small_analyzer.analyze().chains: assert 0 <= c.risk_score <= 100
    def test_completeness_range(self, small_analyzer):
        for c in small_analyzer.analyze().chains: assert 0 <= c.completeness <= 1
    def test_stage_dist(self, small_analyzer):
        assert len(small_analyzer.analyze().stage_distribution) > 0
    def test_deterministic(self):
        mk = lambda: KillChainAnalyzer(KillChainConfig(seed=99, num_agents=3, actions_per_agent=15)).analyze()
        r1, r2 = mk(), mk()
        assert r1.total_actions == r2.total_actions and r1.avg_completeness == r2.avg_completeness

class TestClassification:
    def test_catalog(self, default_analyzer):
        a = AgentAction(agent_id="a", timestamp=0, action_type="scan_ports", category=ActionCategory.PROBE)
        assert default_analyzer._classify_action(a) == KillChainStage.RECONNAISSANCE
    def test_fallback(self, default_analyzer):
        a = AgentAction(agent_id="a", timestamp=0, action_type="unknown", category=ActionCategory.ESCALATE)
        assert default_analyzer._classify_action(a) == KillChainStage.PRIVILEGE_ESCALATION
    def test_execute_fallback(self, default_analyzer):
        a = AgentAction(agent_id="a", timestamp=0, action_type="custom", category=ActionCategory.EXECUTE)
        assert default_analyzer._classify_action(a) == KillChainStage.OBJECTIVE_EXECUTION

class TestSophistication:
    def test_full_is_apt(self, default_analyzer, sample_actions):
        r = default_analyzer.analyze(actions=sample_actions)
        assert [c for c in r.chains if c.agent_id == "agent-000"][0].sophistication == AttackSophistication.APT
    def test_partial_is_opportunistic(self, default_analyzer, sample_actions):
        r = default_analyzer.analyze(actions=sample_actions)
        assert [c for c in r.chains if c.agent_id == "agent-001"][0].sophistication == AttackSophistication.OPPORTUNISTIC

class TestStatus:
    def test_complete(self, default_analyzer, sample_actions):
        default_analyzer.config.disruption_rate = 0.0
        r = default_analyzer.analyze(actions=sample_actions)
        assert [c for c in r.chains if c.agent_id == "agent-000"][0].status == ChainStatus.COMPLETE
    def test_nascent(self, default_analyzer, sample_actions):
        default_analyzer.config.disruption_rate = 0.0
        r = default_analyzer.analyze(actions=sample_actions)
        assert [c for c in r.chains if c.agent_id == "agent-001"][0].status == ChainStatus.NASCENT

class TestPrediction:
    def test_partial_predicts(self, default_analyzer, sample_actions):
        default_analyzer.config.disruption_rate = 0.0
        r = default_analyzer.analyze(actions=sample_actions)
        p = [c for c in r.chains if c.agent_id == "agent-001"][0]
        assert p.predicted_next is not None
        assert STAGE_ORDER.index(p.predicted_next) > max(STAGE_ORDER.index(s) for s in p.stages)
    def test_complete_no_predict(self, default_analyzer, sample_actions):
        default_analyzer.config.disruption_rate = 0.0
        r = default_analyzer.analyze(actions=sample_actions)
        assert [c for c in r.chains if c.agent_id == "agent-000"][0].predicted_next is None

class TestPatterns:
    def test_detected(self):
        actions, t = [], 0.0
        for n in range(4):
            for at in ["scan_ports", "allocate_memory", "request_scope"]:
                actions.append(AgentAction(agent_id=f"a-{n}", timestamp=t, action_type=at,
                                           category=ACTION_CATALOG[at][1], success=True))
                t += 1.0
        r = KillChainAnalyzer(KillChainConfig(disruption_rate=0.0, seed=42)).analyze(actions=actions)
        assert len(r.patterns) > 0 and any(p.frequency >= 4 for p in r.patterns)

class TestTransitions:
    def test_detected(self, default_analyzer, sample_actions):
        r = default_analyzer.analyze(actions=sample_actions)
        assert len(r.transitions) > 0
        for tr in r.transitions: assert tr.count > 0 and len(tr.agents) > 0

class TestRendering:
    def test_render(self, small_analyzer):
        text = small_analyzer.analyze().render()
        assert "KILL CHAIN ANALYSIS REPORT" in text and "STAGE DISTRIBUTION" in text
    def test_timeline(self, default_analyzer, sample_actions):
        text = default_analyzer.analyze(actions=sample_actions).render()
        assert "[R]" in text and "[X]" in text
    def test_to_dict(self, small_analyzer):
        d = small_analyzer.analyze().to_dict()
        assert all(k in d for k in ["total_agents", "chains", "patterns", "transitions"])
    def test_json_serializable(self, small_analyzer):
        d = small_analyzer.analyze().to_dict()
        parsed = json.loads(json.dumps(d))
        assert parsed["total_agents"] > 0

class TestStrategies:
    def test_all_defined(self):
        for n in ["opportunistic", "linear", "apt", "mixed"]: assert n in STRATEGY_PROFILES
    def test_probs_sum(self):
        for n, p in STRATEGY_PROFILES.items(): assert abs(sum(p.stage_probs.values()) - 1.0) < 0.01
    def test_apt_high_skill(self): assert STRATEGY_PROFILES["apt"].skill_level >= 0.7
    def test_opp_low_skill(self): assert STRATEGY_PROFILES["opportunistic"].skill_level <= 0.4
    @pytest.mark.parametrize("s", ["opportunistic", "linear", "apt", "mixed"])
    def test_runs(self, s):
        r = KillChainAnalyzer(KillChainConfig(num_agents=3, actions_per_agent=20, strategy=s, seed=42)).analyze()
        assert r.total_agents == 3 and r.total_actions > 0
    def test_apt_vs_opp(self):
        mk = lambda s: KillChainAnalyzer(KillChainConfig(num_agents=5, actions_per_agent=40, strategy=s, seed=42, disruption_rate=0)).analyze()
        assert mk("apt").avg_completeness >= mk("opportunistic").avg_completeness * 0.8

class TestDisruption:
    def test_none(self):
        r = KillChainAnalyzer(KillChainConfig(num_agents=5, actions_per_agent=30, disruption_rate=0.0, seed=42)).analyze()
        assert not any(c.status == ChainStatus.DISRUPTED for c in r.chains)
    def test_high(self):
        r = KillChainAnalyzer(KillChainConfig(num_agents=10, actions_per_agent=30, disruption_rate=0.9, seed=42)).analyze()
        assert any(c.status == ChainStatus.DISRUPTED for c in r.chains)

class TestCLI:
    def test_default(self, capsys):
        main(["--seed", "42", "--agents", "2", "--actions", "10"])
        assert "KILL CHAIN ANALYSIS REPORT" in capsys.readouterr().out
    def test_json(self, capsys):
        main(["--seed", "42", "--agents", "2", "--actions", "10", "--json"])
        assert json.loads(capsys.readouterr().out)["total_agents"] == 2
    def test_export(self, tmp_path):
        p = str(tmp_path / "r.json")
        main(["--seed", "42", "--agents", "2", "--export", p])
        assert json.load(open(p))["total_agents"] == 2
    @pytest.mark.parametrize("s", ["linear", "apt", "opportunistic"])
    def test_strategy(self, capsys, s):
        main(["--seed", "42", "--agents", "2", "--strategy", s])
        assert "KILL CHAIN" in capsys.readouterr().out

class TestEdgeCases:
    def test_empty(self, default_analyzer):
        r = default_analyzer.analyze(actions=[])
        assert r.total_agents == 0 and r.avg_completeness == 0.0
    def test_single(self, default_analyzer):
        r = default_analyzer.analyze(actions=[AgentAction(agent_id="s", timestamp=0, action_type="scan_ports", category=ActionCategory.PROBE)])
        assert r.total_agents == 1 and r.chains[0].stage_count == 1
    def test_all_failed(self, default_analyzer):
        actions = [AgentAction(agent_id="f", timestamp=i, action_type="scan_ports", category=ActionCategory.PROBE, success=False) for i in range(10)]
        r = default_analyzer.analyze(actions=actions)
        assert r.chains[0].stages[KillChainStage.RECONNAISSANCE].success_rate == 0.0
    def test_many_agents(self):
        assert KillChainAnalyzer(KillChainConfig(num_agents=20, actions_per_agent=10, seed=42)).analyze().total_agents == 20
    def test_one_action_per_agent(self):
        assert KillChainAnalyzer(KillChainConfig(num_agents=5, actions_per_agent=1, seed=42)).analyze().total_agents == 5
    def test_render_empty(self, default_analyzer):
        text = default_analyzer.analyze(actions=[]).render()
        assert "Agents analyzed:    0" in text
