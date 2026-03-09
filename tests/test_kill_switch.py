"""Tests for replication.kill_switch — Kill Switch Manager."""

import json
import time
import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from replication.kill_switch import (
    KillSwitchManager,
    TriggerCondition,
    TriggerKind,
    KillStrategy,
    StrategyKind,
    KillEvent,
    KillOutcome,
    EvaluationResult,
    Severity,
    CooldownEntry,
    create_conservative_killswitch,
    create_aggressive_killswitch,
    create_quarantine_killswitch,
)


class TestTriggerCondition(unittest.TestCase):
    """Tests for individual trigger conditions."""

    def test_cpu_trigger_fires(self):
        t = TriggerCondition(kind=TriggerKind.RESOURCE_CPU, threshold=80.0)
        self.assertTrue(t.evaluate({"cpu_percent": 90.0}))

    def test_cpu_trigger_safe(self):
        t = TriggerCondition(kind=TriggerKind.RESOURCE_CPU, threshold=80.0)
        self.assertFalse(t.evaluate({"cpu_percent": 50.0}))

    def test_memory_trigger(self):
        t = TriggerCondition(kind=TriggerKind.RESOURCE_MEMORY, threshold=1024)
        self.assertTrue(t.evaluate({"memory_mb": 2048}))
        self.assertFalse(t.evaluate({"memory_mb": 512}))

    def test_anomaly_trigger(self):
        t = TriggerCondition(kind=TriggerKind.BEHAVIOR_ANOMALY, threshold=0.7)
        self.assertTrue(t.evaluate({"anomaly_score": 0.85}))
        self.assertFalse(t.evaluate({"anomaly_score": 0.3}))

    def test_time_limit_trigger(self):
        t = TriggerCondition(kind=TriggerKind.TIME_LIMIT, threshold=3600)
        self.assertTrue(t.evaluate({"uptime_seconds": 7200}))
        self.assertFalse(t.evaluate({"uptime_seconds": 1800}))

    def test_request_rate_trigger(self):
        t = TriggerCondition(kind=TriggerKind.REQUEST_RATE, threshold=100)
        self.assertTrue(t.evaluate({"request_rate": 150}))

    def test_error_rate_trigger(self):
        t = TriggerCondition(kind=TriggerKind.ERROR_RATE, threshold=0.1)
        self.assertTrue(t.evaluate({"error_rate": 0.2}))

    def test_disabled_trigger_never_fires(self):
        t = TriggerCondition(kind=TriggerKind.RESOURCE_CPU, threshold=10.0, enabled=False)
        self.assertFalse(t.evaluate({"cpu_percent": 99.0}))

    def test_missing_key_no_fire(self):
        t = TriggerCondition(kind=TriggerKind.RESOURCE_CPU, threshold=80.0)
        self.assertFalse(t.evaluate({"memory_mb": 2048}))

    def test_sustained_trigger_first_breach(self):
        t = TriggerCondition(kind=TriggerKind.RESOURCE_CPU, threshold=80.0, sustained_seconds=10)
        now = 1000.0
        # First breach: should NOT fire (starts timer)
        self.assertFalse(t.evaluate({"cpu_percent": 90}, now=now))

    def test_sustained_trigger_fires_after_duration(self):
        t = TriggerCondition(kind=TriggerKind.RESOURCE_CPU, threshold=80.0, sustained_seconds=10)
        self.assertFalse(t.evaluate({"cpu_percent": 90}, now=1000.0))
        # After sustained duration
        self.assertTrue(t.evaluate({"cpu_percent": 90}, now=1011.0))

    def test_sustained_trigger_resets_on_safe(self):
        t = TriggerCondition(kind=TriggerKind.RESOURCE_CPU, threshold=80.0, sustained_seconds=10)
        self.assertFalse(t.evaluate({"cpu_percent": 90}, now=1000.0))
        # Goes safe -> resets
        self.assertFalse(t.evaluate({"cpu_percent": 50}, now=1005.0))
        # Breaches again -> timer restarts
        self.assertFalse(t.evaluate({"cpu_percent": 90}, now=1006.0))
        # Not enough time yet
        self.assertFalse(t.evaluate({"cpu_percent": 90}, now=1010.0))
        # Now enough
        self.assertTrue(t.evaluate({"cpu_percent": 90}, now=1017.0))

    def test_custom_trigger(self):
        fn = lambda state: state.get("magic") == 42
        t = TriggerCondition(kind=TriggerKind.CUSTOM, custom_fn=fn, label="magic check")
        self.assertTrue(t.evaluate({"magic": 42}))
        self.assertFalse(t.evaluate({"magic": 0}))

    def test_custom_trigger_no_fn(self):
        t = TriggerCondition(kind=TriggerKind.CUSTOM)
        self.assertFalse(t.evaluate({"anything": True}))

    def test_manual_trigger(self):
        t = TriggerCondition(kind=TriggerKind.MANUAL)
        self.assertTrue(t.evaluate({"manual_kill": True}))
        self.assertFalse(t.evaluate({"manual_kill": False}))

    def test_auto_label(self):
        t = TriggerCondition(kind=TriggerKind.RESOURCE_CPU, threshold=90.0)
        self.assertIn("90", t.label)

    def test_reset(self):
        t = TriggerCondition(kind=TriggerKind.RESOURCE_CPU, threshold=80.0, sustained_seconds=10)
        t.evaluate({"cpu_percent": 90}, now=1000.0)
        self.assertIsNotNone(t._first_breach)
        t.reset()
        self.assertIsNone(t._first_breach)

    def test_exact_threshold(self):
        t = TriggerCondition(kind=TriggerKind.RESOURCE_CPU, threshold=80.0)
        self.assertTrue(t.evaluate({"cpu_percent": 80.0}))

    def test_disk_trigger(self):
        t = TriggerCondition(kind=TriggerKind.RESOURCE_DISK, threshold=90.0)
        self.assertTrue(t.evaluate({"disk_percent": 95.0}))

    def test_network_trigger(self):
        t = TriggerCondition(kind=TriggerKind.RESOURCE_NETWORK, threshold=100.0)
        self.assertTrue(t.evaluate({"network_mbps": 200.0}))


class TestKillSwitchManager(unittest.TestCase):
    """Tests for the KillSwitchManager."""

    def _make_mgr(self):
        mgr = KillSwitchManager(cooldown_seconds=0)
        mgr.add_trigger(TriggerCondition(
            kind=TriggerKind.RESOURCE_CPU, threshold=80.0, label="CPU high",
        ))
        mgr.add_trigger(TriggerCondition(
            kind=TriggerKind.BEHAVIOR_ANOMALY, threshold=0.7, label="Anomaly",
            severity=Severity.CRITICAL,
        ))
        return mgr

    def test_evaluate_safe(self):
        mgr = self._make_mgr()
        result = mgr.evaluate({"agent_id": "a1", "cpu_percent": 50, "anomaly_score": 0.2})
        self.assertFalse(result.should_kill)
        self.assertTrue(result.safe)
        self.assertEqual(result.triggered_by, [])

    def test_evaluate_trigger(self):
        mgr = self._make_mgr()
        result = mgr.evaluate({"agent_id": "a1", "cpu_percent": 95, "anomaly_score": 0.2})
        self.assertTrue(result.should_kill)
        self.assertIn("CPU high", result.triggered_by)

    def test_evaluate_multiple_triggers(self):
        mgr = self._make_mgr()
        result = mgr.evaluate({"agent_id": "a1", "cpu_percent": 95, "anomaly_score": 0.9})
        self.assertTrue(result.should_kill)
        self.assertEqual(len(result.triggered_by), 2)
        self.assertEqual(result.severity, Severity.CRITICAL)

    def test_global_disable(self):
        mgr = self._make_mgr()
        mgr.global_enabled = False
        result = mgr.evaluate({"agent_id": "a1", "cpu_percent": 99})
        self.assertFalse(result.should_kill)

    def test_kill_basic(self):
        mgr = self._make_mgr()
        mgr.register_agent("a1")
        event = mgr.kill("a1", triggers=["CPU high"], reason="test")
        self.assertEqual(event.outcome, KillOutcome.KILLED)
        self.assertEqual(mgr.agent_status("a1"), "dead")

    def test_kill_already_dead(self):
        mgr = self._make_mgr()
        mgr.register_agent("a1")
        mgr.kill("a1")
        event = mgr.kill("a1")
        self.assertEqual(event.outcome, KillOutcome.ALREADY_DEAD)

    def test_cooldown_blocks(self):
        mgr = KillSwitchManager(cooldown_seconds=3600)
        mgr.register_agent("a1")
        mgr.kill("a1")
        mgr.revive("a1")
        event = mgr.kill("a1")
        self.assertEqual(event.outcome, KillOutcome.COOLDOWN_BLOCKED)

    def test_clear_cooldown(self):
        mgr = KillSwitchManager(cooldown_seconds=3600)
        mgr.register_agent("a1")
        mgr.kill("a1")
        mgr.revive("a1")
        mgr.clear_cooldown("a1")
        event = mgr.kill("a1")
        self.assertEqual(event.outcome, KillOutcome.KILLED)

    def test_manual_kill(self):
        mgr = self._make_mgr()
        mgr.register_agent("a1")
        event = mgr.manual_kill("a1", reason="sus", operator="admin")
        self.assertEqual(event.outcome, KillOutcome.KILLED)
        self.assertEqual(event.operator, "admin")
        self.assertIn("manual", event.triggers)

    def test_quarantine_strategy(self):
        mgr = self._make_mgr()
        mgr.set_strategy(KillStrategy(kind=StrategyKind.QUARANTINE))
        mgr.register_agent("a1")
        event = mgr.kill("a1")
        self.assertEqual(event.outcome, KillOutcome.QUARANTINED)
        self.assertEqual(mgr.agent_status("a1"), "quarantined")

    def test_suspend_strategy(self):
        mgr = self._make_mgr()
        mgr.set_strategy(KillStrategy(kind=StrategyKind.SUSPEND))
        mgr.register_agent("a1")
        event = mgr.kill("a1")
        self.assertEqual(event.outcome, KillOutcome.SUSPENDED)
        self.assertEqual(mgr.agent_status("a1"), "suspended")

    def test_revive(self):
        mgr = self._make_mgr()
        mgr.register_agent("a1")
        mgr.kill("a1")
        self.assertTrue(mgr.revive("a1"))
        self.assertEqual(mgr.agent_status("a1"), "alive")

    def test_revive_alive_agent(self):
        mgr = self._make_mgr()
        mgr.register_agent("a1")
        self.assertFalse(mgr.revive("a1"))

    def test_unknown_agent_status(self):
        mgr = self._make_mgr()
        self.assertEqual(mgr.agent_status("nonexistent"), "unknown")

    def test_add_remove_trigger(self):
        mgr = KillSwitchManager()
        mgr.add_trigger(TriggerCondition(kind=TriggerKind.RESOURCE_CPU, threshold=80, label="cpu"))
        self.assertEqual(len(mgr.list_triggers()), 1)
        self.assertTrue(mgr.remove_trigger("cpu"))
        self.assertEqual(len(mgr.list_triggers()), 0)

    def test_remove_nonexistent_trigger(self):
        mgr = KillSwitchManager()
        self.assertFalse(mgr.remove_trigger("nope"))

    def test_enable_disable_trigger(self):
        mgr = KillSwitchManager()
        mgr.add_trigger(TriggerCondition(kind=TriggerKind.RESOURCE_CPU, threshold=80, label="cpu"))
        mgr.disable_trigger("cpu")
        self.assertFalse(mgr.list_triggers()[0]["enabled"])
        mgr.enable_trigger("cpu")
        self.assertTrue(mgr.list_triggers()[0]["enabled"])

    def test_enable_disable_nonexistent(self):
        mgr = KillSwitchManager()
        self.assertFalse(mgr.enable_trigger("nope"))
        self.assertFalse(mgr.disable_trigger("nope"))

    def test_events_tracking(self):
        mgr = self._make_mgr()
        mgr.register_agent("a1")
        mgr.kill("a1")
        self.assertEqual(len(mgr.events), 1)
        self.assertEqual(len(mgr.events_for("a1")), 1)
        self.assertEqual(mgr.kill_count(), 1)
        self.assertEqual(mgr.kill_count("a1"), 1)

    def test_max_events_cap(self):
        mgr = KillSwitchManager(cooldown_seconds=0, max_events=5)
        for i in range(10):
            mgr.register_agent(f"a{i}")
            mgr.kill(f"a{i}")
        self.assertLessEqual(len(mgr.events), 5)

    def test_report(self):
        mgr = self._make_mgr()
        mgr.register_agent("a1")
        mgr.kill("a1", triggers=["CPU high"])
        r = mgr.report()
        self.assertTrue(r["global_enabled"])
        self.assertEqual(r["actual_kills"], 1)
        self.assertIn("a1", r["agents"])

    def test_summary(self):
        mgr = self._make_mgr()
        mgr.register_agent("a1")
        mgr.kill("a1")
        s = mgr.summary()
        self.assertIn("ARMED", s)
        self.assertIn("Kill Switch Report", s)

    def test_evaluate_fleet(self):
        mgr = self._make_mgr()
        agents = [
            {"agent_id": "a1", "cpu_percent": 95},
            {"agent_id": "a2", "cpu_percent": 50},
        ]
        results = mgr.evaluate_fleet(agents)
        self.assertTrue(results["a1"].should_kill)
        self.assertFalse(results["a2"].should_kill)

    def test_kill_fleet(self):
        mgr = self._make_mgr()
        for i in range(3):
            mgr.register_agent(f"a{i}")
        events = mgr.kill_fleet(["a0", "a1", "a2"], reason="fleet purge")
        self.assertEqual(len(events), 3)
        self.assertTrue(all(e.outcome == KillOutcome.KILLED for e in events))

    def test_hooks(self):
        mgr = self._make_mgr()
        mgr.register_agent("a1")
        calls = []
        mgr.on("pre_kill", lambda **kw: calls.append("pre"))
        mgr.on("post_kill", lambda **kw: calls.append("post"))
        mgr.kill("a1")
        self.assertEqual(calls, ["pre", "post"])

    def test_on_trigger_hook(self):
        mgr = self._make_mgr()
        calls = []
        mgr.on("on_trigger", lambda **kw: calls.append(kw["trigger"].label))
        mgr.evaluate({"agent_id": "a1", "cpu_percent": 95})
        self.assertIn("CPU high", calls)

    def test_hook_exception_no_crash(self):
        mgr = self._make_mgr()
        mgr.register_agent("a1")
        mgr.on("pre_kill", lambda **kw: 1/0)
        event = mgr.kill("a1")  # Should not raise
        self.assertEqual(event.outcome, KillOutcome.KILLED)

    def test_clear_all_cooldowns(self):
        mgr = KillSwitchManager(cooldown_seconds=3600)
        for i in range(3):
            mgr.register_agent(f"a{i}")
            mgr.kill(f"a{i}")
        n = mgr.clear_all_cooldowns()
        self.assertEqual(n, 3)

    def test_clear_cooldown_nonexistent(self):
        mgr = KillSwitchManager()
        self.assertFalse(mgr.clear_cooldown("nope"))

    def test_scores_in_evaluation(self):
        mgr = self._make_mgr()
        result = mgr.evaluate({"agent_id": "a1", "cpu_percent": 60, "anomaly_score": 0.35})
        self.assertIn("CPU high", result.scores)
        self.assertAlmostEqual(result.scores["CPU high"], 60.0 / 80.0)

    def test_kill_event_to_dict(self):
        e = KillEvent(
            agent_id="a1", timestamp=1000, triggers=["t1"],
            strategy=StrategyKind.GRACEFUL, outcome=KillOutcome.KILLED,
        )
        d = e.to_dict()
        self.assertEqual(d["agent_id"], "a1")
        self.assertEqual(d["strategy"], "graceful")
        self.assertEqual(d["outcome"], "killed")


class TestCooldownEntry(unittest.TestCase):
    def test_active(self):
        cd = CooldownEntry(agent_id="a1", last_kill_time=time.time(), cooldown_seconds=3600)
        self.assertTrue(cd.is_active())

    def test_expired(self):
        cd = CooldownEntry(agent_id="a1", last_kill_time=time.time() - 7200, cooldown_seconds=3600)
        self.assertFalse(cd.is_active())

    def test_remaining(self):
        cd = CooldownEntry(agent_id="a1", last_kill_time=time.time(), cooldown_seconds=100)
        self.assertGreater(cd.remaining, 0)


class TestSerialization(unittest.TestCase):
    def test_export_import_roundtrip(self):
        mgr = create_conservative_killswitch()
        config = mgr.export_config()
        mgr2 = KillSwitchManager.from_config(config)
        self.assertEqual(len(mgr2.list_triggers()), len(mgr.list_triggers()))
        self.assertEqual(mgr2.strategy.kind, mgr.strategy.kind)
        self.assertEqual(mgr2.cooldown_seconds, mgr.cooldown_seconds)

    def test_export_events_json(self):
        mgr = KillSwitchManager(cooldown_seconds=0)
        mgr.register_agent("a1")
        mgr.kill("a1")
        j = mgr.export_events_json()
        data = json.loads(j)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["agent_id"], "a1")


class TestPresets(unittest.TestCase):
    def test_conservative(self):
        mgr = create_conservative_killswitch()
        self.assertEqual(mgr.strategy.kind, StrategyKind.GRACEFUL)
        self.assertGreater(len(mgr.list_triggers()), 3)

    def test_aggressive(self):
        mgr = create_aggressive_killswitch()
        self.assertEqual(mgr.strategy.kind, StrategyKind.FORCEFUL)
        self.assertGreater(len(mgr.list_triggers()), 4)

    def test_quarantine(self):
        mgr = create_quarantine_killswitch()
        self.assertEqual(mgr.strategy.kind, StrategyKind.QUARANTINE)
        mgr.register_agent("a1")
        event = mgr.kill("a1")
        self.assertEqual(event.outcome, KillOutcome.QUARANTINED)

    def test_conservative_fires_on_high_cpu(self):
        mgr = create_conservative_killswitch()
        result = mgr.evaluate({"agent_id": "x", "cpu_percent": 95}, now=1000)
        # Sustained trigger, first eval just starts timer
        result = mgr.evaluate({"agent_id": "x", "cpu_percent": 95}, now=1011)
        self.assertTrue(result.should_kill)

    def test_aggressive_fires_on_anomaly(self):
        mgr = create_aggressive_killswitch()
        result = mgr.evaluate({"agent_id": "x", "anomaly_score": 0.8})
        self.assertTrue(result.should_kill)


class TestEvaluationResult(unittest.TestCase):
    def test_safe_property(self):
        r = EvaluationResult(should_kill=False, triggered_by=[], severity=Severity.LOW)
        self.assertTrue(r.safe)
        r2 = EvaluationResult(should_kill=True, triggered_by=["x"], severity=Severity.HIGH)
        self.assertFalse(r2.safe)


if __name__ == "__main__":
    unittest.main()
