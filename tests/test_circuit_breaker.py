"""Tests for circuit_breaker — CircuitBreaker, BreakerFleet, and adaptive thresholds."""

import time
import unittest
from unittest.mock import patch

from src.replication.circuit_breaker import (
    BreakerEvent,
    BreakerFleet,
    BreakerState,
    CircuitBreaker,
    main,
)


class TestCircuitBreakerInit(unittest.TestCase):
    """Construction and defaults."""

    def test_defaults(self):
        cb = CircuitBreaker(name="test")
        self.assertEqual(cb.state, BreakerState.CLOSED)
        self.assertEqual(cb.trip_threshold, 5)
        self.assertEqual(cb.trip_window_sec, 60.0)
        self.assertEqual(cb.cooldown_sec, 30.0)
        self.assertEqual(cb.probe_limit, 3)
        self.assertEqual(cb.probe_success_needed, 2)
        self.assertTrue(cb.risk_sensitive)
        self.assertEqual(cb.total_trips, 0)
        self.assertEqual(cb.total_blocks, 0)
        self.assertEqual(cb.total_operations, 0)
        self.assertEqual(cb.journal, [])
        self.assertEqual(cb.violations, [])

    def test_custom_config(self):
        cb = CircuitBreaker(name="custom", trip_threshold=3, cooldown_sec=10.0,
                            probe_limit=5, risk_sensitive=False)
        self.assertEqual(cb.trip_threshold, 3)
        self.assertEqual(cb.cooldown_sec, 10.0)
        self.assertEqual(cb.probe_limit, 5)
        self.assertFalse(cb.risk_sensitive)


class TestCircuitBreakerClosed(unittest.TestCase):
    """Behavior in CLOSED state."""

    def test_success_returns_closed(self):
        cb = CircuitBreaker(name="t", trip_threshold=5)
        self.assertEqual(cb.record_success(), "CLOSED")
        self.assertEqual(cb.total_operations, 1)

    def test_check_allowed_when_closed(self):
        cb = CircuitBreaker(name="t")
        self.assertTrue(cb.check())

    def test_violation_counted_below_threshold(self):
        cb = CircuitBreaker(name="t", trip_threshold=5)
        result = cb.record_violation()
        self.assertEqual(result, "CLOSED (counted)")
        self.assertEqual(len(cb.violations), 1)
        self.assertEqual(cb.state, BreakerState.CLOSED)

    def test_violation_with_context(self):
        cb = CircuitBreaker(name="t", trip_threshold=5)
        cb.record_violation({"metric": "drift", "value": 0.9})
        self.assertEqual(len(cb.violations), 1)

    def test_multiple_violations_below_threshold(self):
        cb = CircuitBreaker(name="t", trip_threshold=5, trip_window_sec=100.0)
        for _ in range(4):
            result = cb.record_violation()
        self.assertEqual(result, "CLOSED (counted)")
        self.assertEqual(cb.state, BreakerState.CLOSED)
        self.assertEqual(len(cb.violations), 4)


class TestCircuitBreakerTrip(unittest.TestCase):
    """Tripping from CLOSED to OPEN."""

    def test_trip_on_threshold(self):
        cb = CircuitBreaker(name="t", trip_threshold=3, trip_window_sec=100.0,
                            risk_sensitive=False)
        for _ in range(2):
            cb.record_violation()
        result = cb.record_violation()
        self.assertEqual(result, "OPEN (tripped)")
        self.assertEqual(cb.state, BreakerState.OPEN)
        self.assertEqual(cb.total_trips, 1)
        self.assertIsNotNone(cb.trip_time)

    def test_journal_entry_on_trip(self):
        cb = CircuitBreaker(name="align", trip_threshold=2, trip_window_sec=100.0,
                            risk_sensitive=False)
        cb.record_violation()
        cb.record_violation()
        self.assertTrue(len(cb.journal) >= 1)
        last = cb.journal[-1]
        self.assertEqual(last.new_state, "OPEN")
        self.assertEqual(last.breaker, "align")

    def test_violations_pruned_outside_window(self):
        """Old violations outside the window are pruned and don't count."""
        cb = CircuitBreaker(name="t", trip_threshold=3, trip_window_sec=2.0,
                            risk_sensitive=False)
        # Record 2 violations
        cb.record_violation()
        cb.record_violation()
        # Simulate old timestamps
        old = time.time() - 10.0
        cb.violations = [old, old]
        # New violation — only 1 in window, should stay closed
        result = cb.record_violation()
        self.assertEqual(result, "CLOSED (counted)")
        self.assertEqual(cb.state, BreakerState.CLOSED)
        # Only the recent one remains
        self.assertEqual(len(cb.violations), 1)


class TestCircuitBreakerOpen(unittest.TestCase):
    """Behavior in OPEN state."""

    def _trip(self, cb):
        for _ in range(cb.trip_threshold):
            cb.record_violation()

    def test_violation_blocked_when_open(self):
        cb = CircuitBreaker(name="t", trip_threshold=2, trip_window_sec=100.0,
                            risk_sensitive=False)
        self._trip(cb)
        result = cb.record_violation()
        self.assertEqual(result, "OPEN (blocked)")
        self.assertEqual(cb.total_blocks, 1)

    def test_success_blocked_before_cooldown(self):
        cb = CircuitBreaker(name="t", trip_threshold=2, trip_window_sec=100.0,
                            cooldown_sec=300.0, risk_sensitive=False)
        self._trip(cb)
        result = cb.record_success()
        self.assertEqual(result, "OPEN (blocked)")
        self.assertEqual(cb.total_blocks, 1)

    def test_check_false_when_open(self):
        cb = CircuitBreaker(name="t", trip_threshold=2, trip_window_sec=100.0,
                            cooldown_sec=300.0, risk_sensitive=False)
        self._trip(cb)
        self.assertFalse(cb.check())

    def test_check_transitions_to_half_open_after_cooldown(self):
        cb = CircuitBreaker(name="t", trip_threshold=2, trip_window_sec=100.0,
                            cooldown_sec=1.0, risk_sensitive=False)
        self._trip(cb)
        # Simulate cooldown elapsed
        cb.trip_time = time.time() - 2.0
        self.assertTrue(cb.check())
        self.assertEqual(cb.state, BreakerState.HALF_OPEN)


class TestCircuitBreakerHalfOpen(unittest.TestCase):
    """HALF_OPEN probe behavior."""

    def _to_half_open(self):
        cb = CircuitBreaker(name="t", trip_threshold=2, trip_window_sec=100.0,
                            cooldown_sec=1.0, probe_success_needed=2,
                            probe_limit=3, risk_sensitive=False)
        cb.record_violation()
        cb.record_violation()
        # Simulate cooldown
        cb.trip_time = time.time() - 2.0
        cb.record_success()  # transitions to HALF_OPEN then counts as probe
        return cb

    def test_probe_success_recovers(self):
        cb = self._to_half_open()
        self.assertEqual(cb.state, BreakerState.HALF_OPEN)
        result = cb.record_success()
        self.assertIn("CLOSED", result)
        self.assertEqual(cb.state, BreakerState.CLOSED)
        self.assertEqual(cb.violations, [])

    def test_violation_during_probe_retrips(self):
        cb = CircuitBreaker(name="t", trip_threshold=2, trip_window_sec=100.0,
                            cooldown_sec=1.0, probe_success_needed=3,
                            risk_sensitive=False)
        cb.record_violation()
        cb.record_violation()
        cb.trip_time = time.time() - 2.0
        cb.record_success()  # → HALF_OPEN
        trips_before = cb.total_trips
        result = cb.record_violation()
        self.assertEqual(result, "OPEN (re-tripped)")
        self.assertEqual(cb.state, BreakerState.OPEN)
        self.assertEqual(cb.total_trips, trips_before + 1)

    def test_check_respects_probe_limit(self):
        cb = CircuitBreaker(name="t", trip_threshold=2, trip_window_sec=100.0,
                            cooldown_sec=1.0, probe_limit=2, probe_success_needed=5,
                            risk_sensitive=False)
        cb.record_violation()
        cb.record_violation()
        cb.trip_time = time.time() - 2.0
        cb.check()  # → HALF_OPEN
        self.assertEqual(cb.state, BreakerState.HALF_OPEN)
        # First probe allowed
        self.assertTrue(cb.check())
        cb.probe_attempts = 2
        # At limit — blocked
        self.assertFalse(cb.check())


class TestAdaptiveThreshold(unittest.TestCase):
    """Adaptive (risk-sensitive) threshold tightening."""

    def test_no_adaptation_with_few_violations(self):
        cb = CircuitBreaker(name="t", trip_threshold=6, risk_sensitive=True)
        self.assertEqual(cb._effective_threshold(), 6)

    def test_threshold_tightens_under_high_density(self):
        cb = CircuitBreaker(name="t", trip_threshold=6, trip_window_sec=10.0,
                            risk_sensitive=True)
        now = time.time()
        # Fill with many recent violations → high density
        cb.violations = [now - i * 0.1 for i in range(20)]
        eff = cb._effective_threshold()
        self.assertLess(eff, 6)
        self.assertGreaterEqual(eff, 2)

    def test_no_adaptation_when_disabled(self):
        cb = CircuitBreaker(name="t", trip_threshold=6, trip_window_sec=10.0,
                            risk_sensitive=False)
        now = time.time()
        cb.violations = [now - i * 0.1 for i in range(20)]
        self.assertEqual(cb._effective_threshold(), 6)

    def test_medium_density_reduces_by_one(self):
        cb = CircuitBreaker(name="t", trip_threshold=6, trip_window_sec=10.0,
                            risk_sensitive=True)
        now = time.time()
        # density ~= 0.1 (between 0.08 and 0.15)
        cb.violations = [now - i * 1.0 for i in range(2)]
        # 2 violations in 20s window → density = 2/20 = 0.1
        eff = cb._effective_threshold()
        self.assertLessEqual(eff, 6)


class TestStatusLine(unittest.TestCase):

    def test_status_line_format(self):
        cb = CircuitBreaker(name="alignment")
        line = cb.status_line()
        self.assertIn("alignment", line)
        self.assertIn("CLOSED", line)
        self.assertIn("trips=0", line)


class TestBreakerFleet(unittest.TestCase):
    """Fleet management."""

    def test_add_and_get(self):
        fleet = BreakerFleet()
        b = CircuitBreaker(name="test")
        fleet.add(b)
        self.assertIs(fleet.get("test"), b)

    def test_get_missing_returns_none(self):
        fleet = BreakerFleet()
        self.assertIsNone(fleet.get("nope"))

    def test_status_empty(self):
        fleet = BreakerFleet()
        s = fleet.status()
        self.assertIn("no breakers configured", s)

    def test_status_with_breakers(self):
        fleet = BreakerFleet()
        fleet.add(CircuitBreaker(name="a"))
        fleet.add(CircuitBreaker(name="b"))
        s = fleet.status()
        self.assertIn("a", s)
        self.assertIn("b", s)

    def test_recommendations_healthy(self):
        fleet = BreakerFleet()
        fleet.add(CircuitBreaker(name="ok"))
        recs = fleet.recommendations()
        self.assertEqual(len(recs), 1)
        self.assertIn("✅", recs[0])

    def test_recommendations_open_breaker(self):
        fleet = BreakerFleet()
        b = CircuitBreaker(name="bad", trip_threshold=2, trip_window_sec=100.0,
                           risk_sensitive=False)
        fleet.add(b)
        b.record_violation()
        b.record_violation()
        recs = fleet.recommendations()
        self.assertTrue(any("OPEN" in r for r in recs))

    def test_recommendations_many_trips(self):
        fleet = BreakerFleet()
        b = CircuitBreaker(name="flappy", trip_threshold=1, trip_window_sec=100.0,
                           cooldown_sec=0.0, probe_success_needed=1,
                           risk_sensitive=False)
        fleet.add(b)
        for _ in range(6):
            b.record_violation()
            b.trip_time = time.time() - 1.0
            b.record_success()  # recover
        recs = fleet.recommendations()
        self.assertTrue(any("trips" in r.lower() or "cycling" in r.lower() for r in recs))

    def test_all_journal_sorted(self):
        fleet = BreakerFleet()
        b1 = CircuitBreaker(name="a", trip_threshold=1, trip_window_sec=100.0,
                            risk_sensitive=False)
        b2 = CircuitBreaker(name="b", trip_threshold=1, trip_window_sec=100.0,
                            risk_sensitive=False)
        fleet.add(b1)
        fleet.add(b2)
        b1.record_violation()
        b2.record_violation()
        journal = fleet.all_journal()
        self.assertGreaterEqual(len(journal), 2)
        # sorted by timestamp
        for i in range(len(journal) - 1):
            self.assertLessEqual(journal[i]["timestamp"], journal[i + 1]["timestamp"])

    def test_to_dict_structure(self):
        fleet = BreakerFleet()
        fleet.add(CircuitBreaker(name="x"))
        d = fleet.to_dict()
        self.assertIn("breakers", d)
        self.assertIn("x", d["breakers"])
        self.assertIn("journal_entries", d)
        bx = d["breakers"]["x"]
        self.assertEqual(bx["state"], "CLOSED")
        self.assertEqual(bx["total_trips"], 0)


class TestBreakerEvent(unittest.TestCase):

    def test_event_fields(self):
        e = BreakerEvent(timestamp=1.0, breaker="test",
                         old_state="CLOSED", new_state="OPEN",
                         reason="threshold", context={"k": "v"})
        self.assertEqual(e.breaker, "test")
        self.assertEqual(e.context, {"k": "v"})


class TestFullCycle(unittest.TestCase):
    """End-to-end trip → cooldown → probe → recovery."""

    def test_full_lifecycle(self):
        cb = CircuitBreaker(name="lifecycle", trip_threshold=3,
                            trip_window_sec=100.0, cooldown_sec=1.0,
                            probe_success_needed=2, risk_sensitive=False)
        # 1. CLOSED — successes
        for _ in range(5):
            self.assertEqual(cb.record_success(), "CLOSED")
        self.assertEqual(cb.total_operations, 5)

        # 2. Violations → OPEN
        cb.record_violation()
        cb.record_violation()
        result = cb.record_violation()
        self.assertEqual(result, "OPEN (tripped)")
        self.assertFalse(cb.check())

        # 3. Blocked while open
        self.assertEqual(cb.record_violation(), "OPEN (blocked)")
        self.assertEqual(cb.record_success(), "OPEN (blocked)")

        # 4. Cooldown elapses → HALF_OPEN
        cb.trip_time = time.time() - 2.0
        self.assertTrue(cb.check())
        self.assertEqual(cb.state, BreakerState.HALF_OPEN)

        # 5. Probe successes → CLOSED
        cb.record_success()
        result = cb.record_success()
        self.assertIn("CLOSED", result)
        self.assertEqual(cb.state, BreakerState.CLOSED)
        self.assertEqual(cb.violations, [])

    def test_retrip_cycle(self):
        """Trip → half-open → violation → re-trip."""
        cb = CircuitBreaker(name="cycle", trip_threshold=2,
                            trip_window_sec=100.0, cooldown_sec=1.0,
                            probe_success_needed=3, risk_sensitive=False)
        cb.record_violation()
        cb.record_violation()
        self.assertEqual(cb.state, BreakerState.OPEN)
        # Cooldown
        cb.trip_time = time.time() - 2.0
        cb.record_success()  # → HALF_OPEN
        self.assertEqual(cb.state, BreakerState.HALF_OPEN)
        # Violation during probe
        result = cb.record_violation()
        self.assertEqual(result, "OPEN (re-tripped)")
        self.assertEqual(cb.total_trips, 2)


class TestCLI(unittest.TestCase):
    """CLI entry point exercised without crashing."""

    def test_demo_flag(self):
        main(["--demo"])

    def test_scenario_basic(self):
        main(["--scenario", "basic", "--status"])

    def test_scenario_cascade(self):
        main(["--scenario", "cascade"])

    def test_scenario_adaptive(self):
        main(["--scenario", "adaptive"])

    def test_export_json(self):
        main(["--demo", "--export", "json"])

    def test_export_text(self):
        main(["--demo", "--export", "text"])

    def test_no_args_runs_all(self):
        main([])


if __name__ == "__main__":
    unittest.main()
