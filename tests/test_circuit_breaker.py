"""Tests for circuit_breaker module — CircuitBreaker state machine and BreakerFleet."""

import time
from unittest.mock import patch

import pytest

from replication.circuit_breaker import (
    BreakerEvent,
    BreakerFleet,
    BreakerState,
    CircuitBreaker,
)


# ── CircuitBreaker state transitions ────────────────────────────────


class TestCircuitBreakerBasic:
    """CLOSED → OPEN → HALF_OPEN → CLOSED lifecycle."""

    def test_initial_state_is_closed(self):
        cb = CircuitBreaker(name="test")
        assert cb.state == BreakerState.CLOSED
        assert cb.check() is True

    def test_violations_below_threshold_stay_closed(self):
        cb = CircuitBreaker(name="test", trip_threshold=5, trip_window_sec=60)
        for _ in range(4):
            result = cb.record_violation()
            assert "CLOSED" in result
        assert cb.state == BreakerState.CLOSED

    def test_violations_at_threshold_trip_to_open(self):
        cb = CircuitBreaker(name="test", trip_threshold=3, trip_window_sec=60)
        for _ in range(2):
            cb.record_violation()
        result = cb.record_violation()
        assert "OPEN" in result and "tripped" in result
        assert cb.state == BreakerState.OPEN
        assert cb.total_trips == 1

    def test_open_blocks_operations(self):
        cb = CircuitBreaker(name="test", trip_threshold=1, cooldown_sec=999)
        cb.record_violation()
        assert cb.state == BreakerState.OPEN
        assert cb.check() is False
        result = cb.record_success()
        assert "blocked" in result
        assert cb.total_blocks == 1

    def test_open_blocks_further_violations(self):
        cb = CircuitBreaker(name="test", trip_threshold=1, cooldown_sec=999)
        cb.record_violation()
        result = cb.record_violation()
        assert "blocked" in result
        assert cb.total_blocks == 1

    def test_cooldown_transitions_to_half_open(self):
        cb = CircuitBreaker(name="test", trip_threshold=1, cooldown_sec=0.01)
        cb.record_violation()
        assert cb.state == BreakerState.OPEN
        time.sleep(0.02)
        assert cb.check() is True
        assert cb.state == BreakerState.HALF_OPEN

    def test_probe_success_recovers_to_closed(self):
        cb = CircuitBreaker(name="test", trip_threshold=1, cooldown_sec=0.01,
                            probe_success_needed=2)
        cb.record_violation()
        time.sleep(0.02)
        cb.check()  # trigger half-open
        cb.record_success()
        assert cb.state == BreakerState.HALF_OPEN
        result = cb.record_success()
        assert "recovered" in result
        assert cb.state == BreakerState.CLOSED

    def test_violation_during_half_open_retrips(self):
        cb = CircuitBreaker(name="test", trip_threshold=1, cooldown_sec=0.01)
        cb.record_violation()
        time.sleep(0.02)
        cb.check()  # half-open
        assert cb.state == BreakerState.HALF_OPEN
        result = cb.record_violation()
        assert "re-tripped" in result
        assert cb.state == BreakerState.OPEN
        assert cb.total_trips == 2

    def test_probe_limit_blocks_excess_attempts(self):
        cb = CircuitBreaker(name="test", trip_threshold=1, cooldown_sec=0.01,
                            probe_limit=2, probe_success_needed=3)
        cb.record_violation()
        time.sleep(0.02)
        cb.check()  # half-open
        cb.record_success()
        cb.record_success()
        # probe_attempts == 2 == probe_limit, so check should block
        assert cb.check() is False


class TestCircuitBreakerWindow:
    """Violation window expiry."""

    def test_old_violations_expire(self):
        cb = CircuitBreaker(name="test", trip_threshold=3, trip_window_sec=0.05,
                            risk_sensitive=False)
        cb.record_violation()
        cb.record_violation()
        time.sleep(0.06)
        # Old violations expired; this is only the 1st in the new window
        result = cb.record_violation()
        assert "CLOSED" in result
        assert cb.state == BreakerState.CLOSED


class TestAdaptiveThreshold:
    """Risk-sensitive adaptive threshold tightening."""

    def test_adaptive_tightens_under_high_density(self):
        cb = CircuitBreaker(name="test", trip_threshold=6,
                            trip_window_sec=100.0, risk_sensitive=True)
        # Inject many recent violations to raise density > 0.15
        now = time.time()
        cb.violations = [now - i * 0.1 for i in range(30)]
        eff = cb._effective_threshold()
        assert eff < cb.trip_threshold
        assert eff >= 2

    def test_non_adaptive_ignores_density(self):
        cb = CircuitBreaker(name="test", trip_threshold=6,
                            trip_window_sec=100.0, risk_sensitive=False)
        now = time.time()
        cb.violations = [now - i * 0.1 for i in range(30)]
        assert cb._effective_threshold() == 6


class TestJournal:
    """Event journal recording."""

    def test_trip_records_journal_entry(self):
        cb = CircuitBreaker(name="j", trip_threshold=1, cooldown_sec=999)
        cb.record_violation({"metric": "score"})
        assert len(cb.journal) == 1
        evt = cb.journal[0]
        assert evt.old_state == "CLOSED"
        assert evt.new_state == "OPEN"
        assert evt.breaker == "j"
        assert "metric" in evt.context

    def test_full_cycle_records_multiple_entries(self):
        cb = CircuitBreaker(name="j", trip_threshold=1, cooldown_sec=0.01,
                            probe_success_needed=1)
        cb.record_violation()
        time.sleep(0.02)
        cb.record_success()  # triggers half-open transition + probe
        cb.record_success()  # closes
        # Should have: CLOSED→OPEN, OPEN→HALF_OPEN, HALF_OPEN→CLOSED
        states = [(e.old_state, e.new_state) for e in cb.journal]
        assert ("CLOSED", "OPEN") in states
        assert ("OPEN", "HALF_OPEN") in states
        assert ("HALF_OPEN", "CLOSED") in states


class TestStatusLine:
    """status_line formatting."""

    def test_status_line_contains_name_and_state(self):
        cb = CircuitBreaker(name="align")
        line = cb.status_line()
        assert "align" in line
        assert "CLOSED" in line
        assert "🟢" in line

    def test_open_status_shows_red(self):
        cb = CircuitBreaker(name="x", trip_threshold=1)
        cb.record_violation()
        assert "🔴" in cb.status_line()


# ── BreakerFleet ─────────────────────────────────────────────────────


class TestBreakerFleet:

    def test_add_and_get(self):
        fleet = BreakerFleet()
        b = CircuitBreaker(name="test")
        fleet.add(b)
        assert fleet.get("test") is b
        assert fleet.get("nonexistent") is None

    def test_status_includes_all_breakers(self):
        fleet = BreakerFleet()
        fleet.add(CircuitBreaker(name="alpha"))
        fleet.add(CircuitBreaker(name="beta"))
        status = fleet.status()
        assert "alpha" in status
        assert "beta" in status

    def test_empty_fleet_status(self):
        fleet = BreakerFleet()
        status = fleet.status()
        assert "no breakers" in status

    def test_recommendations_healthy(self):
        fleet = BreakerFleet()
        fleet.add(CircuitBreaker(name="ok"))
        recs = fleet.recommendations()
        assert any("healthy" in r or "✅" in r for r in recs)

    def test_recommendations_open_breaker(self):
        fleet = BreakerFleet()
        b = CircuitBreaker(name="bad", trip_threshold=1)
        fleet.add(b)
        b.record_violation()
        recs = fleet.recommendations()
        assert any("OPEN" in r for r in recs)

    def test_recommendations_many_trips(self):
        fleet = BreakerFleet()
        b = CircuitBreaker(name="flaky", trip_threshold=1, cooldown_sec=0.001,
                           probe_success_needed=1)
        fleet.add(b)
        for _ in range(6):
            b.record_violation()
            time.sleep(0.002)
            b.record_success()  # half-open
            b.record_success()  # close
        recs = fleet.recommendations()
        assert any("trips" in r.lower() or "🔁" in r for r in recs)

    def test_all_journal_sorted(self):
        fleet = BreakerFleet()
        b1 = CircuitBreaker(name="a", trip_threshold=1)
        b2 = CircuitBreaker(name="b", trip_threshold=1)
        fleet.add(b1)
        fleet.add(b2)
        b1.record_violation()
        b2.record_violation()
        journal = fleet.all_journal()
        assert len(journal) >= 2
        timestamps = [e["timestamp"] for e in journal]
        assert timestamps == sorted(timestamps)

    def test_to_dict_structure(self):
        fleet = BreakerFleet()
        fleet.add(CircuitBreaker(name="x", trip_threshold=3))
        d = fleet.to_dict()
        assert "breakers" in d
        assert "x" in d["breakers"]
        assert d["breakers"]["x"]["state"] == "CLOSED"
        assert "journal_entries" in d


class TestRecordSuccessFromOpen:
    """record_success while OPEN should transition via cooldown."""

    def test_success_after_cooldown_enters_half_open(self):
        cb = CircuitBreaker(name="test", trip_threshold=1, cooldown_sec=0.01,
                            probe_success_needed=2)
        cb.record_violation()
        time.sleep(0.02)
        result = cb.record_success()
        # Should have transitioned to HALF_OPEN and counted as probe success
        assert cb.state == BreakerState.HALF_OPEN
        assert "HALF_OPEN" in result
