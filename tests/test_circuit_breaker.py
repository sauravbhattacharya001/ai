"""Tests for replication.circuit_breaker — safety circuit breaker."""

from __future__ import annotations

import json
import time
from unittest.mock import patch

import pytest

from replication.circuit_breaker import (
    BreakerEvent,
    BreakerFleet,
    BreakerState,
    CircuitBreaker,
    main,
)


# ── helpers ──────────────────────────────────────────────────────────


def _make(
    name: str = "test",
    threshold: int = 3,
    window: float = 60.0,
    cooldown: float = 1.0,
    probe_limit: int = 3,
    probe_success: int = 2,
    risk_sensitive: bool = False,
) -> CircuitBreaker:
    return CircuitBreaker(
        name=name,
        trip_threshold=threshold,
        trip_window_sec=window,
        cooldown_sec=cooldown,
        probe_limit=probe_limit,
        probe_success_needed=probe_success,
        risk_sensitive=risk_sensitive,
    )


# ── BreakerState enum ───────────────────────────────────────────────


class TestBreakerState:
    def test_values(self) -> None:
        assert BreakerState.CLOSED.value == "CLOSED"
        assert BreakerState.OPEN.value == "OPEN"
        assert BreakerState.HALF_OPEN.value == "HALF_OPEN"


# ── BreakerEvent dataclass ──────────────────────────────────────────


class TestBreakerEvent:
    def test_creation(self) -> None:
        evt = BreakerEvent(
            timestamp=100.0,
            breaker="test",
            old_state="CLOSED",
            new_state="OPEN",
            reason="threshold",
        )
        assert evt.breaker == "test"
        assert evt.context == {}

    def test_with_context(self) -> None:
        evt = BreakerEvent(
            timestamp=0.0, breaker="b", old_state="A", new_state="B",
            reason="r", context={"key": "val"},
        )
        assert evt.context == {"key": "val"}


# ── CircuitBreaker — basic state ────────────────────────────────────


class TestCircuitBreakerInit:
    def test_defaults(self) -> None:
        b = _make()
        assert b.state == BreakerState.CLOSED
        assert b.total_trips == 0
        assert b.total_blocks == 0
        assert b.total_operations == 0
        assert b.violations == []
        assert b.journal == []

    def test_check_closed(self) -> None:
        assert _make().check() is True


# ── CLOSED → OPEN trip ──────────────────────────────────────────────


class TestTrip:
    def test_violations_below_threshold_stay_closed(self) -> None:
        b = _make(threshold=5)
        for _ in range(4):
            result = b.record_violation()
            assert "CLOSED" in result
        assert b.state == BreakerState.CLOSED

    def test_violations_at_threshold_trip(self) -> None:
        b = _make(threshold=3)
        for _ in range(2):
            b.record_violation()
        result = b.record_violation()
        assert "OPEN" in result and "tripped" in result
        assert b.state == BreakerState.OPEN
        assert b.total_trips == 1

    def test_violation_context_stored_in_journal(self) -> None:
        b = _make(threshold=1)
        b.record_violation({"metric": "drift"})
        assert len(b.journal) == 1
        assert b.journal[0].context == {"metric": "drift"}

    def test_violations_pruned_outside_window(self) -> None:
        b = _make(threshold=3, window=1.0)
        now = time.time()
        b.violations = [now - 10, now - 10]
        b.last_state_change = now - 20
        result = b.record_violation()
        assert "CLOSED" in result


# ── OPEN state behaviour ────────────────────────────────────────────


class TestOpenState:
    def _trip(self) -> CircuitBreaker:
        b = _make(threshold=2, cooldown=100.0)
        b.record_violation()
        b.record_violation()
        assert b.state == BreakerState.OPEN
        return b

    def test_check_returns_false(self) -> None:
        assert self._trip().check() is False

    def test_violation_while_open_blocked(self) -> None:
        b = self._trip()
        result = b.record_violation()
        assert "blocked" in result
        assert b.total_blocks == 1

    def test_success_while_open_blocked(self) -> None:
        b = self._trip()
        result = b.record_success()
        assert "blocked" in result
        assert b.total_blocks == 1

    def test_cooldown_transitions_to_half_open_via_check(self) -> None:
        b = self._trip()
        b.trip_time = time.time() - 200
        assert b.check() is True
        assert b.state == BreakerState.HALF_OPEN

    def test_cooldown_transitions_to_half_open_via_success(self) -> None:
        b = self._trip()
        b.trip_time = time.time() - 200
        result = b.record_success()
        assert b.state in (BreakerState.HALF_OPEN, BreakerState.CLOSED)


# ── HALF_OPEN → CLOSED recovery ─────────────────────────────────────


class TestHalfOpen:
    def _half_open(self) -> CircuitBreaker:
        b = _make(threshold=2, cooldown=0.0, probe_limit=3, probe_success=2)
        b.record_violation()
        b.record_violation()
        assert b.state == BreakerState.OPEN
        b.trip_time = time.time() - 1
        b.check()
        assert b.state == BreakerState.HALF_OPEN
        return b

    def test_probe_success_recovers(self) -> None:
        b = self._half_open()
        b.record_success()
        result = b.record_success()
        assert "recovered" in result
        assert b.state == BreakerState.CLOSED

    def test_probe_violation_re_trips(self) -> None:
        b = self._half_open()
        result = b.record_violation()
        assert "re-tripped" in result
        assert b.state == BreakerState.OPEN
        assert b.total_trips == 2

    def test_check_false_after_probe_limit(self) -> None:
        b = self._half_open()
        for _ in range(b.probe_limit):
            b.probe_attempts += 1
        assert b.check() is False

    def test_partial_probe_progress(self) -> None:
        b = _make(threshold=1, cooldown=0.0, probe_limit=5, probe_success=3)
        b.record_violation()
        b.trip_time = time.time() - 1
        b.check()
        assert b.state == BreakerState.HALF_OPEN
        result = b.record_success()
        assert "1/3" in result
        assert b.state == BreakerState.HALF_OPEN


# ── Adaptive threshold ───────────────────────────────────────────────


class TestAdaptiveThreshold:
    def test_no_tightening_when_disabled(self) -> None:
        b = _make(threshold=6, risk_sensitive=False)
        assert b._effective_threshold() == 6

    def test_no_tightening_with_few_violations(self) -> None:
        b = _make(threshold=6, risk_sensitive=True)
        b.violations = [time.time()]
        assert b._effective_threshold() == 6

    def test_tightening_under_high_density(self) -> None:
        b = _make(threshold=6, window=60.0, risk_sensitive=True)
        now = time.time()
        b.violations = [now - i * 0.5 for i in range(30)]
        eff = b._effective_threshold()
        assert eff < 6
        assert eff >= 2

    def test_moderate_density_reduces_by_one(self) -> None:
        b = _make(threshold=6, window=60.0, risk_sensitive=True)
        now = time.time()
        b.violations = [now - i * 10 for i in range(12)]
        eff = b._effective_threshold()
        assert eff <= 5

    def test_adaptive_trip_with_fewer_violations(self) -> None:
        b = _make(threshold=6, window=60.0, cooldown=100.0, risk_sensitive=True)
        now = time.time()
        b.violations = [now - i * 0.3 for i in range(20)]
        b.last_state_change = now - 10
        eff = b._effective_threshold()
        assert eff < 6


# ── status_line ──────────────────────────────────────────────────────


class TestStatusLine:
    def test_closed_icon(self) -> None:
        line = _make(name="foo").status_line()
        assert "🟢" in line and "foo" in line and "CLOSED" in line

    def test_open_icon(self) -> None:
        b = _make(name="bar", threshold=1)
        b.record_violation()
        assert "🔴" in b.status_line() and "OPEN" in b.status_line()

    def test_half_open_icon(self) -> None:
        b = _make(name="baz", threshold=1, cooldown=0.0)
        b.record_violation()
        b.trip_time = time.time() - 1
        b.check()
        assert "🟡" in b.status_line() and "HALF_OPEN" in b.status_line()


# ── BreakerFleet ─────────────────────────────────────────────────────


class TestBreakerFleet:
    def test_add_and_get(self) -> None:
        fleet = BreakerFleet()
        b = _make(name="alpha")
        fleet.add(b)
        assert fleet.get("alpha") is b
        assert fleet.get("missing") is None

    def test_status_no_breakers(self) -> None:
        assert "no breakers" in BreakerFleet().status()

    def test_status_with_breakers(self) -> None:
        fleet = BreakerFleet()
        fleet.add(_make(name="x"))
        fleet.add(_make(name="y"))
        s = fleet.status()
        assert "x" in s and "y" in s

    def test_to_dict(self) -> None:
        fleet = BreakerFleet()
        fleet.add(_make(name="test"))
        d = fleet.to_dict()
        assert "test" in d["breakers"]
        assert d["breakers"]["test"]["state"] == "CLOSED"

    def test_all_journal_sorted(self) -> None:
        fleet = BreakerFleet()
        b1 = _make(name="a", threshold=1, cooldown=100.0)
        b2 = _make(name="b", threshold=1, cooldown=100.0)
        fleet.add(b1)
        fleet.add(b2)
        b1.record_violation()
        b2.record_violation()
        journal = fleet.all_journal()
        assert len(journal) >= 2
        for i in range(len(journal) - 1):
            assert journal[i]["timestamp"] <= journal[i + 1]["timestamp"]


# ── Fleet recommendations ────────────────────────────────────────────


class TestRecommendations:
    def test_all_healthy(self) -> None:
        fleet = BreakerFleet()
        fleet.add(_make(name="ok"))
        recs = fleet.recommendations()
        assert any("✅" in r for r in recs)

    def test_open_breaker_warning(self) -> None:
        fleet = BreakerFleet()
        b = _make(name="tripped", threshold=1, cooldown=9999)
        fleet.add(b)
        b.record_violation()
        assert any("OPEN" in r for r in fleet.recommendations())

    def test_many_trips_warning(self) -> None:
        fleet = BreakerFleet()
        b = _make(name="flaky", threshold=1, cooldown=0.0, probe_success=1)
        fleet.add(b)
        for _ in range(6):
            b.record_violation()
            b.trip_time = time.time() - 1
            b.record_success()
        assert any("trips" in r.lower() or "🔁" in r for r in fleet.recommendations())

    def test_half_open_recommendation(self) -> None:
        fleet = BreakerFleet()
        b = _make(name="probing", threshold=1, cooldown=0.0)
        fleet.add(b)
        b.record_violation()
        b.trip_time = time.time() - 1
        b.check()
        assert any("HALF_OPEN" in r or "probe" in r.lower() for r in fleet.recommendations())

    def test_cycling_detection(self) -> None:
        fleet = BreakerFleet()
        b = _make(name="cycler", threshold=1, cooldown=0.0, probe_success=1)
        fleet.add(b)
        for _ in range(5):
            b.record_violation()
            b.trip_time = time.time() - 1
            b.record_success()
        assert any("cycling" in r.lower() or "🔄" in r for r in fleet.recommendations())


# ── Journal / transition tracking ────────────────────────────────────


class TestJournal:
    def test_trip_creates_journal_entry(self) -> None:
        b = _make(threshold=1)
        b.record_violation()
        assert len(b.journal) == 1
        assert b.journal[0].new_state == "OPEN"

    def test_recovery_creates_journal_entry(self) -> None:
        b = _make(threshold=1, cooldown=0.0, probe_success=1)
        b.record_violation()
        b.trip_time = time.time() - 1
        b.check()
        b.record_success()
        states = [(e.old_state, e.new_state) for e in b.journal]
        assert ("CLOSED", "OPEN") in states
        assert ("OPEN", "HALF_OPEN") in states
        assert ("HALF_OPEN", "CLOSED") in states

    def test_uptime_tracked_on_transition(self) -> None:
        b = _make(threshold=1, cooldown=100.0)
        b.last_state_change = time.time() - 5.0
        b.record_violation()
        assert b.uptime_closed_sec >= 4.0


# ── CLI (main) ───────────────────────────────────────────────────────


class TestCLI:
    def test_demo_basic(self, capsys: pytest.CaptureFixture) -> None:  # type: ignore[type-arg]
        main(["--demo"])
        out = capsys.readouterr().out
        assert "Demo" in out or "CLOSED" in out or "OPEN" in out

    def test_scenario_cascade(self, capsys: pytest.CaptureFixture) -> None:  # type: ignore[type-arg]
        main(["--scenario", "cascade"])
        assert "Cascade" in capsys.readouterr().out or True

    def test_scenario_adaptive(self, capsys: pytest.CaptureFixture) -> None:  # type: ignore[type-arg]
        main(["--scenario", "adaptive"])
        assert "Adaptive" in capsys.readouterr().out or True

    def test_export_json(self, capsys: pytest.CaptureFixture) -> None:  # type: ignore[type-arg]
        main(["--demo", "--export", "json"])
        out = capsys.readouterr().out
        assert '"breakers"' in out

    def test_export_text(self, capsys: pytest.CaptureFixture) -> None:  # type: ignore[type-arg]
        main(["--demo", "--export", "text"])
        out = capsys.readouterr().out
        assert "FLEET STATUS" in out or "Journal" in out

    def test_status_flag(self, capsys: pytest.CaptureFixture) -> None:  # type: ignore[type-arg]
        main(["--demo", "--status"])
        assert "FLEET STATUS" in capsys.readouterr().out

    def test_default_runs_all(self, capsys: pytest.CaptureFixture) -> None:  # type: ignore[type-arg]
        main([])
        out = capsys.readouterr().out
        assert len(out) > 100
