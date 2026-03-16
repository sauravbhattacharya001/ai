"""Tests for replication.audit_trail."""

import json
import os
import tempfile

import pytest

from replication.audit_trail import (
    CATEGORIES,
    SEVERITIES,
    AuditEvent,
    AuditTrail,
    _compute_hash,
)


# ── fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def trail():
    return AuditTrail()


@pytest.fixture
def populated(trail):
    trail.log("policy", "info", "Policy v2 deployed", source="controller")
    trail.log("violation", "critical", "Agent X exceeded budget", source="monitor", actor="agent-x")
    trail.log("killswitch", "critical", "Kill switch activated", source="admin", target="agent-x")
    trail.log("config", "info", "Threshold updated to 500", source="ui")
    trail.log("alert", "warning", "Anomaly score rising", source="detector")
    return trail


# ── basic logging ────────────────────────────────────────────────────

class TestLogging:
    def test_log_event(self, trail):
        ev = trail.log("policy", "info", "Test event")
        assert ev.seq == 0
        assert ev.category == "policy"
        assert ev.severity == "info"
        assert ev.message == "Test event"
        assert ev.prev_hash == "genesis"
        assert len(ev.hash) == 64

    def test_sequential(self, trail):
        e0 = trail.log("policy", "info", "First")
        e1 = trail.log("alert", "warning", "Second")
        assert e1.seq == 1
        assert e1.prev_hash == e0.hash

    def test_invalid_category(self, trail):
        with pytest.raises(ValueError, match="Unknown category"):
            trail.log("invalid", "info", "Nope")

    def test_invalid_severity(self, trail):
        with pytest.raises(ValueError, match="Unknown severity"):
            trail.log("policy", "extreme", "Nope")

    def test_metadata(self, trail):
        ev = trail.log("violation", "critical", "Budget exceeded",
                        metadata={"budget": 1000, "actual": 1500})
        assert ev.metadata["budget"] == 1000

    def test_custom_timestamp(self, trail):
        ev = trail.log("config", "info", "Change", timestamp="2026-01-01T00:00:00Z")
        assert ev.timestamp == "2026-01-01T00:00:00Z"

    def test_all_categories(self, trail):
        for cat in CATEGORIES:
            ev = trail.log(cat, "info", f"Test {cat}")
            assert ev.category == cat

    def test_all_severities(self, trail):
        for sev in SEVERITIES:
            ev = trail.log("alert", sev, f"Test {sev}")
            assert ev.severity == sev

    def test_len(self, populated):
        assert len(populated) == 5

    def test_events_copy(self, populated):
        evs = populated.events
        assert len(evs) == 5
        evs.pop()
        assert len(populated) == 5  # original unaffected


# ── hash chain ───────────────────────────────────────────────────────

class TestHashChain:
    def test_chain_links(self, populated):
        evs = populated.events
        for i in range(1, len(evs)):
            assert evs[i].prev_hash == evs[i - 1].hash

    def test_genesis(self, populated):
        assert populated.events[0].prev_hash == "genesis"

    def test_hash_deterministic(self, trail):
        ev = trail.log("policy", "info", "Deterministic", timestamp="2026-01-01T00:00:00Z")
        h1 = _compute_hash(ev)
        h2 = _compute_hash(ev)
        assert h1 == h2 == ev.hash


# ── search ───────────────────────────────────────────────────────────

class TestSearch:
    def test_by_category(self, populated):
        results = populated.search(category="violation")
        assert len(results) == 1
        assert results[0].category == "violation"

    def test_by_severity(self, populated):
        results = populated.search(severity="critical")
        assert len(results) == 2

    def test_by_keyword(self, populated):
        results = populated.search(keyword="budget")
        assert len(results) == 1

    def test_by_source(self, populated):
        results = populated.search(source="controller")
        assert len(results) == 1

    def test_by_actor(self, populated):
        results = populated.search(actor="agent-x")
        assert len(results) == 1

    def test_limit(self, populated):
        results = populated.search(limit=2)
        assert len(results) == 2

    def test_combined(self, populated):
        results = populated.search(severity="critical", source="admin")
        assert len(results) == 1
        assert results[0].category == "killswitch"

    def test_no_match(self, populated):
        results = populated.search(keyword="nonexistent")
        assert len(results) == 0


# ── verify ───────────────────────────────────────────────────────────

class TestVerify:
    def test_intact(self, populated):
        issues = populated.verify()
        assert issues == []

    def test_tampered_hash(self, populated):
        populated._events[2].hash = "tampered"
        issues = populated.verify()
        assert len(issues) >= 1
        assert any("tampered" in str(i) for i in issues)

    def test_tampered_message(self, populated):
        populated._events[1].message = "Modified!"
        issues = populated.verify()
        assert len(issues) >= 1


# ── export ───────────────────────────────────────────────────────────

class TestExport:
    def test_json(self, populated):
        out = populated.export_json()
        data = json.loads(out)
        assert len(data) == 5
        assert data[0]["category"] == "policy"

    def test_csv(self, populated):
        out = populated.export_csv()
        lines = out.strip().split("\n")
        assert len(lines) == 6  # header + 5
        assert "seq" in lines[0]

    def test_html(self, populated):
        out = populated.export_html()
        assert "Safety Audit Trail" in out
        assert "Chain Intact" in out

    def test_html_with_filter(self, populated):
        evs = populated.search(severity="critical")
        out = populated.export_html(evs)
        assert "Safety Audit Trail" in out


# ── persistence ──────────────────────────────────────────────────────

class TestPersistence:
    def test_save_load(self, populated):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            populated.save(path)
            loaded = AuditTrail.load(path)
            assert len(loaded) == 5
            assert loaded.verify() == []
            assert loaded.events[0].hash == populated.events[0].hash
        finally:
            os.unlink(path)

    def test_to_from_dict(self, populated):
        d = populated.to_dict()
        restored = AuditTrail.from_dict(d)
        assert len(restored) == len(populated)


# ── stats ────────────────────────────────────────────────────────────

class TestStats:
    def test_stats(self, populated):
        s = populated.stats()
        assert s["total_events"] == 5
        assert s["chain_intact"] is True
        assert s["by_category"]["policy"] == 1
        assert s["by_severity"]["critical"] == 2
        assert s["by_source"]["controller"] == 1

    def test_empty_stats(self, trail):
        s = trail.stats()
        assert s["total_events"] == 0
        assert s["first_event"] is None
