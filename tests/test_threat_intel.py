"""Tests for the Agent Threat Intelligence Feed module."""

import time
import pytest
from replication.threat_intel import (
    IOC, IOCType, Severity, AlertAction, AlertRule,
    CorrelationGroup, FeedConfig, ThreatIntelFeed, FeedReport,
    TrendPoint, _score_to_grade,
)


# ── Helpers ────────────────────────────────────────────────────────────

def _ioc(
    ioc_type=IOCType.BEHAVIOR, source="test", agent_id="agent-001",
    description="test ioc", score=50.0, confidence=0.5,
    timestamp=1000.0, tags=None, correlation_keys=None, **kw
):
    return IOC(
        ioc_type=ioc_type, source=source, agent_id=agent_id,
        description=description, score=score, confidence=confidence,
        timestamp=timestamp, tags=tags or [], correlation_keys=correlation_keys or [],
        **kw,
    )


def _feed(**kw):
    config = FeedConfig(source_weights={"test": 1.0, "killchain": 1.5, "src_a": 1.0, "src_b": 1.0}, **kw)
    return ThreatIntelFeed(config)


# ── IOC ────────────────────────────────────────────────────────────────

class TestIOC:
    def test_auto_id(self):
        ioc = _ioc()
        assert len(ioc.id) == 16

    def test_unique_ids(self):
        a = _ioc(description="a")
        b = _ioc(description="b")
        assert a.id != b.id

    def test_to_dict(self):
        ioc = _ioc(tags=["t1"])
        d = ioc.to_dict()
        assert d["type"] == "behavior"
        assert d["source"] == "test"
        assert d["tags"] == ["t1"]

    def test_timestamp_auto(self):
        ioc = IOC(ioc_type=IOCType.BEHAVIOR, source="x", agent_id="a", description="d")
        assert ioc.timestamp > 0


# ── Severity ───────────────────────────────────────────────────────────

class TestSeverity:
    def test_numeric(self):
        assert Severity.INFO.numeric == 1
        assert Severity.CRITICAL.numeric == 5

    def test_from_score(self):
        assert Severity.from_score(95) == Severity.CRITICAL
        assert Severity.from_score(75) == Severity.HIGH
        assert Severity.from_score(55) == Severity.MEDIUM
        assert Severity.from_score(35) == Severity.LOW
        assert Severity.from_score(10) == Severity.INFO


# ── Grade ──────────────────────────────────────────────────────────────

class TestGrade:
    def test_grades(self):
        assert _score_to_grade(95) == "A"
        assert _score_to_grade(85) == "B"
        assert _score_to_grade(75) == "C"
        assert _score_to_grade(65) == "D"
        assert _score_to_grade(50) == "F"


# ── Ingestion ──────────────────────────────────────────────────────────

class TestIngestion:
    def test_ingest_returns_id(self):
        feed = _feed()
        ioc_id = feed.ingest(_ioc())
        assert ioc_id is not None
        assert feed.count == 1

    def test_dedup_within_window(self):
        feed = _feed(dedup_window_seconds=100)
        feed.ingest(_ioc(timestamp=1000))
        result = feed.ingest(_ioc(timestamp=1050))  # same desc, within window
        assert result is None
        assert feed.count == 1

    def test_dedup_outside_window(self):
        feed = _feed(dedup_window_seconds=10)
        feed.ingest(_ioc(timestamp=1000))
        result = feed.ingest(_ioc(timestamp=1020))
        assert result is not None
        assert feed.count == 2

    def test_source_weight_applied(self):
        config = FeedConfig(source_weights={"killchain": 1.5})
        feed = ThreatIntelFeed(config)
        feed.ingest(_ioc(source="killchain", score=60))
        ioc = feed.query()[0]
        assert ioc.score == 90.0  # 60 * 1.5

    def test_score_capped_at_100(self):
        config = FeedConfig(source_weights={"killchain": 1.5})
        feed = ThreatIntelFeed(config)
        feed.ingest(_ioc(source="killchain", score=80))
        assert feed.query()[0].score == 100.0

    def test_max_capacity_eviction(self):
        feed = _feed(max_iocs=3)
        for i in range(5):
            feed.ingest(_ioc(description=f"ioc-{i}", timestamp=1000 + i * 100))
        assert feed.count == 3

    def test_batch_ingest(self):
        feed = _feed()
        results = feed.ingest_batch([_ioc(description=f"b-{i}", timestamp=1000 + i * 100) for i in range(5)])
        assert len(results) == 5
        assert feed.count == 5


# ── Query ──────────────────────────────────────────────────────────────

class TestQuery:
    def test_filter_by_type(self):
        feed = _feed()
        feed.ingest(_ioc(ioc_type=IOCType.BEHAVIOR, description="a"))
        feed.ingest(_ioc(ioc_type=IOCType.RESOURCE, description="b", timestamp=1001))
        results = feed.query(ioc_type=IOCType.RESOURCE)
        assert len(results) == 1
        assert results[0].ioc_type == IOCType.RESOURCE

    def test_filter_by_severity(self):
        feed = _feed()
        feed.ingest(_ioc(score=95, description="crit"))
        feed.ingest(_ioc(score=20, description="low", timestamp=1001))
        results = feed.query(severity=Severity.CRITICAL)
        assert len(results) == 1

    def test_filter_min_severity(self):
        feed = _feed()
        feed.ingest(_ioc(score=95, description="a"))
        feed.ingest(_ioc(score=75, description="b", timestamp=1001))
        feed.ingest(_ioc(score=20, description="c", timestamp=1002))
        results = feed.query(min_severity=Severity.HIGH)
        assert len(results) == 2

    def test_filter_by_source(self):
        feed = _feed()
        feed.ingest(_ioc(source="test", description="a"))
        feed.ingest(_ioc(source="killchain", description="b", timestamp=1001))
        results = feed.query(source="killchain")
        assert len(results) == 1

    def test_filter_by_agent(self):
        feed = _feed()
        feed.ingest(_ioc(agent_id="a1", description="x"))
        feed.ingest(_ioc(agent_id="a2", description="y", timestamp=1001))
        results = feed.query(agent_id="a1")
        assert len(results) == 1

    def test_filter_by_tag(self):
        feed = _feed()
        feed.ingest(_ioc(tags=["urgent"], description="a"))
        feed.ingest(_ioc(tags=["routine"], description="b", timestamp=1001))
        results = feed.query(tag="urgent")
        assert len(results) == 1

    def test_filter_min_score(self):
        feed = _feed()
        feed.ingest(_ioc(score=80, description="a"))
        feed.ingest(_ioc(score=30, description="b", timestamp=1001))
        results = feed.query(min_score=50)
        assert len(results) == 1

    def test_filter_since(self):
        feed = _feed()
        feed.ingest(_ioc(timestamp=500, description="old"))
        feed.ingest(_ioc(timestamp=1500, description="new"))
        results = feed.query(since=1000)
        assert len(results) == 1

    def test_limit(self):
        feed = _feed()
        for i in range(10):
            feed.ingest(_ioc(description=f"i-{i}", timestamp=1000 + i * 100))
        results = feed.query(limit=3)
        assert len(results) == 3

    def test_sorted_by_score_desc(self):
        feed = _feed()
        feed.ingest(_ioc(score=30, description="low"))
        feed.ingest(_ioc(score=90, description="high", timestamp=1001))
        results = feed.query()
        assert results[0].score > results[1].score

    def test_get_by_id(self):
        feed = _feed()
        ioc_id = feed.ingest(_ioc())
        assert feed.get(ioc_id) is not None
        assert feed.get("nonexistent") is None


# ── Correlation ────────────────────────────────────────────────────────

class TestCorrelation:
    def test_basic_correlation(self):
        feed = _feed(correlation_window_seconds=200)
        feed.ingest(_ioc(description="a", correlation_keys=["campaign-1"], timestamp=1000))
        feed.ingest(_ioc(description="b", correlation_keys=["campaign-1"], timestamp=1100, agent_id="agent-002"))
        groups = feed.correlate()
        assert len(groups) >= 1
        assert len(groups[0].ioc_ids) == 2

    def test_no_correlation_without_keys(self):
        feed = _feed()
        feed.ingest(_ioc(description="a"))
        feed.ingest(_ioc(description="b", timestamp=1001))
        groups = feed.correlate()
        assert len(groups) == 0

    def test_time_window_split(self):
        feed = _feed(correlation_window_seconds=50)
        feed.ingest(_ioc(description="a", correlation_keys=["k1"], timestamp=1000))
        feed.ingest(_ioc(description="b", correlation_keys=["k1"], timestamp=1200))
        groups = feed.correlate()
        assert len(groups) == 0  # too far apart

    def test_diversity_bonus(self):
        feed = _feed(correlation_window_seconds=500)
        feed.ingest(_ioc(description="a", source="src_a", ioc_type=IOCType.BEHAVIOR,
                         correlation_keys=["k1"], timestamp=1000))
        feed.ingest(_ioc(description="b", source="src_b", ioc_type=IOCType.RESOURCE,
                         correlation_keys=["k1"], timestamp=1100))
        groups = feed.correlate()
        assert len(groups) == 1
        # Diversity bonus: 1 extra source + 1 extra type = +10
        assert groups[0].composite_score > 50


# ── Aging ──────────────────────────────────────────────────────────────

class TestAging:
    def test_age_out(self):
        feed = _feed()
        feed.ingest(_ioc(timestamp=100, description="old"))
        feed.ingest(_ioc(timestamp=9000, description="new"))
        removed = feed.age_out(max_age_seconds=1000, now=9500)
        assert removed == 1
        assert feed.count == 1

    def test_decay_scores(self):
        feed = _feed(aging_half_life_seconds=100)
        feed.ingest(_ioc(score=80, timestamp=1000, description="a"))
        feed.decay_scores(now=1100)  # 1 half-life later
        ioc = feed.query()[0]
        assert 35 <= ioc.score <= 45  # ~40 after half-life


# ── Alerts ─────────────────────────────────────────────────────────────

class TestAlerts:
    def test_critical_spike_rule(self):
        feed = _feed()
        feed.add_alert_rule(ThreatIntelFeed.rule_critical_spike(2))
        for i in range(3):
            feed.ingest(_ioc(score=95, description=f"crit-{i}", timestamp=1000 + i * 100))
        alerts = feed.check_alerts(now=2000)
        assert len(alerts) == 1
        assert alerts[0]["rule"] == "critical_spike"

    def test_cooldown(self):
        feed = _feed()
        rule = ThreatIntelFeed.rule_critical_spike(1)
        feed.add_alert_rule(rule)
        feed.ingest(_ioc(score=95, description="c"))
        feed.check_alerts(now=1000)
        alerts = feed.check_alerts(now=1010)  # within cooldown
        assert len(alerts) == 0

    def test_multi_agent_correlation_rule(self):
        feed = _feed(correlation_window_seconds=500)
        feed.add_alert_rule(ThreatIntelFeed.rule_multi_agent_correlation(2))
        feed.ingest(_ioc(agent_id="a1", description="x", correlation_keys=["k"], timestamp=1000))
        feed.ingest(_ioc(agent_id="a2", description="y", correlation_keys=["k"], timestamp=1100))
        feed.correlate()
        alerts = feed.check_alerts(now=2000)
        assert len(alerts) == 1

    def test_volume_surge_rule(self):
        feed = _feed()
        now = time.time()
        feed.add_alert_rule(ThreatIntelFeed.rule_volume_surge(5, 600))
        for i in range(6):
            feed.ingest(_ioc(description=f"v-{i}", timestamp=now - 100 + i))
        alerts = feed.check_alerts(now=now)
        assert len(alerts) == 1


# ── Trends ─────────────────────────────────────────────────────────────

class TestTrends:
    def test_trend_buckets(self):
        feed = _feed(trend_window_seconds=100, trend_buckets=3)
        now = 1000.0
        feed.ingest(_ioc(timestamp=now - 50, description="a"))
        feed.ingest(_ioc(timestamp=now - 150, description="b"))
        trends = feed.compute_trends(now=now)
        assert len(trends) == 3
        assert trends[-1].count == 1  # most recent bucket

    def test_empty_buckets(self):
        feed = _feed(trend_window_seconds=100, trend_buckets=2)
        trends = feed.compute_trends(now=1000)
        assert all(t.count == 0 for t in trends)


# ── STIX Export ────────────────────────────────────────────────────────

class TestSTIX:
    def test_stix_bundle_structure(self):
        feed = _feed()
        feed.ingest(_ioc())
        bundle = feed.export_stix()
        assert bundle["type"] == "bundle"
        assert bundle["spec_version"] == "2.1"
        assert len(bundle["objects"]) == 1
        assert bundle["objects"][0]["type"] == "indicator"

    def test_stix_with_correlations(self):
        feed = _feed(correlation_window_seconds=500)
        feed.ingest(_ioc(description="a", correlation_keys=["k"], timestamp=1000))
        feed.ingest(_ioc(description="b", correlation_keys=["k"], timestamp=1100))
        feed.correlate()
        bundle = feed.export_stix()
        types = [o["type"] for o in bundle["objects"]]
        assert "relationship" in types


# ── Agent Risk Profile ─────────────────────────────────────────────────

class TestAgentRisk:
    def test_risk_profile(self):
        feed = _feed()
        feed.ingest(_ioc(agent_id="a1", score=80, description="a"))
        feed.ingest(_ioc(agent_id="a1", score=40, description="b", timestamp=1001))
        profile = feed.agent_risk_profile("a1")
        assert profile["agent_id"] == "a1"
        assert profile["ioc_count"] == 2
        assert profile["risk_score"] > 0
        assert profile["grade"] in "ABCDF"

    def test_unknown_agent(self):
        feed = _feed()
        profile = feed.agent_risk_profile("unknown")
        assert profile["risk_score"] == 0
        assert profile["grade"] == "A"


# ── Full Analysis ──────────────────────────────────────────────────────

class TestAnalysis:
    def test_analyze_report(self):
        feed = _feed()
        for i in range(5):
            feed.ingest(_ioc(description=f"a-{i}", timestamp=1000 + i * 100,
                             score=30 + i * 15))
        report = feed.analyze(now=2000)
        assert isinstance(report, FeedReport)
        assert report.total_iocs == 5
        assert report.overall_threat_level > 0
        assert report.threat_grade in "ABCDF"
        assert len(report.recommendations) > 0

    def test_render(self):
        feed = _feed()
        feed.ingest(_ioc(score=95, description="crit"))
        report = feed.analyze(now=2000)
        text = report.render()
        assert "THREAT INTELLIGENCE" in text
        assert "crit" in text

    def test_empty_feed_analysis(self):
        feed = _feed()
        report = feed.analyze(now=1000)
        assert report.total_iocs == 0
        assert report.overall_threat_level == 0.0


# ── Persistence ────────────────────────────────────────────────────────

class TestPersistence:
    def test_export_import(self):
        feed = _feed()
        feed.ingest(_ioc(description="persist-test"))
        state = feed.export_state()
        feed2 = _feed()
        count = feed2.import_state(state)
        assert count == 1
        assert feed2.count == 1

    def test_roundtrip_preserves_data(self):
        feed = _feed()
        feed.ingest(_ioc(description="rt", tags=["x"], score=77, source="test"))
        state = feed.export_state()
        feed2 = _feed()
        feed2.import_state(state)
        ioc = feed2.query()[0]
        assert ioc.description == "rt"
        assert ioc.tags == ["x"]


# ── Edge Cases ─────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_combined_filters(self):
        feed = _feed()
        feed.ingest(_ioc(ioc_type=IOCType.RESOURCE, agent_id="a1", score=80, description="x"))
        feed.ingest(_ioc(ioc_type=IOCType.RESOURCE, agent_id="a2", score=80, description="y", timestamp=1001))
        feed.ingest(_ioc(ioc_type=IOCType.BEHAVIOR, agent_id="a1", score=80, description="z", timestamp=1002))
        results = feed.query(ioc_type=IOCType.RESOURCE, agent_id="a1")
        assert len(results) == 1

    def test_all_ioc_types(self):
        feed = _feed()
        for i, t in enumerate(IOCType):
            feed.ingest(_ioc(ioc_type=t, description=f"type-{t.value}", timestamp=1000 + i * 100))
        assert feed.count == len(IOCType)

    def test_high_volume(self):
        feed = _feed(max_iocs=1000)
        for i in range(200):
            feed.ingest(_ioc(description=f"bulk-{i}", timestamp=1000 + i))
        assert feed.count == 200
        report = feed.analyze(now=2000)
        assert report.total_iocs == 200
