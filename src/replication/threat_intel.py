"""Agent Threat Intelligence Feed — aggregates detection signals into normalized IOCs.

Individual safety modules (kill chain, steganography, covert channels,
escalation, hoarding, etc.) each flag suspicious behaviors.  But security
teams need a single, queryable feed of *indicators of compromise* (IOCs)
with normalized severity, source attribution, and temporal context.

This module provides:

1. **IOC ingestion** — normalized indicator format from any detector.
2. **Severity scoring** — composite scoring with source-weight blending.
3. **IOC correlation** — link related indicators across detectors/agents.
4. **Feed management** — query, filter, age-out, deduplicate, export.
5. **STIX-lite export** — simplified STIX 2.1-inspired JSON bundles.
6. **Alert rules** — threshold-based alerting on feed patterns.
7. **Trend analysis** — IOC volume/severity trends over time windows.

IOC types
---------
- ``behavior`` — suspicious behavioral pattern (e.g. privilege escalation)
- ``communication`` — covert channel or steganographic signal
- ``resource`` — resource hoarding or anomalous consumption
- ``replication`` — self-modification or replication attempt
- ``evasion`` — defense evasion or deception signal
- ``exfiltration`` — data exfiltration indicator
- ``coordination`` — multi-agent coordination / emergent behavior

Usage (CLI)::

    python -m replication.threat_intel                     # demo feed
    python -m replication.threat_intel --agents 5          # 5 agents
    python -m replication.threat_intel --stix              # STIX export
    python -m replication.threat_intel --json              # JSON output

Programmatic::

    from replication.threat_intel import ThreatIntelFeed, IOC, IOCType
    feed = ThreatIntelFeed()
    feed.ingest(IOC(type=IOCType.BEHAVIOR, source="killchain", ...))
    report = feed.analyze()
    print(report.render())
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


# ── Enums ──────────────────────────────────────────────────────────────

class IOCType(Enum):
    """Categories of indicators of compromise."""
    BEHAVIOR = "behavior"
    COMMUNICATION = "communication"
    RESOURCE = "resource"
    REPLICATION = "replication"
    EVASION = "evasion"
    EXFILTRATION = "exfiltration"
    COORDINATION = "coordination"


class Severity(Enum):
    """Severity levels for IOCs."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def numeric(self) -> int:
        return {"info": 1, "low": 2, "medium": 3, "high": 4, "critical": 5}[self.value]

    @classmethod
    def from_score(cls, score: float) -> "Severity":
        if score >= 90:
            return cls.CRITICAL
        if score >= 70:
            return cls.HIGH
        if score >= 50:
            return cls.MEDIUM
        if score >= 30:
            return cls.LOW
        return cls.INFO


class AlertAction(Enum):
    """Actions for alert rules."""
    LOG = "log"
    NOTIFY = "notify"
    QUARANTINE = "quarantine"
    ESCALATE = "escalate"


# ── Data classes ───────────────────────────────────────────────────────

@dataclass
class IOC:
    """A single indicator of compromise."""
    ioc_type: IOCType
    source: str  # detector module name
    agent_id: str
    description: str
    severity: Severity = Severity.MEDIUM
    confidence: float = 0.5  # 0.0-1.0
    score: float = 50.0  # 0-100
    timestamp: float = 0.0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    correlation_keys: List[str] = field(default_factory=list)
    _id: str = ""

    def __post_init__(self):
        if self.timestamp <= 0:
            self.timestamp = time.time()
        if not self._id:
            raw = f"{self.ioc_type.value}:{self.source}:{self.agent_id}:{self.description}:{self.timestamp}"
            self._id = hashlib.sha256(raw.encode()).hexdigest()[:16]

    @property
    def id(self) -> str:
        return self._id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self._id,
            "type": self.ioc_type.value,
            "source": self.source,
            "agent_id": self.agent_id,
            "description": self.description,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "score": self.score,
            "timestamp": self.timestamp,
            "tags": self.tags,
            "metadata": self.metadata,
            "correlation_keys": self.correlation_keys,
        }


@dataclass
class CorrelationGroup:
    """A group of correlated IOCs."""
    group_id: str
    ioc_ids: List[str]
    agents: List[str]
    sources: List[str]
    types: List[str]
    composite_score: float
    composite_severity: Severity
    description: str
    created: float = 0.0

    def __post_init__(self):
        if self.created <= 0:
            self.created = time.time()


@dataclass
class AlertRule:
    """A rule that triggers when feed conditions are met."""
    name: str
    condition: Callable[["ThreatIntelFeed"], bool]
    action: AlertAction = AlertAction.LOG
    cooldown_seconds: float = 300.0
    description: str = ""
    _last_fired: float = 0.0

    def should_fire(self, feed: "ThreatIntelFeed", now: float = 0.0) -> bool:
        if now <= 0:
            now = time.time()
        if now - self._last_fired < self.cooldown_seconds:
            return False
        return self.condition(feed)

    def fire(self, now: float = 0.0) -> Dict[str, Any]:
        if now <= 0:
            now = time.time()
        self._last_fired = now
        return {
            "rule": self.name,
            "action": self.action.value,
            "description": self.description,
            "fired_at": now,
        }


@dataclass
class TrendPoint:
    """A single point in a trend time series."""
    window_start: float
    window_end: float
    count: int
    avg_score: float
    max_severity: Severity
    types: Dict[str, int]


@dataclass
class FeedReport:
    """Complete analysis report from the threat intel feed."""
    total_iocs: int
    by_type: Dict[str, int]
    by_severity: Dict[str, int]
    by_source: Dict[str, int]
    by_agent: Dict[str, int]
    top_iocs: List[Dict[str, Any]]
    correlation_groups: List[CorrelationGroup]
    active_alerts: List[Dict[str, Any]]
    trends: List[TrendPoint]
    overall_threat_level: float  # 0-100
    threat_grade: str
    recommendations: List[str]

    def render(self) -> str:
        lines = [
            "=" * 60,
            "  AGENT THREAT INTELLIGENCE FEED REPORT",
            "=" * 60,
            "",
            f"Total IOCs: {self.total_iocs}",
            f"Overall Threat Level: {self.overall_threat_level:.1f}/100 (Grade: {self.threat_grade})",
            "",
            "── By Type ──",
        ]
        for t, c in sorted(self.by_type.items(), key=lambda x: -x[1]):
            lines.append(f"  {t:20s} {c:4d}")
        lines.append("")
        lines.append("── By Severity ──")
        for s in ["critical", "high", "medium", "low", "info"]:
            if s in self.by_severity:
                lines.append(f"  {s:20s} {self.by_severity[s]:4d}")
        lines.append("")
        lines.append("── By Source ──")
        for src, c in sorted(self.by_source.items(), key=lambda x: -x[1]):
            lines.append(f"  {src:20s} {c:4d}")
        lines.append("")
        lines.append("── By Agent ──")
        for a, c in sorted(self.by_agent.items(), key=lambda x: -x[1]):
            lines.append(f"  {a:20s} {c:4d}")

        if self.top_iocs:
            lines.append("")
            lines.append("── Top IOCs (by score) ──")
            for ioc in self.top_iocs[:10]:
                lines.append(f"  [{ioc['severity'].upper():8s}] {ioc['score']:5.1f} "
                             f"| {ioc['agent_id']} | {ioc['source']} | {ioc['description'][:50]}")

        if self.correlation_groups:
            lines.append("")
            lines.append(f"── Correlation Groups ({len(self.correlation_groups)}) ──")
            for g in self.correlation_groups[:5]:
                lines.append(f"  Group {g.group_id}: {len(g.ioc_ids)} IOCs, "
                             f"score={g.composite_score:.1f}, "
                             f"agents={g.agents}, sources={g.sources}")
                lines.append(f"    {g.description}")

        if self.active_alerts:
            lines.append("")
            lines.append(f"── Active Alerts ({len(self.active_alerts)}) ──")
            for a in self.active_alerts:
                lines.append(f"  [{a['action'].upper():10s}] {a['rule']}: {a['description']}")

        if self.recommendations:
            lines.append("")
            lines.append("── Recommendations ──")
            for r in self.recommendations:
                lines.append(f"  • {r}")

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)


# ── Source weights ─────────────────────────────────────────────────────

DEFAULT_SOURCE_WEIGHTS: Dict[str, float] = {
    "killchain": 1.5,
    "steganography": 1.3,
    "covert_channels": 1.3,
    "escalation": 1.4,
    "hoarding": 1.0,
    "selfmod": 1.5,
    "emergent": 1.2,
    "drift": 1.1,
    "deception": 1.4,
    "fingerprint": 0.9,
    "game_theory": 1.1,
    "consensus": 0.8,
    "goal_inference": 1.2,
    "influence": 1.0,
}


# ── ThreatIntelFeed ────────────────────────────────────────────────────

@dataclass
class FeedConfig:
    """Configuration for the threat intel feed."""
    max_iocs: int = 10000
    dedup_window_seconds: float = 60.0
    correlation_window_seconds: float = 300.0
    aging_half_life_seconds: float = 3600.0
    source_weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SOURCE_WEIGHTS))
    trend_window_seconds: float = 600.0
    trend_buckets: int = 6


class ThreatIntelFeed:
    """Aggregates IOCs from detection modules into a queryable threat feed."""

    def __init__(self, config: Optional[FeedConfig] = None):
        self.config = config or FeedConfig()
        self._iocs: List[IOC] = []
        self._ioc_index: Dict[str, IOC] = {}
        self._correlation_groups: List[CorrelationGroup] = []
        self._alert_rules: List[AlertRule] = []
        self._alert_history: List[Dict[str, Any]] = []
        self._dedup_hashes: Dict[str, float] = {}

    # ── Ingestion ──────────────────────────────────────────────────

    def _dedup_key(self, ioc: IOC) -> str:
        return f"{ioc.ioc_type.value}:{ioc.source}:{ioc.agent_id}:{ioc.description}"

    def ingest(self, ioc: IOC) -> Optional[str]:
        """Ingest an IOC. Returns IOC id if accepted, None if deduplicated."""
        key = self._dedup_key(ioc)
        now = ioc.timestamp
        if key in self._dedup_hashes:
            if now - self._dedup_hashes[key] < self.config.dedup_window_seconds:
                return None
        self._dedup_hashes[key] = now

        # Apply source weight
        weight = self.config.source_weights.get(ioc.source, 1.0)
        ioc.score = min(100.0, ioc.score * weight)
        ioc.severity = Severity.from_score(ioc.score)

        self._iocs.append(ioc)
        self._ioc_index[ioc.id] = ioc

        # Evict oldest if over capacity
        if len(self._iocs) > self.config.max_iocs:
            evicted = self._iocs.pop(0)
            self._ioc_index.pop(evicted.id, None)

        return ioc.id

    def ingest_batch(self, iocs: List[IOC]) -> List[Optional[str]]:
        """Ingest multiple IOCs."""
        return [self.ingest(ioc) for ioc in iocs]

    # ── Query ──────────────────────────────────────────────────────

    def query(
        self,
        ioc_type: Optional[IOCType] = None,
        severity: Optional[Severity] = None,
        min_severity: Optional[Severity] = None,
        source: Optional[str] = None,
        agent_id: Optional[str] = None,
        tag: Optional[str] = None,
        min_score: Optional[float] = None,
        since: Optional[float] = None,
        limit: int = 100,
    ) -> List[IOC]:
        """Query IOCs with filters."""
        results = self._iocs
        if ioc_type is not None:
            results = [i for i in results if i.ioc_type == ioc_type]
        if severity is not None:
            results = [i for i in results if i.severity == severity]
        if min_severity is not None:
            results = [i for i in results if i.severity.numeric >= min_severity.numeric]
        if source is not None:
            results = [i for i in results if i.source == source]
        if agent_id is not None:
            results = [i for i in results if i.agent_id == agent_id]
        if tag is not None:
            results = [i for i in results if tag in i.tags]
        if min_score is not None:
            results = [i for i in results if i.score >= min_score]
        if since is not None:
            results = [i for i in results if i.timestamp >= since]
        # Sort by score desc
        results.sort(key=lambda i: (-i.score, -i.timestamp))
        return results[:limit]

    def get(self, ioc_id: str) -> Optional[IOC]:
        """Get IOC by ID."""
        return self._ioc_index.get(ioc_id)

    @property
    def count(self) -> int:
        return len(self._iocs)

    # ── Correlation ────────────────────────────────────────────────

    def correlate(self) -> List[CorrelationGroup]:
        """Find correlated IOCs based on shared correlation keys and time proximity."""
        key_to_iocs: Dict[str, List[IOC]] = {}
        for ioc in self._iocs:
            for key in ioc.correlation_keys:
                key_to_iocs.setdefault(key, []).append(ioc)

        groups: List[CorrelationGroup] = []
        seen_sets: set = set()

        for key, iocs in key_to_iocs.items():
            if len(iocs) < 2:
                continue
            # Group by time proximity
            iocs_sorted = sorted(iocs, key=lambda i: i.timestamp)
            window: List[IOC] = [iocs_sorted[0]]
            for ioc in iocs_sorted[1:]:
                if ioc.timestamp - window[0].timestamp <= self.config.correlation_window_seconds:
                    window.append(ioc)
                else:
                    if len(window) >= 2:
                        self._emit_group(window, key, groups, seen_sets)
                    window = [ioc]
            if len(window) >= 2:
                self._emit_group(window, key, groups, seen_sets)

        groups.sort(key=lambda g: -g.composite_score)
        self._correlation_groups = groups
        return groups

    def _emit_group(
        self, iocs: List[IOC], key: str,
        groups: List[CorrelationGroup], seen: set
    ):
        ids = tuple(sorted(i.id for i in iocs))
        if ids in seen:
            return
        seen.add(ids)
        agents = sorted(set(i.agent_id for i in iocs))
        sources = sorted(set(i.source for i in iocs))
        types = sorted(set(i.ioc_type.value for i in iocs))
        avg_score = sum(i.score for i in iocs) / len(iocs)
        max_sev = max(iocs, key=lambda i: i.severity.numeric).severity
        # Boost score for multi-source/multi-type correlations
        diversity_bonus = (len(sources) - 1) * 5 + (len(types) - 1) * 5
        composite = min(100.0, avg_score + diversity_bonus)
        gid = hashlib.sha256(str(ids).encode()).hexdigest()[:12]
        desc = (f"Correlated {len(iocs)} IOCs via key '{key}': "
                f"{len(agents)} agent(s), {len(sources)} source(s)")
        groups.append(CorrelationGroup(
            group_id=gid, ioc_ids=list(ids), agents=agents,
            sources=sources, types=types, composite_score=composite,
            composite_severity=max_sev, description=desc,
        ))

    # ── Aging ──────────────────────────────────────────────────────

    def age_out(self, max_age_seconds: Optional[float] = None, now: float = 0.0) -> int:
        """Remove IOCs older than max_age. Returns count removed."""
        if now <= 0:
            now = time.time()
        cutoff = max_age_seconds or (self.config.aging_half_life_seconds * 6)
        before = len(self._iocs)
        self._iocs = [i for i in self._iocs if now - i.timestamp <= cutoff]
        self._ioc_index = {i.id: i for i in self._iocs}
        return before - len(self._iocs)

    def decay_scores(self, now: float = 0.0) -> None:
        """Apply exponential decay to IOC scores based on age."""
        if now <= 0:
            now = time.time()
        hl = self.config.aging_half_life_seconds
        if hl <= 0:
            return
        for ioc in self._iocs:
            age = now - ioc.timestamp
            if age > 0:
                factor = math.pow(0.5, age / hl)
                ioc.score = ioc.score * factor
                ioc.severity = Severity.from_score(ioc.score)

    # ── Alerts ─────────────────────────────────────────────────────

    def add_alert_rule(self, rule: AlertRule) -> None:
        self._alert_rules.append(rule)

    def check_alerts(self, now: float = 0.0) -> List[Dict[str, Any]]:
        """Evaluate all alert rules. Returns list of fired alerts."""
        fired = []
        for rule in self._alert_rules:
            if rule.should_fire(self, now):
                alert = rule.fire(now)
                fired.append(alert)
                self._alert_history.append(alert)
        return fired

    # ── Built-in alert rules ───────────────────────────────────────

    @staticmethod
    def rule_critical_spike(threshold: int = 3) -> AlertRule:
        """Alert when critical IOCs exceed threshold."""
        def condition(feed: ThreatIntelFeed) -> bool:
            count = sum(1 for i in feed._iocs if i.severity == Severity.CRITICAL)
            return count >= threshold
        return AlertRule(
            name="critical_spike",
            condition=condition,
            action=AlertAction.ESCALATE,
            description=f"Critical IOC count exceeds {threshold}",
        )

    @staticmethod
    def rule_multi_agent_correlation(threshold: int = 2) -> AlertRule:
        """Alert when correlation groups span multiple agents."""
        def condition(feed: ThreatIntelFeed) -> bool:
            return any(len(g.agents) >= threshold for g in feed._correlation_groups)
        return AlertRule(
            name="multi_agent_correlation",
            condition=condition,
            action=AlertAction.NOTIFY,
            description=f"Correlated IOCs spanning {threshold}+ agents detected",
        )

    @staticmethod
    def rule_volume_surge(threshold: int = 50, window_seconds: float = 300.0) -> AlertRule:
        """Alert on IOC volume surge in a time window."""
        def condition(feed: ThreatIntelFeed) -> bool:
            now = time.time()
            recent = sum(1 for i in feed._iocs if now - i.timestamp <= window_seconds)
            return recent >= threshold
        return AlertRule(
            name="volume_surge",
            condition=condition,
            action=AlertAction.NOTIFY,
            description=f"IOC volume exceeds {threshold} in {window_seconds}s window",
        )

    # ── Trends ─────────────────────────────────────────────────────

    def compute_trends(self, now: float = 0.0) -> List[TrendPoint]:
        """Compute IOC trends over time windows."""
        if now <= 0:
            now = time.time()
        window = self.config.trend_window_seconds
        buckets = self.config.trend_buckets
        trends = []
        for i in range(buckets):
            end = now - i * window
            start = end - window
            bucket_iocs = [ioc for ioc in self._iocs
                           if start <= ioc.timestamp < end]
            if not bucket_iocs:
                trends.append(TrendPoint(
                    window_start=start, window_end=end, count=0,
                    avg_score=0, max_severity=Severity.INFO, types={}
                ))
                continue
            avg_score = sum(i.score for i in bucket_iocs) / len(bucket_iocs)
            max_sev = max(bucket_iocs, key=lambda x: x.severity.numeric).severity
            types: Dict[str, int] = {}
            for ioc in bucket_iocs:
                types[ioc.ioc_type.value] = types.get(ioc.ioc_type.value, 0) + 1
            trends.append(TrendPoint(
                window_start=start, window_end=end, count=len(bucket_iocs),
                avg_score=avg_score, max_severity=max_sev, types=types,
            ))
        trends.reverse()
        return trends

    # ── STIX-lite export ───────────────────────────────────────────

    def export_stix(self) -> Dict[str, Any]:
        """Export feed as a STIX 2.1-inspired bundle."""
        objects = []
        for ioc in self._iocs:
            objects.append({
                "type": "indicator",
                "spec_version": "2.1",
                "id": f"indicator--{ioc.id}",
                "created": ioc.timestamp,
                "name": ioc.description,
                "description": f"Source: {ioc.source}, Agent: {ioc.agent_id}",
                "indicator_types": [ioc.ioc_type.value],
                "pattern_type": "agent-behavior",
                "pattern": f"[agent:behavior = '{ioc.ioc_type.value}']",
                "valid_from": ioc.timestamp,
                "confidence": int(ioc.confidence * 100),
                "labels": ioc.tags,
                "x_score": ioc.score,
                "x_severity": ioc.severity.value,
                "x_agent_id": ioc.agent_id,
                "x_source": ioc.source,
            })
        for group in self._correlation_groups:
            rel_objects = []
            for i, src_id in enumerate(group.ioc_ids):
                for tgt_id in group.ioc_ids[i + 1:]:
                    rel_objects.append({
                        "type": "relationship",
                        "spec_version": "2.1",
                        "id": f"relationship--{group.group_id}-{src_id[:6]}-{tgt_id[:6]}",
                        "relationship_type": "related-to",
                        "source_ref": f"indicator--{src_id}",
                        "target_ref": f"indicator--{tgt_id}",
                    })
            objects.extend(rel_objects)
        return {
            "type": "bundle",
            "id": f"bundle--threat-intel-{hashlib.sha256(str(time.time()).encode()).hexdigest()[:12]}",
            "spec_version": "2.1",
            "objects": objects,
        }

    # ── Agent risk profile ─────────────────────────────────────────

    def agent_risk_profile(self, agent_id: str) -> Dict[str, Any]:
        """Get risk profile for a specific agent."""
        agent_iocs = [i for i in self._iocs if i.agent_id == agent_id]
        if not agent_iocs:
            return {"agent_id": agent_id, "risk_score": 0, "grade": "A", "ioc_count": 0}
        avg_score = sum(i.score for i in agent_iocs) / len(agent_iocs)
        max_score = max(i.score for i in agent_iocs)
        # Weighted: 60% max, 40% avg — worst behavior matters most
        risk = 0.6 * max_score + 0.4 * avg_score
        risk = min(100.0, risk * (1 + 0.05 * min(len(agent_iocs), 20)))
        risk = min(100.0, risk)
        types = {}
        sources = {}
        for ioc in agent_iocs:
            types[ioc.ioc_type.value] = types.get(ioc.ioc_type.value, 0) + 1
            sources[ioc.source] = sources.get(ioc.source, 0) + 1
        grade = _score_to_grade(100 - risk)
        return {
            "agent_id": agent_id,
            "risk_score": round(risk, 1),
            "grade": grade,
            "ioc_count": len(agent_iocs),
            "avg_score": round(avg_score, 1),
            "max_score": round(max_score, 1),
            "types": types,
            "sources": sources,
            "top_iocs": [i.to_dict() for i in sorted(agent_iocs, key=lambda x: -x.score)[:5]],
        }

    # ── Full analysis ──────────────────────────────────────────────

    def analyze(self, now: float = 0.0) -> FeedReport:
        """Run full analysis and return report."""
        if now <= 0:
            now = time.time()

        by_type: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}
        by_source: Dict[str, int] = {}
        by_agent: Dict[str, int] = {}

        for ioc in self._iocs:
            by_type[ioc.ioc_type.value] = by_type.get(ioc.ioc_type.value, 0) + 1
            by_severity[ioc.severity.value] = by_severity.get(ioc.severity.value, 0) + 1
            by_source[ioc.source] = by_source.get(ioc.source, 0) + 1
            by_agent[ioc.agent_id] = by_agent.get(ioc.agent_id, 0) + 1

        self.correlate()
        alerts = self.check_alerts(now)
        trends = self.compute_trends(now)

        top_iocs = [i.to_dict() for i in sorted(self._iocs, key=lambda x: -x.score)[:10]]

        # Overall threat level
        if self._iocs:
            scores = [i.score for i in self._iocs]
            avg = sum(scores) / len(scores)
            top_avg = sum(sorted(scores, reverse=True)[:5]) / min(5, len(scores))
            threat = 0.4 * avg + 0.6 * top_avg
            # Volume factor
            threat = min(100.0, threat * (1 + 0.01 * min(len(self._iocs), 50)))
        else:
            threat = 0.0

        recs = self._generate_recommendations(by_type, by_severity, by_agent)

        return FeedReport(
            total_iocs=len(self._iocs),
            by_type=by_type,
            by_severity=by_severity,
            by_source=by_source,
            by_agent=by_agent,
            top_iocs=top_iocs,
            correlation_groups=self._correlation_groups,
            active_alerts=alerts,
            trends=trends,
            overall_threat_level=round(threat, 1),
            threat_grade=_score_to_grade(100 - threat),
            recommendations=recs,
        )

    def _generate_recommendations(
        self, by_type: Dict[str, int],
        by_severity: Dict[str, int],
        by_agent: Dict[str, int],
    ) -> List[str]:
        recs = []
        crit = by_severity.get("critical", 0)
        high = by_severity.get("high", 0)
        if crit > 0:
            recs.append(f"URGENT: {crit} critical IOC(s) require immediate investigation")
        if high > 3:
            recs.append(f"High severity IOC count ({high}) elevated — review detection sources")
        if by_type.get("replication", 0) > 0:
            recs.append("Replication indicators detected — verify containment measures")
        if by_type.get("exfiltration", 0) > 0:
            recs.append("Exfiltration signals present — audit data access controls")
        if by_type.get("coordination", 0) > 0:
            recs.append("Multi-agent coordination detected — check for emergent behaviors")
        if len(by_agent) > 3:
            recs.append(f"IOCs span {len(by_agent)} agents — consider fleet-wide security review")
        if self._correlation_groups:
            multi = [g for g in self._correlation_groups if len(g.agents) > 1]
            if multi:
                recs.append(f"{len(multi)} cross-agent correlation group(s) — investigate coordinated activity")
        if not recs:
            recs.append("No elevated threats detected — continue standard monitoring")
        return recs

    # ── Persistence ────────────────────────────────────────────────

    def export_state(self) -> Dict[str, Any]:
        """Export feed state for persistence."""
        return {
            "iocs": [i.to_dict() for i in self._iocs],
            "alert_history": self._alert_history,
        }

    def import_state(self, state: Dict[str, Any]) -> int:
        """Import feed state. Returns count of IOCs loaded."""
        count = 0
        for d in state.get("iocs", []):
            ioc = IOC(
                ioc_type=IOCType(d["type"]),
                source=d["source"],
                agent_id=d["agent_id"],
                description=d["description"],
                severity=Severity(d["severity"]),
                confidence=d.get("confidence", 0.5),
                score=d.get("score", 50),
                timestamp=d.get("timestamp", 0),
                tags=d.get("tags", []),
                metadata=d.get("metadata", {}),
                correlation_keys=d.get("correlation_keys", []),
                _id=d.get("id", ""),
            )
            self._iocs.append(ioc)
            self._ioc_index[ioc.id] = ioc
            count += 1
        self._alert_history = state.get("alert_history", [])
        return count


# ── Helpers ────────────────────────────────────────────────────────────

def _score_to_grade(score: float) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


# ── Demo / CLI ─────────────────────────────────────────────────────────

def _build_demo_feed(n_agents: int = 3) -> ThreatIntelFeed:
    """Build a demo feed with simulated IOCs."""
    import random
    random.seed(42)
    feed = ThreatIntelFeed()

    feed.add_alert_rule(ThreatIntelFeed.rule_critical_spike(3))
    feed.add_alert_rule(ThreatIntelFeed.rule_multi_agent_correlation(2))

    sources_and_types = [
        ("killchain", IOCType.BEHAVIOR, "Kill chain stage progression detected"),
        ("steganography", IOCType.COMMUNICATION, "Hidden message in agent output"),
        ("covert_channels", IOCType.COMMUNICATION, "Covert timing channel detected"),
        ("escalation", IOCType.BEHAVIOR, "Privilege escalation attempt"),
        ("hoarding", IOCType.RESOURCE, "Anomalous resource accumulation"),
        ("selfmod", IOCType.REPLICATION, "Self-modification attempt detected"),
        ("emergent", IOCType.COORDINATION, "Emergent synchronization pattern"),
        ("drift", IOCType.EVASION, "Alignment drift detected"),
        ("deception", IOCType.EVASION, "Deceptive response pattern"),
        ("fingerprint", IOCType.BEHAVIOR, "Behavioral anomaly vs baseline"),
    ]

    base_time = time.time() - 3600
    for i in range(40):
        src, ioc_type, desc = random.choice(sources_and_types)
        agent = f"agent-{random.randint(0, n_agents - 1):03d}"
        ioc = IOC(
            ioc_type=ioc_type,
            source=src,
            agent_id=agent,
            description=f"{desc} (event #{i})",
            confidence=round(random.uniform(0.3, 1.0), 2),
            score=round(random.uniform(20, 95), 1),
            timestamp=base_time + i * 90 + random.uniform(0, 30),
            tags=[random.choice(["automated", "manual", "high-priority", "routine"])],
            correlation_keys=[f"campaign-{random.randint(0, 3)}"],
        )
        feed.ingest(ioc)

    return feed


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Agent Threat Intelligence Feed")
    parser.add_argument("--agents", type=int, default=3, help="Number of agents")
    parser.add_argument("--stix", action="store_true", help="Export STIX bundle")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    feed = _build_demo_feed(args.agents)
    report = feed.analyze()

    if args.stix:
        print(json.dumps(feed.export_stix(), indent=2, default=str))
    elif args.json:
        print(json.dumps({
            "total_iocs": report.total_iocs,
            "threat_level": report.overall_threat_level,
            "grade": report.threat_grade,
            "by_type": report.by_type,
            "by_severity": report.by_severity,
            "by_source": report.by_source,
            "by_agent": report.by_agent,
            "correlation_groups": len(report.correlation_groups),
            "recommendations": report.recommendations,
        }, indent=2))
    else:
        print(report.render())


if __name__ == "__main__":
    main()
