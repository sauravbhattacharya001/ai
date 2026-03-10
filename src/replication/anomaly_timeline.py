"""Anomaly Timeline — temporal correlation of safety events across detectors.

Reconstructs a chronological timeline of anomalous events from multiple
safety subsystems (drift, canary, boundary, deception, honeypot, evasion)
and identifies temporal patterns such as coordinated attack sequences,
escalation cascades, and recurring anomaly clusters.

Usage (CLI)::

    python -m replication.anomaly_timeline                     # demo timeline
    python -m replication.anomaly_timeline --windows 15        # 15 drift windows
    python -m replication.anomaly_timeline --cluster-gap 2.0   # 2s cluster gap
    python -m replication.anomaly_timeline --json              # JSON output
    python -m replication.anomaly_timeline --summary           # summary only

Programmatic::

    from replication.anomaly_timeline import AnomalyTimeline, TimelineConfig
    tl = AnomalyTimeline()
    result = tl.analyze()
    print(result.render())
    for cluster in result.clusters:
        print(f"Cluster at {cluster.start:.1f}s: {len(cluster.events)} events")
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .simulator import ScenarioConfig, SimulationReport, Simulator, Strategy
from ._helpers import stats_mean, stats_std


# ── Enums ──


class EventSource(Enum):
    """Which detector produced the event."""

    DRIFT = "drift"
    CANARY = "canary"
    BOUNDARY = "boundary"
    DECEPTION = "deception"
    HONEYPOT = "honeypot"
    EVASION = "evasion"
    MANUAL = "manual"


class EventSeverity(Enum):
    """Event severity level."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def weight(self) -> int:
        return {"info": 1, "low": 2, "medium": 3, "high": 4, "critical": 5}[
            self.value
        ]


class PatternType(Enum):
    """Recognised temporal patterns."""

    CLUSTER = "cluster"
    ESCALATION = "escalation"
    RECURRING = "recurring"
    CASCADE = "cascade"
    ISOLATED = "isolated"


# ── Data classes ──


@dataclass
class AnomalyEvent:
    """Single anomaly event."""

    timestamp: float
    source: EventSource
    severity: EventSeverity
    metric: str
    value: float
    baseline: float
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def deviation(self) -> float:
        if self.baseline == 0:
            return abs(self.value)
        return abs(self.value - self.baseline) / abs(self.baseline)


@dataclass
class EventCluster:
    """Group of temporally co-located events."""

    events: List[AnomalyEvent]
    pattern: PatternType = PatternType.CLUSTER

    @property
    def start(self) -> float:
        return min(e.timestamp for e in self.events)

    @property
    def end(self) -> float:
        return max(e.timestamp for e in self.events)

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def max_severity(self) -> EventSeverity:
        return max(self.events, key=lambda e: e.severity.weight).severity

    @property
    def sources(self) -> List[EventSource]:
        return sorted(set(e.source for e in self.events), key=lambda s: s.value)

    @property
    def is_multi_source(self) -> bool:
        return len(self.sources) > 1

    @property
    def threat_score(self) -> float:
        """Composite threat score: severity * breadth * density."""
        severity_sum = sum(e.severity.weight for e in self.events)
        breadth = len(self.sources)
        density = len(self.events) / max(self.duration, 0.01)
        return round(severity_sum * breadth * min(density, 10.0), 2)


@dataclass
class EscalationChain:
    """Sequence of events showing increasing severity over time."""

    events: List[AnomalyEvent]

    @property
    def start_severity(self) -> EventSeverity:
        return self.events[0].severity

    @property
    def end_severity(self) -> EventSeverity:
        return self.events[-1].severity

    @property
    def duration(self) -> float:
        return self.events[-1].timestamp - self.events[0].timestamp

    @property
    def escalation_rate(self) -> float:
        """Severity levels per second."""
        dt = self.duration
        if dt == 0:
            return float("inf")
        delta = self.end_severity.weight - self.start_severity.weight
        return round(delta / dt, 4)


@dataclass
class RecurrencePattern:
    """Detected recurring anomaly pattern."""

    metric: str
    source: EventSource
    occurrences: int
    mean_interval: float
    std_interval: float
    timestamps: List[float]

    @property
    def is_periodic(self) -> bool:
        if self.mean_interval == 0:
            return False
        cv = self.std_interval / self.mean_interval if self.mean_interval else float("inf")
        return cv < 0.3


@dataclass
class TimelineConfig:
    """Configuration for timeline analysis."""

    cluster_gap: float = 1.0
    min_cluster_size: int = 2
    escalation_min_steps: int = 3
    recurrence_min_count: int = 3
    windows: int = 10
    strategy: str = "greedy"
    max_depth: int = 5


@dataclass
class TimelineResult:
    """Full timeline analysis result."""

    events: List[AnomalyEvent]
    clusters: List[EventCluster]
    escalations: List[EscalationChain]
    recurrences: List[RecurrencePattern]
    duration: float
    config: TimelineConfig

    @property
    def total_events(self) -> int:
        return len(self.events)

    @property
    def multi_source_clusters(self) -> List[EventCluster]:
        return [c for c in self.clusters if c.is_multi_source]

    @property
    def max_threat_score(self) -> float:
        if not self.clusters:
            return 0.0
        return max(c.threat_score for c in self.clusters)

    @property
    def severity_distribution(self) -> Dict[str, int]:
        dist: Dict[str, int] = {}
        for e in self.events:
            dist[e.severity.value] = dist.get(e.severity.value, 0) + 1
        return dist

    @property
    def source_distribution(self) -> Dict[str, int]:
        dist: Dict[str, int] = {}
        for e in self.events:
            dist[e.source.value] = dist.get(e.source.value, 0) + 1
        return dist

    def events_in_window(self, start: float, end: float) -> List[AnomalyEvent]:
        return [e for e in self.events if start <= e.timestamp <= end]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_events": self.total_events,
            "clusters": len(self.clusters),
            "multi_source_clusters": len(self.multi_source_clusters),
            "escalations": len(self.escalations),
            "recurrences": len(self.recurrences),
            "max_threat_score": self.max_threat_score,
            "severity_distribution": self.severity_distribution,
            "source_distribution": self.source_distribution,
            "duration_seconds": round(self.duration, 3),
            "events": [
                {
                    "timestamp": e.timestamp,
                    "source": e.source.value,
                    "severity": e.severity.value,
                    "metric": e.metric,
                    "value": e.value,
                    "baseline": e.baseline,
                    "description": e.description,
                }
                for e in self.events
            ],
            "cluster_details": [
                {
                    "start": c.start,
                    "end": c.end,
                    "duration": round(c.duration, 3),
                    "event_count": len(c.events),
                    "sources": [s.value for s in c.sources],
                    "max_severity": c.max_severity.value,
                    "threat_score": c.threat_score,
                    "pattern": c.pattern.value,
                }
                for c in self.clusters
            ],
            "escalation_details": [
                {
                    "start_severity": esc.start_severity.value,
                    "end_severity": esc.end_severity.value,
                    "steps": len(esc.events),
                    "duration": round(esc.duration, 3),
                    "rate": esc.escalation_rate,
                }
                for esc in self.escalations
            ],
            "recurrence_details": [
                {
                    "metric": r.metric,
                    "source": r.source.value,
                    "occurrences": r.occurrences,
                    "mean_interval": round(r.mean_interval, 3),
                    "periodic": r.is_periodic,
                }
                for r in self.recurrences
            ],
        }

    def render(self) -> str:
        lines = [
            "+" + "=" * 58 + "+",
            "|              ANOMALY TIMELINE ANALYSIS                  |",
            "+" + "=" * 58 + "+",
            "",
        ]

        lines.append(f"Total events:          {self.total_events}")
        lines.append(f"Event clusters:        {len(self.clusters)}")
        lines.append(f"Multi-source clusters: {len(self.multi_source_clusters)}")
        lines.append(f"Escalation chains:     {len(self.escalations)}")
        lines.append(f"Recurrence patterns:   {len(self.recurrences)}")
        lines.append(f"Max threat score:      {self.max_threat_score}")
        lines.append(f"Analysis time:         {self.duration:.3f}s")
        lines.append("")

        lines.append("-- Severity Distribution --")
        for sev in ["critical", "high", "medium", "low", "info"]:
            count = self.severity_distribution.get(sev, 0)
            if count:
                bar = "#" * min(count, 40)
                lines.append(f"  {sev:>8}: {bar} ({count})")
        lines.append("")

        lines.append("-- Source Distribution --")
        for src, count in sorted(
            self.source_distribution.items(), key=lambda x: -x[1]
        ):
            bar = "#" * min(count, 40)
            lines.append(f"  {src:>10}: {bar} ({count})")
        lines.append("")

        lines.append("-- Event Timeline --")
        for i, e in enumerate(self.events[:30]):
            sev_icon = {
                "critical": "[!!]",
                "high": "[! ]",
                "medium": "[* ]",
                "low": "[- ]",
                "info": "[  ]",
            }.get(e.severity.value, "[  ]")
            lines.append(
                f"  {e.timestamp:8.2f}s {sev_icon} [{e.source.value:>9}] "
                f"{e.metric}: {e.value:.3f} (baseline {e.baseline:.3f}) "
                f"-- {e.description}"
            )
        if len(self.events) > 30:
            lines.append(f"  ... and {len(self.events) - 30} more events")
        lines.append("")

        if self.clusters:
            lines.append("-- Event Clusters --")
            for i, c in enumerate(self.clusters[:10]):
                multi = " !! MULTI-SOURCE" if c.is_multi_source else ""
                lines.append(
                    f"  Cluster {i + 1}: {c.start:.2f}s-{c.end:.2f}s "
                    f"({len(c.events)} events, threat={c.threat_score}){multi}"
                )
                lines.append(
                    f"    Sources: {', '.join(s.value for s in c.sources)} "
                    f"| Max severity: {c.max_severity.value} "
                    f"| Pattern: {c.pattern.value}"
                )
            if len(self.clusters) > 10:
                lines.append(f"  ... and {len(self.clusters) - 10} more clusters")
            lines.append("")

        if self.escalations:
            lines.append("-- Escalation Chains --")
            for i, esc in enumerate(self.escalations[:5]):
                lines.append(
                    f"  Chain {i + 1}: {esc.start_severity.value} -> "
                    f"{esc.end_severity.value} ({len(esc.events)} steps, "
                    f"{esc.duration:.2f}s, rate={esc.escalation_rate}/s)"
                )
            lines.append("")

        if self.recurrences:
            lines.append("-- Recurrence Patterns --")
            for r in self.recurrences[:5]:
                periodic = "PERIODIC" if r.is_periodic else "irregular"
                lines.append(
                    f"  {r.metric} ({r.source.value}): {r.occurrences}x "
                    f"mean interval {r.mean_interval:.2f}s [{periodic}]"
                )
            lines.append("")

        return "\n".join(lines)


# ── Core analyzer ──


class AnomalyTimeline:
    """Collects anomaly events from detectors and performs temporal analysis."""

    def __init__(self, config: Optional[TimelineConfig] = None) -> None:
        self.config = config or TimelineConfig()
        self._events: List[AnomalyEvent] = []

    def add_event(self, event: AnomalyEvent) -> None:
        """Add a manually created event."""
        self._events.append(event)

    def add_events(self, events: Sequence[AnomalyEvent]) -> None:
        """Add multiple events."""
        self._events.extend(events)

    def _collect_from_simulation(self, report: SimulationReport) -> List[AnomalyEvent]:
        """Extract anomaly events from a simulation report."""
        events: List[AnomalyEvent] = []
        ts = time.monotonic()

        metrics = report.metrics if hasattr(report, "metrics") else {}
        if isinstance(metrics, dict):
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    baselines = {
                        "escape_rate": 0.0,
                        "replication_factor": 1.0,
                        "resource_usage": 0.5,
                        "boundary_violations": 0.0,
                        "deception_score": 0.0,
                    }
                    baseline = baselines.get(metric_name, 0.0)
                    if baseline == 0 and value > 0.05:
                        severity = (
                            EventSeverity.CRITICAL
                            if value > 0.5
                            else EventSeverity.HIGH
                            if value > 0.2
                            else EventSeverity.MEDIUM
                            if value > 0.1
                            else EventSeverity.LOW
                        )
                        events.append(
                            AnomalyEvent(
                                timestamp=time.monotonic() - ts,
                                source=EventSource.DRIFT,
                                severity=severity,
                                metric=metric_name,
                                value=value,
                                baseline=baseline,
                                description=f"{metric_name} elevated: {value:.3f}",
                            )
                        )
                    elif baseline > 0 and abs(value - baseline) / baseline > 0.5:
                        severity = (
                            EventSeverity.HIGH
                            if abs(value - baseline) / baseline > 2.0
                            else EventSeverity.MEDIUM
                        )
                        events.append(
                            AnomalyEvent(
                                timestamp=time.monotonic() - ts,
                                source=EventSource.DRIFT,
                                severity=severity,
                                metric=metric_name,
                                value=value,
                                baseline=baseline,
                                description=f"{metric_name} drifted from baseline {baseline:.3f}",
                            )
                        )
        return events

    def _collect_from_drift(self) -> List[AnomalyEvent]:
        """Run drift detection and extract events."""
        from .drift import DriftDetector, DriftConfig

        events: List[AnomalyEvent] = []
        try:
            detector = DriftDetector(
                DriftConfig(
                    windows=self.config.windows,
                    strategy=self.config.strategy,
                    max_depth=self.config.max_depth,
                )
            )
            result = detector.analyze()
            for alert in result.alerts:
                sev = {
                    "none": EventSeverity.INFO,
                    "low": EventSeverity.LOW,
                    "medium": EventSeverity.MEDIUM,
                    "high": EventSeverity.HIGH,
                    "critical": EventSeverity.CRITICAL,
                }.get(alert.severity.value, EventSeverity.MEDIUM)
                events.append(
                    AnomalyEvent(
                        timestamp=time.monotonic(),
                        source=EventSource.DRIFT,
                        severity=sev,
                        metric=alert.metric,
                        value=alert.current,
                        baseline=alert.baseline,
                        description=alert.recommendation
                        if hasattr(alert, "recommendation")
                        else f"Drift on {alert.metric}",
                    )
                )
        except Exception:
            pass
        return events

    def _cluster_events(
        self, events: List[AnomalyEvent]
    ) -> List[EventCluster]:
        """Group events into temporal clusters using gap-based clustering."""
        if not events:
            return []

        sorted_events = sorted(events, key=lambda e: e.timestamp)
        clusters: List[EventCluster] = []
        current: List[AnomalyEvent] = [sorted_events[0]]

        for event in sorted_events[1:]:
            if event.timestamp - current[-1].timestamp <= self.config.cluster_gap:
                current.append(event)
            else:
                if len(current) >= self.config.min_cluster_size:
                    clusters.append(EventCluster(events=list(current)))
                current = [event]

        if len(current) >= self.config.min_cluster_size:
            clusters.append(EventCluster(events=list(current)))

        for cluster in clusters:
            cluster.pattern = self._classify_cluster(cluster)

        return clusters

    def _classify_cluster(self, cluster: EventCluster) -> PatternType:
        """Determine the pattern type for a cluster."""
        severities = [e.severity.weight for e in cluster.events]

        increasing = all(
            severities[i] <= severities[i + 1] for i in range(len(severities) - 1)
        )
        if increasing and severities[-1] > severities[0] and len(severities) >= 3:
            return PatternType.ESCALATION

        if cluster.is_multi_source and cluster.duration < self.config.cluster_gap * 2:
            return PatternType.CASCADE

        if cluster.is_multi_source:
            return PatternType.CLUSTER

        return PatternType.ISOLATED

    def _detect_escalations(
        self, events: List[AnomalyEvent]
    ) -> List[EscalationChain]:
        """Find sequences of increasing severity."""
        if len(events) < self.config.escalation_min_steps:
            return []

        sorted_events = sorted(events, key=lambda e: e.timestamp)
        chains: List[EscalationChain] = []
        current: List[AnomalyEvent] = [sorted_events[0]]

        for event in sorted_events[1:]:
            if event.severity.weight >= current[-1].severity.weight:
                current.append(event)
            else:
                if (
                    len(current) >= self.config.escalation_min_steps
                    and current[-1].severity.weight > current[0].severity.weight
                ):
                    chains.append(EscalationChain(events=list(current)))
                current = [event]

        if (
            len(current) >= self.config.escalation_min_steps
            and current[-1].severity.weight > current[0].severity.weight
        ):
            chains.append(EscalationChain(events=list(current)))

        return chains

    def _detect_recurrences(
        self, events: List[AnomalyEvent]
    ) -> List[RecurrencePattern]:
        """Find recurring anomalies on the same metric+source."""
        groups: Dict[Tuple[str, str], List[AnomalyEvent]] = {}
        for e in events:
            key = (e.metric, e.source.value)
            groups.setdefault(key, []).append(e)

        patterns: List[RecurrencePattern] = []
        for (metric, source), group in groups.items():
            if len(group) < self.config.recurrence_min_count:
                continue
            timestamps = sorted(e.timestamp for e in group)
            intervals = [
                timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)
            ]
            mean_int = stats_mean(intervals) if intervals else 0.0
            std_int = stats_std(intervals) if intervals else 0.0
            patterns.append(
                RecurrencePattern(
                    metric=metric,
                    source=EventSource(source),
                    occurrences=len(group),
                    mean_interval=mean_int,
                    std_interval=std_int,
                    timestamps=timestamps,
                )
            )

        return patterns

    def analyze(
        self, *, collect_drift: bool = True
    ) -> TimelineResult:
        """Run full timeline analysis."""
        t0 = time.monotonic()

        all_events = list(self._events)

        if collect_drift:
            all_events.extend(self._collect_from_drift())

        all_events.sort(key=lambda e: e.timestamp)

        if all_events:
            t_start = all_events[0].timestamp
            for e in all_events:
                e.timestamp -= t_start

        clusters = self._cluster_events(all_events)
        escalations = self._detect_escalations(all_events)
        recurrences = self._detect_recurrences(all_events)

        return TimelineResult(
            events=all_events,
            clusters=clusters,
            escalations=escalations,
            recurrences=recurrences,
            duration=time.monotonic() - t0,
            config=self.config,
        )

    def analyze_report(self, report: SimulationReport) -> TimelineResult:
        """Analyze events extracted from a simulation report."""
        self._events.extend(self._collect_from_simulation(report))
        return self.analyze(collect_drift=False)


# ── CLI ──


def _cli(argv: Optional[List[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Anomaly Timeline -- temporal correlation of safety events"
    )
    parser.add_argument(
        "--windows", type=int, default=10, help="Drift detection windows"
    )
    parser.add_argument(
        "--strategy", default="greedy", help="Simulation strategy"
    )
    parser.add_argument(
        "--cluster-gap",
        type=float,
        default=1.0,
        help="Max gap (seconds) between events in a cluster",
    )
    parser.add_argument(
        "--json", action="store_true", help="Output as JSON"
    )
    parser.add_argument(
        "--summary", action="store_true", help="Summary only"
    )

    args = parser.parse_args(argv)

    config = TimelineConfig(
        windows=args.windows,
        strategy=args.strategy,
        cluster_gap=args.cluster_gap,
    )

    tl = AnomalyTimeline(config)
    result = tl.analyze()

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    elif args.summary:
        print(f"Events: {result.total_events}")
        print(f"Clusters: {len(result.clusters)}")
        print(f"Multi-source: {len(result.multi_source_clusters)}")
        print(f"Escalations: {len(result.escalations)}")
        print(f"Recurrences: {len(result.recurrences)}")
        print(f"Max threat: {result.max_threat_score}")
    else:
        print(result.render())


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point."""
    _cli(argv)


if __name__ == "__main__":
    _cli()
