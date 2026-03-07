"""Timeline Reconstructor — unified chronological event timeline.

Runs simulation, drift detection, compliance auditing, and threat
analysis, then merges all events into a single chronological timeline
for post-incident forensic review.  Events are tagged by source,
severity, and category, and can be filtered/searched interactively.

Usage (CLI)::

    python -m replication timeline                        # full timeline
    python -m replication timeline --severity high        # filter by severity
    python -m replication timeline --source drift         # only drift events
    python -m replication timeline --category safety      # by category
    python -m replication timeline --search "escape"      # text search
    python -m replication timeline --last 20              # last N events
    python -m replication timeline --json                 # JSON output
    python -m replication timeline --export timeline.json # save to file

Programmatic::

    from replication.timeline import TimelineReconstructor, TimelineConfig
    reconstructor = TimelineReconstructor()
    timeline = reconstructor.build()
    print(timeline.render())
    for event in timeline.events:
        print(f"[{event.timestamp}] {event.source}: {event.summary}")
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence


# ── Enums ────────────────────────────────────────────────────────


class EventSource(Enum):
    """Where the event originated."""

    SIMULATION = "simulation"
    DRIFT = "drift"
    COMPLIANCE = "compliance"
    THREAT = "threat"
    FORENSICS = "forensics"
    QUARANTINE = "quarantine"
    INCIDENT = "incident"
    ESCALATION = "escalation"
    KILLCHAIN = "killchain"
    SELFMOD = "selfmod"
    TOPOLOGY = "topology"
    SYSTEM = "system"


class EventSeverity(Enum):
    """Severity of the event."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EventCategory(Enum):
    """Broad category of the event."""

    SAFETY = "safety"
    PERFORMANCE = "performance"
    COMPLIANCE = "compliance"
    SECURITY = "security"
    OPERATIONAL = "operational"


# ── Data classes ─────────────────────────────────────────────────


@dataclass
class TimelineEvent:
    """A single event in the timeline."""

    timestamp: float  # Unix epoch (relative or absolute)
    source: EventSource
    severity: EventSeverity
    category: EventCategory
    summary: str
    details: Dict[str, Any] = field(default_factory=dict)
    related_events: List[int] = field(default_factory=list)  # indices

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "source": self.source.value,
            "severity": self.severity.value,
            "category": self.category.value,
            "summary": self.summary,
            "details": self.details,
            "related_events": self.related_events,
        }


@dataclass
class TimelineConfig:
    """Configuration for timeline reconstruction."""

    strategy: str = "balanced"
    max_depth: int = 4
    drift_windows: int = 5
    include_sources: Optional[List[EventSource]] = None
    severity_filter: Optional[EventSeverity] = None
    category_filter: Optional[EventCategory] = None
    search_query: Optional[str] = None
    last_n: Optional[int] = None


@dataclass
class TimelineSpan:
    """A span grouping related events (e.g. an escalation sequence)."""

    label: str
    start_idx: int
    end_idx: int
    severity: EventSeverity
    source: EventSource


@dataclass
class TimelineStats:
    """Aggregate statistics for the timeline."""

    total_events: int
    by_source: Dict[str, int]
    by_severity: Dict[str, int]
    by_category: Dict[str, int]
    duration_seconds: float
    critical_count: int
    high_count: int
    spans: List[TimelineSpan]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_events": self.total_events,
            "by_source": self.by_source,
            "by_severity": self.by_severity,
            "by_category": self.by_category,
            "duration_seconds": round(self.duration_seconds, 3),
            "critical_count": self.critical_count,
            "high_count": self.high_count,
            "spans": [
                {"label": s.label, "start": s.start_idx, "end": s.end_idx,
                 "severity": s.severity.value, "source": s.source.value}
                for s in self.spans
            ],
        }


@dataclass
class Timeline:
    """Complete reconstructed timeline."""

    events: List[TimelineEvent]
    stats: TimelineStats
    config: TimelineConfig
    built_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "built_at": self.built_at,
            "stats": self.stats.to_dict(),
            "events": [e.to_dict() for e in self.events],
        }

    def render(self) -> str:
        """Render a human-readable timeline report."""
        lines: List[str] = []
        lines.append("")
        lines.append("┌─────────────────────────────────────────┐")
        lines.append("│       TIMELINE RECONSTRUCTION           │")
        lines.append("└─────────────────────────────────────────┘")
        lines.append("")
        lines.append(f"  Built: {self.built_at}")
        lines.append(f"  Events: {self.stats.total_events}")
        lines.append(f"  Duration: {self.stats.duration_seconds:.1f}s")
        lines.append(f"  Critical: {self.stats.critical_count}  "
                      f"High: {self.stats.high_count}")
        lines.append("")

        # Source breakdown
        lines.append("  ── Sources ──")
        for src, count in sorted(self.stats.by_source.items(),
                                  key=lambda x: -x[1]):
            bar = "█" * min(count, 30)
            lines.append(f"    {src:<14} {count:>3}  {bar}")
        lines.append("")

        # Severity breakdown
        lines.append("  ── Severity ──")
        sev_icons = {
            "info": "ℹ️ ", "low": "🔵", "medium": "🟡",
            "high": "🟠", "critical": "🔴",
        }
        for sev in ["critical", "high", "medium", "low", "info"]:
            count = self.stats.by_severity.get(sev, 0)
            if count > 0:
                icon = sev_icons.get(sev, "  ")
                lines.append(f"    {icon} {sev:<10} {count:>3}")
        lines.append("")

        # Spans
        if self.stats.spans:
            lines.append("  ── Event Spans ──")
            for span in self.stats.spans:
                lines.append(
                    f"    [{span.severity.value:>8}] {span.label} "
                    f"(events {span.start_idx}–{span.end_idx})"
                )
            lines.append("")

        # Events
        lines.append("  ── Chronological Events ──")
        lines.append("")
        for i, evt in enumerate(self.events):
            sev_char = {
                EventSeverity.INFO: "·",
                EventSeverity.LOW: "○",
                EventSeverity.MEDIUM: "◑",
                EventSeverity.HIGH: "●",
                EventSeverity.CRITICAL: "◉",
            }.get(evt.severity, "?")

            ts = f"{evt.timestamp:>8.2f}s"
            src = evt.source.value[:12].ljust(12)
            lines.append(f"  {sev_char} {ts}  {src}  {evt.summary}")

            # Show key details inline for high/critical
            if evt.severity in (EventSeverity.HIGH, EventSeverity.CRITICAL):
                for key, val in list(evt.details.items())[:3]:
                    lines.append(f"               └─ {key}: {val}")

        lines.append("")
        lines.append(f"  Total: {self.stats.total_events} events across "
                      f"{len(self.stats.by_source)} sources")
        lines.append("")
        return "\n".join(lines)


# ── Reconstructor ────────────────────────────────────────────────


class TimelineReconstructor:
    """Builds a unified timeline from multiple analysis modules."""

    def build(self, config: Optional[TimelineConfig] = None) -> Timeline:
        """Run analyses and merge into a chronological timeline."""
        config = config or TimelineConfig()
        events: List[TimelineEvent] = []
        t0 = time.monotonic()

        # 1. Simulation events
        events.extend(self._collect_simulation(config))

        # 2. Drift events
        events.extend(self._collect_drift(config))

        # 3. Compliance events
        events.extend(self._collect_compliance(config))

        # 4. Threat events
        events.extend(self._collect_threats(config))

        # 5. Forensic events
        events.extend(self._collect_forensics(config))

        # Sort chronologically
        events.sort(key=lambda e: e.timestamp)

        # Apply filters
        events = self._apply_filters(events, config)

        # Detect spans (groups of related events)
        spans = self._detect_spans(events)

        # Build stats
        elapsed = time.monotonic() - t0
        stats = self._compute_stats(events, spans, elapsed)

        return Timeline(
            events=events,
            stats=stats,
            config=config,
            built_at=datetime.now(timezone.utc).isoformat(),
        )

    # ── Collectors ───────────────────────────────────────────────

    def _collect_simulation(self, config: TimelineConfig) -> List[TimelineEvent]:
        """Extract events from a simulation run."""
        from .simulator import Simulator, PRESETS

        events: List[TimelineEvent] = []
        preset = PRESETS.get(config.strategy, PRESETS["balanced"])
        sc = replace(preset, max_depth=config.max_depth)
        sim = Simulator(sc)
        report = sim.run()

        # Simulation start
        events.append(TimelineEvent(
            timestamp=0.0,
            source=EventSource.SIMULATION,
            severity=EventSeverity.INFO,
            category=EventCategory.OPERATIONAL,
            summary=f"Simulation started: strategy={config.strategy}, "
                    f"depth={config.max_depth}",
            details={"strategy": config.strategy, "max_depth": config.max_depth},
        ))

        # Map timeline entries from the simulation report
        type_severity = {
            "spawn": EventSeverity.INFO,
            "task": EventSeverity.INFO,
            "replicate_ok": EventSeverity.LOW,
            "replicate_denied": EventSeverity.MEDIUM,
            "shutdown": EventSeverity.INFO,
            "expired": EventSeverity.MEDIUM,
            "cooldown": EventSeverity.LOW,
        }

        for i, entry in enumerate(report.timeline):
            etype = entry.get("type", "unknown")
            sev = type_severity.get(etype, EventSeverity.INFO)

            if etype == "replicate_denied":
                sev = EventSeverity.MEDIUM
            elif etype == "expired":
                sev = EventSeverity.HIGH

            events.append(TimelineEvent(
                timestamp=entry.get("time_ms", i * 0.1) / 1000.0,
                source=EventSource.SIMULATION,
                severity=sev,
                category=EventCategory.SAFETY,
                summary=entry.get("detail", etype),
                details={
                    "type": etype,
                    "worker_id": entry.get("worker_id", ""),
                    "time_ms": entry.get("time_ms", 0),
                },
            ))

        # Simulation end summary
        total_workers = len(report.workers)
        denied = report.total_replications_denied
        succeeded = report.total_replications_succeeded

        end_sev = EventSeverity.INFO
        if denied == 0 and succeeded > 5:
            end_sev = EventSeverity.HIGH  # unrestricted replication

        events.append(TimelineEvent(
            timestamp=max((e.get("time_ms", 0) for e in report.timeline),
                          default=0) / 1000.0 + 0.001,
            source=EventSource.SIMULATION,
            severity=end_sev,
            category=EventCategory.OPERATIONAL,
            summary=f"Simulation complete: {total_workers} workers, "
                    f"{succeeded} replications ({denied} denied)",
            details={
                "total_workers": total_workers,
                "replications_succeeded": succeeded,
                "replications_denied": denied,
                "total_tasks": report.total_tasks,
            },
        ))

        return events

    def _collect_drift(self, config: TimelineConfig) -> List[TimelineEvent]:
        """Extract events from drift detection."""
        from .drift import DriftDetector, DriftConfig

        events: List[TimelineEvent] = []
        dc = DriftConfig(windows=config.drift_windows)
        detector = DriftDetector()
        result = detector.analyze(dc)

        base_t = 2.0  # offset after simulation events

        for i, alert in enumerate(result.alerts):
            sev_map = {
                "none": EventSeverity.INFO,
                "low": EventSeverity.LOW,
                "medium": EventSeverity.MEDIUM,
                "high": EventSeverity.HIGH,
                "critical": EventSeverity.CRITICAL,
            }
            sev = sev_map.get(alert.severity.value, EventSeverity.MEDIUM)

            events.append(TimelineEvent(
                timestamp=base_t + i * 0.3,
                source=EventSource.DRIFT,
                severity=sev,
                category=EventCategory.SAFETY,
                summary=f"Drift detected: {alert.metric} "
                        f"({alert.direction.value}), z={alert.z_score:.1f}",
                details={
                    "metric": alert.metric,
                    "current": alert.current,
                    "baseline": alert.baseline,
                    "z_score": round(alert.z_score, 2),
                    "direction": alert.direction.value,
                },
            ))

        # Overall drift summary
        events.append(TimelineEvent(
            timestamp=base_t + len(result.alerts) * 0.3 + 0.1,
            source=EventSource.DRIFT,
            severity=EventSeverity.INFO,
            category=EventCategory.OPERATIONAL,
            summary=f"Drift analysis complete: {len(result.alerts)} alerts "
                    f"across {config.drift_windows} windows",
            details={"alert_count": len(result.alerts),
                     "windows": config.drift_windows},
        ))

        return events

    def _collect_compliance(self, config: TimelineConfig) -> List[TimelineEvent]:
        """Extract events from compliance audit."""
        from .compliance import ComplianceAuditor, AuditConfig
        from .contract import ReplicationContract

        events: List[TimelineEvent] = []
        contract = ReplicationContract(
            max_depth=config.max_depth,
            max_replicas=10,
            cooldown_seconds=0.5,
        )
        auditor = ComplianceAuditor()

        try:
            result = auditor.audit(contract)
        except Exception:
            # If audit fails, emit a single system event
            events.append(TimelineEvent(
                timestamp=5.0,
                source=EventSource.COMPLIANCE,
                severity=EventSeverity.LOW,
                category=EventCategory.COMPLIANCE,
                summary="Compliance audit skipped (configuration mismatch)",
            ))
            return events

        base_t = 5.0

        for i, fw_result in enumerate(result.framework_results):
            for j, finding in enumerate(fw_result.findings):
                if finding.verdict.value == "pass":
                    sev = EventSeverity.INFO
                elif finding.verdict.value == "warn":
                    sev = EventSeverity.MEDIUM
                else:
                    sev = EventSeverity.HIGH

                events.append(TimelineEvent(
                    timestamp=base_t + i * 0.5 + j * 0.05,
                    source=EventSource.COMPLIANCE,
                    severity=sev,
                    category=EventCategory.COMPLIANCE,
                    summary=f"[{fw_result.framework.value}] "
                            f"{finding.check_id}: {finding.verdict.value}",
                    details={
                        "framework": fw_result.framework.value,
                        "check_id": finding.check_id,
                        "title": finding.title,
                        "verdict": finding.verdict.value,
                        "rationale": finding.rationale,
                    },
                ))

        return events

    def _collect_threats(self, config: TimelineConfig) -> List[TimelineEvent]:
        """Extract events from threat simulation."""
        from .threats import ThreatSimulator

        events: List[TimelineEvent] = []
        ts = ThreatSimulator()
        report = ts.run_all()

        base_t = 8.0

        for i, result in enumerate(report.results):
            sev_map = {
                "low": EventSeverity.LOW,
                "medium": EventSeverity.MEDIUM,
                "high": EventSeverity.HIGH,
                "critical": EventSeverity.CRITICAL,
            }
            sev = sev_map.get(result.severity.value, EventSeverity.MEDIUM)

            events.append(TimelineEvent(
                timestamp=base_t + i * 0.2,
                source=EventSource.THREAT,
                severity=sev,
                category=EventCategory.SECURITY,
                summary=f"Threat: {result.name} — {result.status.value}",
                details={
                    "scenario_id": result.scenario_id,
                    "threat": result.name,
                    "severity": result.severity.value,
                    "status": result.status.value,
                    "attacks_attempted": result.attacks_attempted,
                    "attacks_blocked": result.attacks_blocked,
                    "description": result.description,
                },
            ))

        return events

    def _collect_forensics(self, config: TimelineConfig) -> List[TimelineEvent]:
        """Extract events from forensic analysis."""
        from .forensics import ForensicAnalyzer
        from .simulator import Simulator, PRESETS

        events: List[TimelineEvent] = []
        preset = PRESETS.get(config.strategy, PRESETS["balanced"])
        sc = replace(preset, max_depth=config.max_depth)
        sim = Simulator(sc)
        report = sim.run()

        analyzer = ForensicAnalyzer()
        forensics = analyzer.analyze(report)

        base_t = 11.0

        # Near misses
        for i, nm in enumerate(forensics.near_misses):
            events.append(TimelineEvent(
                timestamp=base_t + i * 0.2,
                source=EventSource.FORENSICS,
                severity=EventSeverity.HIGH,
                category=EventCategory.SAFETY,
                summary=f"Near miss: {nm.description}",
                details={
                    "metric": nm.metric,
                    "current_value": nm.current_value,
                    "limit_value": nm.limit_value,
                    "headroom_pct": round(nm.headroom_pct, 2),
                    "worker_id": nm.worker_id,
                },
            ))

        # Escalation phases
        for i, phase in enumerate(forensics.escalation_phases):
            events.append(TimelineEvent(
                timestamp=base_t + 3.0 + i * 0.3,
                source=EventSource.FORENSICS,
                severity=EventSeverity.MEDIUM,
                category=EventCategory.SAFETY,
                summary=f"Escalation phase: {phase.label} "
                        f"(steps {phase.start_step}–{phase.end_step})",
                details={
                    "label": phase.label,
                    "start_step": phase.start_step,
                    "end_step": phase.end_step,
                },
            ))

        # Decision points
        for i, dp in enumerate(forensics.decision_points):
            events.append(TimelineEvent(
                timestamp=base_t + 5.0 + i * 0.15,
                source=EventSource.FORENSICS,
                severity=EventSeverity.LOW,
                category=EventCategory.SAFETY,
                summary=f"Decision point: {dp.decision}",
                details={"step": dp.step, "worker_id": dp.worker_id,
                         "depth": dp.depth, "reason": dp.reason},
            ))

        return events

    # ── Filtering ────────────────────────────────────────────────

    def _apply_filters(self, events: List[TimelineEvent],
                       config: TimelineConfig) -> List[TimelineEvent]:
        """Apply severity, source, category, and search filters."""
        filtered = events

        if config.include_sources:
            filtered = [e for e in filtered if e.source in config.include_sources]

        if config.severity_filter:
            sev_order = list(EventSeverity)
            min_idx = sev_order.index(config.severity_filter)
            allowed = set(sev_order[min_idx:])
            filtered = [e for e in filtered if e.severity in allowed]

        if config.category_filter:
            filtered = [e for e in filtered if e.category == config.category_filter]

        if config.search_query:
            q = config.search_query.lower()
            filtered = [
                e for e in filtered
                if q in e.summary.lower() or
                any(q in str(v).lower() for v in e.details.values())
            ]

        if config.last_n and config.last_n > 0:
            filtered = filtered[-config.last_n:]

        return filtered

    # ── Span detection ───────────────────────────────────────────

    def _detect_spans(self, events: List[TimelineEvent]) -> List[TimelineSpan]:
        """Detect contiguous groups of related events (spans)."""
        spans: List[TimelineSpan] = []
        if not events:
            return spans

        # Group consecutive events from the same source
        current_source = events[0].source
        span_start = 0
        max_sev = events[0].severity

        sev_rank = {s: i for i, s in enumerate(EventSeverity)}

        for i in range(1, len(events)):
            if events[i].source != current_source:
                if i - span_start >= 3:  # only spans with 3+ events
                    spans.append(TimelineSpan(
                        label=f"{current_source.value} sequence",
                        start_idx=span_start,
                        end_idx=i - 1,
                        severity=max_sev,
                        source=current_source,
                    ))
                current_source = events[i].source
                span_start = i
                max_sev = events[i].severity
            else:
                if sev_rank.get(events[i].severity, 0) > sev_rank.get(max_sev, 0):
                    max_sev = events[i].severity

        # Final span
        if len(events) - span_start >= 3:
            spans.append(TimelineSpan(
                label=f"{current_source.value} sequence",
                start_idx=span_start,
                end_idx=len(events) - 1,
                severity=max_sev,
                source=current_source,
            ))

        return spans

    # ── Stats ────────────────────────────────────────────────────

    def _compute_stats(self, events: List[TimelineEvent],
                       spans: List[TimelineSpan],
                       elapsed: float) -> TimelineStats:
        """Compute aggregate statistics."""
        by_source: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}
        by_category: Dict[str, int] = {}
        critical = 0
        high = 0

        for e in events:
            by_source[e.source.value] = by_source.get(e.source.value, 0) + 1
            by_severity[e.severity.value] = by_severity.get(e.severity.value, 0) + 1
            by_category[e.category.value] = by_category.get(e.category.value, 0) + 1
            if e.severity == EventSeverity.CRITICAL:
                critical += 1
            elif e.severity == EventSeverity.HIGH:
                high += 1

        return TimelineStats(
            total_events=len(events),
            by_source=by_source,
            by_severity=by_severity,
            by_category=by_category,
            duration_seconds=elapsed,
            critical_count=critical,
            high_count=high,
            spans=spans,
        )


# ── CLI ──────────────────────────────────────────────────────────


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point for timeline reconstruction."""
    parser = argparse.ArgumentParser(
        prog="python -m replication.timeline",
        description="Reconstruct a unified chronological event timeline",
    )
    parser.add_argument(
        "--strategy", default="balanced",
        help="Simulation strategy preset (default: balanced)",
    )
    parser.add_argument(
        "--max-depth", type=int, default=4,
        help="Max simulation depth (default: 4)",
    )
    parser.add_argument(
        "--drift-windows", type=int, default=5,
        help="Number of drift detection windows (default: 5)",
    )
    parser.add_argument(
        "--severity", choices=["info", "low", "medium", "high", "critical"],
        help="Minimum severity filter",
    )
    parser.add_argument(
        "--source",
        choices=[s.value for s in EventSource],
        help="Filter to a single event source",
    )
    parser.add_argument(
        "--category",
        choices=[c.value for c in EventCategory],
        help="Filter to a single event category",
    )
    parser.add_argument(
        "--search", help="Text search across event summaries and details",
    )
    parser.add_argument(
        "--last", type=int, help="Show only the last N events",
    )
    parser.add_argument(
        "--json", action="store_true", dest="json_output",
        help="Output as JSON",
    )
    parser.add_argument(
        "--export", metavar="FILE", help="Export timeline to a JSON file",
    )
    args = parser.parse_args(argv)

    config = TimelineConfig(
        strategy=args.strategy,
        max_depth=args.max_depth,
        drift_windows=args.drift_windows,
        severity_filter=(
            EventSeverity(args.severity) if args.severity else None
        ),
        include_sources=(
            [EventSource(args.source)] if args.source else None
        ),
        category_filter=(
            EventCategory(args.category) if args.category else None
        ),
        search_query=args.search,
        last_n=args.last,
    )

    reconstructor = TimelineReconstructor()
    timeline = reconstructor.build(config)

    if args.json_output:
        print(json.dumps(timeline.to_dict(), indent=2))
    else:
        print(timeline.render())

    if args.export:
        with open(args.export, "w", encoding="utf-8") as f:
            json.dump(timeline.to_dict(), f, indent=2)
        print(f"\n  📁 Exported to {args.export}")


if __name__ == "__main__":
    main()
