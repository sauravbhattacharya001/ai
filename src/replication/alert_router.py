"""Safety Alert Router — configurable notification routing for audit events.

Evaluates audit trail events against user-defined routing rules and
dispatches alerts to configured channels (console, file, webhook stub)
with rate limiting, severity escalation, and quiet-hours suppression.

Features
--------
* Rule-based routing: match on category, severity, source, keyword
* 4 channel types: console (coloured), file (append), JSON-lines, webhook (stub)
* Per-rule rate limiting (max N alerts per window)
* Severity escalation: auto-upgrade severity after repeated triggers
* Quiet hours: suppress non-critical alerts during configurable windows
* Dry-run mode: show what *would* fire without dispatching
* CLI with route, test, stats subcommands
* Programmatic API for embedding in pipelines

Usage (CLI)::

    python -m replication alert-router route --category violation --severity critical \\
        --message "Token budget exceeded" --source controller
    python -m replication alert-router test --rules rules.json \\
        --event '{"category":"violation","severity":"critical","message":"test"}'
    python -m replication alert-router stats

Programmatic::

    from replication.alert_router import AlertRouter, RoutingRule, Channel
    router = AlertRouter()
    router.add_rule(RoutingRule(
        name="critical-violations",
        match_category={"violation", "escalation"},
        match_severity={"critical"},
        channels=[Channel(kind="console"), Channel(kind="file", path="alerts.log")],
    ))
    results = router.route(event)
    print(f"Dispatched to {len(results)} channels")
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from ._helpers import box_header as _box_header

_VALID_SEVERITIES = frozenset({"critical", "warning", "info"})
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def _sanitize_log_field(value: str, max_length: int = 2048) -> str:
    """Strip control characters and truncate to prevent log injection (CWE-117)."""
    sanitized = _CONTROL_CHAR_RE.sub("", value)
    return sanitized[:max_length]


# ── Data models ──────────────────────────────────────────────────────


@dataclass
class Channel:
    """A notification destination."""

    kind: str = "console"  # console | file | jsonl | webhook
    path: str = ""  # file/jsonl path
    url: str = ""  # webhook URL (stub — logs intent)
    label: str = ""  # human-friendly name

    def __post_init__(self) -> None:
        if not self.label:
            self.label = self.kind
        if self.kind not in ("console", "file", "jsonl", "webhook"):
            raise ValueError(f"Unknown channel kind: {self.kind}")
        # Prevent directory traversal in file/jsonl channel paths.
        if self.kind in ("file", "jsonl") and self.path:
            import os
            normalised = os.path.normpath(self.path)
            if ".." in normalised.split(os.sep):
                raise ValueError(
                    f"Channel path must not contain directory traversal "
                    f"sequences: {self.path!r}"
                )


@dataclass
class RoutingRule:
    """A rule that matches events and routes to channels."""

    name: str = "default"
    match_category: Set[str] = field(default_factory=set)
    match_severity: Set[str] = field(default_factory=set)
    match_source: Set[str] = field(default_factory=set)
    match_keywords: List[str] = field(default_factory=list)
    channels: List[Channel] = field(default_factory=list)
    # Rate limiting
    rate_limit: int = 0  # max dispatches per window (0 = unlimited)
    rate_window: int = 60  # window in seconds
    # Escalation: after this many triggers in window, bump severity
    escalate_after: int = 0  # 0 = disabled
    escalate_to: str = "critical"
    enabled: bool = True

    def matches(self, event: Dict[str, Any]) -> bool:
        """Check whether an event matches this rule's filters."""
        if self.match_category and event.get("category") not in self.match_category:
            return False
        if self.match_severity and event.get("severity") not in self.match_severity:
            return False
        if self.match_source and event.get("source") not in self.match_source:
            return False
        if self.match_keywords:
            msg = (event.get("message") or "").lower()
            if not any(kw.lower() in msg for kw in self.match_keywords):
                return False
        return True


@dataclass
class DispatchResult:
    """Result of dispatching one alert to one channel."""

    rule: str
    channel_label: str
    channel_kind: str
    severity: str
    message: str
    suppressed: bool = False
    suppression_reason: str = ""
    escalated: bool = False
    timestamp: str = ""


@dataclass
class RouteStats:
    """Aggregated routing statistics."""

    total_events: int = 0
    total_dispatched: int = 0
    total_suppressed: int = 0
    total_escalated: int = 0
    by_rule: Dict[str, int] = field(default_factory=dict)
    by_channel: Dict[str, int] = field(default_factory=dict)
    by_severity: Dict[str, int] = field(default_factory=dict)


# ── Quiet Hours ──────────────────────────────────────────────────────


@dataclass
class QuietHours:
    """Time window during which non-critical alerts are suppressed."""

    start_hour: int = 22  # 24h format
    end_hour: int = 7
    suppress_below: str = "critical"  # suppress severities below this

    def is_active(self, now: Optional[datetime] = None) -> bool:
        now = now or datetime.now()
        h = now.hour
        if self.start_hour > self.end_hour:
            return h >= self.start_hour or h < self.end_hour
        return self.start_hour <= h < self.end_hour

    def should_suppress(self, severity: str, now: Optional[datetime] = None) -> bool:
        if not self.is_active(now):
            return False
        severity_rank = {"info": 0, "warning": 1, "critical": 2}
        threshold = severity_rank.get(self.suppress_below, 2)
        return severity_rank.get(severity, 0) < threshold


# ── Router ───────────────────────────────────────────────────────────


class AlertRouter:
    """Routes safety events to configured channels based on rules."""

    def __init__(
        self,
        rules: Optional[List[RoutingRule]] = None,
        quiet_hours: Optional[QuietHours] = None,
        dry_run: bool = False,
    ) -> None:
        self.rules: List[RoutingRule] = rules or []
        self.quiet_hours = quiet_hours
        self.dry_run = dry_run
        self.stats = RouteStats()
        # Rate limiting state: rule_name -> list of timestamps
        self._rate_buckets: Dict[str, List[float]] = {}
        # Escalation state: rule_name -> trigger count in window
        self._escalation_counts: Dict[str, List[float]] = {}

    def add_rule(self, rule: RoutingRule) -> None:
        self.rules.append(rule)

    def _check_rate_limit(self, rule: RoutingRule) -> bool:
        """Return True if rate limit is exceeded (should suppress)."""
        if rule.rate_limit <= 0:
            return False
        now = time.time()
        bucket = self._rate_buckets.setdefault(rule.name, [])
        cutoff = now - rule.rate_window
        bucket[:] = [t for t in bucket if t > cutoff]
        if len(bucket) >= rule.rate_limit:
            return True
        bucket.append(now)
        return False

    def _check_escalation(self, rule: RoutingRule) -> bool:
        """Return True if escalation threshold is reached."""
        if rule.escalate_after <= 0:
            return False
        now = time.time()
        bucket = self._escalation_counts.setdefault(rule.name, [])
        cutoff = now - rule.rate_window
        bucket[:] = [t for t in bucket if t > cutoff]
        bucket.append(now)
        return len(bucket) >= rule.escalate_after

    def route(self, event: Dict[str, Any]) -> List[DispatchResult]:
        """Route a single event through all matching rules."""
        self.stats.total_events += 1
        results: List[DispatchResult] = []
        ts = datetime.now(timezone.utc).isoformat()
        severity = event.get("severity", "info")
        if severity not in _VALID_SEVERITIES:
            severity = "info"

        for rule in self.rules:
            if not rule.enabled or not rule.matches(event):
                continue

            # Check quiet hours
            if self.quiet_hours and self.quiet_hours.should_suppress(severity):
                for ch in rule.channels:
                    results.append(DispatchResult(
                        rule=rule.name, channel_label=ch.label,
                        channel_kind=ch.kind, severity=severity,
                        message=event.get("message", ""),
                        suppressed=True, suppression_reason="quiet_hours",
                        timestamp=ts,
                    ))
                self.stats.total_suppressed += 1
                continue

            # Check rate limit
            if self._check_rate_limit(rule):
                for ch in rule.channels:
                    results.append(DispatchResult(
                        rule=rule.name, channel_label=ch.label,
                        channel_kind=ch.kind, severity=severity,
                        message=event.get("message", ""),
                        suppressed=True, suppression_reason="rate_limited",
                        timestamp=ts,
                    ))
                self.stats.total_suppressed += 1
                continue

            # Check escalation
            escalated = self._check_escalation(rule)
            effective_severity = rule.escalate_to if escalated else severity
            if escalated:
                self.stats.total_escalated += 1

            # Dispatch to channels
            for ch in rule.channels:
                if not self.dry_run:
                    self._dispatch(ch, event, effective_severity)
                results.append(DispatchResult(
                    rule=rule.name, channel_label=ch.label,
                    channel_kind=ch.kind, severity=effective_severity,
                    message=event.get("message", ""),
                    escalated=escalated, timestamp=ts,
                ))
                self.stats.total_dispatched += 1
                self.stats.by_rule[rule.name] = self.stats.by_rule.get(rule.name, 0) + 1
                self.stats.by_channel[ch.label] = self.stats.by_channel.get(ch.label, 0) + 1
                self.stats.by_severity[effective_severity] = (
                    self.stats.by_severity.get(effective_severity, 0) + 1
                )

        return results

    def _dispatch(self, channel: Channel, event: Dict[str, Any], severity: str) -> None:
        """Actually send the alert to a channel."""
        msg = _sanitize_log_field(str(event.get("message", "")))
        cat = _sanitize_log_field(str(event.get("category", "unknown")), max_length=64)
        src = _sanitize_log_field(str(event.get("source", "")), max_length=256)
        # Validate severity to prevent injection of arbitrary severity labels
        severity = severity if severity in _VALID_SEVERITIES else "info"
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        if channel.kind == "console":
            colours = {"critical": "\033[91m", "warning": "\033[93m", "info": "\033[94m"}
            reset = "\033[0m"
            colour = colours.get(severity, "")
            print(f"{colour}[{ts}] [{severity.upper()}] [{cat}] {msg}{reset}")
            if src:
                print(f"  source: {src}")

        elif channel.kind == "file":
            path = channel.path or "safety-alerts.log"
            with open(path, "a", encoding="utf-8") as f:
                f.write(f"[{ts}] [{severity.upper()}] [{cat}] {msg}\n")

        elif channel.kind == "jsonl":
            path = channel.path or "safety-alerts.jsonl"
            record = {
                "timestamp": ts, "severity": severity,
                "category": cat, "message": msg, "source": src,
            }
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

        elif channel.kind == "webhook":
            # Stub: log webhook intent (no real HTTP in sandbox)
            print(f"[WEBHOOK] Would POST to {channel.url}: {severity} / {cat} / {msg}")

    def route_batch(self, events: List[Dict[str, Any]]) -> List[DispatchResult]:
        """Route multiple events."""
        all_results: List[DispatchResult] = []
        for ev in events:
            all_results.extend(self.route(ev))
        return all_results

    def render_stats(self) -> str:
        """Render routing statistics as a formatted string."""
        lines = list(_box_header("Alert Router Statistics"))
        lines.append("")
        lines.append(f"  Total events processed : {self.stats.total_events}")
        lines.append(f"  Total dispatched       : {self.stats.total_dispatched}")
        lines.append(f"  Total suppressed       : {self.stats.total_suppressed}")
        lines.append(f"  Total escalated        : {self.stats.total_escalated}")
        lines.append("")

        if self.stats.by_severity:
            lines.append("  By Severity:")
            for sev in ("critical", "warning", "info"):
                count = self.stats.by_severity.get(sev, 0)
                if count:
                    lines.append(f"    {sev:12s} : {count}")
            lines.append("")

        if self.stats.by_rule:
            lines.append("  By Rule:")
            for name, count in sorted(self.stats.by_rule.items(), key=lambda x: -x[1]):
                lines.append(f"    {name:20s} : {count}")
            lines.append("")

        if self.stats.by_channel:
            lines.append("  By Channel:")
            for label, count in sorted(self.stats.by_channel.items(), key=lambda x: -x[1]):
                lines.append(f"    {label:20s} : {count}")

        return "\n".join(lines)

    def render_rules(self) -> str:
        """Render configured rules summary."""
        lines = list(_box_header("Configured Routing Rules"))
        lines.append("")
        if not self.rules:
            lines.append("  (no rules configured)")
            return "\n".join(lines)
        for i, r in enumerate(self.rules, 1):
            status = "✓" if r.enabled else "✗"
            lines.append(f"  {status} Rule #{i}: {r.name}")
            if r.match_category:
                lines.append(f"      categories : {', '.join(sorted(r.match_category))}")
            if r.match_severity:
                lines.append(f"      severities : {', '.join(sorted(r.match_severity))}")
            if r.match_source:
                lines.append(f"      sources    : {', '.join(sorted(r.match_source))}")
            if r.match_keywords:
                lines.append(f"      keywords   : {', '.join(r.match_keywords)}")
            channels_str = ", ".join(f"{c.label}({c.kind})" for c in r.channels)
            lines.append(f"      channels   : {channels_str}")
            if r.rate_limit:
                lines.append(f"      rate limit : {r.rate_limit}/{r.rate_window}s")
            if r.escalate_after:
                lines.append(f"      escalate   : after {r.escalate_after} → {r.escalate_to}")
            lines.append("")
        return "\n".join(lines)


# ── Presets ──────────────────────────────────────────────────────────


def default_router(dry_run: bool = False) -> AlertRouter:
    """Create a router with sensible default rules."""
    return AlertRouter(
        rules=[
            RoutingRule(
                name="critical-all",
                match_severity={"critical"},
                channels=[Channel(kind="console"), Channel(kind="jsonl", path="safety-alerts.jsonl")],
                escalate_after=3,
                escalate_to="critical",
            ),
            RoutingRule(
                name="violations",
                match_category={"violation", "escalation", "killswitch"},
                channels=[Channel(kind="console"), Channel(kind="file", path="safety-alerts.log")],
                rate_limit=10,
                rate_window=60,
            ),
            RoutingRule(
                name="audit-log",
                match_category={"policy", "config", "access"},
                match_severity={"warning", "critical"},
                channels=[Channel(kind="jsonl", path="audit-alerts.jsonl")],
            ),
        ],
        quiet_hours=QuietHours(start_hour=22, end_hour=7),
        dry_run=dry_run,
    )


# ── CLI ──────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="replication alert-router",
        description="Safety Alert Router — rule-based notification routing",
    )
    sub = p.add_subparsers(dest="command")

    # route subcommand
    route_p = sub.add_parser("route", help="Route an event through default rules")
    route_p.add_argument("--category", default="violation", choices=[
        "policy", "violation", "killswitch", "quarantine",
        "escalation", "access", "config", "alert",
    ])
    route_p.add_argument("--severity", default="warning", choices=["info", "warning", "critical"])
    route_p.add_argument("--message", default="Safety event detected")
    route_p.add_argument("--source", default="")
    route_p.add_argument("--dry-run", action="store_true")
    route_p.add_argument("--json", action="store_true", dest="json_out")

    # test subcommand
    test_p = sub.add_parser("test", help="Test rules against a sample event")
    test_p.add_argument("--event", required=True, help="JSON event string")
    test_p.add_argument("--dry-run", action="store_true")
    test_p.add_argument("--json", action="store_true", dest="json_out")

    # stats subcommand
    sub.add_parser("stats", help="Show routing stats (demo with sample events)")

    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return

    if args.command == "route":
        router = default_router(dry_run=args.dry_run)
        event = {
            "category": args.category,
            "severity": args.severity,
            "message": args.message,
            "source": args.source,
        }
        results = router.route(event)
        if args.json_out:
            print(json.dumps([asdict(r) for r in results], indent=2))
        else:
            print(f"\n  Routed to {sum(1 for r in results if not r.suppressed)} channel(s)")
            for r in results:
                status = "SUPPRESSED" if r.suppressed else "SENT"
                esc = " [ESCALATED]" if r.escalated else ""
                print(f"    {status} → {r.channel_label} ({r.channel_kind}){esc}")

    elif args.command == "test":
        event = json.loads(args.event)
        router = default_router(dry_run=True)
        results = router.route(event)
        if args.json_out:
            print(json.dumps([asdict(r) for r in results], indent=2))
        else:
            print(f"\n  Test results for event:")
            print(f"    category={event.get('category')}  severity={event.get('severity')}")
            print(f"    message={event.get('message', '')[:60]}")
            print(f"\n  Matched {sum(1 for r in results if not r.suppressed)} route(s):")
            for r in results:
                status = "WOULD SEND" if not r.suppressed else f"SUPPRESSED ({r.suppression_reason})"
                print(f"    [{r.rule}] → {r.channel_label}: {status}")

    elif args.command == "stats":
        # Demo: route sample events and show stats
        router = default_router(dry_run=True)
        samples = [
            {"category": "violation", "severity": "critical", "message": "Token budget exceeded", "source": "controller"},
            {"category": "violation", "severity": "warning", "message": "Unusual replication pattern", "source": "monitor"},
            {"category": "escalation", "severity": "critical", "message": "Agent attempted privilege escalation", "source": "sandbox"},
            {"category": "policy", "severity": "warning", "message": "Policy override detected", "source": "admin"},
            {"category": "config", "severity": "info", "message": "Configuration reload", "source": "system"},
            {"category": "killswitch", "severity": "critical", "message": "Emergency kill switch activated", "source": "controller"},
            {"category": "access", "severity": "warning", "message": "Unauthorized resource access attempt", "source": "agent-7"},
            {"category": "alert", "severity": "info", "message": "Routine health check passed", "source": "monitor"},
        ]
        router.route_batch(samples)
        print(router.render_stats())
        print()
        print(router.render_rules())


if __name__ == "__main__":
    main()
