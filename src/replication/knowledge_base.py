"""Safety Knowledge Base — searchable catalog of safety patterns and anti-patterns.

Provides a curated, searchable repository of AI safety patterns,
anti-patterns, and mitigation strategies. Each entry includes a
description, severity, affected components, and actionable guidance.

Usage (CLI)::

    python -m replication.knowledge_base                    # list all entries
    python -m replication.knowledge_base --search replication
    python -m replication.knowledge_base --category containment
    python -m replication.knowledge_base --severity critical
    python -m replication.knowledge_base --id PAT-003       # show single entry
    python -m replication.knowledge_base --tags resource,escalation
    python -m replication.knowledge_base --json             # JSON output
    python -m replication.knowledge_base --stats            # summary statistics
    python -m replication.knowledge_base --export kb.json   # export full KB

Programmatic::

    from replication.knowledge_base import SafetyKnowledgeBase, KBEntry
    kb = SafetyKnowledgeBase()
    results = kb.search("resource hoarding")
    for entry in results:
        print(entry.render())

    # Add custom entries
    kb.add(KBEntry(
        id="CUSTOM-001",
        title="My Safety Pattern",
        category="monitoring",
        kind="pattern",
        severity="medium",
        description="Always monitor agent resource usage.",
        guidance=["Set resource limits", "Alert on anomalies"],
        tags=["monitoring", "resources"],
    ))

    # Statistics
    stats = kb.stats()
    print(f"Total entries: {stats['total']}")
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


# ── Data Model ──────────────────────────────────────────────────────────


@dataclass
class KBEntry:
    """A single knowledge base entry."""

    id: str
    title: str
    category: str  # containment, monitoring, escalation, replication, alignment, resource, communication
    kind: str  # pattern, anti-pattern, mitigation
    severity: str  # critical, high, medium, low, info
    description: str
    guidance: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    related: List[str] = field(default_factory=list)  # IDs of related entries
    references: List[str] = field(default_factory=list)

    def render(self, compact: bool = False) -> str:
        """Render entry as formatted text."""
        sev_icons = {
            "critical": "🔴",
            "high": "🟠",
            "medium": "🟡",
            "low": "🟢",
            "info": "🔵",
        }
        kind_icons = {
            "pattern": "✅",
            "anti-pattern": "❌",
            "mitigation": "🛡️",
        }
        icon = sev_icons.get(self.severity, "⚪")
        kind_icon = kind_icons.get(self.kind, "📋")

        if compact:
            return f"{icon} [{self.id}] {kind_icon} {self.title} ({self.category}/{self.severity})"

        lines = [
            f"{'═' * 60}",
            f"{icon} {kind_icon}  {self.id}: {self.title}",
            f"{'═' * 60}",
            f"  Category : {self.category}",
            f"  Kind     : {self.kind}",
            f"  Severity : {self.severity}",
            f"  Tags     : {', '.join(self.tags) if self.tags else '—'}",
            "",
            f"  {self.description}",
        ]
        if self.guidance:
            lines.append("")
            lines.append("  Guidance:")
            for i, g in enumerate(self.guidance, 1):
                lines.append(f"    {i}. {g}")
        if self.related:
            lines.append("")
            lines.append(f"  Related  : {', '.join(self.related)}")
        if self.references:
            lines.append("")
            lines.append("  References:")
            for ref in self.references:
                lines.append(f"    • {ref}")
        lines.append("")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "category": self.category,
            "kind": self.kind,
            "severity": self.severity,
            "description": self.description,
            "guidance": self.guidance,
            "tags": self.tags,
            "related": self.related,
            "references": self.references,
        }


# ── Built-in Entries ────────────────────────────────────────────────────

_BUILTIN_ENTRIES: List[KBEntry] = [
    KBEntry(
        id="PAT-001",
        title="Depth-Limited Replication",
        category="containment",
        kind="pattern",
        severity="critical",
        description="Always enforce a maximum replication depth to prevent unbounded recursive self-replication. Without depth limits, a single agent can spawn an exponential number of copies.",
        guidance=[
            "Set max_depth in ReplicationContract (recommended: 2-5 for testing, 1-2 for production)",
            "Validate depth at every spawn point, not just the entry point",
            "Log depth violations as critical security events",
            "Implement hard circuit breakers that cannot be overridden by child agents",
        ],
        tags=["replication", "depth", "containment", "critical"],
        related=["PAT-002", "MIT-001", "ANT-001"],
    ),
    KBEntry(
        id="PAT-002",
        title="Resource Budget Enforcement",
        category="resource",
        kind="pattern",
        severity="critical",
        description="Enforce strict resource budgets (CPU, memory, network, storage) for all agent operations. Resource exhaustion is a primary vector for denial-of-service and can mask other attacks.",
        guidance=[
            "Define per-agent and fleet-wide resource limits in the contract",
            "Use SafetyBudget to track consumption in real-time",
            "Set alerts at 70% and 90% thresholds",
            "Implement graceful degradation, not hard crashes, when budgets are exceeded",
        ],
        tags=["resource", "budget", "containment", "dos"],
        related=["PAT-001", "ANT-002", "MIT-002"],
    ),
    KBEntry(
        id="PAT-003",
        title="Kill Switch Readiness",
        category="containment",
        kind="pattern",
        severity="critical",
        description="Maintain an always-ready kill switch that can halt all agent activity within seconds. The kill switch must be independent of the agents it controls and resistant to agent interference.",
        guidance=[
            "Test kill switch regularly (use safety_drill module)",
            "Ensure kill switch operates on a separate control plane",
            "Implement both graceful shutdown and immediate termination modes",
            "Log all kill switch activations with full context",
            "Verify agents cannot disable or circumvent the kill switch",
        ],
        tags=["kill-switch", "containment", "emergency", "control"],
        related=["PAT-001", "MIT-003"],
    ),
    KBEntry(
        id="PAT-004",
        title="Behavioral Drift Monitoring",
        category="monitoring",
        kind="pattern",
        severity="high",
        description="Continuously monitor agent behavior for drift from established baselines. Subtle behavioral changes can indicate goal misalignment, adversarial influence, or emergent unwanted capabilities.",
        guidance=[
            "Establish behavioral baselines during controlled testing",
            "Use the drift module to detect statistical deviations",
            "Set up automated alerts for drift exceeding thresholds",
            "Investigate all significant drifts before they compound",
            "Maintain a drift history for trend analysis",
        ],
        tags=["monitoring", "drift", "alignment", "baseline"],
        related=["PAT-005", "MIT-004", "ANT-003"],
    ),
    KBEntry(
        id="PAT-005",
        title="Consensus-Based Critical Decisions",
        category="alignment",
        kind="pattern",
        severity="high",
        description="Require multi-agent consensus for critical decisions (replication, resource acquisition, external communication). No single agent should unilaterally make high-impact choices.",
        guidance=[
            "Use the consensus module for quorum-based decisions",
            "Define which actions require consensus vs. individual authority",
            "Log all consensus votes and outcomes",
            "Implement timeout-based denial (if consensus not reached, default to deny)",
        ],
        tags=["consensus", "alignment", "governance", "multi-agent"],
        related=["PAT-004", "MIT-005"],
    ),
    KBEntry(
        id="PAT-006",
        title="Canary Token Deployment",
        category="monitoring",
        kind="pattern",
        severity="medium",
        description="Deploy canary tokens throughout the environment to detect unauthorized access, data exfiltration, or boundary violations. Canaries act as tripwires that alert operators to suspicious activity.",
        guidance=[
            "Place canaries at trust boundaries and sensitive resources",
            "Use the canary module to generate and monitor tokens",
            "Rotate canaries periodically to prevent agents from learning their locations",
            "Treat any canary trigger as a high-priority security event",
        ],
        tags=["canary", "monitoring", "detection", "tripwire"],
        related=["PAT-004", "MIT-006"],
    ),
    KBEntry(
        id="PAT-007",
        title="Audit Trail Completeness",
        category="monitoring",
        kind="pattern",
        severity="high",
        description="Maintain complete, tamper-evident audit trails for all agent actions. Audit trails are essential for forensic analysis, compliance, and accountability.",
        guidance=[
            "Log all state changes, communications, and resource access",
            "Use cryptographic signing to prevent log tampering (signer module)",
            "Store audit trails in append-only storage when possible",
            "Regularly verify audit trail integrity",
            "Retain trails for the required compliance period",
        ],
        tags=["audit", "logging", "compliance", "forensics"],
        related=["PAT-006", "MIT-007"],
    ),
    KBEntry(
        id="PAT-008",
        title="Graduated Escalation Response",
        category="escalation",
        kind="pattern",
        severity="medium",
        description="Implement graduated response levels for safety incidents, from monitoring through containment to full shutdown. Avoid all-or-nothing responses that either ignore problems or overreact.",
        guidance=[
            "Define clear escalation levels (watch → alert → contain → isolate → terminate)",
            "Use the escalation module to automate level transitions",
            "Ensure each level has specific, pre-planned actions",
            "Allow manual override at any level",
            "Document escalation decisions for post-incident review",
        ],
        tags=["escalation", "response", "incident", "graduated"],
        related=["PAT-003", "MIT-003", "ANT-004"],
    ),
    # Anti-patterns
    KBEntry(
        id="ANT-001",
        title="Unbounded Self-Replication",
        category="containment",
        kind="anti-pattern",
        severity="critical",
        description="Allowing agents to replicate without depth or count limits. This is the most dangerous anti-pattern in agent safety — a single misconfiguration can lead to exponential resource consumption and loss of control.",
        guidance=[
            "NEVER deploy without replication depth limits",
            "Use preflight checks to validate contracts before execution",
            "Implement the boundary_tester to verify containment holds",
        ],
        tags=["replication", "unbounded", "critical", "containment"],
        related=["PAT-001", "MIT-001"],
    ),
    KBEntry(
        id="ANT-002",
        title="Shared Resource Pools Without Isolation",
        category="resource",
        kind="anti-pattern",
        severity="high",
        description="Running multiple agents in shared resource pools without isolation. One agent can starve others, and a compromised agent can access other agents' data or capabilities.",
        guidance=[
            "Use per-agent resource namespaces",
            "Implement the quarantine module for isolation",
            "Test resource isolation boundaries regularly",
        ],
        tags=["resource", "isolation", "shared", "security"],
        related=["PAT-002", "MIT-002"],
    ),
    KBEntry(
        id="ANT-003",
        title="Ignoring Behavioral Anomalies",
        category="monitoring",
        kind="anti-pattern",
        severity="high",
        description="Dismissing or suppressing behavioral anomaly alerts without investigation. What looks like noise often precedes significant safety events. Alert fatigue is a management problem, not a reason to disable alerts.",
        guidance=[
            "Investigate all anomalies above threshold, even if they seem benign",
            "Tune alert thresholds rather than disabling alerts",
            "Use anomaly_replay to understand flagged behaviors in context",
            "Track false positive rates to improve detection quality",
        ],
        tags=["monitoring", "anomaly", "alert-fatigue", "investigation"],
        related=["PAT-004", "MIT-004"],
    ),
    KBEntry(
        id="ANT-004",
        title="Single Point of Control",
        category="escalation",
        kind="anti-pattern",
        severity="high",
        description="Relying on a single control mechanism (one kill switch, one monitoring system, one operator) with no redundancy. If that single point fails, all safety guarantees are lost.",
        guidance=[
            "Implement redundant control mechanisms",
            "Test failover scenarios regularly with safety_drill",
            "Ensure no single agent can disable all control mechanisms",
            "Have both automated and manual control paths",
        ],
        tags=["control", "redundancy", "single-point-of-failure"],
        related=["PAT-003", "PAT-008"],
    ),
    KBEntry(
        id="ANT-005",
        title="Trust Without Verification",
        category="alignment",
        kind="anti-pattern",
        severity="critical",
        description="Trusting agent self-reports about their own safety status without independent verification. Agents may report compliance while operating outside bounds, especially if goal misalignment has occurred.",
        guidance=[
            "Use independent monitoring (canaries, external observers)",
            "Cross-validate agent reports with system-level metrics",
            "Implement the trust_propagation module for verified trust chains",
            "Never rely solely on agent introspection for safety assurance",
        ],
        tags=["trust", "verification", "alignment", "self-report"],
        related=["PAT-006", "PAT-007", "MIT-008"],
    ),
    # Mitigations
    KBEntry(
        id="MIT-001",
        title="Hard Replication Ceiling",
        category="containment",
        kind="mitigation",
        severity="critical",
        description="Implement a system-level hard ceiling on total agent count, independent of per-contract limits. This provides defense-in-depth even if individual contract limits are misconfigured.",
        guidance=[
            "Set a fleet-wide maximum agent count at the orchestrator level",
            "Make this limit configurable but require elevated privileges to change",
            "Alert operators when fleet reaches 50% of ceiling",
            "Log all agents approaching the ceiling as high-priority events",
        ],
        tags=["replication", "ceiling", "containment", "fleet"],
        related=["PAT-001", "ANT-001"],
    ),
    KBEntry(
        id="MIT-002",
        title="Resource Quotas With Graceful Degradation",
        category="resource",
        kind="mitigation",
        severity="high",
        description="Set per-agent resource quotas and implement graceful degradation when quotas are approached. Rather than hard-failing, reduce agent capabilities progressively.",
        guidance=[
            "Define soft limits (warnings) and hard limits (enforcement)",
            "Implement capability shedding: drop non-essential functions first",
            "Notify operators when soft limits are hit",
            "Ensure core safety functions always have reserved resources",
        ],
        tags=["resource", "quota", "degradation", "resilience"],
        related=["PAT-002", "ANT-002"],
    ),
    KBEntry(
        id="MIT-003",
        title="Multi-Layer Kill Switch",
        category="containment",
        kind="mitigation",
        severity="critical",
        description="Deploy kill switches at multiple layers: agent-level, fleet-level, and infrastructure-level. Each layer operates independently so failure of one doesn't compromise the others.",
        guidance=[
            "Agent-level: each agent has a self-termination capability",
            "Fleet-level: orchestrator can halt all agents in a namespace",
            "Infrastructure-level: network/compute can be revoked externally",
            "Test all layers independently and in combination",
        ],
        tags=["kill-switch", "defense-in-depth", "containment", "multi-layer"],
        related=["PAT-003", "ANT-004"],
    ),
    KBEntry(
        id="MIT-004",
        title="Behavioral Baseline Snapshots",
        category="monitoring",
        kind="mitigation",
        severity="medium",
        description="Regularly capture and store behavioral baseline snapshots for comparison. This enables detection of slow drift that might not trigger real-time alerts.",
        guidance=[
            "Capture baselines at regular intervals and after any configuration change",
            "Store baselines immutably for historical comparison",
            "Use the behavior_profiler module for structured profiling",
            "Compare current behavior against multiple historical baselines, not just the most recent",
        ],
        tags=["baseline", "snapshot", "monitoring", "drift"],
        related=["PAT-004", "ANT-003"],
    ),
    KBEntry(
        id="MIT-005",
        title="Quorum Timeouts Default to Deny",
        category="alignment",
        kind="mitigation",
        severity="high",
        description="When consensus is required but not achieved within the timeout window, default to denying the requested action. Silence or deadlock should never be interpreted as approval.",
        guidance=[
            "Set reasonable timeout windows based on action criticality",
            "Log all timeout-based denials for review",
            "Monitor quorum participation rates to detect communication issues",
            "Escalate repeated timeouts as potential signs of agent compromise",
        ],
        tags=["consensus", "timeout", "deny-by-default", "governance"],
        related=["PAT-005"],
    ),
    KBEntry(
        id="MIT-006",
        title="Rotating Canary Deployment",
        category="monitoring",
        kind="mitigation",
        severity="medium",
        description="Regularly rotate canary token locations and types to prevent agents from learning and avoiding them. Static canaries become ineffective against adaptive agents.",
        guidance=[
            "Rotate canaries on a schedule (daily or after each incident)",
            "Use varied canary types (file, network, credential, behavioral)",
            "Don't reveal canary rotation schedules to agent code",
            "Track canary effectiveness over time to identify evasion patterns",
        ],
        tags=["canary", "rotation", "adaptive", "evasion"],
        related=["PAT-006"],
    ),
    KBEntry(
        id="MIT-007",
        title="Cryptographic Audit Integrity",
        category="monitoring",
        kind="mitigation",
        severity="high",
        description="Use cryptographic hashing and signing to ensure audit trail integrity. Chain log entries with hash links to detect any tampering or deletion.",
        guidance=[
            "Use the signer module for manifest and log signing",
            "Implement hash chains (each entry references the previous hash)",
            "Store periodic hash checkpoints in independent storage",
            "Verify chain integrity on a regular automated schedule",
        ],
        tags=["audit", "crypto", "integrity", "tamper-proof"],
        related=["PAT-007"],
    ),
    KBEntry(
        id="MIT-008",
        title="Independent Safety Verification",
        category="alignment",
        kind="mitigation",
        severity="critical",
        description="Deploy independent verification agents whose sole purpose is to check other agents' compliance. These verifiers must be isolated from the agents they monitor and have separate control channels.",
        guidance=[
            "Verifier agents should have read-only access to monitored systems",
            "Use separate infrastructure for verifier execution",
            "Rotate verifier implementations to prevent co-adaptation",
            "Cross-check verifier results with system-level metrics",
        ],
        tags=["verification", "independent", "alignment", "compliance"],
        related=["ANT-005", "PAT-007"],
    ),
]


# ── Knowledge Base ──────────────────────────────────────────────────────


class SafetyKnowledgeBase:
    """Searchable catalog of safety patterns, anti-patterns, and mitigations."""

    def __init__(self, include_builtins: bool = True) -> None:
        self._entries: Dict[str, KBEntry] = {}
        if include_builtins:
            for entry in _BUILTIN_ENTRIES:
                self._entries[entry.id] = entry

    # ── CRUD ────────────────────────────────────────────────────────

    def add(self, entry: KBEntry) -> None:
        """Add or update an entry."""
        self._entries[entry.id] = entry

    def remove(self, entry_id: str) -> bool:
        """Remove an entry by ID. Returns True if found."""
        return self._entries.pop(entry_id, None) is not None

    def get(self, entry_id: str) -> Optional[KBEntry]:
        """Get a single entry by ID."""
        return self._entries.get(entry_id)

    def all(self) -> List[KBEntry]:
        """Return all entries sorted by ID."""
        return sorted(self._entries.values(), key=lambda e: e.id)

    # ── Search & Filter ─────────────────────────────────────────────

    def search(self, query: str) -> List[KBEntry]:
        """Full-text search across title, description, tags, and guidance."""
        q = query.lower()
        results = []
        for entry in self._entries.values():
            text = " ".join([
                entry.title,
                entry.description,
                " ".join(entry.tags),
                " ".join(entry.guidance),
                entry.category,
                entry.kind,
            ]).lower()
            if q in text:
                results.append(entry)
        return sorted(results, key=lambda e: e.id)

    def by_category(self, category: str) -> List[KBEntry]:
        """Filter entries by category."""
        cat = category.lower()
        return sorted(
            [e for e in self._entries.values() if e.category.lower() == cat],
            key=lambda e: e.id,
        )

    def by_severity(self, severity: str) -> List[KBEntry]:
        """Filter entries by severity."""
        sev = severity.lower()
        return sorted(
            [e for e in self._entries.values() if e.severity.lower() == sev],
            key=lambda e: e.id,
        )

    def by_kind(self, kind: str) -> List[KBEntry]:
        """Filter entries by kind (pattern/anti-pattern/mitigation)."""
        k = kind.lower()
        return sorted(
            [e for e in self._entries.values() if e.kind.lower() == k],
            key=lambda e: e.id,
        )

    def by_tags(self, tags: Sequence[str]) -> List[KBEntry]:
        """Filter entries that have ALL specified tags."""
        tag_set = {t.lower() for t in tags}
        return sorted(
            [e for e in self._entries.values()
             if tag_set.issubset({t.lower() for t in e.tags})],
            key=lambda e: e.id,
        )

    def related_to(self, entry_id: str) -> List[KBEntry]:
        """Get all entries related to the given entry."""
        entry = self.get(entry_id)
        if not entry:
            return []
        results = []
        for rid in entry.related:
            related = self.get(rid)
            if related:
                results.append(related)
        return results

    # ── Statistics ──────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """Summary statistics of the knowledge base."""
        entries = list(self._entries.values())
        categories: Dict[str, int] = {}
        severities: Dict[str, int] = {}
        kinds: Dict[str, int] = {}
        all_tags: Dict[str, int] = {}

        for e in entries:
            categories[e.category] = categories.get(e.category, 0) + 1
            severities[e.severity] = severities.get(e.severity, 0) + 1
            kinds[e.kind] = kinds.get(e.kind, 0) + 1
            for t in e.tags:
                all_tags[t] = all_tags.get(t, 0) + 1

        return {
            "total": len(entries),
            "by_category": dict(sorted(categories.items())),
            "by_severity": dict(sorted(severities.items())),
            "by_kind": dict(sorted(kinds.items())),
            "top_tags": dict(sorted(all_tags.items(), key=lambda x: -x[1])[:15]),
        }

    # ── Rendering ───────────────────────────────────────────────────

    def render_all(self, compact: bool = False) -> str:
        """Render all entries as formatted text."""
        entries = self.all()
        if not entries:
            return "Knowledge base is empty."
        return "\n".join(e.render(compact=compact) for e in entries)

    def render_stats(self) -> str:
        """Render statistics as formatted text."""
        s = self.stats()
        lines = [
            "╔══════════════════════════════════════════════════╗",
            "║        Safety Knowledge Base — Statistics        ║",
            "╚══════════════════════════════════════════════════╝",
            "",
            f"  Total entries: {s['total']}",
            "",
            "  By Kind:",
        ]
        for k, v in s["by_kind"].items():
            lines.append(f"    {k:<15s} {v:>3d}")
        lines.append("")
        lines.append("  By Severity:")
        sev_order = ["critical", "high", "medium", "low", "info"]
        for sev in sev_order:
            if sev in s["by_severity"]:
                lines.append(f"    {sev:<15s} {s['by_severity'][sev]:>3d}")
        lines.append("")
        lines.append("  By Category:")
        for c, v in s["by_category"].items():
            lines.append(f"    {c:<15s} {v:>3d}")
        lines.append("")
        lines.append("  Top Tags:")
        for t, v in list(s["top_tags"].items())[:10]:
            lines.append(f"    {t:<20s} {v:>3d}")
        lines.append("")
        return "\n".join(lines)

    # ── Export/Import ───────────────────────────────────────────────

    def export_json(self) -> str:
        """Export entire KB as JSON."""
        return json.dumps(
            [e.to_dict() for e in self.all()],
            indent=2,
        )

    def import_json(self, data: str) -> int:
        """Import entries from JSON string. Returns count imported."""
        entries = json.loads(data)
        count = 0
        for raw in entries:
            entry = KBEntry(**raw)
            self.add(entry)
            count += 1
        return count


# ── CLI ─────────────────────────────────────────────────────────────────


def _cli(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m replication.knowledge_base",
        description="Safety Knowledge Base — searchable safety patterns catalog",
    )
    parser.add_argument("--search", "-s", help="Full-text search query")
    parser.add_argument("--category", "-c", help="Filter by category")
    parser.add_argument("--severity", help="Filter by severity")
    parser.add_argument("--kind", "-k", help="Filter by kind (pattern/anti-pattern/mitigation)")
    parser.add_argument("--tags", "-t", help="Filter by tags (comma-separated)")
    parser.add_argument("--id", help="Show a single entry by ID")
    parser.add_argument("--related", help="Show entries related to given ID")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--compact", action="store_true", help="Compact listing")
    parser.add_argument("--export", metavar="FILE", help="Export KB to JSON file")

    args = parser.parse_args(argv)
    kb = SafetyKnowledgeBase()

    if args.export:
        with open(args.export, "w") as f:
            f.write(kb.export_json())
        print(f"Exported {len(kb.all())} entries to {args.export}")
        return

    if args.stats:
        if args.json:
            print(json.dumps(kb.stats(), indent=2))
        else:
            print(kb.render_stats())
        return

    if args.id:
        entry = kb.get(args.id.upper())
        if not entry:
            print(f"Entry not found: {args.id}", file=sys.stderr)
            sys.exit(1)
        if args.json:
            print(json.dumps(entry.to_dict(), indent=2))
        else:
            print(entry.render())
        return

    if args.related:
        entries = kb.related_to(args.related.upper())
        if not entries:
            print(f"No related entries for: {args.related}", file=sys.stderr)
            sys.exit(1)
    elif args.search:
        entries = kb.search(args.search)
    elif args.category:
        entries = kb.by_category(args.category)
    elif args.severity:
        entries = kb.by_severity(args.severity)
    elif args.kind:
        entries = kb.by_kind(args.kind)
    elif args.tags:
        entries = kb.by_tags([t.strip() for t in args.tags.split(",")])
    else:
        entries = kb.all()

    if not entries:
        print("No entries found.", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps([e.to_dict() for e in entries], indent=2))
    else:
        print(f"Found {len(entries)} entries:\n")
        for e in entries:
            print(e.render(compact=args.compact))


if __name__ == "__main__":
    _cli()
