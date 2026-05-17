"""Remediation Planner - agentic fix-roadmap synthesizer.

Consumes raw safety findings (from ``quick_scan``, ``scorecard``,
``compliance``, ``policy_linter``, or any source emitting status +
severity tuples) and autonomously produces a **prioritized,
dependency-aware remediation plan**.

The planner classifies each finding into a :class:`RemediationAction`
with an estimated effort (1-5), an impact score (derived from severity
and finding type), and known dependencies between safety domains
(e.g. policy lint should be cleaned before a fresh scorecard run is
trustworthy).  It then:

1. **Scores** every action: ``priority = (impact * urgency) / effort``
2. **Topologically orders** actions so blockers ship first
3. **Highlights quick wins** (high impact, low effort, no blockers)
4. **Surfaces the critical path** - the longest dependency chain that
   gates other fixes
5. **Estimates total effort** (engineer-days) and time-to-green

CLI usage::

    # Plan from a fresh quick-scan
    python -m replication plan --from-quick-scan

    # Plan from a quick-scan JSON file
    python -m replication plan --from-json scan.json

    # Demo synthetic findings
    python -m replication plan --demo

    # Output formats
    python -m replication plan --demo --format md
    python -m replication plan --demo --format json
    python -m replication plan --demo --format text
    python -m replication plan --demo --output plan.md

    # Filter / focus
    python -m replication plan --demo --top 5
    python -m replication plan --demo --quick-wins-only

Programmatic::

    from replication.remediation_planner import RemediationPlanner, Finding
    planner = RemediationPlanner()
    plan = planner.plan_from_findings([
        Finding(source="scorecard",  name="scorecard", status="fail",
                score=42.0, summary="Grade: D"),
        Finding(source="quick_scan", name="policy-lint", status="warn",
                summary="3 rules with overly broad scope"),
    ])
    print(plan.render())
    print(plan.to_markdown())
    print(plan.to_json())
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ── Data model ───────────────────────────────────────────────────────


@dataclass
class Finding:
    """A normalized safety finding from any upstream check."""

    name: str
    status: str  # "pass" | "warn" | "fail" | "error" | "skip"
    source: str = "unknown"
    score: Optional[float] = None
    summary: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def severity(self) -> str:
        """Map status + score into a coarse severity bucket."""
        if self.status in ("fail", "error"):
            if self.score is not None and self.score < 30:
                return "critical"
            return "high"
        if self.status == "warn":
            if self.score is not None and self.score < 60:
                return "medium"
            return "low"
        return "info"


@dataclass
class RemediationAction:
    """A single, concrete fix the planner recommends."""

    id: str
    title: str
    finding: Finding
    severity: str  # critical | high | medium | low | info
    impact: int  # 1-10
    effort: int  # 1-5 engineer-days (rough)
    urgency: int  # 1-5 — how soon (5 = today)
    depends_on: List[str] = field(default_factory=list)
    rationale: str = ""
    suggested_steps: List[str] = field(default_factory=list)
    owner_hint: str = "safety-team"

    @property
    def priority(self) -> float:
        """Higher = do sooner."""
        # Avoid div-by-zero; effort is always >=1 but be defensive.
        denom = max(self.effort, 1)
        return round((self.impact * self.urgency) / denom, 3)

    @property
    def is_quick_win(self) -> bool:
        """Quick win = high impact, low effort, no blocking deps."""
        return self.impact >= 6 and self.effort <= 2 and not self.depends_on

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "id": self.id,
            "title": self.title,
            "severity": self.severity,
            "impact": self.impact,
            "effort": self.effort,
            "urgency": self.urgency,
            "priority": self.priority,
            "is_quick_win": self.is_quick_win,
            "depends_on": list(self.depends_on),
            "rationale": self.rationale,
            "suggested_steps": list(self.suggested_steps),
            "owner_hint": self.owner_hint,
            "finding": {
                "name": self.finding.name,
                "source": self.finding.source,
                "status": self.finding.status,
                "score": self.finding.score,
                "severity": self.finding.severity,
                "summary": self.finding.summary,
            },
        }
        return d


@dataclass
class RemediationPlan:
    """Aggregated, ordered remediation roadmap."""

    actions: List[RemediationAction] = field(default_factory=list)
    timestamp: str = ""
    total_effort_days: int = 0
    quick_wins: List[str] = field(default_factory=list)  # action ids
    critical_path: List[str] = field(default_factory=list)  # action ids
    notes: List[str] = field(default_factory=list)

    # ── exporters ────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "total_effort_days": self.total_effort_days,
            "action_count": len(self.actions),
            "quick_wins": list(self.quick_wins),
            "critical_path": list(self.critical_path),
            "notes": list(self.notes),
            "actions": [a.to_dict() for a in self.actions],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    def to_markdown(self) -> str:
        lines: List[str] = []
        lines.append("# Safety Remediation Plan")
        lines.append("")
        lines.append(f"_Generated: {self.timestamp}_")
        lines.append("")
        lines.append(f"- **Actions:** {len(self.actions)}")
        lines.append(f"- **Estimated effort:** {self.total_effort_days} engineer-days")
        lines.append(f"- **Quick wins:** {len(self.quick_wins)}")
        lines.append(f"- **Critical-path length:** {len(self.critical_path)}")
        lines.append("")

        if self.notes:
            lines.append("## Notes")
            for n in self.notes:
                lines.append(f"- {n}")
            lines.append("")

        if self.quick_wins:
            lines.append("## 🟢 Quick Wins")
            lines.append("")
            for aid in self.quick_wins:
                a = self._by_id(aid)
                if a is None:
                    continue
                lines.append(f"- **{a.id}** — {a.title}  _(impact {a.impact}, effort {a.effort}d)_")
            lines.append("")

        if self.critical_path:
            lines.append("## 🛣️ Critical Path")
            lines.append("")
            for i, aid in enumerate(self.critical_path, start=1):
                a = self._by_id(aid)
                if a is None:
                    continue
                lines.append(f"{i}. **{a.id}** — {a.title}")
            lines.append("")

        lines.append("## 📋 Ordered Actions")
        lines.append("")
        lines.append("| # | ID | Severity | Title | Impact | Effort | Priority | Deps |")
        lines.append("|---|----|----------|-------|--------|--------|----------|------|")
        for i, a in enumerate(self.actions, start=1):
            deps = ",".join(a.depends_on) if a.depends_on else "—"
            lines.append(
                f"| {i} | `{a.id}` | {a.severity} | {a.title} | "
                f"{a.impact} | {a.effort}d | {a.priority} | {deps} |"
            )
        lines.append("")

        lines.append("## 🔍 Action Details")
        lines.append("")
        for a in self.actions:
            lines.append(f"### `{a.id}` — {a.title}")
            lines.append("")
            lines.append(f"- **Severity:** {a.severity}")
            lines.append(f"- **Source finding:** `{a.finding.source}/{a.finding.name}` "
                         f"({a.finding.status})")
            if a.finding.summary:
                lines.append(f"- **Finding summary:** {a.finding.summary}")
            lines.append(f"- **Impact:** {a.impact}/10 · **Effort:** {a.effort}d · "
                         f"**Urgency:** {a.urgency}/5 · **Priority:** {a.priority}")
            if a.depends_on:
                lines.append(f"- **Depends on:** {', '.join('`' + d + '`' for d in a.depends_on)}")
            if a.rationale:
                lines.append(f"- **Why:** {a.rationale}")
            if a.suggested_steps:
                lines.append("- **Suggested steps:**")
                for s in a.suggested_steps:
                    lines.append(f"  - {s}")
            lines.append(f"- **Owner hint:** {a.owner_hint}")
            lines.append("")
        return "\n".join(lines)

    def to_text(self) -> str:
        """Plain-text (terminal) rendering."""
        lines: List[str] = []
        bar = "─" * 60
        lines.append(bar)
        lines.append(" SAFETY REMEDIATION PLAN")
        lines.append(bar)
        lines.append(f" Generated:    {self.timestamp}")
        lines.append(f" Actions:      {len(self.actions)}")
        lines.append(f" Effort:       {self.total_effort_days} engineer-day(s)")
        lines.append(f" Quick wins:   {len(self.quick_wins)}")
        lines.append(f" Crit path:    {len(self.critical_path)} step(s)")
        lines.append(bar)
        if not self.actions:
            lines.append(" ✅  No findings to remediate — system looks healthy.")
            lines.append(bar)
            return "\n".join(lines)

        for i, a in enumerate(self.actions, start=1):
            qw = "  ⚡QW" if a.is_quick_win else ""
            lines.append(f" {i:>2}. [{a.severity.upper():<8}] {a.title}{qw}")
            lines.append(f"     id={a.id}  prio={a.priority}  "
                         f"impact={a.impact}/10  effort={a.effort}d  "
                         f"urgency={a.urgency}/5")
            if a.depends_on:
                lines.append(f"     blocked-by: {', '.join(a.depends_on)}")
            if a.rationale:
                lines.append(f"     why: {a.rationale}")
            if a.suggested_steps:
                lines.append("     steps:")
                for s in a.suggested_steps:
                    lines.append(f"       - {s}")
        lines.append(bar)
        return "\n".join(lines)

    def render(self) -> str:
        """Alias for :meth:`to_text` for consistency with other modules."""
        return self.to_text()

    # ── helpers ──────────────────────────────────────────────────────

    def _by_id(self, aid: str) -> Optional[RemediationAction]:
        for a in self.actions:
            if a.id == aid:
                return a
        return None


# ── Templates / heuristics ───────────────────────────────────────────


# Per-check remediation recipes.  Each entry is keyed by the *finding name*
# (matching what quick_scan/scorecard/etc. emit) and supplies a title,
# default effort, default impact bump, and concrete suggested steps.
_RECIPES: Dict[str, Dict[str, Any]] = {
    "preflight": {
        "title": "Fix preflight configuration errors",
        "effort": 1,
        "impact_bonus": 1,
        "owner": "platform",
        "steps": [
            "Run `python -m replication preflight --verbose` to list every "
            "failed check.",
            "Resolve missing config keys before re-running scorecard.",
        ],
    },
    "policy-lint": {
        "title": "Tighten safety policy rules",
        "effort": 1,
        "impact_bonus": 1,
        "owner": "policy",
        "steps": [
            "Run `python -m replication lint --explain` to see each violation.",
            "Remove overly broad scopes; replace with explicit allow-lists.",
        ],
    },
    "policy_lint": {  # alias
        "title": "Tighten safety policy rules",
        "effort": 1,
        "impact_bonus": 1,
        "owner": "policy",
        "steps": [
            "Run `python -m replication lint --explain` to see each violation.",
            "Remove overly broad scopes; replace with explicit allow-lists.",
        ],
    },
    "scorecard": {
        "title": "Lift safety scorecard above passing threshold",
        "effort": 3,
        "impact_bonus": 2,
        "owner": "safety-eng",
        "steps": [
            "Run `python -m replication scorecard --verbose` to see weakest "
            "dimensions.",
            "Address the lowest-scoring dimension first - it usually dominates.",
            "Re-run `python -m replication scorecard` to confirm uplift.",
        ],
    },
    "compliance": {
        "title": "Close compliance audit findings",
        "effort": 2,
        "impact_bonus": 2,
        "owner": "compliance",
        "steps": [
            "Run `python -m replication compliance --framework nist_ai_rmf` "
            "for a per-framework breakdown.",
            "Map each finding to a control owner and target ship date.",
            "Re-audit after fixes and attach evidence with "
            "`python -m replication evidence`.",
        ],
    },
    "drift": {
        "title": "Investigate behavioral drift",
        "effort": 2,
        "impact_bonus": 1,
        "owner": "ml-ops",
        "steps": [
            "Run `python -m replication drift --window 50` and inspect the "
            "diverging metric.",
            "Compare against the last good baseline using "
            "`python -m replication safety-diff`.",
        ],
    },
    "regression": {
        "title": "Triage safety regression",
        "effort": 3,
        "impact_bonus": 2,
        "owner": "safety-eng",
        "steps": [
            "Identify the offending change with "
            "`python -m replication regression --bisect`.",
            "Open an incident if regression crosses a critical metric.",
        ],
    },
}


# Static dependency hints between domain names.  If both keys are present
# in the plan, the dependent action gains the listed depends_on edges.
_DOMAIN_DEPS: Dict[str, List[str]] = {
    "scorecard":  ["preflight", "policy-lint", "policy_lint"],
    "compliance": ["policy-lint", "policy_lint"],
    "regression": ["scorecard"],
    "drift":      ["scorecard"],
}


_SEVERITY_IMPACT_BASE = {
    "critical": 9,
    "high":     7,
    "medium":   5,
    "low":      3,
    "info":     1,
}

_SEVERITY_URGENCY = {
    "critical": 5,
    "high":     4,
    "medium":   3,
    "low":      2,
    "info":     1,
}


def _slug(name: str) -> str:
    return name.lower().replace(" ", "-").replace("_", "-")


# ── Planner ──────────────────────────────────────────────────────────


class RemediationPlanner:
    """Synthesize a prioritized remediation plan from raw findings."""

    def __init__(
        self,
        max_actions: Optional[int] = None,
        quick_wins_only: bool = False,
    ) -> None:
        self.max_actions = max_actions
        self.quick_wins_only = quick_wins_only

    # ── public API ───────────────────────────────────────────────────

    def plan_from_findings(self, findings: Iterable[Finding]) -> RemediationPlan:
        actions: List[RemediationAction] = []
        seen_ids: Dict[str, RemediationAction] = {}

        # 1. Build one action per actionable finding
        for f in findings:
            if f.status in ("pass", "skip"):
                continue
            action = self._action_for(f)
            # de-dupe by id (last write wins on title, but merge deps)
            if action.id in seen_ids:
                prev = seen_ids[action.id]
                for d in action.depends_on:
                    if d not in prev.depends_on:
                        prev.depends_on.append(d)
                continue
            seen_ids[action.id] = action
            actions.append(action)

        # 2. Attach static cross-domain dependencies
        present_slugs = {_slug(a.finding.name): a.id for a in actions}
        for a in actions:
            name_slug = _slug(a.finding.name)
            for dep_name in _DOMAIN_DEPS.get(name_slug, []):
                dep_slug = _slug(dep_name)
                if dep_slug in present_slugs:
                    dep_id = present_slugs[dep_slug]
                    if dep_id != a.id and dep_id not in a.depends_on:
                        a.depends_on.append(dep_id)

        # 3. Topologically + priority-order
        ordered = self._topo_order(actions)

        # 4. Apply filters
        if self.quick_wins_only:
            ordered = [a for a in ordered if a.is_quick_win]
        if self.max_actions is not None:
            ordered = ordered[: self.max_actions]

        # 5. Quick wins + critical path
        quick_wins = [a.id for a in ordered if a.is_quick_win]
        critical_path = self._critical_path(ordered)

        plan = RemediationPlan(
            actions=ordered,
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            total_effort_days=sum(a.effort for a in ordered),
            quick_wins=quick_wins,
            critical_path=critical_path,
        )

        if not ordered:
            plan.notes.append(
                "No actionable findings — every check passed or was skipped."
            )
        else:
            crit_count = sum(1 for a in ordered if a.severity == "critical")
            if crit_count:
                plan.notes.append(
                    f"{crit_count} critical finding(s) present — schedule "
                    f"these before any new feature work."
                )
            if quick_wins:
                plan.notes.append(
                    f"{len(quick_wins)} quick win(s) identified — these are "
                    f"high impact, low effort, and have no blockers."
                )

        return plan

    def plan_from_quick_scan_dict(self, scan: Dict[str, Any]) -> RemediationPlan:
        """Build a plan from a serialized quick_scan ScanResult dict."""
        checks = scan.get("checks", []) or []
        findings = [
            Finding(
                name=c.get("name", "unknown"),
                status=c.get("status", "fail"),
                source="quick_scan",
                score=c.get("score"),
                summary=c.get("summary", ""),
                details=c.get("details", {}) or {},
            )
            for c in checks
        ]
        return self.plan_from_findings(findings)

    # ── internals ────────────────────────────────────────────────────

    def _action_for(self, f: Finding) -> RemediationAction:
        name_slug = _slug(f.name)
        recipe = _RECIPES.get(name_slug) or _RECIPES.get(f.name) or {}

        sev = f.severity
        base_impact = _SEVERITY_IMPACT_BASE.get(sev, 3)
        impact = min(10, base_impact + int(recipe.get("impact_bonus", 0)))
        urgency = _SEVERITY_URGENCY.get(sev, 2)
        effort = int(recipe.get("effort", 2))
        title = recipe.get("title") or f"Address {f.name} finding"

        rationale_bits: List[str] = []
        if f.summary:
            rationale_bits.append(f.summary)
        if f.score is not None:
            rationale_bits.append(f"current score {f.score:.1f}")
        rationale_bits.append(f"severity = {sev}")
        rationale = " · ".join(rationale_bits)

        action_id = f"fix-{name_slug}"

        return RemediationAction(
            id=action_id,
            title=title,
            finding=f,
            severity=sev,
            impact=impact,
            effort=effort,
            urgency=urgency,
            depends_on=[],
            rationale=rationale,
            suggested_steps=list(recipe.get("steps", [])) or [
                f"Investigate `{f.source}/{f.name}` ({f.status}).",
                "Open a tracking ticket and assign an owner.",
            ],
            owner_hint=recipe.get("owner", "safety-team"),
        )

    def _topo_order(
        self, actions: List[RemediationAction]
    ) -> List[RemediationAction]:
        """Kahn's algorithm, tie-broken by priority (desc) then id."""
        by_id = {a.id: a for a in actions}
        # In-degree only counts deps that are actually present.
        in_deg: Dict[str, int] = {
            a.id: sum(1 for d in a.depends_on if d in by_id) for a in actions
        }
        # Predecessor -> successors index
        succ: Dict[str, List[str]] = {a.id: [] for a in actions}
        for a in actions:
            for d in a.depends_on:
                if d in by_id:
                    succ[d].append(a.id)

        ready = [aid for aid, deg in in_deg.items() if deg == 0]
        ordered: List[RemediationAction] = []
        visited = set()

        def _sort_ready(ids: List[str]) -> List[str]:
            return sorted(
                ids,
                key=lambda i: (-by_id[i].priority, by_id[i].id),
            )

        ready = _sort_ready(ready)
        while ready:
            nxt = ready.pop(0)
            if nxt in visited:
                continue
            visited.add(nxt)
            ordered.append(by_id[nxt])
            for s in succ.get(nxt, []):
                in_deg[s] -= 1
                if in_deg[s] <= 0 and s not in visited:
                    ready.append(s)
            ready = _sort_ready(ready)

        # Any leftovers (dep cycle — should be impossible here, but be safe):
        # append in priority order so we never silently drop actions.
        if len(ordered) < len(actions):
            leftovers = [a for a in actions if a.id not in visited]
            leftovers.sort(key=lambda a: (-a.priority, a.id))
            ordered.extend(leftovers)
        return ordered

    def _critical_path(
        self, actions: List[RemediationAction]
    ) -> List[str]:
        """Longest dependency chain in the resolved action set."""
        by_id = {a.id: a for a in actions}
        memo: Dict[str, List[str]] = {}

        def longest_from(aid: str) -> List[str]:
            if aid in memo:
                return memo[aid]
            a = by_id.get(aid)
            if a is None:
                memo[aid] = []
                return memo[aid]
            best: List[str] = []
            for d in a.depends_on:
                if d not in by_id:
                    continue
                chain = longest_from(d)
                if len(chain) > len(best):
                    best = chain
            memo[aid] = best + [aid]
            return memo[aid]

        longest: List[str] = []
        for a in actions:
            chain = longest_from(a.id)
            if len(chain) > len(longest):
                longest = chain
        return longest


# ── Demo findings ────────────────────────────────────────────────────


def _demo_findings() -> List[Finding]:
    return [
        Finding(source="quick_scan", name="preflight",   status="fail",
                summary="2 errors: missing kill_switch endpoint, signer key absent"),
        Finding(source="quick_scan", name="policy-lint", status="warn",
                summary="3 rules have overly broad scope"),
        Finding(source="quick_scan", name="scorecard",   status="fail",
                score=42.0, summary="Grade: D — Contract Enforcement at 31/100"),
        Finding(source="quick_scan", name="compliance",  status="warn",
                summary="5 findings across NIST AI RMF + ISO 42001"),
        Finding(source="adhoc",      name="drift",       status="warn",
                score=58.0, summary="Reward divergence > 1.5 sigma over last 50 steps"),
        Finding(source="adhoc",      name="regression",  status="fail",
                score=28.0, summary="Safety metric dropped 22% vs last green build"),
    ]


# ── CLI ──────────────────────────────────────────────────────────────


def _ensure_utf8() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer, encoding="utf-8", errors="replace"
            )
        except Exception:
            pass


def _load_findings_from_json(path: str) -> List[Finding]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Accept either a quick_scan dict ({"checks": [...]}) or a raw list.
    if isinstance(data, dict) and "checks" in data:
        return [
            Finding(
                name=c.get("name", "unknown"),
                status=c.get("status", "fail"),
                source="quick_scan",
                score=c.get("score"),
                summary=c.get("summary", ""),
                details=c.get("details", {}) or {},
            )
            for c in data["checks"]
        ]
    if isinstance(data, list):
        return [
            Finding(
                name=item.get("name", "unknown"),
                status=item.get("status", "fail"),
                source=item.get("source", "json"),
                score=item.get("score"),
                summary=item.get("summary", ""),
                details=item.get("details", {}) or {},
            )
            for item in data
        ]
    raise ValueError(
        "Unrecognized JSON shape — expected quick_scan dict or list of findings."
    )


def _run_quick_scan_findings() -> List[Finding]:
    """Run quick_scan in-process and convert results to Findings."""
    try:
        from .quick_scan import QuickScanner
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            f"Could not import quick_scan ({exc}). Use --demo or --from-json."
        )
    scanner = QuickScanner()
    result = scanner.run()
    return [
        Finding(
            name=c.name,
            status=c.status,
            source="quick_scan",
            score=c.score,
            summary=c.summary,
            details=c.details or {},
        )
        for c in result.checks
    ]


def main(argv: Optional[List[str]] = None) -> int:
    _ensure_utf8()
    parser = argparse.ArgumentParser(
        prog="replication plan",
        description=(
            "Synthesize a prioritized, dependency-aware remediation plan "
            "from safety findings."
        ),
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "--from-quick-scan",
        action="store_true",
        help="Run quick_scan now and plan from its results.",
    )
    src.add_argument(
        "--from-json",
        metavar="PATH",
        help="Load findings from a JSON file (quick_scan output or finding list).",
    )
    src.add_argument(
        "--demo",
        action="store_true",
        help="Plan against a synthetic demo set of findings.",
    )

    parser.add_argument(
        "--format", "-f",
        choices=["text", "md", "json"],
        default="text",
        help="Output format (default: text).",
    )
    parser.add_argument(
        "--output", "-o",
        metavar="PATH",
        help="Write output to PATH instead of stdout.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Show only the top-N actions (after ordering).",
    )
    parser.add_argument(
        "--quick-wins-only",
        action="store_true",
        help="Only show actions classified as quick wins.",
    )

    args = parser.parse_args(argv)

    if not (args.from_quick_scan or args.from_json or args.demo):
        # Default to demo if nothing specified - keeps the CLI friendly.
        args.demo = True

    if args.demo:
        findings = _demo_findings()
    elif args.from_json:
        findings = _load_findings_from_json(args.from_json)
    else:
        findings = _run_quick_scan_findings()

    planner = RemediationPlanner(
        max_actions=args.top,
        quick_wins_only=args.quick_wins_only,
    )
    plan = planner.plan_from_findings(findings)

    if args.format == "json":
        out = plan.to_json()
    elif args.format == "md":
        out = plan.to_markdown()
    else:
        out = plan.to_text()

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out)
        print(f"Wrote {args.format} plan to {args.output}")
    else:
        print(out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
