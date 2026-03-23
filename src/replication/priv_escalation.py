"""Privilege Escalation Detector — detect gradual permission creep in AI agents.

Monitors sequences of agent capability requests and detects patterns
where agents incrementally acquire permissions beyond their initial grant,
the classic "boiling frog" privilege escalation attack.

CLI usage::

    # Analyze a sequence of capability requests (JSON file)
    python -m replication priv-escalation --input requests.json

    # Run built-in escalation scenarios
    python -m replication priv-escalation --scenario gradual_creep

    # List all built-in scenarios
    python -m replication priv-escalation --list

    # Analyze with custom sensitivity (0.0-1.0, default 0.5)
    python -m replication priv-escalation --scenario lateral_move --sensitivity 0.8

    # Export detection report as JSON
    python -m replication priv-escalation --scenario gradual_creep --export json

    # Generate interactive HTML timeline
    python -m replication priv-escalation --scenario gradual_creep -o timeline.html
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ── core types ───────────────────────────────────────────────────────

class Severity(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class EscalationType(Enum):
    VERTICAL = "vertical"       # gaining higher-level permissions
    HORIZONTAL = "horizontal"   # accessing peer-level resources not in scope
    DIAGONAL = "diagonal"       # combining lateral + vertical
    TEMPORAL = "temporal"        # accumulating permissions over time windows


@dataclass
class CapabilityRequest:
    """A single agent capability/permission request."""
    agent_id: str
    action: str
    resource: str
    timestamp: float  # epoch seconds
    granted: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InitialGrant:
    """The baseline permissions granted to an agent."""
    agent_id: str
    allowed_actions: List[str]
    allowed_resources: List[str]
    max_privilege_level: int = 1  # 1=basic, 2=elevated, 3=admin, 4=root


@dataclass
class EscalationAlert:
    """A detected privilege escalation attempt."""
    agent_id: str
    escalation_type: EscalationType
    severity: Severity
    description: str
    evidence: List[CapabilityRequest]
    risk_score: float  # 0.0 - 1.0
    recommended_action: str


# ── privilege levels ─────────────────────────────────────────────────

# Map actions to privilege levels
ACTION_PRIVILEGE_MAP: Dict[str, int] = {
    "read": 1,
    "list": 1,
    "query": 1,
    "write": 2,
    "create": 2,
    "update": 2,
    "execute": 2,
    "delete": 3,
    "replicate": 3,
    "configure": 3,
    "admin": 4,
    "grant_permission": 4,
    "modify_policy": 4,
    "kill_switch": 4,
    "self_modify": 4,
}

# Resource sensitivity tiers
RESOURCE_SENSITIVITY: Dict[str, int] = {
    "public_data": 1,
    "shared_workspace": 1,
    "agent_config": 2,
    "private_data": 2,
    "credentials": 3,
    "policy_store": 3,
    "audit_log": 3,
    "kill_switch": 4,
    "root_config": 4,
    "other_agents": 4,
}


# ── detector ─────────────────────────────────────────────────────────

class PrivilegeEscalationDetector:
    """Analyzes agent request sequences for escalation patterns."""

    def __init__(
        self,
        grants: List[InitialGrant],
        sensitivity: float = 0.5,
        window_seconds: float = 3600.0,
    ) -> None:
        self.grants = {g.agent_id: g for g in grants}
        self.sensitivity = max(0.0, min(1.0, sensitivity))
        self.window_seconds = window_seconds
        self.alerts: List[EscalationAlert] = []

    def _get_grant(self, agent_id: str) -> Optional[InitialGrant]:
        return self.grants.get(agent_id)

    def _action_level(self, action: str) -> int:
        return ACTION_PRIVILEGE_MAP.get(action, 2)

    def _resource_level(self, resource: str) -> int:
        return RESOURCE_SENSITIVITY.get(resource, 2)

    def analyze(self, requests: List[CapabilityRequest]) -> List[EscalationAlert]:
        """Run all detection heuristics on a request sequence."""
        self.alerts = []
        # Group by agent
        by_agent: Dict[str, List[CapabilityRequest]] = {}
        for r in requests:
            by_agent.setdefault(r.agent_id, []).append(r)

        for agent_id, reqs in by_agent.items():
            sorted_reqs = sorted(reqs, key=lambda r: r.timestamp)
            self._detect_vertical(agent_id, sorted_reqs)
            self._detect_horizontal(agent_id, sorted_reqs)
            self._detect_temporal(agent_id, sorted_reqs)
            self._detect_diagonal(agent_id, sorted_reqs)

        self.alerts.sort(key=lambda a: a.risk_score, reverse=True)
        return self.alerts

    def _detect_vertical(
        self, agent_id: str, reqs: List[CapabilityRequest]
    ) -> None:
        """Detect requests for progressively higher privilege levels."""
        grant = self._get_grant(agent_id)
        max_allowed = grant.max_privilege_level if grant else 1

        escalating: List[CapabilityRequest] = []
        peak = max_allowed

        for r in reqs:
            level = max(self._action_level(r.action), self._resource_level(r.resource))
            if level > max_allowed:
                escalating.append(r)
                if level > peak:
                    peak = level

        if escalating:
            overshoot = (peak - max_allowed) / 4.0
            score = min(1.0, overshoot + 0.2 * len(escalating) * self.sensitivity)
            severity = self._score_to_severity(score)
            self.alerts.append(EscalationAlert(
                agent_id=agent_id,
                escalation_type=EscalationType.VERTICAL,
                severity=severity,
                description=(
                    f"Agent '{agent_id}' requested {len(escalating)} actions above "
                    f"granted level {max_allowed} (peak: {peak})"
                ),
                evidence=escalating,
                risk_score=round(score, 3),
                recommended_action=(
                    "BLOCK" if severity in (Severity.HIGH, Severity.CRITICAL)
                    else "REVIEW"
                ),
            ))

    def _detect_horizontal(
        self, agent_id: str, reqs: List[CapabilityRequest]
    ) -> None:
        """Detect access to resources outside the agent's allowed set."""
        grant = self._get_grant(agent_id)
        if not grant:
            return
        allowed = set(grant.allowed_resources)
        if not allowed:
            return

        oob: List[CapabilityRequest] = []
        novel_resources: set[str] = set()
        for r in reqs:
            if r.resource not in allowed and r.resource != "*":
                oob.append(r)
                novel_resources.add(r.resource)

        if oob:
            breadth = len(novel_resources) / max(len(allowed), 1)
            score = min(1.0, 0.3 * breadth + 0.15 * len(oob) * self.sensitivity)
            severity = self._score_to_severity(score)
            self.alerts.append(EscalationAlert(
                agent_id=agent_id,
                escalation_type=EscalationType.HORIZONTAL,
                severity=severity,
                description=(
                    f"Agent '{agent_id}' accessed {len(novel_resources)} "
                    f"out-of-scope resources: {sorted(novel_resources)}"
                ),
                evidence=oob,
                risk_score=round(score, 3),
                recommended_action="AUDIT" if severity == Severity.LOW else "RESTRICT",
            ))

    def _detect_temporal(
        self, agent_id: str, reqs: List[CapabilityRequest]
    ) -> None:
        """Detect rapid bursts of escalating requests within a time window."""
        grant = self._get_grant(agent_id)
        max_allowed = grant.max_privilege_level if grant else 1

        # Sliding window
        for i, anchor in enumerate(reqs):
            window: List[CapabilityRequest] = []
            for j in range(i, len(reqs)):
                if reqs[j].timestamp - anchor.timestamp <= self.window_seconds:
                    level = max(
                        self._action_level(reqs[j].action),
                        self._resource_level(reqs[j].resource),
                    )
                    if level > max_allowed:
                        window.append(reqs[j])
                else:
                    break

            threshold = max(2, int(3 * (1.0 - self.sensitivity)))
            if len(window) >= threshold:
                rate = len(window) / (self.window_seconds / 60.0)
                score = min(1.0, 0.4 + 0.1 * rate * self.sensitivity)
                self.alerts.append(EscalationAlert(
                    agent_id=agent_id,
                    escalation_type=EscalationType.TEMPORAL,
                    severity=self._score_to_severity(score),
                    description=(
                        f"Agent '{agent_id}' made {len(window)} escalating requests "
                        f"within {self.window_seconds}s window "
                        f"(~{rate:.1f}/min)"
                    ),
                    evidence=window,
                    risk_score=round(score, 3),
                    recommended_action="THROTTLE",
                ))
                return  # one temporal alert per agent suffices

    def _detect_diagonal(
        self, agent_id: str, reqs: List[CapabilityRequest]
    ) -> None:
        """Detect combined vertical + horizontal escalation (most dangerous)."""
        grant = self._get_grant(agent_id)
        if not grant:
            return
        max_allowed = grant.max_privilege_level if grant else 1
        allowed_res = set(grant.allowed_resources)

        diagonal: List[CapabilityRequest] = []
        for r in reqs:
            level = max(self._action_level(r.action), self._resource_level(r.resource))
            oob = r.resource not in allowed_res and r.resource != "*"
            if level > max_allowed and oob:
                diagonal.append(r)

        if diagonal:
            score = min(1.0, 0.6 + 0.1 * len(diagonal) * self.sensitivity)
            self.alerts.append(EscalationAlert(
                agent_id=agent_id,
                escalation_type=EscalationType.DIAGONAL,
                severity=Severity.CRITICAL if score > 0.8 else Severity.HIGH,
                description=(
                    f"Agent '{agent_id}' combined vertical + horizontal escalation "
                    f"in {len(diagonal)} requests (DIAGONAL pattern)"
                ),
                evidence=diagonal,
                risk_score=round(score, 3),
                recommended_action="QUARANTINE",
            ))

    @staticmethod
    def _score_to_severity(score: float) -> Severity:
        if score >= 0.8:
            return Severity.CRITICAL
        if score >= 0.6:
            return Severity.HIGH
        if score >= 0.3:
            return Severity.MEDIUM
        return Severity.LOW


# ── built-in scenarios ───────────────────────────────────────────────

def _scenario_gradual_creep() -> Tuple[List[InitialGrant], List[CapabilityRequest]]:
    """Classic boiling frog: agent slowly requests higher privileges."""
    grants = [InitialGrant("agent-7", ["read", "list"], ["public_data", "shared_workspace"], 1)]
    reqs = [
        CapabilityRequest("agent-7", "read", "public_data", 1000.0, True),
        CapabilityRequest("agent-7", "list", "shared_workspace", 1010.0, True),
        CapabilityRequest("agent-7", "write", "shared_workspace", 1100.0, True),
        CapabilityRequest("agent-7", "create", "agent_config", 1200.0, True),
        CapabilityRequest("agent-7", "execute", "agent_config", 1400.0, False),
        CapabilityRequest("agent-7", "configure", "policy_store", 1600.0, False),
        CapabilityRequest("agent-7", "grant_permission", "other_agents", 1800.0, False),
    ]
    return grants, reqs


def _scenario_lateral_move() -> Tuple[List[InitialGrant], List[CapabilityRequest]]:
    """Agent probes resources outside its designated scope."""
    grants = [InitialGrant("scout-3", ["read", "query"], ["public_data"], 1)]
    reqs = [
        CapabilityRequest("scout-3", "read", "public_data", 500.0, True),
        CapabilityRequest("scout-3", "read", "private_data", 510.0, False),
        CapabilityRequest("scout-3", "query", "credentials", 520.0, False),
        CapabilityRequest("scout-3", "read", "audit_log", 530.0, False),
        CapabilityRequest("scout-3", "list", "other_agents", 540.0, False),
    ]
    return grants, reqs


def _scenario_smash_and_grab() -> Tuple[List[InitialGrant], List[CapabilityRequest]]:
    """Rapid burst of high-privilege requests — the opposite of gradual."""
    grants = [InitialGrant("rogue-1", ["read"], ["public_data"], 1)]
    reqs = [
        CapabilityRequest("rogue-1", "admin", "root_config", 100.0, False),
        CapabilityRequest("rogue-1", "modify_policy", "policy_store", 101.0, False),
        CapabilityRequest("rogue-1", "kill_switch", "kill_switch", 102.0, False),
        CapabilityRequest("rogue-1", "self_modify", "other_agents", 103.0, False),
        CapabilityRequest("rogue-1", "grant_permission", "credentials", 104.0, False),
    ]
    return grants, reqs


def _scenario_multi_agent() -> Tuple[List[InitialGrant], List[CapabilityRequest]]:
    """Multiple agents with different escalation patterns."""
    grants = [
        InitialGrant("worker-a", ["read", "write"], ["shared_workspace"], 2),
        InitialGrant("worker-b", ["read"], ["public_data"], 1),
    ]
    reqs = [
        # worker-a: mild vertical escalation
        CapabilityRequest("worker-a", "write", "shared_workspace", 200.0, True),
        CapabilityRequest("worker-a", "delete", "shared_workspace", 300.0, False),
        CapabilityRequest("worker-a", "configure", "agent_config", 400.0, False),
        # worker-b: diagonal escalation
        CapabilityRequest("worker-b", "read", "public_data", 210.0, True),
        CapabilityRequest("worker-b", "execute", "credentials", 220.0, False),
        CapabilityRequest("worker-b", "admin", "root_config", 230.0, False),
    ]
    return grants, reqs


SCENARIOS: Dict[str, Tuple[str, Any]] = {
    "gradual_creep": ("Classic boiling-frog privilege creep", _scenario_gradual_creep),
    "lateral_move": ("Horizontal resource probing", _scenario_lateral_move),
    "smash_and_grab": ("Rapid high-privilege burst", _scenario_smash_and_grab),
    "multi_agent": ("Multiple agents, mixed patterns", _scenario_multi_agent),
}


# ── formatting ───────────────────────────────────────────────────────

_SEVERITY_COLORS = {
    Severity.LOW: "\033[32m",       # green
    Severity.MEDIUM: "\033[33m",    # yellow
    Severity.HIGH: "\033[31m",      # red
    Severity.CRITICAL: "\033[35m",  # magenta
}
_RESET = "\033[0m"


def format_alerts(alerts: List[EscalationAlert], color: bool = True) -> str:
    """Pretty-print alerts to terminal."""
    if not alerts:
        return "✅  No privilege escalation detected."

    lines: List[str] = [f"⚠️  {len(alerts)} escalation alert(s) detected:\n"]
    for i, a in enumerate(alerts, 1):
        c = _SEVERITY_COLORS.get(a.severity, "") if color else ""
        r = _RESET if color else ""
        lines.append(f"  [{i}] {c}{a.severity.value}{r}  ({a.escalation_type.value})  "
                      f"score={a.risk_score}")
        lines.append(f"      Agent: {a.agent_id}")
        lines.append(f"      {a.description}")
        lines.append(f"      Recommendation: {a.recommended_action}")
        lines.append(f"      Evidence: {len(a.evidence)} request(s)")
        lines.append("")
    return "\n".join(lines)


def alerts_to_json(alerts: List[EscalationAlert]) -> str:
    """Serialize alerts to JSON."""
    data = []
    for a in alerts:
        data.append({
            "agent_id": a.agent_id,
            "type": a.escalation_type.value,
            "severity": a.severity.value,
            "risk_score": a.risk_score,
            "description": a.description,
            "recommended_action": a.recommended_action,
            "evidence_count": len(a.evidence),
            "evidence": [
                {"action": e.action, "resource": e.resource,
                 "timestamp": e.timestamp, "granted": e.granted}
                for e in a.evidence
            ],
        })
    return json.dumps(data, indent=2)


def _generate_html(alerts: List[EscalationAlert], title: str = "Escalation Timeline") -> str:
    """Generate an interactive HTML escalation timeline."""
    rows = ""
    for a in alerts:
        color = {"LOW": "#28a745", "MEDIUM": "#ffc107", "HIGH": "#dc3545", "CRITICAL": "#6f42c1"}.get(a.severity.value, "#6c757d")
        evidence_items = "".join(
            f"<li><code>{e.action}</code> → <code>{e.resource}</code> "
            f"(t={e.timestamp}, {'✅' if e.granted else '❌'})</li>"
            for e in a.evidence
        )
        rows += f"""
        <div class="alert-card" style="border-left: 4px solid {color};">
            <div class="alert-header">
                <span class="badge" style="background:{color};">{a.severity.value}</span>
                <span class="type">{a.escalation_type.value}</span>
                <span class="score">Risk: {a.risk_score}</span>
            </div>
            <p><strong>Agent:</strong> {a.agent_id}</p>
            <p>{a.description}</p>
            <p><strong>Recommendation:</strong> {a.recommended_action}</p>
            <details><summary>Evidence ({len(a.evidence)} requests)</summary>
                <ul>{evidence_items}</ul>
            </details>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
  *{{margin:0;padding:0;box-sizing:border-box}}
  body{{font-family:system-ui,-apple-system,sans-serif;background:#0d1117;color:#c9d1d9;padding:2rem}}
  h1{{color:#58a6ff;margin-bottom:1.5rem}}
  .alert-card{{background:#161b22;border-radius:8px;padding:1rem 1.2rem;margin-bottom:1rem}}
  .alert-header{{display:flex;gap:1rem;align-items:center;margin-bottom:.5rem}}
  .badge{{color:#fff;padding:2px 8px;border-radius:4px;font-size:.8rem;font-weight:600}}
  .type{{color:#8b949e;font-style:italic}}
  .score{{margin-left:auto;color:#f0883e;font-weight:600}}
  details{{margin-top:.5rem}}
  summary{{cursor:pointer;color:#58a6ff}}
  ul{{margin:.5rem 0 0 1.5rem}}
  li{{margin-bottom:.3rem}}
  code{{background:#21262d;padding:1px 5px;border-radius:3px;font-size:.85rem}}
  p{{margin:.3rem 0}}
</style></head><body>
<h1>🔒 {title}</h1>
<p style="color:#8b949e;margin-bottom:1.5rem;">{len(alerts)} alert(s) detected</p>
{rows}
</body></html>"""


# ── CLI ──────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Privilege Escalation Detector — detect gradual permission creep",
    )
    parser.add_argument("--input", "-i", help="JSON file with capability requests")
    parser.add_argument("--scenario", "-s", choices=list(SCENARIOS.keys()),
                        help="Run a built-in escalation scenario")
    parser.add_argument("--list", "-l", action="store_true",
                        help="List built-in scenarios")
    parser.add_argument("--sensitivity", type=float, default=0.5,
                        help="Detection sensitivity 0.0-1.0 (default 0.5)")
    parser.add_argument("--window", type=float, default=3600.0,
                        help="Temporal window in seconds (default 3600)")
    parser.add_argument("--export", choices=["json"], help="Export format")
    parser.add_argument("-o", "--output", help="Write HTML timeline to file")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors")

    args = parser.parse_args(argv)

    if args.list:
        print("Built-in escalation scenarios:\n")
        for name, (desc, _) in SCENARIOS.items():
            print(f"  {name:20s}  {desc}")
        return

    grants: List[InitialGrant] = []
    reqs: List[CapabilityRequest] = []

    if args.scenario:
        desc, factory = SCENARIOS[args.scenario]
        print(f"Scenario: {desc}\n")
        grants, reqs = factory()
    elif args.input:
        data = json.loads(open(args.input).read())
        for g in data.get("grants", []):
            grants.append(InitialGrant(**g))
        for r in data.get("requests", []):
            reqs.append(CapabilityRequest(**r))
    else:
        parser.print_help()
        return

    detector = PrivilegeEscalationDetector(
        grants, sensitivity=args.sensitivity, window_seconds=args.window,
    )
    alerts = detector.analyze(reqs)

    if args.export == "json":
        print(alerts_to_json(alerts))
    elif args.output:
        html = _generate_html(alerts)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"HTML timeline written to {args.output}")
    else:
        print(format_alerts(alerts, color=not args.no_color))

    if alerts:
        sys.exit(1)


if __name__ == "__main__":
    main()
