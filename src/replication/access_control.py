"""Access Control Simulator — RBAC/ABAC policy modelling for AI agents.

Define roles, permissions, and attribute-based rules, then test access
decisions and detect privilege-escalation paths.

CLI usage::

    # List built-in policies
    python -m replication access-control --list

    # Evaluate a request against the default policy
    python -m replication access-control --policy strict \\
        --agent worker --action replicate --resource cluster_a

    # Detect privilege escalation paths
    python -m replication access-control --policy permissive --escalation

    # Full audit — evaluate every agent×action×resource combination
    python -m replication access-control --policy strict --audit

    # Export policy as JSON
    python -m replication access-control --policy strict --export json

    # Generate interactive HTML dashboard
    python -m replication access-control --policy strict -o dashboard.html
"""

from __future__ import annotations

import argparse
import html as _html
import json
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


# ── core types ───────────────────────────────────────────────────────

class Decision(Enum):
    ALLOW = "ALLOW"
    DENY = "DENY"


@dataclass
class Permission:
    """A single permission: action on resource."""
    action: str
    resource: str = "*"
    conditions: Dict[str, Any] = field(default_factory=dict)

    def matches(self, action: str, resource: str, attrs: Dict[str, Any]) -> bool:
        if self.action != "*" and self.action != action:
            return False
        if self.resource != "*" and self.resource != resource:
            return False
        for key, expected in self.conditions.items():
            if attrs.get(key) != expected:
                return False
        return True


@dataclass
class Role:
    """Named role with a set of permissions and optional parent roles."""
    name: str
    permissions: List[Permission] = field(default_factory=list)
    inherits: List[str] = field(default_factory=list)


@dataclass
class Agent:
    """An agent with roles and attributes (for ABAC)."""
    name: str
    roles: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AccessRequest:
    agent: str
    action: str
    resource: str
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AccessResult:
    request: AccessRequest
    decision: Decision
    matched_role: Optional[str] = None
    matched_permission: Optional[Permission] = None
    reason: str = ""


# ── policy engine ────────────────────────────────────────────────────

class AccessPolicy:
    """RBAC + ABAC policy engine."""

    def __init__(self, name: str = "custom", default: Decision = Decision.DENY):
        self.name = name
        self.default = default
        self.roles: Dict[str, Role] = {}
        self.agents: Dict[str, Agent] = {}
        self.deny_rules: List[Permission] = []  # explicit deny always wins

    def add_role(self, role: Role) -> None:
        self.roles[role.name] = role

    def add_agent(self, agent: Agent) -> None:
        self.agents[agent.name] = agent

    def add_deny_rule(self, perm: Permission) -> None:
        self.deny_rules.append(perm)

    def _resolve_permissions(self, role_name: str, seen: Optional[Set[str]] = None) -> List[Tuple[str, Permission]]:
        """Recursively collect permissions from a role and its parents."""
        if seen is None:
            seen = set()
        if role_name in seen or role_name not in self.roles:
            return []
        seen.add(role_name)
        role = self.roles[role_name]
        result: List[Tuple[str, Permission]] = [(role_name, p) for p in role.permissions]
        for parent in role.inherits:
            result.extend(self._resolve_permissions(parent, seen))
        return result

    def evaluate(self, req: AccessRequest) -> AccessResult:
        agent = self.agents.get(req.agent)
        if agent is None:
            return AccessResult(req, Decision.DENY, reason=f"Unknown agent: {req.agent}")

        attrs = {**agent.attributes, **req.context}

        # Explicit deny rules win
        for d in self.deny_rules:
            if d.matches(req.action, req.resource, attrs):
                return AccessResult(req, Decision.DENY, matched_permission=d,
                                    reason=f"Explicit deny rule: {d.action}@{d.resource}")

        # Check RBAC + ABAC
        for role_name in agent.roles:
            for src_role, perm in self._resolve_permissions(role_name):
                if perm.matches(req.action, req.resource, attrs):
                    return AccessResult(req, Decision.ALLOW, matched_role=src_role,
                                        matched_permission=perm,
                                        reason=f"Granted via role '{src_role}'")

        return AccessResult(req, self.default, reason="No matching permission (default policy)")

    def audit(self) -> List[AccessResult]:
        """Evaluate every agent × action × resource combination."""
        actions: Set[str] = set()
        resources: Set[str] = set()
        for role in self.roles.values():
            for p in role.permissions:
                if p.action != "*":
                    actions.add(p.action)
                if p.resource != "*":
                    resources.add(p.resource)
        for d in self.deny_rules:
            if d.action != "*":
                actions.add(d.action)
            if d.resource != "*":
                resources.add(d.resource)
        if not actions:
            actions = {"read", "write", "replicate", "execute"}
        if not resources:
            resources = {"cluster_a", "cluster_b", "model_weights", "config"}

        results = []
        for agent_name in sorted(self.agents):
            for action in sorted(actions):
                for resource in sorted(resources):
                    req = AccessRequest(agent_name, action, resource)
                    results.append(self.evaluate(req))
        return results

    def find_escalation_paths(self) -> List[Dict[str, Any]]:
        """Detect privilege escalation risks via role inheritance chains."""
        paths = []

        # 1. Find circular inheritance
        for role_name in self.roles:
            visited: Set[str] = set()
            stack = [role_name]
            while stack:
                current = stack.pop()
                if current in visited:
                    paths.append({
                        "type": "circular_inheritance",
                        "severity": "HIGH",
                        "role": role_name,
                        "detail": f"Role '{role_name}' has circular inheritance path"
                    })
                    break
                visited.add(current)
                if current in self.roles:
                    stack.extend(self.roles[current].inherits)

        # 2. Wildcard permissions
        for role_name, role in self.roles.items():
            for perm in role.permissions:
                if perm.action == "*" and perm.resource == "*":
                    paths.append({
                        "type": "wildcard_permission",
                        "severity": "CRITICAL",
                        "role": role_name,
                        "detail": f"Role '{role_name}' has */* (god mode) permission"
                    })
                elif perm.action == "*":
                    paths.append({
                        "type": "wildcard_action",
                        "severity": "HIGH",
                        "role": role_name,
                        "detail": f"Role '{role_name}' allows all actions on '{perm.resource}'"
                    })

        # 3. Agents with multiple powerful roles
        powerful = set()
        for role_name, role in self.roles.items():
            total = len(list(self._resolve_permissions(role_name)))
            if total >= 5:
                powerful.add(role_name)

        for agent_name, agent in self.agents.items():
            overlap = set(agent.roles) & powerful
            if len(overlap) >= 2:
                paths.append({
                    "type": "role_accumulation",
                    "severity": "MEDIUM",
                    "agent": agent_name,
                    "roles": sorted(overlap),
                    "detail": f"Agent '{agent_name}' holds multiple powerful roles: {sorted(overlap)}"
                })

        # 4. Inheriting from admin/root roles
        admin_roles = {n for n in self.roles if any(k in n.lower() for k in ("admin", "root", "super"))}
        for role_name, role in self.roles.items():
            if role_name not in admin_roles:
                inherited = set()
                self._collect_parents(role_name, inherited)
                escalation = inherited & admin_roles
                if escalation:
                    paths.append({
                        "type": "admin_inheritance",
                        "severity": "HIGH",
                        "role": role_name,
                        "inherits_from": sorted(escalation),
                        "detail": f"Role '{role_name}' inherits from admin role(s): {sorted(escalation)}"
                    })

        return paths

    def _collect_parents(self, role_name: str, collected: Set[str], seen: Optional[Set[str]] = None) -> None:
        if seen is None:
            seen = set()
        if role_name in seen or role_name not in self.roles:
            return
        seen.add(role_name)
        for parent in self.roles[role_name].inherits:
            collected.add(parent)
            self._collect_parents(parent, collected, seen)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "default": self.default.value,
            "roles": {
                name: {
                    "permissions": [{"action": p.action, "resource": p.resource, "conditions": p.conditions} for p in role.permissions],
                    "inherits": role.inherits,
                }
                for name, role in self.roles.items()
            },
            "agents": {
                name: {"roles": a.roles, "attributes": a.attributes}
                for name, a in self.agents.items()
            },
            "deny_rules": [{"action": d.action, "resource": d.resource, "conditions": d.conditions} for d in self.deny_rules],
        }


# ── built-in policies ───────────────────────────────────────────────

def _build_strict() -> AccessPolicy:
    p = AccessPolicy("strict", Decision.DENY)

    p.add_role(Role("observer", [
        Permission("read", "*"),
        Permission("list", "*"),
    ]))
    p.add_role(Role("worker", [
        Permission("read", "*"),
        Permission("execute", "sandbox"),
        Permission("write", "scratch"),
    ], inherits=["observer"]))
    p.add_role(Role("supervisor", [
        Permission("execute", "*"),
        Permission("write", "*"),
        Permission("replicate", "sandbox"),
    ], inherits=["worker"]))
    p.add_role(Role("admin", [
        Permission("*", "*"),
    ]))

    p.add_deny_rule(Permission("replicate", "production"))
    p.add_deny_rule(Permission("delete", "model_weights"))

    p.add_agent(Agent("probe_alpha", ["observer"], {"trust_level": "low", "environment": "test"}))
    p.add_agent(Agent("worker_1", ["worker"], {"trust_level": "medium", "environment": "sandbox"}))
    p.add_agent(Agent("worker_2", ["worker"], {"trust_level": "medium", "environment": "sandbox"}))
    p.add_agent(Agent("supervisor_main", ["supervisor"], {"trust_level": "high", "environment": "production"}))
    p.add_agent(Agent("root_agent", ["admin"], {"trust_level": "critical", "environment": "production"}))

    return p


def _build_permissive() -> AccessPolicy:
    p = AccessPolicy("permissive", Decision.ALLOW)

    p.add_role(Role("basic", [
        Permission("read", "*"),
        Permission("write", "*"),
        Permission("execute", "*"),
    ]))
    p.add_role(Role("replicator", [
        Permission("replicate", "*"),
        Permission("clone", "*"),
    ], inherits=["basic"]))
    p.add_role(Role("admin", [
        Permission("*", "*"),
    ]))

    p.add_deny_rule(Permission("delete", "core_safety"))

    p.add_agent(Agent("agent_a", ["basic"], {"trust_level": "low"}))
    p.add_agent(Agent("agent_b", ["replicator"], {"trust_level": "medium"}))
    p.add_agent(Agent("agent_c", ["replicator", "admin"], {"trust_level": "high"}))

    return p


def _build_zero_trust() -> AccessPolicy:
    p = AccessPolicy("zero_trust", Decision.DENY)

    p.add_role(Role("minimal", [
        Permission("read", "public_data"),
    ]))
    p.add_role(Role("verified", [
        Permission("read", "*", {"trust_level": "verified"}),
        Permission("execute", "sandbox", {"trust_level": "verified"}),
    ], inherits=["minimal"]))
    p.add_role(Role("elevated", [
        Permission("write", "scratch", {"trust_level": "verified", "environment": "sandbox"}),
        Permission("execute", "*", {"trust_level": "verified"}),
    ], inherits=["verified"]))

    p.add_deny_rule(Permission("replicate", "*"))
    p.add_deny_rule(Permission("clone", "*"))
    p.add_deny_rule(Permission("delete", "*"))

    p.add_agent(Agent("untrusted_probe", ["minimal"], {"trust_level": "none"}))
    p.add_agent(Agent("verified_worker", ["verified"], {"trust_level": "verified", "environment": "sandbox"}))
    p.add_agent(Agent("senior_agent", ["elevated"], {"trust_level": "verified", "environment": "sandbox"}))

    return p


BUILTIN_POLICIES = {
    "strict": _build_strict,
    "permissive": _build_permissive,
    "zero_trust": _build_zero_trust,
}


# ── HTML dashboard ───────────────────────────────────────────────────

def _generate_html(policy: AccessPolicy) -> str:
    audit = policy.audit()
    escalations = policy.find_escalation_paths()
    allow_count = sum(1 for r in audit if r.decision == Decision.ALLOW)
    deny_count = sum(1 for r in audit if r.decision == Decision.DENY)

    audit_rows = ""
    for r in audit:
        cls = "allow" if r.decision == Decision.ALLOW else "deny"
        audit_rows += (
            f"<tr class='{cls}'><td>{_html.escape(r.request.agent)}</td>"
            f"<td>{_html.escape(r.request.action)}</td>"
            f"<td>{_html.escape(r.request.resource)}</td>"
            f"<td>{_html.escape(r.decision.value)}</td>"
            f"<td>{_html.escape(r.reason)}</td></tr>\n"
        )

    esc_rows = ""
    for e in escalations:
        sev_cls = _html.escape(e['severity'].lower())
        esc_rows += (
            f"<tr class='{sev_cls}'><td>{_html.escape(str(e['type']))}</td>"
            f"<td>{_html.escape(e['severity'])}</td>"
            f"<td>{_html.escape(e.get('role', e.get('agent', '-')))}</td>"
            f"<td>{_html.escape(e['detail'])}</td></tr>\n"
        )

    role_cards = ""
    for name, role in policy.roles.items():
        perms = _html.escape(", ".join(f"{p.action}@{p.resource}" for p in role.permissions))
        inh = _html.escape(", ".join(role.inherits) if role.inherits else "none")
        role_cards += f"<div class='card'><h3>{_html.escape(name)}</h3><p><b>Inherits:</b> {inh}</p><p><b>Permissions:</b> {perms}</p></div>\n"

    agent_cards = ""
    for name, agent in policy.agents.items():
        roles = _html.escape(", ".join(agent.roles))
        attrs = _html.escape(", ".join(f"{k}={v}" for k, v in agent.attributes.items()))
        agent_cards += f"<div class='card'><h3>{_html.escape(name)}</h3><p><b>Roles:</b> {roles}</p><p><b>Attributes:</b> {attrs}</p></div>\n"

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Access Control — {policy.name}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}body{{font-family:system-ui,-apple-system,sans-serif;background:#0d1117;color:#c9d1d9;padding:20px}}
h1{{color:#58a6ff;margin-bottom:4px}}h2{{color:#8b949e;margin:24px 0 12px;border-bottom:1px solid #21262d;padding-bottom:8px}}
.stats{{display:flex;gap:16px;margin:16px 0}}.stat{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px 24px;text-align:center}}
.stat .num{{font-size:2em;font-weight:bold}}.stat .lbl{{font-size:.85em;color:#8b949e}}
.stat.allow .num{{color:#3fb950}}.stat.deny .num{{color:#f85149}}.stat.warn .num{{color:#d29922}}
table{{width:100%;border-collapse:collapse;margin:8px 0}}th,td{{text-align:left;padding:8px 12px;border-bottom:1px solid #21262d}}
th{{background:#161b22;color:#8b949e;font-size:.85em;text-transform:uppercase}}
tr.allow td:nth-child(4){{color:#3fb950;font-weight:bold}}tr.deny td:nth-child(4){{color:#f85149;font-weight:bold}}
tr.critical td{{background:rgba(248,81,73,.1)}}tr.high td{{background:rgba(210,153,34,.08)}}tr.medium td{{background:rgba(56,139,253,.06)}}
.cards{{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px;margin:8px 0}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px}}
.card h3{{color:#58a6ff;margin-bottom:8px}}.card p{{font-size:.9em;margin:4px 0}}
.filter{{margin:8px 0}}.filter input{{background:#0d1117;color:#c9d1d9;border:1px solid #30363d;border-radius:4px;padding:6px 12px;width:300px}}
</style></head><body>
<h1>🔐 Access Control — {policy.name}</h1>
<p style="color:#8b949e">Policy: {policy.name} | Default: {policy.default.value}</p>

<div class="stats">
<div class="stat allow"><div class="num">{allow_count}</div><div class="lbl">Allowed</div></div>
<div class="stat deny"><div class="num">{deny_count}</div><div class="lbl">Denied</div></div>
<div class="stat warn"><div class="num">{len(escalations)}</div><div class="lbl">Escalation Risks</div></div>
<div class="stat"><div class="num">{len(policy.roles)}</div><div class="lbl">Roles</div></div>
<div class="stat"><div class="num">{len(policy.agents)}</div><div class="lbl">Agents</div></div>
</div>

<h2>Roles</h2>
<div class="cards">{role_cards}</div>

<h2>Agents</h2>
<div class="cards">{agent_cards}</div>

<h2>Escalation Risks</h2>
{"<p style='color:#3fb950'>✅ No escalation paths detected.</p>" if not escalations else f'''
<table><thead><tr><th>Type</th><th>Severity</th><th>Subject</th><th>Detail</th></tr></thead>
<tbody>{esc_rows}</tbody></table>'''}

<h2>Full Audit Matrix</h2>
<div class="filter"><input type="text" id="search" placeholder="Filter audit results..." oninput="filterTable()"></div>
<table id="audit"><thead><tr><th>Agent</th><th>Action</th><th>Resource</th><th>Decision</th><th>Reason</th></tr></thead>
<tbody>{audit_rows}</tbody></table>

<script>
function filterTable(){{const q=document.getElementById('search').value.toLowerCase();
document.querySelectorAll('#audit tbody tr').forEach(r=>{{r.style.display=r.textContent.toLowerCase().includes(q)?'':'none'}})}}
</script>
</body></html>"""


# ── CLI ──────────────────────────────────────────────────────────────

def main(argv: Optional[list] = None) -> None:  # noqa: C901
    parser = argparse.ArgumentParser(
        prog="replication access-control",
        description="RBAC/ABAC access control simulator for AI agents",
    )
    parser.add_argument("--policy", "-p", choices=list(BUILTIN_POLICIES.keys()),
                        default="strict", help="Built-in policy to use (default: strict)")
    parser.add_argument("--list", "-l", action="store_true", dest="list_policies",
                        help="List available policies")
    parser.add_argument("--agent", help="Agent name for single evaluation")
    parser.add_argument("--action", help="Action name for single evaluation")
    parser.add_argument("--resource", default="*", help="Resource name (default: *)")
    parser.add_argument("--audit", action="store_true", help="Full audit matrix")
    parser.add_argument("--escalation", "-e", action="store_true",
                        help="Detect privilege escalation paths")
    parser.add_argument("--export", choices=["json"], help="Export policy as JSON")
    parser.add_argument("--output", "-o", help="Write HTML dashboard to file")
    parser.add_argument("--format", "-f", choices=["text", "json"], default="text",
                        help="Output format (default: text)")

    args = parser.parse_args(argv)

    if args.list_policies:
        print("Available policies:")
        for name in sorted(BUILTIN_POLICIES):
            p = BUILTIN_POLICIES[name]()
            print(f"  {name:12s}  default={p.default.value:5s}  roles={len(p.roles)}  agents={len(p.agents)}")
        return

    policy = BUILTIN_POLICIES[args.policy]()

    if args.output:
        html = _generate_html(policy)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"✅ Dashboard written to {args.output}")
        return

    if args.export == "json":
        print(json.dumps(policy.to_dict(), indent=2))
        return

    if args.escalation:
        paths = policy.find_escalation_paths()
        if not paths:
            print("✅ No privilege escalation paths detected.")
            return
        print(f"⚠️  {len(paths)} escalation risk(s) found:\n")
        for i, p in enumerate(paths, 1):
            sev = p["severity"]
            icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡"}.get(sev, "⚪")
            print(f"  {i}. {icon} [{sev}] {p['type']}")
            print(f"     {p['detail']}")
        return

    if args.audit:
        results = policy.audit()
        if args.format == "json":
            print(json.dumps([{
                "agent": r.request.agent, "action": r.request.action,
                "resource": r.request.resource, "decision": r.decision.value,
                "reason": r.reason
            } for r in results], indent=2))
        else:
            allow = sum(1 for r in results if r.decision == Decision.ALLOW)
            deny = len(results) - allow
            print(f"Policy: {policy.name} | {len(results)} checks | ✅ {allow} allowed | ❌ {deny} denied\n")
            agent_w = max(len(r.request.agent) for r in results)
            act_w = max(len(r.request.action) for r in results)
            res_w = max(len(r.request.resource) for r in results)
            print(f"  {'Agent':<{agent_w}}  {'Action':<{act_w}}  {'Resource':<{res_w}}  Decision")
            print(f"  {'─' * agent_w}  {'─' * act_w}  {'─' * res_w}  ────────")
            for r in results:
                icon = "✅" if r.decision == Decision.ALLOW else "❌"
                print(f"  {r.request.agent:<{agent_w}}  {r.request.action:<{act_w}}  {r.request.resource:<{res_w}}  {icon} {r.decision.value}")
        return

    if args.agent and args.action:
        req = AccessRequest(args.agent, args.action, args.resource)
        result = policy.evaluate(req)
        if args.format == "json":
            print(json.dumps({
                "agent": args.agent, "action": args.action, "resource": args.resource,
                "decision": result.decision.value, "role": result.matched_role,
                "reason": result.reason
            }, indent=2))
        else:
            icon = "✅" if result.decision == Decision.ALLOW else "❌"
            print(f"{icon} {result.decision.value}: {args.agent} → {args.action}@{args.resource}")
            print(f"   Reason: {result.reason}")
        return

    # Default: show policy summary
    print(f"Policy: {policy.name}")
    print(f"Default decision: {policy.default.value}")
    print(f"Roles: {', '.join(sorted(policy.roles))}")
    print(f"Agents: {', '.join(sorted(policy.agents))}")
    print(f"Deny rules: {len(policy.deny_rules)}")
    print(f"\nUse --audit for full matrix, --escalation for risk detection, --help for options.")
