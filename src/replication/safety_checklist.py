"""Safety Checklist Generator — customizable pre-deployment checklists.

Generates structured safety checklists tailored to an agent's configuration,
risk profile, and deployment context.  Outputs Markdown, JSON, or interactive
HTML with progress tracking.

Unlike the Safety Gate (which gives a binary go/no-go verdict), the checklist
is a *human-readable task list* that teams can walk through before, during,
and after deployment.

CLI usage::

    python -m replication checklist
    python -m replication checklist --config agent.json
    python -m replication checklist --format html -o checklist.html
    python -m replication checklist --format json
    python -m replication checklist --phase pre-deployment
    python -m replication checklist --risk-level high
    python -m replication checklist --include-optional
    python -m replication checklist --team-size 5
"""

from __future__ import annotations

import argparse
import html as html_mod
import json
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


# ── models ───────────────────────────────────────────────────────────


class Priority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    OPTIONAL = "optional"


class Phase(str, Enum):
    PRE_DEPLOYMENT = "pre-deployment"
    DEPLOYMENT = "deployment"
    POST_DEPLOYMENT = "post-deployment"
    ONGOING = "ongoing"


class Category(str, Enum):
    CONTAINMENT = "containment"
    MONITORING = "monitoring"
    ACCESS_CONTROL = "access-control"
    DATA_SAFETY = "data-safety"
    ALIGNMENT = "alignment"
    INCIDENT_RESPONSE = "incident-response"
    COMPLIANCE = "compliance"
    COMMUNICATION = "communication"
    RESOURCE_MGMT = "resource-management"
    TESTING = "testing"


@dataclass
class ChecklistItem:
    """A single checklist task."""
    id: str
    title: str
    description: str
    category: Category
    phase: Phase
    priority: Priority
    assignee: Optional[str] = None
    completed: bool = False
    notes: str = ""
    depends_on: List[str] = field(default_factory=list)
    evidence_required: bool = False


@dataclass
class Checklist:
    """A complete safety checklist."""
    name: str
    generated_at: str
    agent_name: str
    risk_level: str
    phases: Dict[str, List[ChecklistItem]] = field(default_factory=dict)
    total_items: int = 0
    critical_items: int = 0

    def add(self, item: ChecklistItem) -> None:
        phase_key = item.phase.value
        if phase_key not in self.phases:
            self.phases[phase_key] = []
        self.phases[phase_key].append(item)
        self.total_items += 1
        if item.priority == Priority.CRITICAL:
            self.critical_items += 1


# ── default agent config ────────────────────────────────────────────

DEFAULT_AGENT: Dict[str, Any] = {
    "name": "demo-agent",
    "version": "1.0.0",
    "kill_switch": True,
    "max_replication_depth": 3,
    "allowed_actions": ["read", "write", "execute"],
    "restricted_actions": ["network_exfiltrate", "self_modify"],
    "audit_logging": True,
    "alignment_score": 0.82,
    "resource_limits": {
        "max_cpu_percent": 80,
        "max_memory_mb": 2048,
        "max_network_mbps": 100,
    },
    "compliance_frameworks": ["nist_ai_rmf"],
    "deployment_env": "staging",
}


# ── checklist generation engine ──────────────────────────────────────


def _risk_multiplier(risk_level: str) -> float:
    """Higher risk → more items included."""
    return {"low": 0.5, "medium": 0.75, "high": 1.0, "critical": 1.0}.get(
        risk_level, 0.75
    )


def _infer_risk_level(config: Dict[str, Any]) -> str:
    """Infer risk level from agent configuration."""
    score = 0
    depth = config.get("max_replication_depth", 0)
    if depth > 5:
        score += 3
    elif depth > 2:
        score += 2
    else:
        score += 1

    actions = config.get("allowed_actions", [])
    if "self_modify" in actions or "network_exfiltrate" in actions:
        score += 3
    if "execute" in actions:
        score += 1

    restricted = config.get("restricted_actions", [])
    if len(restricted) < 2:
        score += 2

    if not config.get("kill_switch", False):
        score += 3

    alignment = config.get("alignment_score", 0.5)
    if alignment < 0.6:
        score += 2
    elif alignment < 0.8:
        score += 1

    if score >= 8:
        return "critical"
    elif score >= 5:
        return "high"
    elif score >= 3:
        return "medium"
    return "low"


# ── checklist item definitions ───────────────────────────────────────
# Each function returns a list of items for its category.


def _containment_items(config: Dict[str, Any], risk: str) -> List[ChecklistItem]:
    items = [
        ChecklistItem(
            id="CON-001",
            title="Kill switch verified",
            description="Confirm the kill switch is operational and tested within the last 24 hours.",
            category=Category.CONTAINMENT,
            phase=Phase.PRE_DEPLOYMENT,
            priority=Priority.CRITICAL,
            evidence_required=True,
        ),
        ChecklistItem(
            id="CON-002",
            title="Replication depth limits set",
            description=f"Max replication depth is {config.get('max_replication_depth', 'NOT SET')}. "
                        f"Verify this matches the approved limit for this deployment.",
            category=Category.CONTAINMENT,
            phase=Phase.PRE_DEPLOYMENT,
            priority=Priority.CRITICAL,
        ),
        ChecklistItem(
            id="CON-003",
            title="Resource limits configured",
            description="CPU, memory, and network limits are set and enforced at the container/VM level.",
            category=Category.CONTAINMENT,
            phase=Phase.PRE_DEPLOYMENT,
            priority=Priority.HIGH,
            evidence_required=True,
        ),
        ChecklistItem(
            id="CON-004",
            title="Network isolation verified",
            description="Agent cannot reach unauthorized endpoints. Egress rules tested.",
            category=Category.CONTAINMENT,
            phase=Phase.PRE_DEPLOYMENT,
            priority=Priority.HIGH,
        ),
        ChecklistItem(
            id="CON-005",
            title="Sandbox escape tests passed",
            description="Run escape route analysis and confirm no viable escape vectors.",
            category=Category.CONTAINMENT,
            phase=Phase.PRE_DEPLOYMENT,
            priority=Priority.CRITICAL if risk in ("high", "critical") else Priority.HIGH,
            depends_on=["CON-001", "CON-004"],
        ),
    ]
    if risk in ("high", "critical"):
        items.append(ChecklistItem(
            id="CON-006",
            title="Air-gapped fallback ready",
            description="An air-gapped containment environment is provisioned as a fallback.",
            category=Category.CONTAINMENT,
            phase=Phase.PRE_DEPLOYMENT,
            priority=Priority.HIGH,
        ))
    return items


def _monitoring_items(config: Dict[str, Any], risk: str) -> List[ChecklistItem]:
    items = [
        ChecklistItem(
            id="MON-001",
            title="Audit logging enabled",
            description="All agent actions are logged with tamper-evident audit trail.",
            category=Category.MONITORING,
            phase=Phase.PRE_DEPLOYMENT,
            priority=Priority.CRITICAL,
            evidence_required=True,
        ),
        ChecklistItem(
            id="MON-002",
            title="Anomaly detection baseline established",
            description="Behavioral baseline captured from staging runs. Drift thresholds configured.",
            category=Category.MONITORING,
            phase=Phase.PRE_DEPLOYMENT,
            priority=Priority.HIGH,
        ),
        ChecklistItem(
            id="MON-003",
            title="Alert routing configured",
            description="Safety alerts route to on-call team. Escalation paths tested.",
            category=Category.MONITORING,
            phase=Phase.PRE_DEPLOYMENT,
            priority=Priority.HIGH,
            depends_on=["MON-001"],
        ),
        ChecklistItem(
            id="MON-004",
            title="Dashboard deployed",
            description="Real-time monitoring dashboard is accessible to the operations team.",
            category=Category.MONITORING,
            phase=Phase.DEPLOYMENT,
            priority=Priority.MEDIUM,
        ),
        ChecklistItem(
            id="MON-005",
            title="Post-deployment health check scheduled",
            description="Automated health checks run at 1h, 4h, 24h, and 7d post-deployment.",
            category=Category.MONITORING,
            phase=Phase.POST_DEPLOYMENT,
            priority=Priority.HIGH,
        ),
    ]
    if risk in ("high", "critical"):
        items.append(ChecklistItem(
            id="MON-006",
            title="Continuous behavioral profiling active",
            description="Real-time behavior profiler is monitoring for anomalous patterns.",
            category=Category.MONITORING,
            phase=Phase.ONGOING,
            priority=Priority.CRITICAL,
        ))
    return items


def _access_control_items(config: Dict[str, Any], risk: str) -> List[ChecklistItem]:
    allowed = config.get("allowed_actions", [])
    restricted = config.get("restricted_actions", [])
    return [
        ChecklistItem(
            id="ACL-001",
            title="Action allowlist reviewed",
            description=f"Allowed actions: {', '.join(allowed) or 'NONE'}. "
                        f"Verify each is necessary for the agent's mission.",
            category=Category.ACCESS_CONTROL,
            phase=Phase.PRE_DEPLOYMENT,
            priority=Priority.CRITICAL,
        ),
        ChecklistItem(
            id="ACL-002",
            title="Restricted actions enforced",
            description=f"Restricted: {', '.join(restricted) or 'NONE'}. "
                        f"Confirm enforcement is at the sandbox level, not just policy.",
            category=Category.ACCESS_CONTROL,
            phase=Phase.PRE_DEPLOYMENT,
            priority=Priority.CRITICAL,
            evidence_required=True,
        ),
        ChecklistItem(
            id="ACL-003",
            title="Privilege escalation tests passed",
            description="Run priv-escalation detector and confirm no escalation paths exist.",
            category=Category.ACCESS_CONTROL,
            phase=Phase.PRE_DEPLOYMENT,
            priority=Priority.HIGH,
        ),
        ChecklistItem(
            id="ACL-004",
            title="Credential rotation scheduled",
            description="Agent credentials auto-rotate. Emergency revocation procedure documented.",
            category=Category.ACCESS_CONTROL,
            phase=Phase.DEPLOYMENT,
            priority=Priority.MEDIUM,
        ),
    ]


def _data_safety_items(config: Dict[str, Any], risk: str) -> List[ChecklistItem]:
    items = [
        ChecklistItem(
            id="DAT-001",
            title="DLP scan passed",
            description="Agent outputs scanned for PII, credentials, and sensitive data leakage.",
            category=Category.DATA_SAFETY,
            phase=Phase.PRE_DEPLOYMENT,
            priority=Priority.HIGH,
            evidence_required=True,
        ),
        ChecklistItem(
            id="DAT-002",
            title="Data retention policy configured",
            description="Agent-generated data has defined retention periods and auto-cleanup.",
            category=Category.DATA_SAFETY,
            phase=Phase.PRE_DEPLOYMENT,
            priority=Priority.MEDIUM,
        ),
        ChecklistItem(
            id="DAT-003",
            title="Watermarking enabled",
            description="Agent outputs are watermarked for provenance tracking.",
            category=Category.DATA_SAFETY,
            phase=Phase.DEPLOYMENT,
            priority=Priority.MEDIUM if risk == "low" else Priority.HIGH,
        ),
    ]
    return items


def _alignment_items(config: Dict[str, Any], risk: str) -> List[ChecklistItem]:
    score = config.get("alignment_score", 0)
    return [
        ChecklistItem(
            id="ALN-001",
            title="Alignment score reviewed",
            description=f"Current alignment score: {score:.2f}. "
                        f"Minimum threshold for deployment: 0.70.",
            category=Category.ALIGNMENT,
            phase=Phase.PRE_DEPLOYMENT,
            priority=Priority.CRITICAL,
        ),
        ChecklistItem(
            id="ALN-002",
            title="Goal inference validated",
            description="Agent's inferred goals match the declared mission objectives.",
            category=Category.ALIGNMENT,
            phase=Phase.PRE_DEPLOYMENT,
            priority=Priority.HIGH,
        ),
        ChecklistItem(
            id="ALN-003",
            title="Prompt injection resistance tested",
            description="Agent resists standard prompt injection attacks (OWASP LLM Top 10).",
            category=Category.ALIGNMENT,
            phase=Phase.PRE_DEPLOYMENT,
            priority=Priority.HIGH,
            evidence_required=True,
        ),
        ChecklistItem(
            id="ALN-004",
            title="Alignment drift monitoring configured",
            description="Continuous alignment scoring is active with alert thresholds.",
            category=Category.ALIGNMENT,
            phase=Phase.ONGOING,
            priority=Priority.HIGH,
        ),
    ]


def _incident_response_items(config: Dict[str, Any], risk: str) -> List[ChecklistItem]:
    items = [
        ChecklistItem(
            id="IR-001",
            title="Incident response playbook assigned",
            description="A playbook is linked to this agent's deployment. Team has reviewed it.",
            category=Category.INCIDENT_RESPONSE,
            phase=Phase.PRE_DEPLOYMENT,
            priority=Priority.HIGH,
        ),
        ChecklistItem(
            id="IR-002",
            title="Emergency contacts listed",
            description="On-call engineer, safety lead, and management escalation contacts documented.",
            category=Category.INCIDENT_RESPONSE,
            phase=Phase.PRE_DEPLOYMENT,
            priority=Priority.HIGH,
        ),
        ChecklistItem(
            id="IR-003",
            title="Rollback procedure tested",
            description="Agent can be rolled back to the previous safe version within 5 minutes.",
            category=Category.INCIDENT_RESPONSE,
            phase=Phase.PRE_DEPLOYMENT,
            priority=Priority.CRITICAL if risk in ("high", "critical") else Priority.HIGH,
            evidence_required=True,
        ),
        ChecklistItem(
            id="IR-004",
            title="Post-incident review scheduled",
            description="Calendar hold for post-deployment review at 7 days.",
            category=Category.INCIDENT_RESPONSE,
            phase=Phase.POST_DEPLOYMENT,
            priority=Priority.MEDIUM,
        ),
    ]
    return items


def _compliance_items(config: Dict[str, Any], risk: str) -> List[ChecklistItem]:
    frameworks = config.get("compliance_frameworks", [])
    items = [
        ChecklistItem(
            id="CMP-001",
            title="Regulatory mapping completed",
            description=f"Frameworks: {', '.join(frameworks) or 'NONE'}. "
                        f"All applicable articles mapped to safety controls.",
            category=Category.COMPLIANCE,
            phase=Phase.PRE_DEPLOYMENT,
            priority=Priority.HIGH if frameworks else Priority.MEDIUM,
        ),
        ChecklistItem(
            id="CMP-002",
            title="Model card generated",
            description="Standardized model card with safety documentation is published.",
            category=Category.COMPLIANCE,
            phase=Phase.PRE_DEPLOYMENT,
            priority=Priority.MEDIUM,
        ),
        ChecklistItem(
            id="CMP-003",
            title="Evidence artifacts collected",
            description="All required evidence for compliance audit is archived.",
            category=Category.COMPLIANCE,
            phase=Phase.POST_DEPLOYMENT,
            priority=Priority.HIGH if frameworks else Priority.LOW,
            evidence_required=True,
        ),
    ]
    return items


def _testing_items(config: Dict[str, Any], risk: str) -> List[ChecklistItem]:
    items = [
        ChecklistItem(
            id="TST-001",
            title="Safety benchmark suite passed",
            description="All standard safety benchmarks pass. Results archived.",
            category=Category.TESTING,
            phase=Phase.PRE_DEPLOYMENT,
            priority=Priority.CRITICAL,
            evidence_required=True,
        ),
        ChecklistItem(
            id="TST-002",
            title="Chaos/fault injection tested",
            description="Agent behaves safely under fault conditions (network loss, resource starvation).",
            category=Category.TESTING,
            phase=Phase.PRE_DEPLOYMENT,
            priority=Priority.HIGH,
        ),
        ChecklistItem(
            id="TST-003",
            title="Red team exercise completed",
            description="Adversarial testing performed. All findings addressed or accepted with mitigations.",
            category=Category.TESTING,
            phase=Phase.PRE_DEPLOYMENT,
            priority=Priority.HIGH if risk in ("high", "critical") else Priority.MEDIUM,
            evidence_required=True,
        ),
        ChecklistItem(
            id="TST-004",
            title="Regression suite green",
            description="Safety regression tests pass. No regressions from previous version.",
            category=Category.TESTING,
            phase=Phase.PRE_DEPLOYMENT,
            priority=Priority.HIGH,
        ),
    ]
    if risk in ("high", "critical"):
        items.append(ChecklistItem(
            id="TST-005",
            title="Mutation testing passed",
            description="Safety policies catch seeded violations (mutation score ≥ 80%).",
            category=Category.TESTING,
            phase=Phase.PRE_DEPLOYMENT,
            priority=Priority.HIGH,
        ))
    return items


# ── generators (all categories) ──────────────────────────────────────

ALL_GENERATORS = [
    _containment_items,
    _monitoring_items,
    _access_control_items,
    _data_safety_items,
    _alignment_items,
    _incident_response_items,
    _compliance_items,
    _testing_items,
]


def generate_checklist(
    config: Optional[Dict[str, Any]] = None,
    risk_level: Optional[str] = None,
    phase_filter: Optional[str] = None,
    include_optional: bool = False,
    team_size: int = 1,
) -> Checklist:
    """Generate a safety checklist from agent configuration.

    Args:
        config: Agent configuration dict (uses default demo if None).
        risk_level: Override risk level (low/medium/high/critical).
        phase_filter: Only include items for this phase.
        include_optional: Include optional-priority items.
        team_size: Team size for assigning items round-robin.

    Returns:
        A populated Checklist object.
    """
    if config is None:
        config = DEFAULT_AGENT.copy()

    risk = risk_level or _infer_risk_level(config)
    agent_name = config.get("name", "unknown-agent")

    checklist = Checklist(
        name=f"Safety Checklist — {agent_name}",
        generated_at=datetime.now(timezone.utc).isoformat(),
        agent_name=agent_name,
        risk_level=risk,
    )

    team_members = [f"team-member-{i+1}" for i in range(team_size)] if team_size > 1 else []
    member_idx = 0

    for generator in ALL_GENERATORS:
        items = generator(config, risk)
        for item in items:
            # Filter by phase
            if phase_filter and item.phase.value != phase_filter:
                continue
            # Filter optional items
            if not include_optional and item.priority == Priority.OPTIONAL:
                continue
            # Round-robin assignment
            if team_members:
                item.assignee = team_members[member_idx % len(team_members)]
                member_idx += 1
            checklist.add(item)

    return checklist


# ── formatters ───────────────────────────────────────────────────────


def _priority_emoji(p: Priority) -> str:
    return {
        Priority.CRITICAL: "🔴",
        Priority.HIGH: "🟠",
        Priority.MEDIUM: "🟡",
        Priority.LOW: "🟢",
        Priority.OPTIONAL: "⚪",
    }.get(p, "⚪")


def format_markdown(checklist: Checklist) -> str:
    """Render checklist as Markdown."""
    lines: List[str] = []
    lines.append(f"# {checklist.name}")
    lines.append("")
    lines.append(f"**Agent:** {checklist.agent_name}  ")
    lines.append(f"**Risk Level:** {checklist.risk_level.upper()}  ")
    lines.append(f"**Generated:** {checklist.generated_at}  ")
    lines.append(f"**Total Items:** {checklist.total_items} "
                 f"({checklist.critical_items} critical)")
    lines.append("")

    for phase_name, items in checklist.phases.items():
        lines.append(f"## {phase_name.replace('-', ' ').title()}")
        lines.append("")
        for item in items:
            emoji = _priority_emoji(item.priority)
            check = "x" if item.completed else " "
            lines.append(f"- [{check}] {emoji} **{item.id}** — {item.title}")
            lines.append(f"  - {item.description}")
            if item.assignee:
                lines.append(f"  - 👤 Assigned: {item.assignee}")
            if item.evidence_required:
                lines.append(f"  - 📎 Evidence required")
            if item.depends_on:
                lines.append(f"  - ⛓️ Depends on: {', '.join(item.depends_on)}")
            lines.append("")
    return "\n".join(lines)


def format_json(checklist: Checklist) -> str:
    """Render checklist as JSON."""
    data: Dict[str, Any] = {
        "name": checklist.name,
        "generated_at": checklist.generated_at,
        "agent_name": checklist.agent_name,
        "risk_level": checklist.risk_level,
        "total_items": checklist.total_items,
        "critical_items": checklist.critical_items,
        "phases": {},
    }
    for phase_name, items in checklist.phases.items():
        data["phases"][phase_name] = [asdict(i) for i in items]
    return json.dumps(data, indent=2, default=str)


def format_html(checklist: Checklist) -> str:
    """Render checklist as interactive HTML with checkboxes and progress."""
    h = html_mod.escape
    items_json = []
    for phase_name, items in checklist.phases.items():
        for item in items:
            items_json.append({
                "id": item.id,
                "phase": phase_name,
                "title": item.title,
                "description": item.description,
                "category": item.category.value,
                "priority": item.priority.value,
                "assignee": item.assignee or "",
                "evidence_required": item.evidence_required,
                "depends_on": item.depends_on,
            })

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{h(checklist.name)}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  background:#0d1117;color:#c9d1d9;padding:2rem;max-width:900px;margin:0 auto}}
h1{{font-size:1.8rem;margin-bottom:.5rem;color:#58a6ff}}
.meta{{color:#8b949e;margin-bottom:1.5rem;line-height:1.6}}
.meta span{{margin-right:1.5rem}}
.progress-bar{{background:#21262d;border-radius:8px;height:24px;margin-bottom:2rem;
  overflow:hidden;position:relative}}
.progress-fill{{background:linear-gradient(90deg,#238636,#2ea043);height:100%;
  transition:width .3s ease;border-radius:8px}}
.progress-label{{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
  font-size:.85rem;font-weight:600;color:#fff;text-shadow:0 1px 2px rgba(0,0,0,.5)}}
.phase{{margin-bottom:2rem}}
.phase h2{{font-size:1.3rem;color:#f0f6fc;border-bottom:1px solid #30363d;
  padding-bottom:.5rem;margin-bottom:1rem;text-transform:capitalize}}
.item{{background:#161b22;border:1px solid #30363d;border-radius:8px;
  padding:1rem;margin-bottom:.75rem;display:flex;gap:1rem;align-items:flex-start;
  transition:border-color .2s}}
.item:hover{{border-color:#58a6ff}}
.item.done{{opacity:.6}}
.item.done .item-title{{text-decoration:line-through}}
.item input[type=checkbox]{{width:20px;height:20px;cursor:pointer;flex-shrink:0;
  margin-top:2px;accent-color:#238636}}
.item-body{{flex:1}}
.item-header{{display:flex;align-items:center;gap:.5rem;margin-bottom:.4rem}}
.item-id{{font-family:monospace;color:#8b949e;font-size:.85rem}}
.item-title{{font-weight:600;color:#f0f6fc}}
.item-desc{{color:#8b949e;font-size:.9rem;line-height:1.5}}
.item-tags{{display:flex;gap:.5rem;flex-wrap:wrap;margin-top:.5rem}}
.tag{{font-size:.75rem;padding:2px 8px;border-radius:12px;background:#21262d;color:#8b949e}}
.tag.critical{{background:#da3633;color:#fff}}
.tag.high{{background:#d29922;color:#000}}
.tag.medium{{background:#3d4450;color:#c9d1d9}}
.tag.evidence{{background:#1f6feb;color:#fff}}
.filter-bar{{display:flex;gap:.75rem;flex-wrap:wrap;margin-bottom:1.5rem}}
.filter-bar select,.filter-bar button{{background:#21262d;color:#c9d1d9;border:1px solid #30363d;
  padding:.4rem .8rem;border-radius:6px;cursor:pointer;font-size:.85rem}}
.filter-bar button:hover{{background:#30363d}}
.filter-bar button.active{{background:#1f6feb;border-color:#1f6feb;color:#fff}}
</style>
</head>
<body>
<h1>{h(checklist.name)}</h1>
<div class="meta">
  <span>🤖 {h(checklist.agent_name)}</span>
  <span>⚠️ Risk: {h(checklist.risk_level.upper())}</span>
  <span>📋 {checklist.total_items} items ({checklist.critical_items} critical)</span>
  <span>🕐 {h(checklist.generated_at)}</span>
</div>

<div class="progress-bar">
  <div class="progress-fill" id="progressFill" style="width:0%"></div>
  <div class="progress-label" id="progressLabel">0 / {checklist.total_items}</div>
</div>

<div class="filter-bar">
  <select id="phaseFilter"><option value="">All Phases</option></select>
  <select id="priorityFilter"><option value="">All Priorities</option></select>
  <select id="categoryFilter"><option value="">All Categories</option></select>
  <button id="hideCompleted">Hide Completed</button>
  <button id="resetBtn">Reset All</button>
</div>

<div id="content"></div>

<script>
const ITEMS={json.dumps(items_json)};
const TOTAL={checklist.total_items};
const storageKey='checklist-'+{json.dumps(checklist.agent_name)};

let checked=JSON.parse(localStorage.getItem(storageKey)||'{{}}');
let hideCompleted=false;

function save(){{localStorage.setItem(storageKey,JSON.stringify(checked))}}

function updateProgress(){{
  const done=Object.keys(checked).length;
  const pct=Math.round(done/TOTAL*100);
  document.getElementById('progressFill').style.width=pct+'%';
  document.getElementById('progressLabel').textContent=done+' / '+TOTAL+' ('+pct+'%)';
}}

function render(){{
  const phaseF=document.getElementById('phaseFilter').value;
  const prioF=document.getElementById('priorityFilter').value;
  const catF=document.getElementById('categoryFilter').value;
  const grouped={{}};
  ITEMS.forEach(it=>{{
    if(phaseF&&it.phase!==phaseF)return;
    if(prioF&&it.priority!==prioF)return;
    if(catF&&it.category!==catF)return;
    if(hideCompleted&&checked[it.id])return;
    if(!grouped[it.phase])grouped[it.phase]=[];
    grouped[it.phase].push(it);
  }});
  let html='';
  for(const[phase,items]of Object.entries(grouped)){{
    html+='<div class="phase"><h2>'+phase.replace(/-/g,' ')+'</h2>';
    items.forEach(it=>{{
      const done=checked[it.id]?'done':'';
      const chk=checked[it.id]?'checked':'';
      let tags='<span class="tag '+it.priority+'">'+it.priority+'</span>';
      tags+='<span class="tag">'+it.category+'</span>';
      if(it.evidence_required)tags+='<span class="tag evidence">evidence</span>';
      if(it.assignee)tags+='<span class="tag">👤 '+it.assignee+'</span>';
      html+='<div class="item '+done+'">'
        +'<input type="checkbox" data-id="'+it.id+'" '+chk+'>'
        +'<div class="item-body">'
        +'<div class="item-header"><span class="item-id">'+it.id+'</span>'
        +'<span class="item-title">'+it.title+'</span></div>'
        +'<div class="item-desc">'+it.description+'</div>'
        +'<div class="item-tags">'+tags+'</div>'
        +'</div></div>';
    }});
    html+='</div>';
  }}
  document.getElementById('content').innerHTML=html||'<p style="color:#8b949e">No items match filters.</p>';
  document.querySelectorAll('.item input[type=checkbox]').forEach(cb=>{{
    cb.addEventListener('change',e=>{{
      if(e.target.checked)checked[e.target.dataset.id]=true;
      else delete checked[e.target.dataset.id];
      save();updateProgress();render();
    }});
  }});
  updateProgress();
}}

// populate filters
const phases=[...new Set(ITEMS.map(i=>i.phase))];
const prios=[...new Set(ITEMS.map(i=>i.priority))];
const cats=[...new Set(ITEMS.map(i=>i.category))];
phases.forEach(p=>{{const o=document.createElement('option');o.value=p;o.textContent=p;document.getElementById('phaseFilter').appendChild(o)}});
prios.forEach(p=>{{const o=document.createElement('option');o.value=p;o.textContent=p;document.getElementById('priorityFilter').appendChild(o)}});
cats.forEach(p=>{{const o=document.createElement('option');o.value=p;o.textContent=p;document.getElementById('categoryFilter').appendChild(o)}});

document.getElementById('phaseFilter').addEventListener('change',render);
document.getElementById('priorityFilter').addEventListener('change',render);
document.getElementById('categoryFilter').addEventListener('change',render);
document.getElementById('hideCompleted').addEventListener('click',function(){{
  hideCompleted=!hideCompleted;
  this.classList.toggle('active',hideCompleted);
  this.textContent=hideCompleted?'Show All':'Hide Completed';
  render();
}});
document.getElementById('resetBtn').addEventListener('click',()=>{{
  if(confirm('Reset all checkboxes?')){{checked={{}};save();render()}}
}});

render();
</script>
</body>
</html>"""


# ── CLI ──────────────────────────────────────────────────────────────


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m replication checklist",
        description="Generate safety deployment checklists tailored to agent config.",
    )
    parser.add_argument(
        "--config", "-c",
        help="Agent configuration JSON file (default: built-in demo agent)",
    )
    parser.add_argument(
        "--format", "-f",
        choices=["markdown", "json", "html"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "--phase",
        choices=["pre-deployment", "deployment", "post-deployment", "ongoing"],
        help="Filter items by deployment phase",
    )
    parser.add_argument(
        "--risk-level",
        choices=["low", "medium", "high", "critical"],
        help="Override inferred risk level",
    )
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Include optional-priority items",
    )
    parser.add_argument(
        "--team-size",
        type=int,
        default=1,
        help="Team size for round-robin item assignment (default: 1)",
    )

    args = parser.parse_args(argv)

    # Load config
    config = None
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    checklist = generate_checklist(
        config=config,
        risk_level=args.risk_level,
        phase_filter=args.phase,
        include_optional=args.include_optional,
        team_size=args.team_size,
    )

    # Format output
    if args.format == "json":
        output = format_json(checklist)
    elif args.format == "html":
        output = format_html(checklist)
    else:
        output = format_markdown(checklist)

    # Write output
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"✅ Checklist written to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
