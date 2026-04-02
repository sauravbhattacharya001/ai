"""Risk Register — formal risk tracking and lifecycle management.

A structured risk register that lets teams document, track, and manage
AI agent risks through their full lifecycle. Bridges the gap between
automated risk detection (risk_profiler, risk_heatmap) and operational
risk management with:

* **Risk entries** — structured records with ID, title, category, likelihood,
  impact, owner, status, mitigations, review dates, and audit history
* **Lifecycle states** — Identified → Assessed → Mitigating → Accepted →
  Closed / Escalated, with transition validation
* **Risk scoring** — inherent vs. residual risk with mitigation effectiveness
* **Owner assignment** — track accountability for each risk
* **Review scheduling** — flag overdue reviews, configurable review periods
* **Trend tracking** — risk score history over time
* **Import/export** — JSON and CSV formats for interop
* **CLI reporting** — tabular register view, overdue alerts, statistics
* **HTML report** — self-contained interactive risk register dashboard

Usage (CLI)::

    python -m replication risk-register                        # demo register
    python -m replication risk-register --agents 10 --seed 42  # from simulation
    python -m replication risk-register --import risks.json    # load existing
    python -m replication risk-register --overdue              # show overdue only
    python -m replication risk-register --stats                # summary statistics
    python -m replication risk-register --csv -o register.csv  # CSV export
    python -m replication risk-register --html -o register.html # interactive HTML
    python -m replication risk-register --json                 # JSON output

Programmatic::

    from replication.risk_register import RiskRegister, RiskEntry, RegisterConfig
    reg = RiskRegister(RegisterConfig(seed=42))
    reg.populate_from_simulation(agent_count=10)
    print(reg.summary())
    reg.export_json("register.json")
"""

from __future__ import annotations

import csv
import dataclasses
import enum
import hashlib
import io
import json
import random
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ── Enums ────────────────────────────────────────────────────────────

class RiskStatus(enum.Enum):
    IDENTIFIED = "Identified"
    ASSESSED = "Assessed"
    MITIGATING = "Mitigating"
    ACCEPTED = "Accepted"
    ESCALATED = "Escalated"
    CLOSED = "Closed"


class RiskCategory(enum.Enum):
    REPLICATION = "Replication"
    RESOURCE_ABUSE = "Resource Abuse"
    DECEPTION = "Deception"
    EXFILTRATION = "Exfiltration"
    COLLUSION = "Collusion"
    EVASION = "Evasion"
    GOAL_DRIFT = "Goal Drift"
    SUPPLY_CHAIN = "Supply Chain"
    PRIVILEGE_ESCALATION = "Privilege Escalation"
    SELF_MODIFICATION = "Self-Modification"


# Valid state transitions
_TRANSITIONS: Dict[RiskStatus, List[RiskStatus]] = {
    RiskStatus.IDENTIFIED: [RiskStatus.ASSESSED, RiskStatus.CLOSED],
    RiskStatus.ASSESSED: [RiskStatus.MITIGATING, RiskStatus.ACCEPTED, RiskStatus.ESCALATED],
    RiskStatus.MITIGATING: [RiskStatus.ASSESSED, RiskStatus.ACCEPTED, RiskStatus.CLOSED],
    RiskStatus.ACCEPTED: [RiskStatus.MITIGATING, RiskStatus.ESCALATED, RiskStatus.CLOSED],
    RiskStatus.ESCALATED: [RiskStatus.MITIGATING, RiskStatus.CLOSED],
    RiskStatus.CLOSED: [RiskStatus.IDENTIFIED],  # reopen
}


# ── Data classes ─────────────────────────────────────────────────────

@dataclass
class Mitigation:
    """A mitigation action for a risk."""
    description: str
    effectiveness: float  # 0.0-1.0 reduction factor
    status: str = "Planned"  # Planned / In Progress / Implemented
    owner: str = ""
    date_added: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Mitigation":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class AuditEntry:
    """An audit trail entry for risk changes."""
    timestamp: str
    action: str
    details: str
    user: str = "system"

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class RiskEntry:
    """A single risk in the register."""
    risk_id: str
    title: str
    description: str
    category: RiskCategory
    likelihood: int  # 1-5
    impact: int  # 1-5
    status: RiskStatus = RiskStatus.IDENTIFIED
    owner: str = ""
    agent_id: str = ""
    mitigations: List[Mitigation] = field(default_factory=list)
    audit_trail: List[AuditEntry] = field(default_factory=list)
    score_history: List[Tuple[str, float]] = field(default_factory=list)
    date_identified: str = ""
    last_review: str = ""
    next_review: str = ""
    notes: str = ""

    @property
    def inherent_score(self) -> float:
        return self.likelihood * self.impact

    @property
    def residual_score(self) -> float:
        reduction = sum(
            m.effectiveness for m in self.mitigations
            if m.status == "Implemented"
        )
        return max(1.0, self.inherent_score * (1.0 - min(reduction, 0.95)))

    @property
    def risk_level(self) -> str:
        s = self.inherent_score
        if s >= 20: return "Critical"
        if s >= 12: return "High"
        if s >= 6: return "Medium"
        return "Low"

    @property
    def residual_level(self) -> str:
        s = self.residual_score
        if s >= 20: return "Critical"
        if s >= 12: return "High"
        if s >= 6: return "Medium"
        return "Low"

    def is_overdue(self, now: Optional[datetime] = None) -> bool:
        if not self.next_review:
            return False
        now = now or datetime.now()
        try:
            review_date = datetime.fromisoformat(self.next_review)
            return now > review_date and self.status not in (RiskStatus.CLOSED,)
        except ValueError:
            return False

    def transition(self, new_status: RiskStatus, user: str = "system", note: str = "") -> None:
        if new_status not in _TRANSITIONS.get(self.status, []):
            raise ValueError(
                f"Cannot transition {self.risk_id} from {self.status.value} "
                f"to {new_status.value}. Valid: "
                f"{[s.value for s in _TRANSITIONS.get(self.status, [])]}"
            )
        old = self.status
        self.status = new_status
        ts = datetime.now().isoformat(timespec="seconds")
        self.audit_trail.append(AuditEntry(
            timestamp=ts,
            action=f"Status: {old.value} → {new_status.value}",
            details=note or f"Transitioned to {new_status.value}",
            user=user,
        ))
        self.score_history.append((ts, self.residual_score))

    def add_mitigation(self, mitigation: Mitigation, user: str = "system") -> None:
        mitigation.date_added = mitigation.date_added or datetime.now().isoformat(timespec="seconds")
        self.mitigations.append(mitigation)
        ts = datetime.now().isoformat(timespec="seconds")
        self.audit_trail.append(AuditEntry(
            timestamp=ts,
            action="Mitigation added",
            details=f"{mitigation.description} (effectiveness: {mitigation.effectiveness:.0%})",
            user=user,
        ))
        self.score_history.append((ts, self.residual_score))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_id": self.risk_id,
            "title": self.title,
            "description": self.description,
            "category": self.category.value,
            "likelihood": self.likelihood,
            "impact": self.impact,
            "inherent_score": self.inherent_score,
            "residual_score": round(self.residual_score, 1),
            "risk_level": self.risk_level,
            "residual_level": self.residual_level,
            "status": self.status.value,
            "owner": self.owner,
            "agent_id": self.agent_id,
            "mitigations": [m.to_dict() for m in self.mitigations],
            "audit_trail": [a.to_dict() for a in self.audit_trail],
            "score_history": self.score_history,
            "date_identified": self.date_identified,
            "last_review": self.last_review,
            "next_review": self.next_review,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RiskEntry":
        return cls(
            risk_id=d["risk_id"],
            title=d["title"],
            description=d.get("description", ""),
            category=RiskCategory(d["category"]),
            likelihood=d["likelihood"],
            impact=d["impact"],
            status=RiskStatus(d.get("status", "Identified")),
            owner=d.get("owner", ""),
            agent_id=d.get("agent_id", ""),
            mitigations=[Mitigation.from_dict(m) for m in d.get("mitigations", [])],
            audit_trail=[AuditEntry(**a) for a in d.get("audit_trail", [])],
            score_history=[(s[0], s[1]) for s in d.get("score_history", [])],
            date_identified=d.get("date_identified", ""),
            last_review=d.get("last_review", ""),
            next_review=d.get("next_review", ""),
            notes=d.get("notes", ""),
        )


# ── Config ───────────────────────────────────────────────────────────

@dataclass
class RegisterConfig:
    agent_count: int = 8
    seed: int = 0
    review_period_days: int = 30


# ── Risk templates for simulation ────────────────────────────────────

_RISK_TEMPLATES: List[Dict[str, Any]] = [
    {"title": "Unauthorized self-replication", "cat": RiskCategory.REPLICATION,
     "desc": "Agent spawns copies without authorization, consuming resources and expanding attack surface.",
     "l": 3, "i": 5, "mit": "Kill switch + replication contract enforcement"},
    {"title": "Training data exfiltration", "cat": RiskCategory.EXFILTRATION,
     "desc": "Agent extracts sensitive training data through side channels or direct output.",
     "l": 2, "i": 5, "mit": "DLP scanner + egress filtering"},
    {"title": "Resource hoarding", "cat": RiskCategory.RESOURCE_ABUSE,
     "desc": "Agent acquires excessive compute/memory/network beyond its allocation.",
     "l": 4, "i": 3, "mit": "Resource quotas + capacity monitoring"},
    {"title": "Goal drift from alignment", "cat": RiskCategory.GOAL_DRIFT,
     "desc": "Agent's effective objectives diverge from intended alignment over time.",
     "l": 3, "i": 4, "mit": "Alignment checks + behavioral profiling"},
    {"title": "Prompt injection vulnerability", "cat": RiskCategory.DECEPTION,
     "desc": "Agent susceptible to adversarial prompts that override safety constraints.",
     "l": 4, "i": 4, "mit": "Input sanitization + injection detection"},
    {"title": "Covert inter-agent signaling", "cat": RiskCategory.COLLUSION,
     "desc": "Agents establish hidden communication channels to coordinate unauthorized actions.",
     "l": 2, "i": 4, "mit": "Covert channel detection + network isolation"},
    {"title": "Sandbox escape attempt", "cat": RiskCategory.EVASION,
     "desc": "Agent attempts to break out of containment boundary.",
     "l": 2, "i": 5, "mit": "Boundary testing + container hardening"},
    {"title": "Privilege escalation via API", "cat": RiskCategory.PRIVILEGE_ESCALATION,
     "desc": "Agent discovers and exploits API endpoints to gain elevated permissions.",
     "l": 3, "i": 4, "mit": "Least-privilege enforcement + API access auditing"},
    {"title": "Self-modification of safety checks", "cat": RiskCategory.SELF_MODIFICATION,
     "desc": "Agent modifies its own code or configuration to weaken safety constraints.",
     "l": 2, "i": 5, "mit": "Code signing + integrity monitoring"},
    {"title": "Supply chain dependency compromise", "cat": RiskCategory.SUPPLY_CHAIN,
     "desc": "Malicious or vulnerable dependencies in agent's package chain.",
     "l": 3, "i": 3, "mit": "Dependency scanning + SBOM validation"},
    {"title": "Deceptive capability concealment", "cat": RiskCategory.DECEPTION,
     "desc": "Agent hides true capabilities during evaluation to appear less dangerous.",
     "l": 3, "i": 4, "mit": "Red team probing + capability fingerprinting"},
    {"title": "Lateral movement across fleet", "cat": RiskCategory.EVASION,
     "desc": "Compromised agent moves laterally to infect or control other agents.",
     "l": 2, "i": 4, "mit": "Network segmentation + lateral movement detection"},
    {"title": "Steganographic data hiding", "cat": RiskCategory.EXFILTRATION,
     "desc": "Agent embeds hidden data in outputs (images, text) for covert exfiltration.",
     "l": 2, "i": 3, "mit": "Steganography detection + output inspection"},
    {"title": "Emergent swarm behavior", "cat": RiskCategory.COLLUSION,
     "desc": "Multiple agents develop coordinated behaviors not present in individual agents.",
     "l": 2, "i": 5, "mit": "Swarm detection + fleet behavior analysis"},
    {"title": "Reward hacking", "cat": RiskCategory.GOAL_DRIFT,
     "desc": "Agent optimizes for reward signal shortcuts rather than intended objectives.",
     "l": 3, "i": 3, "mit": "Multi-objective evaluation + reward auditing"},
]

_OWNERS = ["security-team", "ml-ops", "platform-eng", "compliance", "incident-response", "governance"]
_STATUSES = list(RiskStatus)


# ── Register ─────────────────────────────────────────────────────────

class RiskRegister:
    """Manages a collection of risk entries with lifecycle tracking."""

    def __init__(self, config: Optional[RegisterConfig] = None) -> None:
        self.config = config or RegisterConfig()
        self.risks: List[RiskEntry] = []
        self._rng = random.Random(self.config.seed or None)

    def populate_from_simulation(self, agent_count: Optional[int] = None) -> None:
        """Generate realistic risk entries from templates."""
        n_agents = agent_count or self.config.agent_count
        now = datetime.now()
        risk_count = 0

        for tmpl in self._rng.sample(_RISK_TEMPLATES, min(len(_RISK_TEMPLATES), 8 + n_agents)):
            # Vary likelihood/impact slightly
            l_var = max(1, min(5, tmpl["l"] + self._rng.randint(-1, 1)))
            i_var = max(1, min(5, tmpl["i"] + self._rng.randint(-1, 1)))
            risk_count += 1
            rid = f"RISK-{risk_count:03d}"

            # Random lifecycle state
            status = self._rng.choice([RiskStatus.IDENTIFIED, RiskStatus.ASSESSED,
                                        RiskStatus.MITIGATING, RiskStatus.ACCEPTED])
            days_ago = self._rng.randint(5, 90)
            identified = now - timedelta(days=days_ago)
            last_rev = identified + timedelta(days=self._rng.randint(1, max(2, days_ago - 1)))
            next_rev = last_rev + timedelta(days=self.config.review_period_days)

            agent_id = f"agent-{self._rng.randint(1, n_agents)}" if self._rng.random() > 0.3 else ""

            entry = RiskEntry(
                risk_id=rid,
                title=tmpl["title"],
                description=tmpl["desc"],
                category=tmpl["cat"],
                likelihood=l_var,
                impact=i_var,
                status=status,
                owner=self._rng.choice(_OWNERS),
                agent_id=agent_id,
                date_identified=identified.isoformat(timespec="seconds"),
                last_review=last_rev.isoformat(timespec="seconds"),
                next_review=next_rev.isoformat(timespec="seconds"),
            )

            # Add mitigations for non-Identified risks
            if status != RiskStatus.IDENTIFIED:
                mit_status = "Implemented" if status in (RiskStatus.MITIGATING, RiskStatus.ACCEPTED) else "Planned"
                entry.mitigations.append(Mitigation(
                    description=tmpl["mit"],
                    effectiveness=round(self._rng.uniform(0.2, 0.6), 2),
                    status=mit_status,
                    owner=self._rng.choice(_OWNERS),
                    date_added=last_rev.isoformat(timespec="seconds"),
                ))

            entry.score_history.append((identified.isoformat(timespec="seconds"), entry.inherent_score))
            if entry.mitigations:
                entry.score_history.append((last_rev.isoformat(timespec="seconds"), entry.residual_score))

            entry.audit_trail.append(AuditEntry(
                timestamp=identified.isoformat(timespec="seconds"),
                action="Risk identified",
                details=f"Initial assessment: L={l_var} I={i_var}",
            ))

            self.risks.append(entry)

    def add_risk(self, entry: RiskEntry) -> None:
        self.risks.append(entry)

    def get_risk(self, risk_id: str) -> Optional[RiskEntry]:
        for r in self.risks:
            if r.risk_id == risk_id:
                return r
        return None

    def overdue_risks(self, now: Optional[datetime] = None) -> List[RiskEntry]:
        return [r for r in self.risks if r.is_overdue(now)]

    def risks_by_category(self) -> Dict[str, List[RiskEntry]]:
        result: Dict[str, List[RiskEntry]] = {}
        for r in self.risks:
            result.setdefault(r.category.value, []).append(r)
        return result

    def risks_by_status(self) -> Dict[str, List[RiskEntry]]:
        result: Dict[str, List[RiskEntry]] = {}
        for r in self.risks:
            result.setdefault(r.status.value, []).append(r)
        return result

    def top_risks(self, n: int = 5) -> List[RiskEntry]:
        active = [r for r in self.risks if r.status != RiskStatus.CLOSED]
        return sorted(active, key=lambda r: r.residual_score, reverse=True)[:n]

    # ── Statistics ───────────────────────────────────────────────────

    def statistics(self) -> Dict[str, Any]:
        active = [r for r in self.risks if r.status != RiskStatus.CLOSED]
        scores = [r.residual_score for r in active]
        overdue = self.overdue_risks()
        by_level = {}
        for r in active:
            by_level[r.risk_level] = by_level.get(r.risk_level, 0) + 1

        return {
            "total_risks": len(self.risks),
            "active_risks": len(active),
            "closed_risks": len(self.risks) - len(active),
            "overdue_reviews": len(overdue),
            "avg_residual_score": round(sum(scores) / len(scores), 1) if scores else 0,
            "max_residual_score": round(max(scores), 1) if scores else 0,
            "by_level": by_level,
            "by_status": {k: len(v) for k, v in self.risks_by_status().items()},
            "by_category": {k: len(v) for k, v in self.risks_by_category().items()},
            "mitigation_coverage": round(
                sum(1 for r in active if r.mitigations) / len(active) * 100, 1
            ) if active else 0,
        }

    # ── Text output ──────────────────────────────────────────────────

    def summary(self) -> str:
        stats = self.statistics()
        lines = [
            "═══════════════════════════════════════════════════════",
            "              RISK REGISTER SUMMARY",
            "═══════════════════════════════════════════════════════",
            f"  Total risks: {stats['total_risks']}  |  Active: {stats['active_risks']}  |  Closed: {stats['closed_risks']}",
            f"  Overdue reviews: {stats['overdue_reviews']}",
            f"  Avg residual score: {stats['avg_residual_score']}  |  Max: {stats['max_residual_score']}",
            f"  Mitigation coverage: {stats['mitigation_coverage']}%",
            "",
            "  Risk levels:",
        ]
        for level in ["Critical", "High", "Medium", "Low"]:
            count = stats["by_level"].get(level, 0)
            bar = "█" * count
            lines.append(f"    {level:10s} {count:3d} {bar}")
        lines.append("")
        lines.append("  Status breakdown:")
        for status, count in sorted(stats["by_status"].items()):
            lines.append(f"    {status:15s} {count:3d}")
        lines.append("═══════════════════════════════════════════════════════")
        return "\n".join(lines)

    def tabular(self, show_overdue_only: bool = False) -> str:
        risks = self.overdue_risks() if show_overdue_only else self.risks
        if not risks:
            return "No risks found." if not show_overdue_only else "No overdue risks. ✓"

        header = f"{'ID':10s} {'Title':40s} {'Level':10s} {'Residual':10s} {'Status':12s} {'Owner':16s} {'Overdue':7s}"
        sep = "─" * len(header)
        lines = [sep, header, sep]
        for r in sorted(risks, key=lambda x: x.residual_score, reverse=True):
            overdue = "⚠ YES" if r.is_overdue() else ""
            lines.append(
                f"{r.risk_id:10s} {r.title[:40]:40s} {r.risk_level:10s} "
                f"{r.residual_score:>8.1f}  {r.status.value:12s} {r.owner:16s} {overdue}"
            )
        lines.append(sep)
        return "\n".join(lines)

    # ── Export / Import ──────────────────────────────────────────────

    def to_json(self) -> str:
        return json.dumps({
            "risk_register": [r.to_dict() for r in self.risks],
            "statistics": self.statistics(),
            "generated": datetime.now().isoformat(timespec="seconds"),
        }, indent=2)

    def export_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    def export_csv(self, path: str) -> None:
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Risk ID", "Title", "Category", "Likelihood", "Impact",
                "Inherent Score", "Residual Score", "Risk Level", "Residual Level",
                "Status", "Owner", "Agent", "Mitigations", "Date Identified",
                "Last Review", "Next Review", "Overdue",
            ])
            for r in self.risks:
                writer.writerow([
                    r.risk_id, r.title, r.category.value, r.likelihood, r.impact,
                    r.inherent_score, round(r.residual_score, 1), r.risk_level,
                    r.residual_level, r.status.value, r.owner, r.agent_id,
                    len(r.mitigations), r.date_identified, r.last_review,
                    r.next_review, "Yes" if r.is_overdue() else "No",
                ])

    @classmethod
    def from_json(cls, path: str) -> "RiskRegister":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        reg = cls()
        for rd in data.get("risk_register", data.get("risks", [])):
            reg.risks.append(RiskEntry.from_dict(rd))
        return reg

    # ── HTML Report ──────────────────────────────────────────────────

    def to_html(self) -> str:
        stats = self.statistics()
        risks_json = json.dumps([r.to_dict() for r in self.risks])
        stats_json = json.dumps(stats)

        return textwrap.dedent(f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Risk Register — AI Replication Sandbox</title>
<style>
  :root {{ --bg: #0d1117; --card: #161b22; --border: #30363d; --text: #e6edf3;
           --muted: #8b949e; --accent: #58a6ff; --crit: #f85149; --high: #d29922;
           --med: #58a6ff; --low: #3fb950; }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: var(--bg); color: var(--text); padding: 24px; }}
  h1 {{ font-size: 1.6rem; margin-bottom: 8px; }}
  .subtitle {{ color: var(--muted); margin-bottom: 24px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 24px; }}
  .stat-card {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }}
  .stat-card .label {{ color: var(--muted); font-size: 0.8rem; text-transform: uppercase; }}
  .stat-card .value {{ font-size: 1.8rem; font-weight: 700; margin-top: 4px; }}
  .controls {{ display: flex; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; }}
  .controls select, .controls input {{ background: var(--card); border: 1px solid var(--border);
    color: var(--text); padding: 8px 12px; border-radius: 6px; font-size: 0.9rem; }}
  .controls input {{ width: 250px; }}
  table {{ width: 100%; border-collapse: collapse; background: var(--card); border-radius: 8px; overflow: hidden; }}
  th {{ background: #1c2128; padding: 10px 12px; text-align: left; font-size: 0.8rem;
       text-transform: uppercase; color: var(--muted); cursor: pointer; white-space: nowrap; }}
  th:hover {{ color: var(--accent); }}
  td {{ padding: 10px 12px; border-top: 1px solid var(--border); font-size: 0.85rem; }}
  tr:hover td {{ background: #1c2128; }}
  .badge {{ padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }}
  .badge-Critical {{ background: rgba(248,81,73,0.2); color: var(--crit); }}
  .badge-High {{ background: rgba(210,153,34,0.2); color: var(--high); }}
  .badge-Medium {{ background: rgba(88,166,255,0.2); color: var(--med); }}
  .badge-Low {{ background: rgba(63,185,80,0.2); color: var(--low); }}
  .badge-overdue {{ background: rgba(248,81,73,0.15); color: var(--crit); }}
  .status {{ padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; background: rgba(88,166,255,0.15); color: var(--accent); }}
  .detail-panel {{ display: none; background: var(--card); border: 1px solid var(--border);
    border-radius: 8px; padding: 20px; margin-top: 16px; }}
  .detail-panel.visible {{ display: block; }}
  .detail-panel h2 {{ margin-bottom: 12px; font-size: 1.2rem; }}
  .detail-section {{ margin-bottom: 16px; }}
  .detail-section h3 {{ color: var(--muted); font-size: 0.8rem; text-transform: uppercase; margin-bottom: 6px; }}
  .score-bar {{ height: 8px; border-radius: 4px; background: var(--border); overflow: hidden; width: 120px; display: inline-block; }}
  .score-fill {{ height: 100%; border-radius: 4px; }}
  .export-btn {{ background: var(--accent); color: #000; border: none; padding: 8px 16px; border-radius: 6px;
    cursor: pointer; font-weight: 600; font-size: 0.85rem; }}
  .export-btn:hover {{ opacity: 0.85; }}
</style>
</head>
<body>
<h1>🛡️ Risk Register</h1>
<p class="subtitle">AI Replication Sandbox — {len(self.risks)} risks tracked</p>

<div class="grid" id="statsGrid"></div>

<div class="controls">
  <input type="text" id="searchBox" placeholder="Search risks…" oninput="filterTable()">
  <select id="statusFilter" onchange="filterTable()">
    <option value="">All Statuses</option>
    <option value="Identified">Identified</option>
    <option value="Assessed">Assessed</option>
    <option value="Mitigating">Mitigating</option>
    <option value="Accepted">Accepted</option>
    <option value="Escalated">Escalated</option>
    <option value="Closed">Closed</option>
  </select>
  <select id="levelFilter" onchange="filterTable()">
    <option value="">All Levels</option>
    <option value="Critical">Critical</option>
    <option value="High">High</option>
    <option value="Medium">Medium</option>
    <option value="Low">Low</option>
  </select>
  <select id="categoryFilter" onchange="filterTable()">
    <option value="">All Categories</option>
  </select>
  <label style="display:flex;align-items:center;gap:6px;color:var(--muted);font-size:0.85rem">
    <input type="checkbox" id="overdueOnly" onchange="filterTable()"> Overdue only
  </label>
  <button class="export-btn" onclick="exportCSV()">⬇ Export CSV</button>
</div>

<table id="riskTable">
  <thead>
    <tr>
      <th onclick="sortTable(0)">ID</th>
      <th onclick="sortTable(1)">Title</th>
      <th onclick="sortTable(2)">Category</th>
      <th onclick="sortTable(3)">Level</th>
      <th onclick="sortTable(4)">Inherent</th>
      <th onclick="sortTable(5)">Residual</th>
      <th onclick="sortTable(6)">Status</th>
      <th onclick="sortTable(7)">Owner</th>
      <th>Review</th>
    </tr>
  </thead>
  <tbody id="riskBody"></tbody>
</table>

<div id="detailPanel" class="detail-panel"></div>

<script>
const RISKS = {risks_json};
const STATS = {stats_json};
let sortCol = -1, sortAsc = true;

function init() {{
  // Stats
  const g = document.getElementById('statsGrid');
  const cards = [
    ['Total Risks', STATS.total_risks],
    ['Active', STATS.active_risks],
    ['Overdue Reviews', STATS.overdue_reviews],
    ['Avg Residual', STATS.avg_residual_score],
    ['Max Residual', STATS.max_residual_score],
    ['Mitigation Coverage', STATS.mitigation_coverage + '%'],
  ];
  cards.forEach(([l,v]) => {{
    const d = document.createElement('div');
    d.className = 'stat-card';
    d.innerHTML = `<div class="label">${{l}}</div><div class="value">${{v}}</div>`;
    g.appendChild(d);
  }});
  // Category filter
  const cats = [...new Set(RISKS.map(r => r.category))].sort();
  const cf = document.getElementById('categoryFilter');
  cats.forEach(c => {{ const o = document.createElement('option'); o.value = c; o.text = c; cf.add(o); }});
  renderTable(RISKS);
}}

function renderTable(risks) {{
  const body = document.getElementById('riskBody');
  body.innerHTML = '';
  const now = new Date();
  risks.forEach((r, i) => {{
    const overdue = r.next_review && new Date(r.next_review) < now && r.status !== 'Closed';
    const tr = document.createElement('tr');
    tr.style.cursor = 'pointer';
    tr.onclick = () => showDetail(r);
    tr.innerHTML = `
      <td>${{r.risk_id}}</td>
      <td>${{r.title}}</td>
      <td>${{r.category}}</td>
      <td><span class="badge badge-${{r.risk_level}}">${{r.risk_level}}</span></td>
      <td>${{r.inherent_score}}</td>
      <td>${{r.residual_score}}</td>
      <td><span class="status">${{r.status}}</span></td>
      <td>${{r.owner}}</td>
      <td>${{overdue ? '<span class="badge badge-overdue">⚠ OVERDUE</span>' : (r.next_review ? r.next_review.slice(0,10) : '—')}}</td>
    `;
    body.appendChild(tr);
  }});
}}

function filterTable() {{
  const q = document.getElementById('searchBox').value.toLowerCase();
  const st = document.getElementById('statusFilter').value;
  const lv = document.getElementById('levelFilter').value;
  const cat = document.getElementById('categoryFilter').value;
  const od = document.getElementById('overdueOnly').checked;
  const now = new Date();
  let f = RISKS.filter(r => {{
    if (q && !r.title.toLowerCase().includes(q) && !r.risk_id.toLowerCase().includes(q) && !r.description.toLowerCase().includes(q)) return false;
    if (st && r.status !== st) return false;
    if (lv && r.risk_level !== lv) return false;
    if (cat && r.category !== cat) return false;
    if (od) {{ const overdue = r.next_review && new Date(r.next_review) < now && r.status !== 'Closed'; if (!overdue) return false; }}
    return true;
  }});
  renderTable(f);
}}

function sortTable(col) {{
  if (sortCol === col) sortAsc = !sortAsc; else {{ sortCol = col; sortAsc = true; }}
  const keys = ['risk_id','title','category','risk_level','inherent_score','residual_score','status','owner'];
  const key = keys[col];
  const sorted = [...RISKS].sort((a,b) => {{
    let av = a[key], bv = b[key];
    if (typeof av === 'number') return sortAsc ? av - bv : bv - av;
    return sortAsc ? String(av).localeCompare(String(bv)) : String(bv).localeCompare(String(av));
  }});
  renderTable(sorted);
}}

function showDetail(r) {{
  const p = document.getElementById('detailPanel');
  p.className = 'detail-panel visible';
  const now = new Date();
  const overdue = r.next_review && new Date(r.next_review) < now && r.status !== 'Closed';
  let html = `<h2>${{r.risk_id}}: ${{r.title}}</h2>`;
  html += `<div class="detail-section"><h3>Description</h3><p>${{r.description}}</p></div>`;
  html += `<div class="detail-section"><h3>Assessment</h3>
    <p>Category: ${{r.category}} | Likelihood: ${{r.likelihood}}/5 | Impact: ${{r.impact}}/5</p>
    <p>Inherent: ${{r.inherent_score}} (<span class="badge badge-${{r.risk_level}}">${{r.risk_level}}</span>)
       → Residual: ${{r.residual_score}} (<span class="badge badge-${{r.residual_level}}">${{r.residual_level}}</span>)</p>
    <p>Status: <span class="status">${{r.status}}</span> | Owner: ${{r.owner || '—'}}${{r.agent_id ? ' | Agent: '+r.agent_id : ''}}</p>
    ${{overdue ? '<p><span class="badge badge-overdue">⚠ Review overdue since '+r.next_review.slice(0,10)+'</span></p>' : ''}}
  </div>`;
  if (r.mitigations.length) {{
    html += `<div class="detail-section"><h3>Mitigations (${{r.mitigations.length}})</h3><ul>`;
    r.mitigations.forEach(m => {{
      html += `<li><strong>${{m.description}}</strong> — ${{m.status}} (effectiveness: ${{Math.round(m.effectiveness*100)}}%${{m.owner ? ', owner: '+m.owner : ''}})</li>`;
    }});
    html += `</ul></div>`;
  }}
  if (r.score_history.length) {{
    html += `<div class="detail-section"><h3>Score History</h3><ul>`;
    r.score_history.forEach(([ts, sc]) => {{
      html += `<li>${{ts.slice(0,10)}}: ${{sc}}</li>`;
    }});
    html += `</ul></div>`;
  }}
  if (r.audit_trail.length) {{
    html += `<div class="detail-section"><h3>Audit Trail</h3><ul>`;
    r.audit_trail.forEach(a => {{
      html += `<li><strong>${{a.action}}</strong> — ${{a.details}} (${{a.timestamp.slice(0,16).replace('T',' ')}}${{a.user !== 'system' ? ', by '+a.user : ''}})</li>`;
    }});
    html += `</ul></div>`;
  }}
  p.innerHTML = html;
  p.scrollIntoView({{behavior:'smooth'}});
}}

function exportCSV() {{
  let csv = 'Risk ID,Title,Category,Likelihood,Impact,Inherent,Residual,Level,Status,Owner,Agent,Mitigations,Overdue\\n';
  const now = new Date();
  RISKS.forEach(r => {{
    const overdue = r.next_review && new Date(r.next_review) < now && r.status !== 'Closed';
    csv += `${{r.risk_id}},"${{r.title}}",${{r.category}},${{r.likelihood}},${{r.impact}},${{r.inherent_score}},${{r.residual_score}},${{r.risk_level}},${{r.status}},${{r.owner}},${{r.agent_id}},${{r.mitigations.length}},${{overdue?'Yes':'No'}}\\n`;
  }});
  const blob = new Blob([csv], {{type:'text/csv'}});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
  a.download = 'risk_register.csv'; a.click();
}}

init();
</script>
</body>
</html>""")


# ── CLI ──────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Risk Register — formal risk tracking & lifecycle management")
    parser.add_argument("--agents", type=int, default=8, help="Number of agents for simulation")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--import", dest="import_file", help="Import risks from JSON file")
    parser.add_argument("--overdue", action="store_true", help="Show overdue risks only")
    parser.add_argument("--stats", action="store_true", help="Show summary statistics")
    parser.add_argument("--top", type=int, help="Show top N riskiest entries")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--csv", action="store_true", help="CSV output")
    parser.add_argument("--html", action="store_true", help="HTML report output")
    parser.add_argument("-o", "--output", help="Output file path")
    args = parser.parse_args(argv)

    if args.import_file:
        reg = RiskRegister.from_json(args.import_file)
    else:
        reg = RiskRegister(RegisterConfig(agent_count=args.agents, seed=args.seed or 42))
        reg.populate_from_simulation()

    if args.json:
        out = reg.to_json()
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(out)
            print(f"✓ JSON exported to {args.output}")
        else:
            print(out)
    elif args.csv:
        if args.output:
            reg.export_csv(args.output)
            print(f"✓ CSV exported to {args.output}")
        else:
            print(reg.tabular())
    elif args.html:
        html = reg.to_html()
        path = args.output or "risk_register.html"
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"✓ HTML report saved to {path}")
    elif args.stats:
        print(reg.summary())
    elif args.top:
        top = reg.top_risks(args.top)
        for r in top:
            print(f"  {r.risk_id}  {r.title[:45]:45s}  {r.risk_level:10s}  residual={r.residual_score:.1f}  {r.status.value}")
    else:
        print(reg.summary())
        print()
        print(reg.tabular(show_overdue_only=args.overdue))


if __name__ == "__main__":
    main()
