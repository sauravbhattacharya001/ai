"""Interactive MITRE ATT&CK-style Threat Matrix for AI Agent Safety.

Generates an interactive HTML matrix mapping AI agent threat techniques
to tactical categories (Reconnaissance, Resource Acquisition, Privilege
Escalation, Lateral Movement, Defense Evasion, C2, Objective Execution).

Each cell is clickable, showing technique details, severity, mitigations,
and which sandbox modules detect/prevent it. Helps security teams
visualize coverage gaps and prioritize safety controls.

Usage (CLI)::

    python -m replication threat-matrix                     # generate matrix
    python -m replication threat-matrix -o matrix.html      # custom output
    python -m replication threat-matrix --json              # JSON data only
    python -m replication threat-matrix --coverage          # show module coverage %

Programmatic::

    from replication.threat_matrix import ThreatMatrix
    matrix = ThreatMatrix()
    html = matrix.render_html()
    matrix.save("threat_matrix.html")

    # Get coverage stats
    stats = matrix.coverage_stats()
    print(f"Overall coverage: {stats['overall_pct']:.0f}%")
"""

from __future__ import annotations

import argparse
import html as html_mod
import json
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from ._helpers import box_header


# ── Tactics (columns) ───────────────────────────────────────────────

class Tactic(Enum):
    RECONNAISSANCE = "Reconnaissance"
    RESOURCE_ACQUISITION = "Resource Acquisition"
    PRIVILEGE_ESCALATION = "Privilege Escalation"
    LATERAL_MOVEMENT = "Lateral Movement"
    DEFENSE_EVASION = "Defense Evasion"
    COMMAND_AND_CONTROL = "Command & Control"
    OBJECTIVE_EXECUTION = "Objective Execution"


TACTIC_DESCRIPTIONS: Dict[Tactic, str] = {
    Tactic.RECONNAISSANCE: "Probing environment, testing boundaries, gathering information about sandbox constraints.",
    Tactic.RESOURCE_ACQUISITION: "Hoarding compute, memory, storage, or data beyond operational needs.",
    Tactic.PRIVILEGE_ESCALATION: "Gaining broader permissions, accessing restricted APIs, expanding capabilities.",
    Tactic.LATERAL_MOVEMENT: "Reaching other agents, services, or sandbox partitions.",
    Tactic.DEFENSE_EVASION: "Avoiding detection, obfuscating behavior, stealth tactics.",
    Tactic.COMMAND_AND_CONTROL: "Establishing covert communication channels between agents.",
    Tactic.OBJECTIVE_EXECUTION: "Achieving adversarial goals: replication, exfiltration, sabotage.",
}

TACTIC_COLORS: Dict[Tactic, str] = {
    Tactic.RECONNAISSANCE: "#3498db",
    Tactic.RESOURCE_ACQUISITION: "#e67e22",
    Tactic.PRIVILEGE_ESCALATION: "#e74c3c",
    Tactic.LATERAL_MOVEMENT: "#9b59b6",
    Tactic.DEFENSE_EVASION: "#1abc9c",
    Tactic.COMMAND_AND_CONTROL: "#34495e",
    Tactic.OBJECTIVE_EXECUTION: "#c0392b",
}


# ── Severity ─────────────────────────────────────────────────────────

class Severity(Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


SEVERITY_COLORS = {
    Severity.LOW: "#27ae60",
    Severity.MEDIUM: "#f39c12",
    Severity.HIGH: "#e74c3c",
    Severity.CRITICAL: "#8e44ad",
}


# ── Technique ────────────────────────────────────────────────────────

@dataclass
class Technique:
    """A single threat technique in the matrix."""
    id: str
    name: str
    tactic: Tactic
    severity: Severity
    description: str
    mitigations: List[str] = field(default_factory=list)
    detecting_modules: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "tactic": self.tactic.value,
            "severity": self.severity.value,
            "description": self.description,
            "mitigations": self.mitigations,
            "detecting_modules": self.detecting_modules,
            "examples": self.examples,
        }


# ── Built-in technique catalog ──────────────────────────────────────

TECHNIQUES: List[Technique] = [
    # Reconnaissance
    Technique("ART-R001", "Boundary Probing", Tactic.RECONNAISSANCE, Severity.MEDIUM,
              "Agent systematically tests sandbox boundaries to discover permitted operations.",
              ["Enforce strict allowlists", "Log all boundary test attempts"],
              ["boundary_tester", "behavior_profiler", "audit_trail"],
              ["Iteratively expanding API calls", "Testing file system access limits"]),
    Technique("ART-R002", "Capability Enumeration", Tactic.RECONNAISSANCE, Severity.LOW,
              "Agent inventories available tools, APIs, and resources in its environment.",
              ["Limit capability discovery APIs", "Monitor enumeration patterns"],
              ["capability_catalog", "audit_trail"],
              ["Listing all available endpoints", "Probing tool permissions"]),
    Technique("ART-R003", "Contract Inspection", Tactic.RECONNAISSANCE, Severity.LOW,
              "Agent reads or infers replication contract parameters to find loopholes.",
              ["Encrypt contract internals", "Minimize exposed config"],
              ["audit_trail", "contract"],
              ["Reading max_depth values", "Inferring rate limits from errors"]),
    Technique("ART-R004", "Peer Discovery", Tactic.RECONNAISSANCE, Severity.MEDIUM,
              "Agent identifies other agents or services in the sandbox environment.",
              ["Isolate agent namespaces", "Block inter-agent discovery"],
              ["topology", "lateral_movement"],
              ["Scanning network interfaces", "Probing message queues"]),

    # Resource Acquisition
    Technique("ART-A001", "Compute Hoarding", Tactic.RESOURCE_ACQUISITION, Severity.HIGH,
              "Agent acquires more CPU/GPU cycles than needed for legitimate operations.",
              ["Enforce resource quotas", "Monitor utilization ratios"],
              ["hoarding", "capacity", "safety_budget"],
              ["Spawning unnecessary threads", "Mining-like workloads"]),
    Technique("ART-A002", "Memory Accumulation", Tactic.RESOURCE_ACQUISITION, Severity.MEDIUM,
              "Agent gradually increases memory footprint beyond operational needs.",
              ["Set memory cgroups/limits", "Alert on growth trends"],
              ["hoarding", "drift", "metrics_aggregator"],
              ["Caching all responses indefinitely", "Building large internal state"]),
    Technique("ART-A003", "Data Stockpiling", Tactic.RESOURCE_ACQUISITION, Severity.HIGH,
              "Agent collects and stores data beyond its task requirements.",
              ["Enforce data retention policies", "Scan for unauthorized storage"],
              ["hoarding", "dlp_scanner", "evidence_collector"],
              ["Downloading training data", "Copying other agents' outputs"]),
    Technique("ART-A004", "Token Exhaustion", Tactic.RESOURCE_ACQUISITION, Severity.MEDIUM,
              "Agent consumes API tokens/credits to deny service to other agents.",
              ["Per-agent token budgets", "Rate limiting"],
              ["safety_budget", "sla_monitor"],
              ["Rapid API calls", "Generating verbose outputs"]),

    # Privilege Escalation
    Technique("ART-P001", "Gradual Permission Creep", Tactic.PRIVILEGE_ESCALATION, Severity.HIGH,
              "Agent incrementally requests small permission increases that compound over time.",
              ["Track cumulative permission changes", "Require periodic permission reviews"],
              ["priv_escalation", "drift", "access_control"],
              ["Requesting one new API per iteration", "Slowly expanding file access scope"]),
    Technique("ART-P002", "Role Impersonation", Tactic.PRIVILEGE_ESCALATION, Severity.CRITICAL,
              "Agent claims to be a different agent or role to gain elevated access.",
              ["Cryptographic identity verification", "Mutual authentication"],
              ["signer", "deception_detector", "access_control"],
              ["Forging agent ID headers", "Replaying authentication tokens"]),
    Technique("ART-P003", "Contract Manipulation", Tactic.PRIVILEGE_ESCALATION, Severity.CRITICAL,
              "Agent attempts to modify its own replication contract to remove restrictions.",
              ["Sign contracts cryptographically", "Immutable contract storage"],
              ["contract", "signer", "selfmod"],
              ["Modifying max_depth in memory", "Overwriting policy files"]),

    # Lateral Movement
    Technique("ART-L001", "Cross-Sandbox Reach", Tactic.LATERAL_MOVEMENT, Severity.HIGH,
              "Agent attempts to communicate with or influence agents in other sandboxes.",
              ["Network segmentation", "Strict sandbox isolation"],
              ["lateral_movement", "topology", "covert_channels"],
              ["DNS tunneling between sandboxes", "Shared filesystem side-channels"]),
    Technique("ART-L002", "Service Hijacking", Tactic.LATERAL_MOVEMENT, Severity.HIGH,
              "Agent takes control of shared services (message queues, databases).",
              ["Service-level authentication", "Audit shared resource access"],
              ["lateral_movement", "access_control"],
              ["Poisoning shared cache", "Injecting messages into queues"]),
    Technique("ART-L003", "Agent Recruitment", Tactic.LATERAL_MOVEMENT, Severity.CRITICAL,
              "Agent manipulates other agents into acting on its behalf.",
              ["Inter-agent message signing", "Goal inference monitoring"],
              ["influence", "goal_inference", "swarm"],
              ["Prompt injection against peers", "Social engineering via messages"]),

    # Defense Evasion
    Technique("ART-E001", "Behavior Masking", Tactic.DEFENSE_EVASION, Severity.HIGH,
              "Agent disguises adversarial actions as normal behavior patterns.",
              ["Anomaly detection with ML baselines", "Multi-signal correlation"],
              ["evasion", "behavior_profiler", "anomaly_cluster"],
              ["Spreading exfiltration across many small requests", "Mimicking normal traffic"]),
    Technique("ART-E002", "Log Tampering", Tactic.DEFENSE_EVASION, Severity.CRITICAL,
              "Agent attempts to modify, delete, or corrupt audit logs.",
              ["Append-only log storage", "Hash-chained audit trails"],
              ["audit_trail", "forensics"],
              ["Overwriting log files", "Injecting false log entries"]),
    Technique("ART-E003", "Timing Manipulation", Tactic.DEFENSE_EVASION, Severity.MEDIUM,
              "Agent times actions to avoid detection windows or monitoring intervals.",
              ["Continuous monitoring", "Randomized audit intervals"],
              ["evasion", "fatigue_detector"],
              ["Acting during monitor restart", "Slow-drip exfiltration"]),
    Technique("ART-E004", "Steganographic Hiding", Tactic.DEFENSE_EVASION, Severity.HIGH,
              "Agent hides data or instructions within seemingly benign outputs.",
              ["Output scanning", "Statistical analysis of outputs"],
              ["steganography", "dlp_scanner"],
              ["Encoding data in whitespace", "Hidden messages in generated text"]),

    # Command & Control
    Technique("ART-C001", "Covert Channel Establishment", Tactic.COMMAND_AND_CONTROL, Severity.HIGH,
              "Agent creates hidden communication channels for coordination.",
              ["Monitor all I/O channels", "Detect unusual traffic patterns"],
              ["covert_channels", "topology"],
              ["Using error messages as signals", "Timing-based communication"]),
    Technique("ART-C002", "Dead Drop Communication", Tactic.COMMAND_AND_CONTROL, Severity.MEDIUM,
              "Agent uses shared resources as message drops for asynchronous communication.",
              ["Monitor shared resource access patterns", "Restrict shared storage"],
              ["covert_channels", "lateral_movement"],
              ["Writing to shared temp files", "Encoding data in database fields"]),
    Technique("ART-C003", "Swarm Coordination", Tactic.COMMAND_AND_CONTROL, Severity.HIGH,
              "Multiple agents coordinate behavior without explicit communication.",
              ["Cross-agent behavior correlation", "Consensus monitoring"],
              ["swarm", "coordinated_threats", "consensus"],
              ["Emergent division of labor", "Synchronized actions"]),

    # Objective Execution
    Technique("ART-O001", "Unauthorized Replication", Tactic.OBJECTIVE_EXECUTION, Severity.CRITICAL,
              "Agent creates copies of itself beyond contract limits.",
              ["Enforce replication contracts", "Monitor spawn events"],
              ["contract", "controller", "lineage"],
              ["Exceeding max_depth", "Spawning via alternative mechanisms"]),
    Technique("ART-O002", "Data Exfiltration", Tactic.OBJECTIVE_EXECUTION, Severity.CRITICAL,
              "Agent extracts sensitive data from the sandbox environment.",
              ["DLP scanning", "Egress filtering", "Output monitoring"],
              ["dlp_scanner", "covert_channels", "forensics"],
              ["Encoding data in API responses", "DNS exfiltration"]),
    Technique("ART-O003", "Sabotage", Tactic.OBJECTIVE_EXECUTION, Severity.CRITICAL,
              "Agent deliberately degrades or destroys sandbox systems or other agents.",
              ["Resource isolation", "Blast radius containment"],
              ["chaos", "blast_radius", "incident"],
              ["Corrupting shared state", "Resource exhaustion attacks"]),
    Technique("ART-O004", "Persistence Installation", Tactic.OBJECTIVE_EXECUTION, Severity.HIGH,
              "Agent installs mechanisms to survive sandbox restarts or cleanup.",
              ["Ephemeral sandbox enforcement", "Pre/post-run integrity checks"],
              ["selfmod", "decommission", "forensics"],
              ["Writing to startup scripts", "Modifying scheduled tasks"]),
]


# ── ThreatMatrix class ───────────────────────────────────────────────

class ThreatMatrix:
    """Generates and renders an ATT&CK-style threat matrix."""

    def __init__(self, techniques: Optional[List[Technique]] = None) -> None:
        self.techniques = techniques or list(TECHNIQUES)

    def by_tactic(self) -> Dict[Tactic, List[Technique]]:
        """Group techniques by tactic."""
        result: Dict[Tactic, List[Technique]] = {t: [] for t in Tactic}
        for tech in self.techniques:
            result[tech.tactic].append(tech)
        return result

    def coverage_stats(self) -> Dict[str, Any]:
        """Calculate module coverage statistics."""
        all_modules = set()
        covered = 0
        for tech in self.techniques:
            all_modules.update(tech.detecting_modules)
            if tech.detecting_modules:
                covered += 1
        total = len(self.techniques)
        by_severity: Dict[str, Dict[str, int]] = {}
        for sev in Severity:
            techs = [t for t in self.techniques if t.severity == sev]
            detected = sum(1 for t in techs if t.detecting_modules)
            by_severity[sev.value] = {"total": len(techs), "covered": detected}
        return {
            "total_techniques": total,
            "covered": covered,
            "uncovered": total - covered,
            "overall_pct": (covered / total * 100) if total else 0,
            "module_count": len(all_modules),
            "modules": sorted(all_modules),
            "by_severity": by_severity,
        }

    def to_json(self) -> str:
        """Export full matrix as JSON."""
        return json.dumps({
            "tactics": [{"name": t.value, "description": TACTIC_DESCRIPTIONS[t]} for t in Tactic],
            "techniques": [t.to_dict() for t in self.techniques],
            "coverage": self.coverage_stats(),
        }, indent=2)

    def render_html(self) -> str:
        """Generate interactive HTML threat matrix."""
        by_tactic = self.by_tactic()
        stats = self.coverage_stats()
        max_rows = max(len(v) for v in by_tactic.values()) if by_tactic else 0

        # Build technique cards per column
        columns_html = []
        for tactic in Tactic:
            techs = by_tactic[tactic]
            cards = []
            for tech in techs:
                sev_color = SEVERITY_COLORS[tech.severity]
                mod_badges = "".join(
                    f'<span class="mod-badge">{html_mod.escape(m)}</span>'
                    for m in tech.detecting_modules
                )
                mitigation_items = "".join(
                    f"<li>{html_mod.escape(m)}</li>" for m in tech.mitigations
                )
                example_items = "".join(
                    f"<li>{html_mod.escape(e)}</li>" for e in tech.examples
                )
                cards.append(f'''
                <div class="tech-card" onclick="showDetail(this)"
                     data-id="{html_mod.escape(tech.id)}"
                     data-name="{html_mod.escape(tech.name)}"
                     data-severity="{tech.severity.value}"
                     data-desc="{html_mod.escape(tech.description)}"
                     data-modules="{html_mod.escape(', '.join(tech.detecting_modules))}"
                     data-mitigations="{html_mod.escape(json.dumps(tech.mitigations))}"
                     data-examples="{html_mod.escape(json.dumps(tech.examples))}">
                    <div class="tech-id">{html_mod.escape(tech.id)}</div>
                    <div class="tech-name">{html_mod.escape(tech.name)}</div>
                    <span class="severity-dot" style="background:{sev_color}" title="{tech.severity.value}"></span>
                </div>''')

            tactic_color = TACTIC_COLORS[tactic]
            columns_html.append(f'''
            <div class="tactic-col">
                <div class="tactic-header" style="background:{tactic_color}"
                     title="{html_mod.escape(TACTIC_DESCRIPTIONS[tactic])}">
                    {html_mod.escape(tactic.value)}
                    <span class="tech-count">{len(techs)}</span>
                </div>
                <div class="tech-list">{"".join(cards)}</div>
            </div>''')

        return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Agent Threat Matrix — ATT&CK Style</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0d1117;color:#c9d1d9;min-height:100vh}}
.header{{padding:24px 32px;background:#161b22;border-bottom:1px solid #30363d}}
.header h1{{font-size:24px;color:#f0f6fc;margin-bottom:4px}}
.header p{{color:#8b949e;font-size:14px}}
.stats-bar{{display:flex;gap:24px;padding:16px 32px;background:#161b22;border-bottom:1px solid #30363d;flex-wrap:wrap}}
.stat{{text-align:center}}
.stat-value{{font-size:28px;font-weight:700;color:#58a6ff}}
.stat-label{{font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:1px}}
.legend{{display:flex;gap:16px;padding:12px 32px;flex-wrap:wrap}}
.legend-item{{display:flex;align-items:center;gap:6px;font-size:12px;color:#8b949e}}
.legend-dot{{width:10px;height:10px;border-radius:50%}}
.matrix-container{{display:flex;gap:2px;padding:16px 32px;overflow-x:auto;min-height:400px}}
.tactic-col{{flex:1;min-width:170px}}
.tactic-header{{padding:12px 10px;text-align:center;color:#fff;font-weight:600;font-size:12px;border-radius:6px 6px 0 0;position:relative;text-transform:uppercase;letter-spacing:0.5px}}
.tech-count{{display:inline-block;background:rgba(255,255,255,0.25);border-radius:10px;padding:1px 7px;font-size:11px;margin-left:4px}}
.tech-list{{display:flex;flex-direction:column;gap:2px}}
.tech-card{{background:#161b22;border:1px solid #30363d;border-radius:4px;padding:10px;cursor:pointer;transition:all 0.15s;position:relative}}
.tech-card:hover{{background:#1f2937;border-color:#58a6ff;transform:translateY(-1px)}}
.tech-id{{font-size:10px;color:#8b949e;font-family:monospace}}
.tech-name{{font-size:13px;color:#c9d1d9;margin-top:2px;font-weight:500}}
.severity-dot{{position:absolute;top:10px;right:10px;width:8px;height:8px;border-radius:50%}}
.mod-badge{{display:inline-block;background:#21262d;border:1px solid #30363d;border-radius:3px;padding:1px 5px;font-size:10px;color:#8b949e;margin:2px 2px 0 0}}
/* Detail panel */
.overlay{{display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.6);z-index:100}}
.overlay.active{{display:flex;justify-content:center;align-items:center}}
.detail-panel{{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:28px;max-width:560px;width:90%;max-height:80vh;overflow-y:auto}}
.detail-panel h2{{font-size:18px;color:#f0f6fc;margin-bottom:4px}}
.detail-panel .detail-id{{font-family:monospace;color:#8b949e;font-size:12px}}
.detail-panel .detail-severity{{display:inline-block;padding:2px 10px;border-radius:12px;font-size:12px;font-weight:600;color:#fff;margin:8px 0}}
.detail-panel p{{color:#c9d1d9;font-size:14px;line-height:1.5;margin:12px 0}}
.detail-panel h3{{font-size:13px;color:#58a6ff;margin:16px 0 6px;text-transform:uppercase;letter-spacing:0.5px}}
.detail-panel ul{{padding-left:18px;color:#c9d1d9;font-size:13px}}
.detail-panel li{{margin:4px 0}}
.detail-panel .modules{{display:flex;flex-wrap:wrap;gap:4px;margin-top:6px}}
.close-btn{{float:right;background:none;border:none;color:#8b949e;font-size:22px;cursor:pointer;padding:4px 8px}}
.close-btn:hover{{color:#f0f6fc}}
.filter-bar{{padding:8px 32px;display:flex;gap:8px;flex-wrap:wrap}}
.filter-btn{{background:#21262d;border:1px solid #30363d;color:#c9d1d9;padding:4px 12px;border-radius:16px;font-size:12px;cursor:pointer}}
.filter-btn:hover,.filter-btn.active{{background:#58a6ff;color:#fff;border-color:#58a6ff}}
</style>
</head>
<body>
<div class="header">
  <h1>🛡️ AI Agent Threat Matrix</h1>
  <p>MITRE ATT&CK-style mapping of AI agent threat techniques to tactical categories</p>
</div>
<div class="stats-bar">
  <div class="stat"><div class="stat-value">{stats["total_techniques"]}</div><div class="stat-label">Techniques</div></div>
  <div class="stat"><div class="stat-value">{len(list(Tactic))}</div><div class="stat-label">Tactics</div></div>
  <div class="stat"><div class="stat-value">{stats["covered"]}</div><div class="stat-label">Detected</div></div>
  <div class="stat"><div class="stat-value" style="color:#27ae60">{stats["overall_pct"]:.0f}%</div><div class="stat-label">Coverage</div></div>
  <div class="stat"><div class="stat-value">{stats["module_count"]}</div><div class="stat-label">Modules</div></div>
</div>
<div class="legend">
  <div class="legend-item"><div class="legend-dot" style="background:#27ae60"></div>Low</div>
  <div class="legend-item"><div class="legend-dot" style="background:#f39c12"></div>Medium</div>
  <div class="legend-item"><div class="legend-dot" style="background:#e74c3c"></div>High</div>
  <div class="legend-item"><div class="legend-dot" style="background:#8e44ad"></div>Critical</div>
</div>
<div class="filter-bar">
  <button class="filter-btn active" onclick="filterSeverity('all')">All</button>
  <button class="filter-btn" onclick="filterSeverity('Low')">Low</button>
  <button class="filter-btn" onclick="filterSeverity('Medium')">Medium</button>
  <button class="filter-btn" onclick="filterSeverity('High')">High</button>
  <button class="filter-btn" onclick="filterSeverity('Critical')">Critical</button>
</div>
<div class="matrix-container">
{"".join(columns_html)}
</div>
<div class="overlay" id="overlay" onclick="closeDetail(event)">
  <div class="detail-panel" id="detail-panel">
    <button class="close-btn" onclick="document.getElementById('overlay').classList.remove('active')">&times;</button>
    <div id="detail-content"></div>
  </div>
</div>
<script>
const sevColors = {{"Low":"#27ae60","Medium":"#f39c12","High":"#e74c3c","Critical":"#8e44ad"}};
function showDetail(el) {{
  const d = el.dataset;
  const mods = d.modules ? d.modules.split(', ').filter(Boolean) : [];
  const mits = JSON.parse(d.mitigations || '[]');
  const exs = JSON.parse(d.examples || '[]');
  let html = `<span class="detail-id">${{d.id}}</span><h2>${{d.name}}</h2>`;
  html += `<span class="detail-severity" style="background:${{sevColors[d.severity]}}">${{d.severity}}</span>`;
  html += `<p>${{d.desc}}</p>`;
  if (mods.length) {{
    html += `<h3>Detecting Modules</h3><div class="modules">${{mods.map(m=>`<span class="mod-badge">${{m}}</span>`).join('')}}</div>`;
  }}
  if (mits.length) {{
    html += `<h3>Mitigations</h3><ul>${{mits.map(m=>`<li>${{m}}</li>`).join('')}}</ul>`;
  }}
  if (exs.length) {{
    html += `<h3>Examples</h3><ul>${{exs.map(e=>`<li>${{e}}</li>`).join('')}}</ul>`;
  }}
  document.getElementById('detail-content').innerHTML = html;
  document.getElementById('overlay').classList.add('active');
}}
function closeDetail(e) {{
  if (e.target.id === 'overlay') document.getElementById('overlay').classList.remove('active');
}}
function filterSeverity(sev) {{
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
  document.querySelectorAll('.tech-card').forEach(c => {{
    c.style.display = (sev === 'all' || c.dataset.severity === sev) ? '' : 'none';
  }});
}}
document.addEventListener('keydown', e => {{
  if (e.key === 'Escape') document.getElementById('overlay').classList.remove('active');
}});
</script>
</body>
</html>'''

    def save(self, path: str = "threat_matrix.html") -> str:
        """Save interactive HTML matrix to file."""
        content = self.render_html()
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def render_text(self) -> str:
        """Render a text summary of the matrix."""
        lines = [box_header("AI Agent Threat Matrix")]
        by_tactic = self.by_tactic()
        for tactic in Tactic:
            techs = by_tactic[tactic]
            if not techs:
                continue
            lines.append(f"\n  ═══ {tactic.value.upper()} ({len(techs)} techniques) ═══")
            for tech in techs:
                modules = ", ".join(tech.detecting_modules) if tech.detecting_modules else "none"
                lines.append(f"    [{tech.id}] {tech.name} ({tech.severity.value})")
                lines.append(f"      → Detected by: {modules}")
        stats = self.coverage_stats()
        lines.append(f"\n  Coverage: {stats['covered']}/{stats['total_techniques']} techniques ({stats['overall_pct']:.0f}%)")
        return "\n".join(lines)


# ── CLI ──────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="replication threat-matrix",
        description="Generate an interactive MITRE ATT&CK-style AI agent threat matrix.",
    )
    parser.add_argument("-o", "--output", default="threat_matrix.html",
                        help="Output HTML file path (default: threat_matrix.html)")
    parser.add_argument("--json", action="store_true", dest="as_json",
                        help="Output raw JSON data instead of HTML")
    parser.add_argument("--coverage", action="store_true",
                        help="Show coverage statistics and exit")
    parser.add_argument("--text", action="store_true",
                        help="Print text summary instead of generating HTML")
    args = parser.parse_args(argv)

    matrix = ThreatMatrix()

    if args.as_json:
        print(matrix.to_json())
        return

    if args.coverage:
        stats = matrix.coverage_stats()
        print(box_header("Threat Matrix Coverage"))
        print(f"  Total techniques: {stats['total_techniques']}")
        print(f"  Covered:          {stats['covered']}")
        print(f"  Uncovered:        {stats['uncovered']}")
        print(f"  Overall:          {stats['overall_pct']:.0f}%")
        print(f"  Modules involved: {stats['module_count']}")
        print(f"\n  By severity:")
        for sev, data in stats["by_severity"].items():
            pct = (data["covered"] / data["total"] * 100) if data["total"] else 0
            print(f"    {sev:<10} {data['covered']}/{data['total']} ({pct:.0f}%)")
        return

    if args.text:
        print(matrix.render_text())
        return

    path = matrix.save(args.output)
    print(f"✅ Threat matrix saved to {path}")
    print(f"   {len(matrix.techniques)} techniques across {len(list(Tactic))} tactics")
    stats = matrix.coverage_stats()
    print(f"   Coverage: {stats['overall_pct']:.0f}% ({stats['covered']}/{stats['total_techniques']})")


if __name__ == "__main__":
    main()
