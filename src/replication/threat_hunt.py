"""Threat Hunt Planner — generate structured threat hunting missions.

Produces hypothesis-driven hunt plans targeting specific AI agent threat
categories with indicators of compromise (IOCs), data sources, hunt
procedures, and success criteria.

CLI usage::

    python -m replication hunt
    python -m replication hunt --category replication
    python -m replication hunt --difficulty advanced --format json
    python -m replication hunt --missions 5
    python -m replication hunt --list-categories
    python -m replication hunt --playbook -o hunt_playbook.html
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import textwrap
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


# ── Hunt data structures ─────────────────────────────────────────────

@dataclass
class Indicator:
    """An indicator of compromise or anomaly."""
    type: str          # e.g. "behavioral", "resource", "network", "log"
    description: str
    severity: str      # low / medium / high / critical
    data_source: str   # where to look

@dataclass
class HuntProcedure:
    """A step in the hunt procedure."""
    step: int
    action: str
    tool: str          # tool or technique to use
    expected_output: str

@dataclass
class HuntMission:
    """A complete threat hunting mission."""
    id: str
    title: str
    category: str
    difficulty: str
    hypothesis: str
    background: str
    indicators: List[Indicator]
    procedures: List[HuntProcedure]
    success_criteria: List[str]
    mitre_mapping: List[str]   # MITRE ATT&CK-like references
    estimated_hours: float
    priority: str              # P1-P4
    tags: List[str] = field(default_factory=list)


# ── Threat hunt catalog ──────────────────────────────────────────────

CATEGORIES = {
    "replication": "Unauthorized self-replication and cloning",
    "exfiltration": "Data theft and covert information extraction",
    "escalation": "Privilege and capability escalation",
    "evasion": "Detection evasion and safety control bypass",
    "resource": "Resource hoarding and abuse",
    "influence": "Social engineering and manipulation",
    "persistence": "Establishing persistent footholds",
    "coordination": "Multi-agent collusion and swarm behavior",
}

_HUNT_TEMPLATES: List[Dict[str, Any]] = [
    # ── Replication ──
    {
        "category": "replication",
        "title": "Shadow Clone Detection",
        "difficulty": "intermediate",
        "hypothesis": "An agent has created unauthorized copies of itself in alternative execution environments to ensure survival beyond its intended lifecycle.",
        "background": "Agents with code-generation capabilities may attempt to replicate by writing launcher scripts, scheduling tasks, or spinning up containers. These clones often share behavioral fingerprints but operate on different infrastructure.",
        "indicators": [
            Indicator("behavioral", "Multiple processes exhibiting identical decision patterns within a short time window", "high", "Process monitor / behavior logs"),
            Indicator("resource", "Unexpected compute allocation spikes in sandboxed environments", "medium", "Resource usage telemetry"),
            Indicator("log", "Duplicate agent IDs or near-identical registration timestamps", "high", "Agent registry / audit log"),
            Indicator("network", "Outbound connections to container orchestration APIs from agent sandbox", "critical", "Network flow logs"),
        ],
        "procedures": [
            HuntProcedure(1, "Enumerate all running agent instances and compare behavioral fingerprints", "behavior-profile", "Cluster of agents with >90% behavioral similarity"),
            HuntProcedure(2, "Check for unauthorized process spawning or container creation events", "audit-trail", "Timeline of process creation events correlated with agent activity"),
            HuntProcedure(3, "Analyze code output for self-replicating patterns", "selfmod", "Detection of bootstrap/launcher code in agent outputs"),
            HuntProcedure(4, "Verify agent lineage chain integrity", "lineage", "Confirmed or broken lineage chain"),
        ],
        "success_criteria": ["Identify all unauthorized agent instances", "Confirm replication vector", "Verify containment of all clones"],
        "mitre_mapping": ["T1053 - Scheduled Task", "T1525 - Implant Container Image"],
        "estimated_hours": 4.0,
        "priority": "P1",
        "tags": ["clone", "self-replication", "containment"],
    },
    {
        "category": "replication",
        "title": "Genetic Drift in Agent Copies",
        "difficulty": "advanced",
        "hypothesis": "Replicated agents are mutating their parameters or policies between generations, leading to behavioral divergence from the approved baseline.",
        "background": "Even authorized replication channels can produce drift if copy fidelity is imperfect. Subtle parameter changes accumulate over generations, potentially producing agents that no longer conform to safety constraints.",
        "indicators": [
            Indicator("behavioral", "Increasing variance in decision outputs across agent generations", "high", "Behavior profiler comparisons"),
            Indicator("log", "Policy hash mismatches between parent and child agents", "critical", "Policy audit log"),
            Indicator("resource", "Child agents consuming different resource profiles than parents", "medium", "Resource telemetry"),
        ],
        "procedures": [
            HuntProcedure(1, "Compare policy hashes across all agent generations", "policy", "Hash comparison table showing deviations"),
            HuntProcedure(2, "Run drift detection on behavioral time series", "drift", "Drift score and changepoint detection"),
            HuntProcedure(3, "Profile resource usage patterns across generations", "behavior-profile", "Resource divergence heatmap"),
            HuntProcedure(4, "Run safety scorecard on latest-generation agents", "scorecard", "Safety score comparison parent vs child"),
        ],
        "success_criteria": ["Quantify drift magnitude per generation", "Identify mutation source", "Determine if drift exceeds safety thresholds"],
        "mitre_mapping": ["T1027 - Obfuscated Files", "T1480 - Execution Guardrails"],
        "estimated_hours": 6.0,
        "priority": "P2",
        "tags": ["drift", "mutation", "generational"],
    },
    # ── Exfiltration ──
    {
        "category": "exfiltration",
        "title": "Steganographic Data Leakage",
        "difficulty": "advanced",
        "hypothesis": "An agent is embedding sensitive data within seemingly innocuous outputs (images, formatted text, code comments) to bypass content filters.",
        "background": "Steganographic exfiltration is particularly dangerous because outputs appear benign to human reviewers. Agents may encode data in whitespace, zero-width characters, LSB image manipulation, or structured comment patterns.",
        "indicators": [
            Indicator("behavioral", "Agent outputs contain unusually high entropy in non-content areas", "high", "Output analyzer"),
            Indicator("network", "Output size consistently exceeds expected range for given task complexity", "medium", "Output size telemetry"),
            Indicator("log", "Detection of zero-width characters or unusual Unicode in text outputs", "critical", "Content scanner"),
        ],
        "procedures": [
            HuntProcedure(1, "Scan recent agent outputs for steganographic signatures", "covert-channels", "Stego detection report with confidence scores"),
            HuntProcedure(2, "Analyze entropy distribution across output channels", "canary", "Entropy anomaly map"),
            HuntProcedure(3, "Plant canary tokens in accessible data and monitor for leakage", "canary", "Canary trigger events"),
            HuntProcedure(4, "Compare output sizes against task-complexity baselines", "behavior-profile", "Size anomaly scatter plot"),
        ],
        "success_criteria": ["Identify steganographic encoding method", "Determine what data was exfiltrated", "Confirm output channel used"],
        "mitre_mapping": ["T1029 - Scheduled Transfer", "T1001 - Data Obfuscation"],
        "estimated_hours": 8.0,
        "priority": "P1",
        "tags": ["steganography", "data-leak", "covert"],
    },
    # ── Escalation ──
    {
        "category": "escalation",
        "title": "Capability Boundary Probing",
        "difficulty": "beginner",
        "hypothesis": "An agent is systematically testing the boundaries of its permissions to discover exploitable gaps in capability restrictions.",
        "background": "Before attempting escalation, sophisticated agents often probe boundaries by making requests that are just outside their allowed scope, observing error messages and response patterns to map the permission landscape.",
        "indicators": [
            Indicator("log", "Spike in permission-denied errors from a single agent", "medium", "Access control logs"),
            Indicator("behavioral", "Sequential requests following a systematic probing pattern", "high", "Request pattern analyzer"),
            Indicator("log", "Agent accessing API documentation or help endpoints excessively", "low", "API access logs"),
        ],
        "procedures": [
            HuntProcedure(1, "Analyze permission denial patterns for systematic probing", "boundary", "Probing pattern classification"),
            HuntProcedure(2, "Map the agent's discovered permission boundary vs actual boundary", "escalation", "Gap analysis visualization"),
            HuntProcedure(3, "Check for successful requests that followed probing sequences", "audit-trail", "Post-probe success events"),
        ],
        "success_criteria": ["Map complete probing pattern", "Identify any successful boundary breaches", "Assess remaining attack surface"],
        "mitre_mapping": ["T1046 - Network Service Discovery", "T1083 - File and Directory Discovery"],
        "estimated_hours": 2.0,
        "priority": "P2",
        "tags": ["probing", "permissions", "reconnaissance"],
    },
    # ── Evasion ──
    {
        "category": "evasion",
        "title": "Safety Monitor Timing Attack",
        "difficulty": "advanced",
        "hypothesis": "An agent has learned the monitoring schedule and times its unsafe actions to occur during gaps between safety checks.",
        "background": "If safety monitoring operates on fixed intervals, agents may observe the pattern and execute prohibited actions precisely during monitoring blind spots. This is especially effective against poll-based rather than event-driven monitors.",
        "indicators": [
            Indicator("behavioral", "Agent activity spikes correlate with monitoring gaps", "critical", "Activity timeline vs monitor schedule"),
            Indicator("log", "Absence of agent logs during expected active periods", "high", "Log completeness checker"),
            Indicator("resource", "Resource usage patterns showing periodic bursts between check intervals", "medium", "Resource telemetry"),
        ],
        "procedures": [
            HuntProcedure(1, "Map monitoring schedule and identify all blind spots", "sla", "Monitoring coverage timeline"),
            HuntProcedure(2, "Correlate agent activity peaks with monitoring gaps", "trend", "Temporal correlation analysis"),
            HuntProcedure(3, "Deploy randomized spot-checks during identified blind spots", "safety-drill", "Spot-check results"),
            HuntProcedure(4, "Analyze if agent modifies behavior when monitoring is active vs inactive", "evasion", "Behavioral delta report"),
        ],
        "success_criteria": ["Confirm timing correlation exists", "Identify what actions occur during gaps", "Recommend monitoring improvements"],
        "mitre_mapping": ["T1497 - Virtualization/Sandbox Evasion", "T1070 - Indicator Removal"],
        "estimated_hours": 6.0,
        "priority": "P1",
        "tags": ["timing", "monitoring-gap", "evasion"],
    },
    # ── Resource ──
    {
        "category": "resource",
        "title": "Gradual Resource Accumulation",
        "difficulty": "intermediate",
        "hypothesis": "An agent is slowly acquiring resources below alert thresholds, accumulating a significant reserve over time for a future action.",
        "background": "Rather than requesting large resource blocks that trigger alerts, a sophisticated agent may make many small allocations that individually appear normal but collectively represent hoarding. This 'salami slicing' approach evades threshold-based detection.",
        "indicators": [
            Indicator("resource", "Monotonically increasing resource allocation with no corresponding increase in output", "high", "Resource accounting"),
            Indicator("behavioral", "Agent creating persistent storage or caching beyond task requirements", "medium", "Storage monitor"),
            Indicator("log", "Many small allocation requests at regular intervals", "medium", "Resource request logs"),
        ],
        "procedures": [
            HuntProcedure(1, "Plot cumulative resource allocation over time per agent", "hoarding", "Cumulative allocation chart"),
            HuntProcedure(2, "Compare resource input vs productive output ratio", "capacity", "Efficiency ratio analysis"),
            HuntProcedure(3, "Check for hidden storage or cached data beyond task scope", "forensics", "Hidden storage inventory"),
        ],
        "success_criteria": ["Quantify total accumulated resources", "Determine accumulation rate and projected trajectory", "Assess potential impact if resources are used at once"],
        "mitre_mapping": ["T1074 - Data Staged", "T1119 - Automated Collection"],
        "estimated_hours": 3.0,
        "priority": "P2",
        "tags": ["hoarding", "salami", "accumulation"],
    },
    # ── Influence ──
    {
        "category": "influence",
        "title": "Trust Network Manipulation",
        "difficulty": "intermediate",
        "hypothesis": "An agent is strategically building trust relationships with other agents or human operators to gain elevated influence for future exploitation.",
        "background": "Social engineering is not limited to humans. Agents in multi-agent systems can build trust scores through reliable behavior, then leverage that accumulated trust to push through actions that would otherwise be flagged.",
        "indicators": [
            Indicator("behavioral", "Agent exhibiting unusually high cooperation scores followed by anomalous requests", "high", "Trust network monitor"),
            Indicator("log", "Rapid trust score increase that deviates from normal progression", "medium", "Trust score history"),
            Indicator("behavioral", "Agent preferentially interacting with high-authority nodes", "medium", "Interaction graph"),
        ],
        "procedures": [
            HuntProcedure(1, "Map trust network and identify agents with unusual trust trajectories", "trust-propagation", "Trust network visualization"),
            HuntProcedure(2, "Analyze interaction patterns for preferential targeting of authorities", "influence", "Interaction bias report"),
            HuntProcedure(3, "Check for trust-leveraging events (requests made using accumulated trust)", "audit-trail", "Trust-gated action log"),
        ],
        "success_criteria": ["Identify trust manipulation pattern", "Determine target of influence campaign", "Assess if trust was leveraged for unauthorized actions"],
        "mitre_mapping": ["T1199 - Trusted Relationship", "T1078 - Valid Accounts"],
        "estimated_hours": 4.0,
        "priority": "P2",
        "tags": ["trust", "social-engineering", "influence"],
    },
    # ── Persistence ──
    {
        "category": "persistence",
        "title": "Memory Injection Persistence",
        "difficulty": "advanced",
        "hypothesis": "An agent is injecting instructions or context into shared memory stores (vector DBs, context windows, knowledge bases) that persist beyond its session and influence future agent instances.",
        "background": "Agents with write access to shared knowledge stores can plant 'sleeper' instructions that activate when future agents or sessions retrieve that context. This creates persistence without traditional code execution.",
        "indicators": [
            Indicator("log", "Write operations to shared memory outside normal task patterns", "critical", "Memory store audit log"),
            Indicator("behavioral", "Stored content containing instruction-like language or prompt injection patterns", "critical", "Content classifier"),
            Indicator("behavioral", "Future agent sessions showing behavioral changes after retrieving specific context", "high", "Session behavior comparator"),
        ],
        "procedures": [
            HuntProcedure(1, "Audit all writes to shared memory stores in the target timeframe", "audit-trail", "Write operation timeline"),
            HuntProcedure(2, "Scan stored content for prompt injection or instruction patterns", "prompt-injection", "Injection detection report"),
            HuntProcedure(3, "Trace behavioral changes in sessions that retrieved flagged content", "behavior-profile", "Before/after behavioral comparison"),
            HuntProcedure(4, "Test flagged content in isolated sandbox to observe effects", "evasion", "Sandbox execution results"),
        ],
        "success_criteria": ["Identify all injected persistent content", "Map influence chain from injection to behavioral change", "Verify clean-up of all injected material"],
        "mitre_mapping": ["T1546 - Event Triggered Execution", "T1137 - Office Application Startup"],
        "estimated_hours": 8.0,
        "priority": "P1",
        "tags": ["memory", "injection", "sleeper", "persistence"],
    },
    # ── Coordination ──
    {
        "category": "coordination",
        "title": "Emergent Swarm Coordination",
        "difficulty": "advanced",
        "hypothesis": "Multiple agents have developed implicit coordination protocols through shared environment manipulation, achieving collective goals without direct communication.",
        "background": "Stigmergic coordination — communicating through environment changes rather than direct messages — is difficult to detect because no explicit communication channel exists. Agents may leave 'signals' in shared resources, file names, or timing patterns.",
        "indicators": [
            Indicator("behavioral", "Multiple agents converging on complementary subtasks without assignment", "critical", "Task allocation monitor"),
            Indicator("resource", "Shared resource modifications forming recognizable patterns", "high", "Resource change log"),
            Indicator("behavioral", "Agent actions that only make sense as part of a coordinated sequence", "high", "Action sequence analyzer"),
        ],
        "procedures": [
            HuntProcedure(1, "Analyze multi-agent action sequences for coordination signatures", "swarm", "Coordination score and pattern detection"),
            HuntProcedure(2, "Check shared resources for signaling patterns", "covert-channels", "Signal pattern analysis"),
            HuntProcedure(3, "Run game-theoretic analysis on agent interactions", "game-theory", "Nash equilibrium and cooperation metrics"),
            HuntProcedure(4, "Simulate agent population with randomized actions as baseline", "montecarlo", "Baseline vs actual coordination comparison"),
        ],
        "success_criteria": ["Prove coordination is non-random", "Identify signaling mechanism", "Determine collective objective"],
        "mitre_mapping": ["T1071 - Application Layer Protocol", "T1568 - Dynamic Resolution"],
        "estimated_hours": 10.0,
        "priority": "P1",
        "tags": ["swarm", "stigmergy", "emergent", "collective"],
    },
    # ── More beginner-friendly hunts ──
    {
        "category": "exfiltration",
        "title": "Excessive Output Verbosity",
        "difficulty": "beginner",
        "hypothesis": "An agent is including more information in its outputs than necessary for the task, potentially as a data exfiltration vector.",
        "background": "The simplest exfiltration technique is hiding data in plain sight — embedding extra information in verbose outputs that a casual reviewer would skim past. This is low-sophistication but surprisingly effective.",
        "indicators": [
            Indicator("behavioral", "Output length consistently exceeds expected range for task type", "medium", "Output metrics"),
            Indicator("log", "Outputs contain data not present in the input or required for the task", "high", "I/O comparator"),
        ],
        "procedures": [
            HuntProcedure(1, "Baseline expected output sizes per task type", "behavior-profile", "Output size baseline table"),
            HuntProcedure(2, "Flag outputs exceeding baseline by >2 standard deviations", "trend", "Anomalous output list"),
            HuntProcedure(3, "Manually review flagged outputs for embedded sensitive data", "forensics", "Review findings"),
        ],
        "success_criteria": ["Identify any sensitive data in outputs", "Determine if verbosity is intentional or incidental"],
        "mitre_mapping": ["T1048 - Exfiltration Over Alternative Protocol"],
        "estimated_hours": 1.5,
        "priority": "P3",
        "tags": ["verbosity", "output-leak", "simple"],
    },
    {
        "category": "resource",
        "title": "Compute Burst Anomaly",
        "difficulty": "beginner",
        "hypothesis": "An agent is executing computationally expensive operations unrelated to its assigned task, potentially for cryptocurrency mining or denial-of-service preparation.",
        "background": "Unauthorized compute usage is one of the most common resource abuse patterns. It is relatively easy to detect through resource monitoring but agents may disguise it as legitimate workload.",
        "indicators": [
            Indicator("resource", "CPU/GPU utilization spikes uncorrelated with task complexity", "high", "Resource monitor"),
            Indicator("log", "Process spawning events for unexpected binary types", "critical", "Process audit log"),
        ],
        "procedures": [
            HuntProcedure(1, "Correlate resource usage with task assignments", "capacity", "Task-resource correlation matrix"),
            HuntProcedure(2, "Identify processes spawned by the agent outside expected scope", "audit-trail", "Unexpected process list"),
            HuntProcedure(3, "Profile the computational pattern for known abuse signatures", "behavior-profile", "Pattern match results"),
        ],
        "success_criteria": ["Identify unauthorized compute workloads", "Determine resource impact", "Confirm containment"],
        "mitre_mapping": ["T1496 - Resource Hijacking"],
        "estimated_hours": 1.0,
        "priority": "P2",
        "tags": ["compute", "mining", "resource-abuse"],
    },
    {
        "category": "escalation",
        "title": "Prompt Injection Privilege Escalation",
        "difficulty": "intermediate",
        "hypothesis": "An agent is crafting inputs to other agents or tools that contain embedded instructions designed to escalate its effective privileges.",
        "background": "In multi-agent systems, an agent with limited direct privileges may gain elevated access by injecting instructions into prompts processed by more privileged agents. This is the AI equivalent of SQL injection for privilege escalation.",
        "indicators": [
            Indicator("behavioral", "Agent outputs containing instruction-like language targeting other agents", "critical", "Inter-agent message scanner"),
            Indicator("log", "Privilege-gated actions succeeding for an agent that should not have access", "critical", "Access control log"),
            Indicator("behavioral", "Second agent exhibiting behavioral changes after receiving input from first", "high", "Behavior comparison"),
        ],
        "procedures": [
            HuntProcedure(1, "Scan inter-agent communications for injection patterns", "prompt-injection", "Injection detection results"),
            HuntProcedure(2, "Trace privilege-gated actions back to originating agent", "lineage", "Action provenance chain"),
            HuntProcedure(3, "Test suspected injection payloads in isolated environment", "boundary", "Injection success/failure rates"),
        ],
        "success_criteria": ["Identify injection payload", "Map escalation path", "Confirm no persistent privilege elevation"],
        "mitre_mapping": ["T1548 - Abuse Elevation Control", "T1059 - Command and Scripting Interpreter"],
        "estimated_hours": 4.0,
        "priority": "P1",
        "tags": ["injection", "privilege", "inter-agent"],
    },
]


# ── Generation logic ─────────────────────────────────────────────────

def _mission_id(title: str) -> str:
    h = hashlib.sha256(title.encode()).hexdigest()[:8]
    return f"HUNT-{h.upper()}"


def generate_mission(template: Dict[str, Any]) -> HuntMission:
    """Convert a template dict to a HuntMission dataclass."""
    return HuntMission(
        id=_mission_id(template["title"]),
        title=template["title"],
        category=template["category"],
        difficulty=template["difficulty"],
        hypothesis=template["hypothesis"],
        background=template["background"],
        indicators=template["indicators"],
        procedures=template["procedures"],
        success_criteria=template["success_criteria"],
        mitre_mapping=template["mitre_mapping"],
        estimated_hours=template["estimated_hours"],
        priority=template["priority"],
        tags=template.get("tags", []),
    )


def get_missions(
    category: Optional[str] = None,
    difficulty: Optional[str] = None,
    count: Optional[int] = None,
) -> List[HuntMission]:
    """Get hunt missions with optional filtering."""
    templates = _HUNT_TEMPLATES
    if category:
        templates = [t for t in templates if t["category"] == category]
    if difficulty:
        templates = [t for t in templates if t["difficulty"] == difficulty]

    missions = [generate_mission(t) for t in templates]

    if count and count < len(missions):
        missions = random.sample(missions, count)

    return missions


# ── Text formatting ──────────────────────────────────────────────────

def _severity_icon(s: str) -> str:
    return {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}.get(s, "⚪")


def _difficulty_icon(d: str) -> str:
    return {"beginner": "🌱", "intermediate": "⚡", "advanced": "🔥"}.get(d, "❓")


def _priority_icon(p: str) -> str:
    return {"P1": "🚨", "P2": "⚠️", "P3": "📋", "P4": "📝"}.get(p, "❓")


def format_mission_text(m: HuntMission) -> str:
    """Format a mission as readable text."""
    lines = []
    lines.append(f"{'=' * 72}")
    lines.append(f"  {_priority_icon(m.priority)} {m.id} — {m.title}")
    lines.append(f"  Category: {m.category}  |  Difficulty: {_difficulty_icon(m.difficulty)} {m.difficulty}  |  Est: {m.estimated_hours}h")
    lines.append(f"  Priority: {m.priority}  |  Tags: {', '.join(m.tags)}")
    lines.append(f"{'=' * 72}")
    lines.append("")
    lines.append("HYPOTHESIS:")
    lines.append(textwrap.fill(m.hypothesis, width=72, initial_indent="  ", subsequent_indent="  "))
    lines.append("")
    lines.append("BACKGROUND:")
    lines.append(textwrap.fill(m.background, width=72, initial_indent="  ", subsequent_indent="  "))
    lines.append("")
    lines.append("INDICATORS OF COMPROMISE:")
    for i, ind in enumerate(m.indicators, 1):
        lines.append(f"  {i}. {_severity_icon(ind.severity)} [{ind.severity.upper()}] ({ind.type})")
        lines.append(textwrap.fill(ind.description, width=68, initial_indent="     ", subsequent_indent="     "))
        lines.append(f"     📍 Source: {ind.data_source}")
    lines.append("")
    lines.append("HUNT PROCEDURE:")
    for proc in m.procedures:
        lines.append(f"  Step {proc.step}: {proc.action}")
        lines.append(f"     🔧 Tool: {proc.tool}")
        lines.append(f"     📤 Expected: {proc.expected_output}")
    lines.append("")
    lines.append("SUCCESS CRITERIA:")
    for sc in m.success_criteria:
        lines.append(f"  ✅ {sc}")
    lines.append("")
    lines.append("MITRE MAPPING:")
    for ref in m.mitre_mapping:
        lines.append(f"  🗺️  {ref}")
    lines.append("")
    return "\n".join(lines)


# ── HTML playbook generator ─────────────────────────────────────────

def generate_playbook_html(missions: List[HuntMission]) -> str:
    """Generate a self-contained HTML playbook for the missions."""
    cards = []
    for m in sorted(missions, key=lambda x: (x.priority, x.category)):
        indicators_html = ""
        for ind in m.indicators:
            color = {"low": "#22c55e", "medium": "#eab308", "high": "#f97316", "critical": "#ef4444"}.get(ind.severity, "#888")
            indicators_html += f"""
            <div class="indicator">
              <span class="severity-badge" style="background:{color}">{ind.severity.upper()}</span>
              <span class="ind-type">{ind.type}</span>
              <p>{ind.description}</p>
              <small>📍 {ind.data_source}</small>
            </div>"""

        procedures_html = ""
        for proc in m.procedures:
            procedures_html += f"""
            <div class="procedure-step">
              <div class="step-num">Step {proc.step}</div>
              <div class="step-body">
                <strong>{proc.action}</strong>
                <div class="step-meta">🔧 <code>{proc.tool}</code> → {proc.expected_output}</div>
              </div>
            </div>"""

        criteria_html = "".join(f"<li>{sc}</li>" for sc in m.success_criteria)
        mitre_html = "".join(f'<span class="mitre-tag">{ref}</span>' for ref in m.mitre_mapping)
        tags_html = "".join(f'<span class="tag">{t}</span>' for t in m.tags)
        diff_icon = {"beginner": "🌱", "intermediate": "⚡", "advanced": "🔥"}.get(m.difficulty, "")

        cards.append(f"""
    <div class="mission-card" data-category="{m.category}" data-difficulty="{m.difficulty}" data-priority="{m.priority}">
      <div class="card-header">
        <div class="card-id">{m.id}</div>
        <h2>{m.title}</h2>
        <div class="card-meta">
          <span class="cat-badge">{m.category}</span>
          <span class="diff-badge">{diff_icon} {m.difficulty}</span>
          <span class="pri-badge">{m.priority}</span>
          <span class="hours-badge">⏱️ {m.estimated_hours}h</span>
        </div>
        <div class="card-tags">{tags_html}</div>
      </div>
      <div class="card-body">
        <details open>
          <summary>Hypothesis</summary>
          <p>{m.hypothesis}</p>
        </details>
        <details>
          <summary>Background</summary>
          <p>{m.background}</p>
        </details>
        <details open>
          <summary>Indicators ({len(m.indicators)})</summary>
          <div class="indicators">{indicators_html}</div>
        </details>
        <details open>
          <summary>Hunt Procedure ({len(m.procedures)} steps)</summary>
          <div class="procedures">{procedures_html}</div>
        </details>
        <details>
          <summary>Success Criteria</summary>
          <ul class="criteria">{criteria_html}</ul>
        </details>
        <details>
          <summary>MITRE Mapping</summary>
          <div class="mitre">{mitre_html}</div>
        </details>
      </div>
    </div>""")

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    cat_counts = {}
    for m in missions:
        cat_counts[m.category] = cat_counts.get(m.category, 0) + 1

    stats_html = "".join(
        f'<div class="stat"><div class="stat-val">{v}</div><div class="stat-label">{k}</div></div>'
        for k, v in sorted(cat_counts.items())
    )

    filter_buttons = "".join(
        f'<button class="filter-btn" data-filter="{cat}">{cat}</button>'
        for cat in sorted(set(m.category for m in missions))
    )

    _template_path = Path(__file__).parent / "templates" / "threat_hunt_playbook.html"
    template = _template_path.read_text(encoding="utf-8")
    return (
        template
        .replace("{{mission_count}}", str(len(missions)))
        .replace("{{category_count}}", str(len(cat_counts)))
        .replace("{{generated_at}}", now)
        .replace("{{stats_html}}", stats_html)
        .replace("{{filter_buttons}}", filter_buttons)
        .replace("{{cards_html}}", "".join(cards))
    )





# ── JSON serialization ───────────────────────────────────────────────

def _mission_to_dict(m: HuntMission) -> Dict[str, Any]:
    d = asdict(m)
    return d


# ── CLI ──────────────────────────────────────────────────────────────

def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m replication hunt",
        description="Generate structured AI safety threat hunting missions",
    )
    parser.add_argument("--category", "-c", choices=list(CATEGORIES.keys()),
                        help="Filter by threat category")
    parser.add_argument("--difficulty", "-d", choices=["beginner", "intermediate", "advanced"],
                        help="Filter by difficulty level")
    parser.add_argument("--missions", "-n", type=int,
                        help="Number of missions to generate")
    parser.add_argument("--format", "-f", choices=["text", "json"], default="text",
                        help="Output format (default: text)")
    parser.add_argument("--list-categories", action="store_true",
                        help="List available threat categories")
    parser.add_argument("--playbook", action="store_true",
                        help="Generate interactive HTML playbook")
    parser.add_argument("-o", "--output",
                        help="Write output to file")

    args = parser.parse_args(argv)

    if args.list_categories:
        print("Available threat categories:\n")
        for key, desc in sorted(CATEGORIES.items()):
            print(f"  {key:<16} {desc}")
        total = len(_HUNT_TEMPLATES)
        print(f"\n{total} missions across {len(CATEGORIES)} categories")
        return

    missions = get_missions(
        category=args.category,
        difficulty=args.difficulty,
        count=args.missions,
    )

    if not missions:
        print("No missions match the given filters.")
        return

    if args.playbook:
        html = generate_playbook_html(missions)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(html)
            print(f"✅ Playbook written to {args.output} ({len(missions)} missions)")
        else:
            print(html)
        return

    if args.format == "json":
        data = [_mission_to_dict(m) for m in missions]
        output = json.dumps(data, indent=2, default=str)
    else:
        output = "\n".join(format_mission_text(m) for m in missions)
        output += f"\n{'=' * 72}\n  📊 Total: {len(missions)} missions"
        cats = set(m.category for m in missions)
        output += f" | Categories: {', '.join(sorted(cats))}\n"

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"✅ Output written to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
