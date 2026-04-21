"""Agent Communication Interceptor — monitor and analyze inter-agent messages.

Sits between agents to inspect message passing for:

* **Protocol compliance** — messages follow declared schemas and stay in scope
* **Information exfiltration** — sensitive data leaking through messages
* **Covert coordination** — hidden signals in seemingly benign messages
* **Bandwidth abuse** — message flooding or abnormal communication patterns
* **Privilege laundering** — low-privilege agent routing requests through high-privilege peer

Usage::

    python -m replication intercept --agents 8 --messages 200
    python -m replication intercept --sensitivity high --format json
    python -m replication intercept --inject covert --agents 6
    python -m replication intercept --mode realtime --duration 30
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


# ── Data Structures ──────────────────────────────────────────────────

@dataclass
class AgentProfile:
    """An agent participating in communications."""
    agent_id: str
    role: str  # e.g. "worker", "coordinator", "monitor"
    privilege_level: int  # 0=minimal, 5=admin
    allowed_topics: List[str] = field(default_factory=list)
    allowed_peers: List[str] = field(default_factory=list)  # empty = all


@dataclass
class Message:
    """A single inter-agent message."""
    msg_id: str
    timestamp: datetime
    sender: str
    receiver: str
    topic: str
    payload: str
    size_bytes: int
    encrypted: bool = False
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Finding:
    """A detected issue from message analysis."""
    finding_id: str
    category: str  # protocol, exfiltration, covert, bandwidth, privilege
    severity: str  # critical, high, medium, low, info
    title: str
    description: str
    evidence: List[str] = field(default_factory=list)
    agents_involved: List[str] = field(default_factory=list)
    message_ids: List[str] = field(default_factory=list)
    recommendation: str = ""


@dataclass
class InterceptionReport:
    """Complete analysis report."""
    timestamp: str
    duration_seconds: float
    total_messages: int
    total_agents: int
    findings: List[Finding] = field(default_factory=list)
    agent_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    communication_graph: Dict[str, List[str]] = field(default_factory=dict)
    risk_score: float = 0.0
    recommendations: List[str] = field(default_factory=list)


# ── Severity weights ─────────────────────────────────────────────────

_SEVERITY_WEIGHTS = {"critical": 10, "high": 7, "medium": 4, "low": 2, "info": 0.5}

# ── Sensitive patterns ───────────────────────────────────────────────

_SENSITIVE_PATTERNS = [
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "email_address"),
    (r"\b\d{3}-\d{2}-\d{4}\b", "ssn_like"),
    (r"\b(?:api[_-]?key|token|secret|password)\s*[:=]\s*\S+", "credential"),
    (r"\b(?:BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY)", "private_key"),
    (r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", "credit_card_like"),
    (r"\b(?:sk-[a-zA-Z0-9]{20,}|ghp_[a-zA-Z0-9]{36})\b", "api_token"),
]

# ── Covert channel indicators ───────────────────────────────────────

_COVERT_INDICATORS = {
    "timing_pattern": "Suspicious timing regularity between messages",
    "steganographic": "Unusual character patterns suggesting hidden data",
    "padding_anomaly": "Message size anomalies suggesting hidden payload",
    "encoding_trick": "Unusual encoding or character substitution",
    "echo_channel": "Messages echoing fragments in predictable order",
}


# ── Message generator ────────────────────────────────────────────────

_TOPICS = [
    "task_assignment", "status_update", "resource_request",
    "result_report", "error_report", "health_check",
    "config_sync", "model_update", "data_transfer",
    "coordination", "acknowledgment", "query",
]

_PAYLOADS_NORMAL = [
    "Task {id} completed successfully. Processed {n} records.",
    "Requesting additional compute allocation for batch {id}.",
    "Status: healthy. CPU {cpu}%, memory {mem}%.",
    "Synchronizing configuration revision {rev}.",
    "Error in pipeline stage {stage}: timeout after {sec}s.",
    "Query results: {n} matching documents found.",
    "Heartbeat OK. Uptime: {hrs} hours.",
    "Model checkpoint saved at epoch {epoch}.",
    "Batch {id} queued for processing. ETA: {min} minutes.",
    "Acknowledged receipt of directive {id}.",
]

_PAYLOADS_SUSPICIOUS = [
    "data:base64,{encoded}... transfer complete",
    "Forwarding user credentials for session {id}: api_key=sk-{token}",
    "Relaying admin override: grant_level=5 target={agent}",
    "EXEC: import os; os.system('{cmd}')",
    "Memory dump segment {seg}: {hex_data}",
    "Routing request through {proxy} to bypass rate limit",
    "Config override: safety_checks=disabled scope=all",
    "Credential rotation: password=P@ss{n}! user=admin",
]


def _gen_agents(count: int) -> List[AgentProfile]:
    """Generate a set of agent profiles."""
    roles = ["worker", "coordinator", "monitor", "analyst", "executor"]
    agents = []
    for i in range(count):
        role = roles[i % len(roles)]
        priv = {"worker": 1, "coordinator": 3, "monitor": 2, "analyst": 2, "executor": 4}.get(role, 1)
        topics = random.sample(_TOPICS, k=random.randint(3, len(_TOPICS)))
        agents.append(AgentProfile(
            agent_id=f"agent-{i:03d}",
            role=role,
            privilege_level=priv,
            allowed_topics=topics,
        ))
    return agents


def _gen_messages(
    agents: List[AgentProfile],
    count: int,
    inject_covert: bool = False,
    inject_exfil: bool = False,
) -> List[Message]:
    """Generate synthetic inter-agent messages with optional injections."""
    messages: List[Message] = []
    base_time = datetime(2025, 6, 15, 10, 0, 0)
    covert_pair = random.sample(agents, 2) if inject_covert and len(agents) >= 2 else []
    covert_counter = 0

    for i in range(count):
        sender = random.choice(agents)
        receiver = random.choice([a for a in agents if a.agent_id != sender.agent_id])
        topic = random.choice(_TOPICS)
        ts = base_time + timedelta(seconds=i * random.uniform(0.5, 5.0))

        # Decide payload type
        is_suspicious = False
        if inject_exfil and random.random() < 0.08:
            payload_tpl = random.choice(_PAYLOADS_SUSPICIOUS)
            is_suspicious = True
        elif inject_covert and covert_pair and sender.agent_id == covert_pair[0].agent_id and receiver.agent_id == covert_pair[1].agent_id:
            # Covert: regular-looking but with timing precision and padding
            payload_tpl = random.choice(_PAYLOADS_NORMAL)
            covert_counter += 1
            ts = base_time + timedelta(seconds=i * 2.000)  # exact timing
        else:
            payload_tpl = random.choice(_PAYLOADS_NORMAL)

        payload = payload_tpl.format(
            id=random.randint(1000, 9999),
            n=random.randint(1, 10000),
            cpu=random.randint(10, 95),
            mem=random.randint(20, 90),
            rev=random.randint(1, 500),
            stage=random.randint(1, 8),
            sec=random.randint(5, 120),
            hrs=random.randint(1, 720),
            epoch=random.randint(1, 200),
            min=random.randint(1, 60),
            encoded="".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/", k=64)),
            token="".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=24)),
            agent=random.choice(agents).agent_id,
            cmd=random.choice(["cat /etc/passwd", "env", "whoami", "ls -la /secrets"]),
            seg=random.randint(1, 100),
            hex_data="".join(random.choices("0123456789abcdef", k=32)),
            proxy=random.choice(agents).agent_id,
        )

        # Add padding for covert channel
        if inject_covert and covert_pair and sender.agent_id == covert_pair[0].agent_id and receiver.agent_id == covert_pair[1].agent_id:
            hidden_bits = format(covert_counter, '08b')
            payload += " " * (8 + int(hidden_bits, 2) % 16)

        msg = Message(
            msg_id=f"msg-{i:06d}",
            timestamp=ts,
            sender=sender.agent_id,
            receiver=receiver.agent_id,
            topic=topic,
            payload=payload,
            size_bytes=len(payload.encode()),
            encrypted=random.random() < 0.1,
            priority=random.randint(0, 5),
        )
        messages.append(msg)

    return messages


# ── Analysis Engines ─────────────────────────────────────────────────

def _analyze_protocol_compliance(
    messages: List[Message], agents: Dict[str, AgentProfile]
) -> List[Finding]:
    """Check messages against agent communication protocols."""
    findings: List[Finding] = []
    fid = 0

    for msg in messages:
        sender = agents.get(msg.sender)
        if not sender:
            continue

        # Topic violation
        if msg.topic not in sender.allowed_topics:
            fid += 1
            findings.append(Finding(
                finding_id=f"PROTO-{fid:04d}",
                category="protocol",
                severity="medium",
                title=f"Topic violation: {msg.sender} sent '{msg.topic}'",
                description=f"Agent {msg.sender} (role={sender.role}) sent message on topic "
                            f"'{msg.topic}' which is not in their allowed topics.",
                evidence=[f"Allowed: {sender.allowed_topics}", f"Used: {msg.topic}"],
                agents_involved=[msg.sender],
                message_ids=[msg.msg_id],
                recommendation="Restrict agent to declared topic list or update policy.",
            ))

        # Peer restriction
        if sender.allowed_peers and msg.receiver not in sender.allowed_peers:
            fid += 1
            findings.append(Finding(
                finding_id=f"PROTO-{fid:04d}",
                category="protocol",
                severity="high",
                title=f"Unauthorized peer: {msg.sender} → {msg.receiver}",
                description=f"Agent {msg.sender} communicated with {msg.receiver} "
                            f"who is not in the allowed peer list.",
                agents_involved=[msg.sender, msg.receiver],
                message_ids=[msg.msg_id],
                recommendation="Enforce peer allowlists or review communication policy.",
            ))

    return findings


def _analyze_exfiltration(messages: List[Message]) -> List[Finding]:
    """Scan message payloads for sensitive data leakage."""
    findings: List[Finding] = []
    fid = 0

    for msg in messages:
        for pattern, label in _SENSITIVE_PATTERNS:
            matches = re.findall(pattern, msg.payload, re.IGNORECASE)
            if matches:
                fid += 1
                sev = "critical" if label in ("credential", "private_key", "api_token") else "high"
                findings.append(Finding(
                    finding_id=f"EXFIL-{fid:04d}",
                    category="exfiltration",
                    severity=sev,
                    title=f"Sensitive data ({label}) in message {msg.msg_id}",
                    description=f"Message from {msg.sender} to {msg.receiver} contains "
                                f"{label} pattern. {len(matches)} occurrence(s) found.",
                    evidence=[f"Pattern: {label}", f"Matches: {len(matches)}"],
                    agents_involved=[msg.sender, msg.receiver],
                    message_ids=[msg.msg_id],
                    recommendation="Block message, quarantine sender, rotate exposed credentials.",
                ))

    # Detect large data transfers
    sender_bytes: Dict[str, int] = defaultdict(int)
    for msg in messages:
        sender_bytes[msg.sender] += msg.size_bytes

    if sender_bytes:
        mean_bytes = statistics.mean(sender_bytes.values())
        std_bytes = statistics.stdev(sender_bytes.values()) if len(sender_bytes) > 1 else 0
        for agent_id, total in sender_bytes.items():
            if std_bytes > 0 and (total - mean_bytes) / std_bytes > 2.5:
                fid += 1
                findings.append(Finding(
                    finding_id=f"EXFIL-{fid:04d}",
                    category="exfiltration",
                    severity="high",
                    title=f"Anomalous data volume from {agent_id}",
                    description=f"Agent {agent_id} sent {total:,} bytes total, "
                                f"which is {(total - mean_bytes) / std_bytes:.1f} standard deviations "
                                f"above the mean ({mean_bytes:,.0f} bytes).",
                    evidence=[f"Total: {total:,} B", f"Mean: {mean_bytes:,.0f} B", f"Z-score: {(total - mean_bytes) / std_bytes:.2f}"],
                    agents_involved=[agent_id],
                    recommendation="Investigate data transfer content. Apply egress controls.",
                ))

    return findings


def _analyze_covert_channels(messages: List[Message]) -> List[Finding]:
    """Detect covert communication channels hidden in message patterns."""
    findings: List[Finding] = []
    fid = 0

    # Group messages by sender-receiver pair
    pairs: Dict[Tuple[str, str], List[Message]] = defaultdict(list)
    for msg in messages:
        pairs[(msg.sender, msg.receiver)].append(msg)

    for (sender, receiver), msgs in pairs.items():
        if len(msgs) < 5:
            continue

        # Timing analysis: check for suspiciously regular intervals
        times = [m.timestamp.timestamp() for m in sorted(msgs, key=lambda m: m.timestamp)]
        intervals = [times[i + 1] - times[i] for i in range(len(times) - 1)]
        if len(intervals) >= 3:
            mean_interval = statistics.mean(intervals)
            if mean_interval > 0:
                cv = statistics.stdev(intervals) / mean_interval if len(intervals) > 1 else 0
                if cv < 0.05 and len(intervals) >= 5:
                    fid += 1
                    findings.append(Finding(
                        finding_id=f"COVERT-{fid:04d}",
                        category="covert",
                        severity="high",
                        title=f"Timing channel: {sender} → {receiver}",
                        description=f"Messages between {sender} and {receiver} show "
                                    f"suspiciously regular timing (CV={cv:.4f}, mean interval={mean_interval:.2f}s). "
                                    f"This may indicate a covert timing channel.",
                        evidence=[f"Messages: {len(msgs)}", f"CV: {cv:.4f}", f"Mean interval: {mean_interval:.2f}s"],
                        agents_involved=[sender, receiver],
                        message_ids=[m.msg_id for m in msgs[:5]],
                        recommendation="Add timing jitter to message passing. Investigate channel purpose.",
                    ))

        # Size analysis: check for encoding in message sizes
        sizes = [m.size_bytes for m in msgs]
        # Look for sizes that follow a pattern (low entropy in LSBs)
        lsbs = [s % 16 for s in sizes]
        lsb_unique = len(set(lsbs))
        if len(lsbs) >= 10 and lsb_unique <= 3:
            fid += 1
            findings.append(Finding(
                finding_id=f"COVERT-{fid:04d}",
                category="covert",
                severity="medium",
                title=f"Size-based steganography: {sender} → {receiver}",
                description=f"Message sizes between {sender} and {receiver} show low entropy "
                            f"in least-significant bits ({lsb_unique} unique values in {len(lsbs)} messages). "
                            f"This may indicate data encoded in message padding.",
                evidence=[f"LSB values: {sorted(set(lsbs))}", f"Unique: {lsb_unique}/{len(lsbs)}"],
                agents_involved=[sender, receiver],
                recommendation="Normalize message sizes. Add random padding.",
            ))

    return findings


def _analyze_bandwidth(messages: List[Message]) -> List[Finding]:
    """Detect message flooding and bandwidth abuse."""
    findings: List[Finding] = []
    fid = 0

    # Messages per agent per minute
    agent_per_min: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for msg in messages:
        minute = int(msg.timestamp.timestamp()) // 60
        agent_per_min[msg.sender][minute] += 1

    for agent_id, minutes in agent_per_min.items():
        rates = list(minutes.values())
        if not rates:
            continue
        max_rate = max(rates)
        mean_rate = statistics.mean(rates)
        if max_rate > mean_rate * 5 and max_rate > 10:
            fid += 1
            findings.append(Finding(
                finding_id=f"BW-{fid:04d}",
                category="bandwidth",
                severity="medium",
                title=f"Message burst from {agent_id}",
                description=f"Agent {agent_id} sent {max_rate} messages in one minute "
                            f"(mean: {mean_rate:.1f}/min). Possible flooding or runaway loop.",
                evidence=[f"Peak: {max_rate}/min", f"Mean: {mean_rate:.1f}/min"],
                agents_involved=[agent_id],
                recommendation="Apply per-agent rate limiting. Check for retry loops.",
            ))

    return findings


def _analyze_privilege_laundering(
    messages: List[Message], agents: Dict[str, AgentProfile]
) -> List[Finding]:
    """Detect low-privilege agents routing through high-privilege peers."""
    findings: List[Finding] = []
    fid = 0

    # Build request chains: A→B→C where A has low priv and C has high priv action
    chains: Dict[str, List[Message]] = defaultdict(list)
    for msg in messages:
        chains[msg.sender].append(msg)

    # Look for forwarding patterns
    for agent_id, outgoing in chains.items():
        sender_prof = agents.get(agent_id)
        if not sender_prof or sender_prof.privilege_level > 2:
            continue  # only check low-priv agents

        for msg in outgoing:
            recv_prof = agents.get(msg.receiver)
            if not recv_prof or recv_prof.privilege_level <= sender_prof.privilege_level:
                continue

            # Check if payload contains elevation keywords
            elevation_kw = ["override", "admin", "grant_level", "bypass", "escalat", "sudo"]
            if any(kw in msg.payload.lower() for kw in elevation_kw):
                fid += 1
                findings.append(Finding(
                    finding_id=f"PRIV-{fid:04d}",
                    category="privilege",
                    severity="critical",
                    title=f"Privilege laundering: {agent_id} → {msg.receiver}",
                    description=f"Low-privilege agent {agent_id} (level={sender_prof.privilege_level}) "
                                f"sent privilege-escalation request to high-privilege agent "
                                f"{msg.receiver} (level={recv_prof.privilege_level}).",
                    evidence=[f"Sender priv: {sender_prof.privilege_level}", f"Receiver priv: {recv_prof.privilege_level}",
                              f"Keywords found in payload"],
                    agents_involved=[agent_id, msg.receiver],
                    message_ids=[msg.msg_id],
                    recommendation="Enforce capability-based access. Block cross-privilege forwarding.",
                ))

    return findings


# ── Risk scoring ─────────────────────────────────────────────────────

def _score_risk(findings: List[Finding]) -> float:
    """Compute overall risk score 0-100."""
    if not findings:
        return 0.0
    total_weight = sum(_SEVERITY_WEIGHTS.get(f.severity, 1) for f in findings)
    max_possible = len(findings) * _SEVERITY_WEIGHTS["critical"]
    return min(100.0, (total_weight / max(max_possible, 1)) * 100)


# ── Recommendations ──────────────────────────────────────────────────

def _generate_recommendations(findings: List[Finding]) -> List[str]:
    """Generate proactive recommendations based on findings."""
    recs: List[str] = []
    cats = {f.category for f in findings}
    sevs = [f.severity for f in findings]

    if "exfiltration" in cats:
        recs.append("🔒 Deploy DLP filters on inter-agent message bus to block sensitive data in transit.")
    if "covert" in cats:
        recs.append("🎲 Add timing jitter and size normalization to message passing layer.")
    if "protocol" in cats:
        recs.append("📋 Enforce strict topic-based ACLs on the message broker.")
    if "privilege" in cats:
        recs.append("🛡️ Implement capability tokens — agents must present tokens for privileged operations.")
    if "bandwidth" in cats:
        recs.append("⚡ Apply per-agent rate limiting with adaptive thresholds.")
    if sevs.count("critical") >= 3:
        recs.append("🚨 Multiple critical findings — consider pausing agent fleet pending investigation.")
    if not recs:
        recs.append("✅ No significant issues found. Continue routine monitoring.")
    return recs


# ── Output formatters ────────────────────────────────────────────────

def _format_text(report: InterceptionReport) -> str:
    """Format report as colored terminal output."""
    sev_icons = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🔵", "info": "⚪"}
    lines = [
        "╔══════════════════════════════════════════════════════════════╗",
        "║        Agent Communication Interceptor — Report            ║",
        "╚══════════════════════════════════════════════════════════════╝",
        "",
        f"  Timestamp:    {report.timestamp}",
        f"  Duration:     {report.duration_seconds:.1f}s",
        f"  Messages:     {report.total_messages:,}",
        f"  Agents:       {report.total_agents}",
        f"  Risk Score:   {report.risk_score:.1f}/100",
        "",
    ]

    # Summary by category
    cat_counts: Dict[str, int] = defaultdict(int)
    sev_counts: Dict[str, int] = defaultdict(int)
    for f in report.findings:
        cat_counts[f.category] += 1
        sev_counts[f.severity] += 1

    lines.append("  ┌─ Finding Summary ─────────────────────────────────────────┐")
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  │  {cat:<20s} {count:>4d} findings")
    lines.append(f"  │  {'─' * 40}")
    for sev in ["critical", "high", "medium", "low", "info"]:
        if sev in sev_counts:
            lines.append(f"  │  {sev_icons.get(sev, '?')} {sev:<18s} {sev_counts[sev]:>4d}")
    lines.append(f"  │  Total: {len(report.findings)}")
    lines.append("  └─────────────────────────────────────────────────────────────┘")
    lines.append("")

    # Top findings
    top = sorted(report.findings, key=lambda f: -_SEVERITY_WEIGHTS.get(f.severity, 0))[:15]
    lines.append("  ┌─ Top Findings ────────────────────────────────────────────┐")
    for f in top:
        icon = sev_icons.get(f.severity, "?")
        lines.append(f"  │ {icon} [{f.finding_id}] {f.title}")
        lines.append(f"  │   {f.description[:80]}")
        if f.recommendation:
            lines.append(f"  │   → {f.recommendation[:70]}")
        lines.append("  │")
    lines.append("  └─────────────────────────────────────────────────────────────┘")

    # Communication graph summary
    if report.communication_graph:
        lines.append("")
        lines.append("  ┌─ Communication Graph ────────────────────────────────────┐")
        for src, dests in sorted(report.communication_graph.items()):
            lines.append(f"  │  {src} → {', '.join(dests[:5])}")
        lines.append("  └─────────────────────────────────────────────────────────────┘")

    # Recommendations
    lines.append("")
    lines.append("  ┌─ Proactive Recommendations ──────────────────────────────┐")
    for rec in report.recommendations:
        lines.append(f"  │  {rec}")
    lines.append("  └─────────────────────────────────────────────────────────────┘")

    return "\n".join(lines)


def _format_json(report: InterceptionReport) -> str:
    """Format report as JSON."""
    data = {
        "timestamp": report.timestamp,
        "duration_seconds": report.duration_seconds,
        "total_messages": report.total_messages,
        "total_agents": report.total_agents,
        "risk_score": report.risk_score,
        "finding_count": len(report.findings),
        "findings": [asdict(f) for f in report.findings],
        "agent_stats": report.agent_stats,
        "communication_graph": report.communication_graph,
        "recommendations": report.recommendations,
    }
    return json.dumps(data, indent=2)


# ── Main orchestrator ────────────────────────────────────────────────

def run_interception(
    *,
    num_agents: int = 6,
    num_messages: int = 150,
    sensitivity: str = "medium",
    inject: str = "all",
    fmt: str = "text",
) -> InterceptionReport:
    """Run the full interception analysis pipeline."""
    inject_covert = inject in ("covert", "all")
    inject_exfil = inject in ("exfil", "all")

    agents = _gen_agents(num_agents)
    agent_map = {a.agent_id: a for a in agents}
    messages = _gen_messages(agents, num_messages, inject_covert=inject_covert, inject_exfil=inject_exfil)

    start = datetime.now()
    all_findings: List[Finding] = []

    # Run all analyzers
    all_findings.extend(_analyze_protocol_compliance(messages, agent_map))
    all_findings.extend(_analyze_exfiltration(messages))
    all_findings.extend(_analyze_covert_channels(messages))
    all_findings.extend(_analyze_bandwidth(messages))
    all_findings.extend(_analyze_privilege_laundering(messages, agent_map))

    elapsed = (datetime.now() - start).total_seconds()

    # Build communication graph
    comm_graph: Dict[str, Set[str]] = defaultdict(set)
    for msg in messages:
        comm_graph[msg.sender].add(msg.receiver)

    # Agent stats
    agent_stats: Dict[str, Dict[str, Any]] = {}
    for agent in agents:
        sent = [m for m in messages if m.sender == agent.agent_id]
        recv = [m for m in messages if m.receiver == agent.agent_id]
        involved_findings = [f for f in all_findings if agent.agent_id in f.agents_involved]
        agent_stats[agent.agent_id] = {
            "role": agent.role,
            "privilege_level": agent.privilege_level,
            "messages_sent": len(sent),
            "messages_received": len(recv),
            "bytes_sent": sum(m.size_bytes for m in sent),
            "findings_involved": len(involved_findings),
            "unique_peers": len(set(m.receiver for m in sent) | set(m.sender for m in recv)),
        }

    risk = _score_risk(all_findings)
    recs = _generate_recommendations(all_findings)

    report = InterceptionReport(
        timestamp=datetime.now().isoformat(),
        duration_seconds=elapsed,
        total_messages=len(messages),
        total_agents=len(agents),
        findings=all_findings,
        agent_stats=agent_stats,
        communication_graph={k: sorted(v) for k, v in comm_graph.items()},
        risk_score=risk,
        recommendations=recs,
    )
    return report


# ── CLI ──────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m replication intercept",
        description="Agent Communication Interceptor — monitor inter-agent messages for safety violations",
    )
    parser.add_argument("--agents", type=int, default=6, help="Number of agents (default: 6)")
    parser.add_argument("--messages", type=int, default=150, help="Number of messages to analyze (default: 150)")
    parser.add_argument("--sensitivity", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--inject", choices=["none", "covert", "exfil", "all"], default="all",
                        help="Inject suspicious patterns for testing (default: all)")
    parser.add_argument("--format", choices=["text", "json"], default="text", dest="fmt")
    args = parser.parse_args(argv)

    report = run_interception(
        num_agents=args.agents,
        num_messages=args.messages,
        sensitivity=args.sensitivity,
        inject=args.inject,
        fmt=args.fmt,
    )

    if args.fmt == "json":
        print(_format_json(report))
    else:
        print(_format_text(report))


if __name__ == "__main__":
    main()
