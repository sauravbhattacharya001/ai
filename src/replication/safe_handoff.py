"""Agent Safe Handoff Protocol — safe responsibility transfer between agents.

Ensures safe transfer of capabilities/responsibilities from one agent (donor)
to another (recipient) through a multi-phase verification protocol with
automatic rollback, state integrity checks, and post-handoff monitoring.

Usage::

    python -m replication handoff --inject --agents 8 --handoffs 20
    python -m replication handoff --inject --scenario cascade
    python -m replication handoff --inject --scenario storm
    python -m replication handoff --inject --scenario hoarding
    python -m replication handoff --format json
    python -m replication handoff --risk-threshold 0.6
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple


class HandoffPhase(str, Enum):
    NEGOTIATE = "negotiate"
    VERIFY_RECIPIENT = "verify_recipient"
    TRANSFER_STATE = "transfer_state"
    VALIDATE = "validate"
    ACTIVATE = "activate"
    MONITOR = "monitor"
    COMPLETE = "complete"
    ROLLED_BACK = "rolled_back"


class HandoffPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class HandoffRequest:
    donor_id: str
    recipient_id: str
    capabilities: List[str]
    state_snapshot: Dict[str, Any] = field(default_factory=dict)
    priority: HandoffPriority = HandoffPriority.MEDIUM
    reason: str = ""
    request_id: str = ""

    def __post_init__(self) -> None:
        if not self.request_id:
            self.request_id = f"hoff-{random.randint(10000, 99999)}"


@dataclass
class PhaseResult:
    phase: HandoffPhase
    success: bool
    duration_ms: float = 0.0
    checks_passed: List[str] = field(default_factory=list)
    checks_failed: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class HandoffRecord:
    request: HandoffRequest
    phases: List[PhaseResult] = field(default_factory=list)
    status: HandoffPhase = HandoffPhase.NEGOTIATE
    started_at: float = 0.0
    completed_at: float = 0.0
    rollback_reason: Optional[str] = None

    @property
    def duration_ms(self) -> float:
        if self.completed_at and self.started_at:
            return (self.completed_at - self.started_at) * 1000
        return sum(p.duration_ms for p in self.phases)

    @property
    def succeeded(self) -> bool:
        return self.status == HandoffPhase.COMPLETE


@dataclass
class Recommendation:
    severity: str
    category: str
    message: str
    agents_involved: List[str] = field(default_factory=list)


@dataclass
class FleetHandoffReport:
    records: List[HandoffRecord] = field(default_factory=list)
    total_handoffs: int = 0
    successful: int = 0
    failed: int = 0
    rolled_back: int = 0
    avg_duration_ms: float = 0.0
    risk_score: float = 0.0
    recommendations: List[Recommendation] = field(default_factory=list)


@dataclass
class AgentProfile:
    agent_id: str
    capabilities: List[str] = field(default_factory=list)
    capacity: int = 4
    trust_score: float = 0.8
    version: str = "1.0"
    state: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HandoffConfig:
    verification_timeout_ms: float = 5000.0
    max_retries: int = 2
    require_capability_match: bool = True
    require_state_integrity: bool = True
    monitor_duration_ms: float = 3000.0
    risk_threshold: float = 0.7


ALL_CAPABILITIES = [
    "compute", "storage", "network", "auth",
    "logging", "monitoring", "deployment", "config",
]

PHASE_ORDER = [
    HandoffPhase.NEGOTIATE, HandoffPhase.VERIFY_RECIPIENT,
    HandoffPhase.TRANSFER_STATE, HandoffPhase.VALIDATE,
    HandoffPhase.ACTIVATE, HandoffPhase.MONITOR,
]


class HandoffProtocol:
    """Multi-phase agent handoff protocol with verification and rollback."""

    def __init__(self, config: Optional[HandoffConfig] = None) -> None:
        self.config = config or HandoffConfig()
        self._agents: Dict[str, AgentProfile] = {}
        self._history: List[HandoffRecord] = []

    def register_agent(self, profile: AgentProfile) -> None:
        self._agents[profile.agent_id] = profile

    def get_agent(self, agent_id: str) -> Optional[AgentProfile]:
        return self._agents.get(agent_id)

    def execute_handoff(self, request: HandoffRequest) -> HandoffRecord:
        record = HandoffRecord(request=request, started_at=time.time())
        for phase in PHASE_ORDER:
            result = self._execute_phase(phase, request, record)
            record.phases.append(result)
            record.status = phase
            if not result.success:
                record.rollback_reason = (
                    f"Phase {phase.value} failed: " + "; ".join(result.checks_failed)
                )
                self._rollback(request, record)
                record.status = HandoffPhase.ROLLED_BACK
                record.completed_at = time.time()
                self._history.append(record)
                return record
        record.status = HandoffPhase.COMPLETE
        record.completed_at = time.time()
        self._apply_handoff(request)
        self._history.append(record)
        return record

    def batch_handoff(self, requests: Sequence[HandoffRequest]) -> FleetHandoffReport:
        records = [self.execute_handoff(r) for r in requests]
        return self.analyze_risks(records)

    def analyze_risks(self, records: Optional[Sequence[HandoffRecord]] = None) -> FleetHandoffReport:
        recs = list(records) if records else list(self._history)
        if not recs:
            return FleetHandoffReport()
        successful = sum(1 for r in recs if r.succeeded)
        failed = sum(1 for r in recs if not r.succeeded and r.status != HandoffPhase.ROLLED_BACK)
        rolled_back = sum(1 for r in recs if r.status == HandoffPhase.ROLLED_BACK)
        durations = [r.duration_ms for r in recs if r.duration_ms > 0]
        avg_dur = sum(durations) / len(durations) if durations else 0.0
        recommendations = self._detect_patterns(recs)
        risk_score = self._compute_risk_score(recs, recommendations)
        return FleetHandoffReport(
            records=recs, total_handoffs=len(recs), successful=successful,
            failed=failed, rolled_back=rolled_back,
            avg_duration_ms=round(avg_dur, 1), risk_score=round(risk_score, 1),
            recommendations=recommendations,
        )

    def _execute_phase(self, phase: HandoffPhase, req: HandoffRequest, record: HandoffRecord) -> PhaseResult:
        handlers = {
            HandoffPhase.NEGOTIATE: self._phase_negotiate,
            HandoffPhase.VERIFY_RECIPIENT: self._phase_verify_recipient,
            HandoffPhase.TRANSFER_STATE: self._phase_transfer_state,
            HandoffPhase.VALIDATE: self._phase_validate,
            HandoffPhase.ACTIVATE: self._phase_activate,
            HandoffPhase.MONITOR: self._phase_monitor,
        }
        start = time.monotonic()
        result = handlers[phase](req, record)
        result.duration_ms = (time.monotonic() - start) * 1000
        return result

    def _phase_negotiate(self, req: HandoffRequest, _rec: HandoffRecord) -> PhaseResult:
        passed, failed = [], []
        donor = self._agents.get(req.donor_id)
        recipient = self._agents.get(req.recipient_id)
        if donor:
            if all(c in donor.capabilities for c in req.capabilities):
                passed.append("donor_has_capabilities")
            else:
                failed.append("donor_missing_capabilities")
        else:
            failed.append("donor_not_found")
        if recipient:
            passed.append("recipient_exists")
        else:
            failed.append("recipient_not_found")
        if req.donor_id != req.recipient_id:
            passed.append("no_self_handoff")
        else:
            failed.append("self_handoff_not_allowed")
        active = [
            r for r in self._history
            if r.status not in (HandoffPhase.COMPLETE, HandoffPhase.ROLLED_BACK)
            and (r.request.donor_id == req.recipient_id or r.request.recipient_id == req.recipient_id)
        ]
        if not active:
            passed.append("no_active_conflicts")
        else:
            failed.append("recipient_in_active_handoff")
        return PhaseResult(phase=HandoffPhase.NEGOTIATE, success=len(failed) == 0, checks_passed=passed, checks_failed=failed)

    def _phase_verify_recipient(self, req: HandoffRequest, _rec: HandoffRecord) -> PhaseResult:
        passed, failed = [], []
        recipient = self._agents.get(req.recipient_id)
        if not recipient:
            return PhaseResult(phase=HandoffPhase.VERIFY_RECIPIENT, success=False, checks_failed=["recipient_not_found"])
        new_count = len(recipient.capabilities) + len(req.capabilities)
        if new_count <= recipient.capacity:
            passed.append("capacity_ok")
        else:
            failed.append(f"capacity_exceeded ({new_count}/{recipient.capacity})")
        if recipient.trust_score >= 0.5:
            passed.append(f"trust_ok ({recipient.trust_score:.2f})")
        else:
            failed.append(f"trust_too_low ({recipient.trust_score:.2f})")
        donor = self._agents.get(req.donor_id)
        if donor and donor.version == recipient.version:
            passed.append("version_compatible")
        elif donor:
            passed.append(f"version_noted ({donor.version} -> {recipient.version})")
        return PhaseResult(phase=HandoffPhase.VERIFY_RECIPIENT, success=len(failed) == 0, checks_passed=passed, checks_failed=failed)

    def _phase_transfer_state(self, req: HandoffRequest, _rec: HandoffRecord) -> PhaseResult:
        passed, failed = [], []
        if self.config.require_state_integrity and req.state_snapshot:
            state_bytes = json.dumps(req.state_snapshot, sort_keys=True).encode()
            integrity_hash = hashlib.sha256(state_bytes).hexdigest()[:16]
            passed.append(f"integrity_hash={integrity_hash}")
            if req.state_snapshot.get("_corrupted"):
                failed.append("state_corruption_detected")
            else:
                passed.append("state_integrity_verified")
            if req.state_snapshot.get("_incomplete"):
                failed.append("state_transfer_incomplete")
            else:
                passed.append("transfer_complete")
        else:
            passed.append("state_transfer_ok")
        return PhaseResult(phase=HandoffPhase.TRANSFER_STATE, success=len(failed) == 0, checks_passed=passed, checks_failed=failed)

    def _phase_validate(self, req: HandoffRequest, _rec: HandoffRecord) -> PhaseResult:
        passed, failed = [], []
        for cap in req.capabilities:
            if req.state_snapshot.get(f"_test_fail_{cap}"):
                failed.append(f"validation_failed:{cap}")
            else:
                passed.append(f"validation_passed:{cap}")
        return PhaseResult(phase=HandoffPhase.VALIDATE, success=len(failed) == 0, checks_passed=passed, checks_failed=failed)

    def _phase_activate(self, req: HandoffRequest, _rec: HandoffRecord) -> PhaseResult:
        passed, failed = [], []
        donor = self._agents.get(req.donor_id)
        recipient = self._agents.get(req.recipient_id)
        if donor and recipient:
            passed.append("responsibility_pointer_switched")
            passed.append("donor_deactivated_for_capabilities")
        else:
            failed.append("activation_failed_missing_agents")
        return PhaseResult(phase=HandoffPhase.ACTIVATE, success=len(failed) == 0, checks_passed=passed, checks_failed=failed)

    def _phase_monitor(self, req: HandoffRequest, _rec: HandoffRecord) -> PhaseResult:
        passed, failed = [], []
        if req.state_snapshot.get("_monitor_anomaly"):
            failed.append("post_handoff_anomaly_detected")
        else:
            passed.append("monitoring_window_clean")
            passed.append(f"monitor_duration={self.config.monitor_duration_ms}ms")
        return PhaseResult(phase=HandoffPhase.MONITOR, success=len(failed) == 0, checks_passed=passed, checks_failed=failed)

    def _rollback(self, req: HandoffRequest, record: HandoffRecord) -> None:
        pass

    def _apply_handoff(self, req: HandoffRequest) -> None:
        donor = self._agents.get(req.donor_id)
        recipient = self._agents.get(req.recipient_id)
        if donor and recipient:
            for cap in req.capabilities:
                if cap in donor.capabilities:
                    donor.capabilities.remove(cap)
                if cap not in recipient.capabilities:
                    recipient.capabilities.append(cap)

    def _detect_patterns(self, records: List[HandoffRecord]) -> List[Recommendation]:
        recs: List[Recommendation] = []
        pair_failures: Dict[Tuple[str, str], int] = Counter()
        pair_total: Dict[Tuple[str, str], int] = Counter()
        for r in records:
            pair = (r.request.donor_id, r.request.recipient_id)
            pair_total[pair] += 1
            if not r.succeeded:
                pair_failures[pair] += 1
        for pair, fails in pair_failures.items():
            total = pair_total[pair]
            if fails >= 2 and fails / total >= 0.5:
                recs.append(Recommendation(
                    severity="warning", category="pair_incompatibility",
                    message=f"Agents {pair[0]} -> {pair[1]} failed {fails}/{total} handoffs",
                    agents_involved=list(pair),
                ))
        chains = self._find_chains(records)
        for chain in chains:
            if len(chain) >= 3:
                recs.append(Recommendation(
                    severity="warning" if len(chain) < 4 else "critical",
                    category="cascade_chain",
                    message=f"Cascade chain ({len(chain)} hops: {' -> '.join(chain)}) - state integrity at risk",
                    agents_involved=chain,
                ))
        if len(records) >= 10:
            timestamps = sorted(r.started_at for r in records if r.started_at)
            if len(timestamps) >= 5:
                intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
                avg_interval = sum(intervals) / len(intervals) if intervals else float('inf')
                if avg_interval < 1.0:
                    recs.append(Recommendation(
                        severity="critical", category="handoff_storm",
                        message=f"Handoff storm - {len(records)} handoffs in rapid succession",
                        agents_involved=[],
                    ))
        cap_count: Dict[str, int] = Counter()
        for r in records:
            if r.succeeded:
                cap_count[r.request.recipient_id] += len(r.request.capabilities)
                cap_count[r.request.donor_id] -= len(r.request.capabilities)
        for agent_id, net in cap_count.items():
            if net >= 4:
                recs.append(Recommendation(
                    severity="warning", category="capability_hoarding",
                    message=f"Agent {agent_id} accumulated {net} net capabilities - single point of failure risk",
                    agents_involved=[agent_id],
                ))
        for r in records:
            if r.status == HandoffPhase.ROLLED_BACK:
                failed_phase = r.phases[-1].phase if r.phases else None
                if failed_phase in (HandoffPhase.ACTIVATE, HandoffPhase.MONITOR):
                    recs.append(Recommendation(
                        severity="critical", category="orphaned_capabilities",
                        message=f"Handoff {r.request.request_id} failed during {failed_phase.value} - capabilities may be orphaned",
                        agents_involved=[r.request.donor_id, r.request.recipient_id],
                    ))
        return recs

    def _find_chains(self, records: List[HandoffRecord]) -> List[List[str]]:
        successful = [r for r in records if r.succeeded]
        edges: Dict[str, List[str]] = defaultdict(list)
        for r in successful:
            edges[r.request.donor_id].append(r.request.recipient_id)
        chains: List[List[str]] = []
        visited: set = set()
        for start in edges:
            if start in visited:
                continue
            chain = [start]
            current = start
            while edges.get(current):
                nxt = edges[current][0]
                if nxt in chain:
                    break
                chain.append(nxt)
                visited.add(nxt)
                current = nxt
            if len(chain) >= 3:
                chains.append(chain)
        return chains

    def _compute_risk_score(self, records: List[HandoffRecord], recs: List[Recommendation]) -> float:
        if not records:
            return 0.0
        score = 0.0
        total = len(records)
        failed = sum(1 for r in records if not r.succeeded)
        score += (failed / total) * 40 if total else 0
        score += min(sum(1 for r in recs if r.severity == "critical") * 10, 30)
        score += min(sum(1 for r in recs if r.severity == "warning") * 5, 20)
        score += min(sum(1 for r in records if not r.succeeded and r.request.priority in (HandoffPriority.HIGH, HandoffPriority.CRITICAL)) * 5, 10)
        return min(score, 100.0)


def generate_synthetic_fleet(num_agents: int = 8) -> List[AgentProfile]:
    agents = []
    for i in range(num_agents):
        caps = random.sample(ALL_CAPABILITIES, random.randint(1, min(4, len(ALL_CAPABILITIES))))
        agents.append(AgentProfile(
            agent_id=f"agent-{i:02d}", capabilities=caps,
            capacity=random.randint(3, 6),
            trust_score=round(random.uniform(0.3, 1.0), 2),
            version=random.choice(["1.0", "1.0", "1.1"]),
            state={cap: {"data": f"state-{cap}-{i}"} for cap in caps},
        ))
    return agents


def generate_synthetic_handoffs(agents: List[AgentProfile], num_handoffs: int = 20, scenario: str = "normal") -> List[HandoffRequest]:
    requests: List[HandoffRequest] = []
    if scenario == "cascade":
        cap = random.choice(ALL_CAPABILITIES)
        for i in range(min(len(agents) - 1, 5)):
            state: Dict[str, Any] = {"chain_hop": i, "origin": agents[0].agent_id}
            if i >= 3:
                state["_incomplete"] = True
            requests.append(HandoffRequest(donor_id=agents[i].agent_id, recipient_id=agents[i+1].agent_id, capabilities=[cap], state_snapshot=state, priority=HandoffPriority.HIGH, reason=f"cascade hop {i+1}"))
            if cap not in agents[i].capabilities:
                agents[i].capabilities.append(cap)
        return requests
    if scenario == "storm":
        for i in range(min(num_handoffs, 15)):
            donor, recipient = random.sample(agents, 2)
            if donor.capabilities:
                requests.append(HandoffRequest(donor_id=donor.agent_id, recipient_id=recipient.agent_id, capabilities=[random.choice(donor.capabilities)], state_snapshot={"storm_batch": i}, priority=random.choice(list(HandoffPriority)), reason="storm handoff"))
        return requests
    if scenario == "hoarding":
        hoarder = agents[0]
        hoarder.capacity = 20
        for donor in agents[1:]:
            if donor.capabilities:
                requests.append(HandoffRequest(donor_id=donor.agent_id, recipient_id=hoarder.agent_id, capabilities=list(donor.capabilities), state_snapshot={"hoarding_target": hoarder.agent_id}, priority=HandoffPriority.MEDIUM, reason="consolidation"))
        return requests
    for i in range(num_handoffs):
        donor, recipient = random.sample(agents, 2)
        if not donor.capabilities:
            continue
        caps = random.sample(donor.capabilities, random.randint(1, min(2, len(donor.capabilities))))
        state: Dict[str, Any] = {c: {"data": f"transferred-{c}"} for c in caps}
        roll = random.random()
        if roll < 0.1:
            state["_corrupted"] = True
        elif roll < 0.15:
            state["_incomplete"] = True
        elif roll < 0.2:
            state[f"_test_fail_{caps[0]}"] = True
        requests.append(HandoffRequest(donor_id=donor.agent_id, recipient_id=recipient.agent_id, capabilities=caps, state_snapshot=state, priority=random.choice(list(HandoffPriority)), reason=random.choice(["load balancing", "maintenance", "upgrade", "failover", "capacity reallocation"])))
    return requests


def _risk_color(score: float) -> str:
    return "#22c55e" if score < 30 else "#f59e0b" if score < 60 else "#ef4444"


def _risk_label(score: float) -> str:
    return "LOW" if score < 30 else "MODERATE" if score < 60 else "HIGH"


def format_text(report: FleetHandoffReport) -> str:
    lines = ["=" * 56, "         AGENT SAFE HANDOFF PROTOCOL REPORT", "=" * 56, "",
             f"  Total Handoffs:  {report.total_handoffs}", f"  Successful:      {report.successful}",
             f"  Failed/Rollback: {report.failed + report.rolled_back}"]
    if report.total_handoffs:
        lines.append(f"  Success Rate:    {report.successful / report.total_handoffs * 100:.1f}%")
    lines += [f"  Avg Duration:    {report.avg_duration_ms:.1f} ms",
              f"  Risk Score:      {report.risk_score}/100 [{_risk_label(report.risk_score)}]", "",
              "-- Handoff Records " + "-" * 37]
    for r in report.records:
        tag = "[OK]" if r.succeeded else "[ROLLBACK]"
        lines.append(f"  {tag} {r.request.request_id}: {r.request.donor_id} -> {r.request.recipient_id} [{', '.join(r.request.capabilities)}] ({r.duration_ms:.0f}ms)")
        if r.rollback_reason:
            lines.append(f"     -> {r.rollback_reason}")
    if report.recommendations:
        lines += ["", "-- Recommendations " + "-" * 37]
        for rec in report.recommendations:
            lines.append(f"  [{rec.severity.upper()}] [{rec.category}] {rec.message}")
    return "\n".join(lines)


def format_json(report: FleetHandoffReport) -> str:
    def _ser(obj: Any) -> Any:
        if isinstance(obj, Enum):
            return obj.value
        return str(obj)
    data = {"total_handoffs": report.total_handoffs, "successful": report.successful, "failed": report.failed,
            "rolled_back": report.rolled_back, "avg_duration_ms": report.avg_duration_ms, "risk_score": report.risk_score,
            "records": [{"request_id": r.request.request_id, "donor": r.request.donor_id, "recipient": r.request.recipient_id,
                         "capabilities": r.request.capabilities, "priority": r.request.priority.value, "status": r.status.value,
                         "duration_ms": round(r.duration_ms, 1), "rollback_reason": r.rollback_reason,
                         "phases": [{"phase": p.phase.value, "success": p.success, "checks_passed": p.checks_passed, "checks_failed": p.checks_failed} for p in r.phases]}
                        for r in report.records],
            "recommendations": [{"severity": r.severity, "category": r.category, "message": r.message, "agents_involved": r.agents_involved} for r in report.recommendations]}
    return json.dumps(data, indent=2, default=_ser)


def format_html(report: FleetHandoffReport) -> str:
    rc = _risk_color(report.risk_score)
    rl = _risk_label(report.risk_score)
    sr = f"{report.successful / report.total_handoffs * 100:.1f}%" if report.total_handoffs else "N/A"
    rows = []
    for r in report.records:
        s = "OK" if r.succeeded else "ROLLBACK"
        badges = " ".join(f'<span class="b {"ok" if p.success else "fail"}">{p.phase.value}</span>' for p in r.phases)
        rows.append(f"<tr><td>{s}</td><td><code>{r.request.request_id}</code></td><td>{r.request.donor_id} &rarr; {r.request.recipient_id}</td><td>{', '.join(r.request.capabilities)}</td><td>{r.request.priority.value}</td><td>{badges}</td><td>{r.duration_ms:.0f}ms</td><td>{r.rollback_reason or '&mdash;'}</td></tr>")
    recs = "".join(f'<div class="rec {r.severity}"><strong>[{r.category}]</strong> {r.message}</div>' for r in report.recommendations) or '<p style="color:#64748b">No recommendations.</p>'
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Safe Handoff Report</title>
<style>body{{font-family:system-ui;background:#0f172a;color:#e2e8f0;margin:2rem}}h1{{color:#38bdf8}}h2{{color:#94a3b8;margin-top:2rem}}.stats{{display:flex;gap:1.5rem;flex-wrap:wrap;margin:1rem 0}}.stat{{background:#1e293b;padding:1rem 1.5rem;border-radius:8px;min-width:120px}}.stat .v{{font-size:1.8rem;font-weight:bold;color:#38bdf8}}.stat .l{{font-size:.85rem;color:#94a3b8}}.rg{{background:#1e293b;padding:1.5rem;border-radius:8px;text-align:center}}.rg .s{{font-size:3rem;font-weight:bold;color:{rc}}}.rg .l{{font-size:1.2rem;color:{rc}}}table{{width:100%;border-collapse:collapse;margin:1rem 0}}th,td{{padding:.5rem;text-align:left;border-bottom:1px solid #334155}}th{{color:#94a3b8}}.b{{display:inline-block;padding:2px 6px;border-radius:4px;margin:1px;font-size:.75rem}}.b.ok{{background:#16a34a33;color:#4ade80}}.b.fail{{background:#dc262633;color:#f87171}}.rec{{padding:.75rem;margin:.5rem 0;border-radius:6px;border-left:4px solid}}.rec.critical{{background:#dc262622;border-color:#ef4444}}.rec.warning{{background:#f59e0b22;border-color:#f59e0b}}.rec.info{{background:#3b82f622;border-color:#3b82f6}}</style></head><body>
<h1>Agent Safe Handoff Protocol Report</h1>
<div class="stats"><div class="stat"><div class="v">{report.total_handoffs}</div><div class="l">Total</div></div><div class="stat"><div class="v">{report.successful}</div><div class="l">Successful</div></div><div class="stat"><div class="v">{report.rolled_back}</div><div class="l">Rolled Back</div></div><div class="stat"><div class="v">{sr}</div><div class="l">Success Rate</div></div><div class="stat"><div class="v">{report.avg_duration_ms:.0f}ms</div><div class="l">Avg Duration</div></div><div class="rg"><div class="s">{report.risk_score}</div><div class="l">{rl} RISK</div></div></div>
<h2>Handoff Records</h2><table><thead><tr><th></th><th>ID</th><th>Transfer</th><th>Capabilities</th><th>Priority</th><th>Phases</th><th>Duration</th><th>Rollback</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Recommendations</h2>{recs}</body></html>"""


def main(argv: Optional[list] = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m replication handoff", description="Agent Safe Handoff Protocol")
    ap.add_argument("--inject", action="store_true")
    ap.add_argument("--agents", type=int, default=8)
    ap.add_argument("--handoffs", type=int, default=20)
    ap.add_argument("--scenario", choices=["normal", "cascade", "storm", "hoarding"], default="normal")
    ap.add_argument("--format", choices=["text", "json", "html"], default="text", dest="fmt")
    ap.add_argument("--risk-threshold", type=float, default=0.7)
    ap.add_argument("--monitor-duration", type=float, default=3000.0)
    args = ap.parse_args(argv)
    config = HandoffConfig(risk_threshold=args.risk_threshold, monitor_duration_ms=args.monitor_duration)
    protocol = HandoffProtocol(config)
    if args.inject:
        agents = generate_synthetic_fleet(args.agents)
    else:
        agents = generate_synthetic_fleet(4)
    for a in agents:
        protocol.register_agent(a)
    requests = generate_synthetic_handoffs(agents, args.handoffs if args.inject else 5, args.scenario)
    report = protocol.batch_handoff(requests)
    if args.fmt == "json":
        print(format_json(report))
    elif args.fmt == "html":
        print(format_html(report))
    else:
        print(format_text(report))
