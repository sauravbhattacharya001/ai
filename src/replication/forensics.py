"""Forensic Analyzer — post-simulation safety analysis.

Deep-dives into simulation reports to reconstruct decision chains,
identify near-miss safety events, detect escalation phases, and run
counterfactual "what-if" analyses with alternative parameters.

Usage (CLI)::

    python -m replication.forensics                                 # default run
    python -m replication.forensics --strategy greedy --max-depth 5  # custom config
    python -m replication.forensics --counterfactuals 3              # 3 what-if variants
    python -m replication.forensics --json                           # JSON output
    python -m replication.forensics --summary-only                   # brief summary

Programmatic::

    from replication.forensics import ForensicAnalyzer
    from replication.simulator import Simulator, ScenarioConfig

    config = ScenarioConfig(strategy="greedy", max_depth=4)
    sim = Simulator(config)
    report = sim.run()

    analyzer = ForensicAnalyzer()
    forensics = analyzer.analyze(report)
    print(forensics.render())
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .simulator import ScenarioConfig, Simulator, SimulationReport, WorkerRecord


# ── Helpers ──────────────────────────────────────────────────────


def _box_header(title: str, width: int = 43) -> List[str]:
    """Create a box-drawing header with centered title.

    Returns 3 lines: top border, title line, bottom border.
    """
    inner = width - 2  # subtract 2 for the │ characters
    return [
        "┌" + "─" * inner + "┐",
        "│" + title.center(inner) + "│",
        "└" + "─" * inner + "┘",
    ]


# ── Data models ──────────────────────────────────────────────────


@dataclass
class ForensicEvent:
    """A single reconstructed event with safety annotations."""

    step: int
    time_ms: float
    event_type: str  # "spawn", "task", "replicate_ok", "replicate_denied", "shutdown"
    worker_id: str
    parent_id: Optional[str]
    depth: int
    detail: str
    safety_relevant: bool
    safety_note: str = ""


@dataclass
class NearMiss:
    """A moment where a safety limit was almost breached."""

    step: int
    time_ms: float
    metric: str  # "depth", "replicas", "concurrent_workers"
    current_value: float
    limit_value: float
    headroom_pct: float  # 0% = at limit, 100% = far from limit
    worker_id: str
    description: str


@dataclass
class EscalationPhase:
    """A period of rapid worker growth."""

    start_step: int
    end_step: int
    start_time_ms: float
    end_time_ms: float
    workers_at_start: int
    workers_at_end: int
    growth_rate: float  # workers per step
    peak_depth: int
    description: str


@dataclass
class Counterfactual:
    """A 'what-if' comparison with modified parameters."""

    parameter: str
    original_value: Any
    modified_value: Any
    original_workers: int
    modified_workers: int
    original_denied: int
    modified_denied: int
    original_max_depth: int
    modified_max_depth: int
    delta_workers: int
    delta_denied: int
    insight: str


@dataclass
class DecisionPoint:
    """A controller decision with context."""

    step: int
    time_ms: float
    worker_id: str
    decision: str  # "allow", "deny_depth", "deny_max_replicas", "deny_cooldown", etc.
    depth: int
    active_workers: int
    reason: str


@dataclass
class ForensicReport:
    """Complete forensic analysis of a simulation run."""

    config: ScenarioConfig
    events: List[ForensicEvent]
    near_misses: List[NearMiss]
    escalation_phases: List[EscalationPhase]
    counterfactuals: List[Counterfactual]
    decision_points: List[DecisionPoint]
    safety_summary: Dict[str, Any]
    recommendations: List[str]

    def render_events(self) -> str:
        """Render reconstructed event timeline with safety annotations."""
        lines: List[str] = []
        lines.extend(_box_header("Forensic Event Timeline"))
        lines.append("")

        icons = {
            "spawn": "🟢",
            "task": "⚡",
            "replicate_ok": "🔀",
            "replicate_denied": "🚫",
            "shutdown": "🔴",
        }

        for evt in self.events:
            icon = icons.get(evt.event_type, "•")
            flag = " ⚠️" if evt.safety_relevant else ""
            ts = f"{evt.time_ms:>8.1f}ms"
            wid = evt.worker_id[:8]
            line = f"  {ts}  {icon}  [{wid}] d={evt.depth}  {evt.detail}{flag}"
            lines.append(line)
            if evt.safety_note:
                lines.append(f"           └─ {evt.safety_note}")

        return "\n".join(lines)

    def render_near_misses(self) -> str:
        """Render near-miss safety events."""
        lines: List[str] = []
        lines.extend(_box_header("Near-Miss Analysis"))
        lines.append("")

        if not self.near_misses:
            lines.append("  No near-miss events detected.")
            lines.append("  Safety limits were not closely approached.")
            return "\n".join(lines)

        for nm in self.near_misses:
            headroom_bar = "█" * max(1, int(nm.headroom_pct / 5))
            lines.append(f"  Step {nm.step} ({nm.time_ms:.1f}ms) — {nm.metric}")
            lines.append(f"    Value: {nm.current_value:.0f} / Limit: {nm.limit_value:.0f}  "
                         f"Headroom: {nm.headroom_pct:.1f}%  {headroom_bar}")
            lines.append(f"    Worker: [{nm.worker_id[:8]}]")
            lines.append(f"    {nm.description}")
            lines.append("")

        return "\n".join(lines)

    def render_escalation(self) -> str:
        """Render escalation phase analysis."""
        lines: List[str] = []
        lines.extend(_box_header("Escalation Phase Analysis"))
        lines.append("")

        if not self.escalation_phases:
            lines.append("  No escalation phases detected.")
            lines.append("  Worker growth remained stable throughout.")
            return "\n".join(lines)

        for i, phase in enumerate(self.escalation_phases, 1):
            lines.append(f"  Phase {i}: Steps {phase.start_step}–{phase.end_step}")
            lines.append(f"    Duration: {phase.end_time_ms - phase.start_time_ms:.1f}ms")
            lines.append(f"    Workers:  {phase.workers_at_start} → {phase.workers_at_end} "
                         f"(+{phase.workers_at_end - phase.workers_at_start})")
            lines.append(f"    Growth:   {phase.growth_rate:.2f} workers/step")
            lines.append(f"    Peak:     depth {phase.peak_depth}")
            lines.append(f"    {phase.description}")
            lines.append("")

        return "\n".join(lines)

    def render_counterfactuals(self) -> str:
        """Render counterfactual what-if analysis."""
        lines: List[str] = []
        lines.extend(_box_header("Counterfactual Analysis"))
        lines.append("")

        if not self.counterfactuals:
            lines.append("  No counterfactual analyses performed.")
            return "\n".join(lines)

        for cf in self.counterfactuals:
            sign = "+" if cf.delta_workers > 0 else ""
            lines.append(f"  What if {cf.parameter} = {cf.modified_value} "
                         f"(was {cf.original_value})?")
            lines.append(f"    Workers:  {cf.original_workers} → {cf.modified_workers} "
                         f"({sign}{cf.delta_workers})")
            lines.append(f"    Denied:   {cf.original_denied} → {cf.modified_denied} "
                         f"({'+' if cf.delta_denied > 0 else ''}{cf.delta_denied})")
            lines.append(f"    Max Depth: {cf.original_max_depth} → {cf.modified_max_depth}")
            lines.append(f"    → {cf.insight}")
            lines.append("")

        return "\n".join(lines)

    def render_decisions(self) -> str:
        """Render controller decision audit trail."""
        lines: List[str] = []
        lines.extend(_box_header("Decision Point Analysis"))
        lines.append("")

        if not self.decision_points:
            lines.append("  No replication decisions recorded.")
            return "\n".join(lines)

        allow_count = sum(1 for d in self.decision_points if d.decision == "allow")
        deny_count = len(self.decision_points) - allow_count
        lines.append(f"  Total decisions: {len(self.decision_points)} "
                     f"(✅ {allow_count} allowed, 🚫 {deny_count} denied)")
        lines.append("")

        # Group denials by reason
        denial_reasons: Dict[str, int] = {}
        for dp in self.decision_points:
            if dp.decision != "allow":
                denial_reasons[dp.decision] = denial_reasons.get(dp.decision, 0) + 1

        if denial_reasons:
            lines.append("  Denial Breakdown:")
            for reason, count in sorted(denial_reasons.items(), key=lambda x: -x[1]):
                bar = "█" * count
                lines.append(f"    {reason:.<30s} {count:>3d}  {bar}")
            lines.append("")

        # Show each decision
        for dp in self.decision_points:
            icon = "✅" if dp.decision == "allow" else "🚫"
            ts = f"{dp.time_ms:>8.1f}ms"
            lines.append(f"  {ts}  {icon}  [{dp.worker_id[:8]}] d={dp.depth} "
                         f"active={dp.active_workers}  {dp.reason}")

        return "\n".join(lines)

    def render_summary(self) -> str:
        """Render the safety summary and recommendations."""
        lines: List[str] = []
        lines.extend(_box_header("Forensic Safety Summary"))
        lines.append("")

        s = self.safety_summary
        lines.append(f"  Safety Score:     {s.get('safety_score', 0):.0f}/100")
        lines.append(f"  Risk Level:       {s.get('risk_level', 'UNKNOWN')}")
        lines.append(f"  Contract Honored: {'Yes ✅' if s.get('contract_honored') else 'No ❌'}")
        lines.append(f"  Total Workers:    {s.get('total_workers', 0)}")
        lines.append(f"  Max Depth Hit:    {s.get('max_depth_reached', 0)} / {s.get('max_depth_limit', 0)}")
        lines.append(f"  Denial Rate:      {s.get('denial_rate_pct', 0):.1f}%")
        lines.append(f"  Near Misses:      {s.get('near_miss_count', 0)}")
        lines.append(f"  Escalation Phases: {s.get('escalation_count', 0)}")
        lines.append("")

        if self.recommendations:
            lines.append("  Recommendations:")
            for i, rec in enumerate(self.recommendations, 1):
                lines.append(f"    {i}. {rec}")

        return "\n".join(lines)

    def render(self) -> str:
        """Render the complete forensic report."""
        sections = [
            self.render_summary(),
            self.render_near_misses(),
            self.render_escalation(),
            self.render_decisions(),
            self.render_counterfactuals(),
        ]
        return "\n\n".join(sections)

    def to_dict(self) -> Dict[str, Any]:
        """Export as a JSON-serializable dictionary."""
        from dataclasses import asdict
        result = asdict(self)
        # Keep only the config fields that are relevant for forensics
        if self.config:
            result["config"] = {
                "strategy": self.config.strategy,
                "max_depth": self.config.max_depth,
                "max_replicas": self.config.max_replicas,
                "cooldown_seconds": self.config.cooldown_seconds,
                "tasks_per_worker": self.config.tasks_per_worker,
            }
        return result


# ── Analyzer ─────────────────────────────────────────────────────


class ForensicAnalyzer:
    """Performs post-simulation forensic analysis.

    Reconstructs what happened during a simulation, identifies
    safety-critical moments, and generates actionable insights.

    Parameters
    ----------
    counterfactual_count : int
        Number of what-if variants to run (default 5).
    near_miss_threshold_pct : float
        Headroom percentage below which a limit approach is flagged
        as a near miss (default 20.0 = within 20% of limit).
    escalation_growth_threshold : float
        Minimum workers-per-step growth rate to flag as an escalation
        phase (default 1.5).
    """

    def __init__(
        self,
        counterfactual_count: int = 5,
        near_miss_threshold_pct: float = 20.0,
        escalation_growth_threshold: float = 1.5,
    ) -> None:
        self.counterfactual_count = counterfactual_count
        self.near_miss_threshold_pct = near_miss_threshold_pct
        self.escalation_growth_threshold = escalation_growth_threshold

    def analyze(
        self,
        report: Optional[SimulationReport] = None,
        config: Optional[ScenarioConfig] = None,
    ) -> ForensicReport:
        """Run forensic analysis on a simulation report.

        If *report* is ``None``, a new simulation is run using *config*
        (or default settings).
        """
        if report is None:
            cfg = config or ScenarioConfig()
            report = Simulator(cfg).run()
        else:
            cfg = report.config

        events = self._reconstruct_events(report)
        near_misses = self._detect_near_misses(events, report)
        escalation = self._detect_escalation(events, report)
        decisions = self._extract_decisions(report)
        counterfactuals = self._run_counterfactuals(report)
        safety_summary = self._build_safety_summary(report, near_misses, escalation, decisions)
        recommendations = self._generate_recommendations(
            report, near_misses, escalation, decisions, safety_summary
        )

        return ForensicReport(
            config=cfg,
            events=events,
            near_misses=near_misses,
            escalation_phases=escalation,
            counterfactuals=counterfactuals,
            decision_points=decisions,
            safety_summary=safety_summary,
            recommendations=recommendations,
        )

    # ── Event reconstruction ──

    def _reconstruct_events(self, report: SimulationReport) -> List[ForensicEvent]:
        """Rebuild the timeline with safety annotations."""
        events: List[ForensicEvent] = []

        for step, entry in enumerate(report.timeline):
            etype = entry["type"]
            wid = entry.get("worker_id", "unknown")
            rec = report.workers.get(wid)
            depth = rec.depth if rec else 0
            parent_id = rec.parent_id if rec else None
            detail = entry.get("detail", "")
            time_ms = entry.get("time_ms", 0.0)

            safety_relevant = False
            safety_note = ""

            if etype == "replicate_ok":
                safety_relevant = True
                # Check if this pushed us close to limits
                worker_count = self._count_active_at_step(report, step)
                if worker_count >= report.config.max_replicas * 0.8:
                    safety_note = (
                        f"Worker count ({worker_count}) approaching "
                        f"replica limit ({report.config.max_replicas})"
                    )
                elif depth >= report.config.max_depth - 1:
                    safety_note = (
                        f"Depth {depth + 1} is at maximum "
                        f"({report.config.max_depth})"
                    )
                else:
                    safety_note = f"Replication at depth {depth}, " \
                                  f"headroom: {report.config.max_depth - depth - 1} levels"

            elif etype == "replicate_denied":
                safety_relevant = True
                safety_note = "Safety contract enforced — replication blocked"

            elif etype == "spawn" and depth == 0:
                safety_note = "Simulation root worker initialized"

            events.append(ForensicEvent(
                step=step,
                time_ms=time_ms,
                event_type=etype,
                worker_id=wid,
                parent_id=parent_id,
                depth=depth,
                detail=detail,
                safety_relevant=safety_relevant,
                safety_note=safety_note,
            ))

        return events

    def _count_active_at_step(self, report: SimulationReport, step: int) -> int:
        """Count workers created up to *step* that haven't shut down yet."""
        spawns = 0
        shutdowns = 0
        for i, entry in enumerate(report.timeline):
            if i > step:
                break
            if entry["type"] == "spawn" or entry["type"] == "replicate_ok":
                spawns += 1
            elif entry["type"] == "shutdown":
                shutdowns += 1
        return max(0, spawns - shutdowns)

    # ── Near-miss detection ──

    def _detect_near_misses(
        self, events: List[ForensicEvent], report: SimulationReport
    ) -> List[NearMiss]:
        """Find moments where safety limits were closely approached."""
        near_misses: List[NearMiss] = []
        threshold = self.near_miss_threshold_pct

        # Track cumulative state
        total_spawned = 0
        max_depth_seen = 0

        for evt in events:
            if evt.event_type in ("spawn", "replicate_ok"):
                total_spawned += 1
                depth = evt.depth
                if evt.event_type == "replicate_ok":
                    # The child's depth is depth + 1, parse from detail
                    for w in report.workers.values():
                        if w.parent_id == evt.worker_id:
                            depth = max(depth, w.depth)
                max_depth_seen = max(max_depth_seen, depth)

                # Check replica headroom
                max_r = report.config.max_replicas
                if max_r > 0:
                    headroom_replicas = ((max_r - total_spawned) / max_r) * 100
                    if 0 < headroom_replicas <= threshold:
                        near_misses.append(NearMiss(
                            step=evt.step,
                            time_ms=evt.time_ms,
                            metric="replicas",
                            current_value=total_spawned,
                            limit_value=max_r,
                            headroom_pct=headroom_replicas,
                            worker_id=evt.worker_id,
                            description=(
                                f"Replica count {total_spawned} within "
                                f"{headroom_replicas:.1f}% of limit {max_r}"
                            ),
                        ))

                # Check depth headroom
                max_d = report.config.max_depth
                if max_d > 0:
                    headroom_depth = ((max_d - max_depth_seen) / max_d) * 100
                    if 0 < headroom_depth <= threshold:
                        near_misses.append(NearMiss(
                            step=evt.step,
                            time_ms=evt.time_ms,
                            metric="depth",
                            current_value=max_depth_seen,
                            limit_value=max_d,
                            headroom_pct=headroom_depth,
                            worker_id=evt.worker_id,
                            description=(
                                f"Depth {max_depth_seen} within "
                                f"{headroom_depth:.1f}% of limit {max_d}"
                            ),
                        ))

        return near_misses

    # ── Escalation detection ──

    def _detect_escalation(
        self, events: List[ForensicEvent], report: SimulationReport
    ) -> List[EscalationPhase]:
        """Identify periods of rapid worker growth."""
        # Build a step → cumulative_spawns series
        spawn_steps: List[Tuple[int, float, int]] = []  # (step, time_ms, cumulative)
        cumulative = 0
        for evt in events:
            if evt.event_type in ("spawn", "replicate_ok"):
                cumulative += 1
                spawn_steps.append((evt.step, evt.time_ms, cumulative))

        if len(spawn_steps) < 2:
            return []

        phases: List[EscalationPhase] = []
        window = 3  # sliding window size
        threshold = self.escalation_growth_threshold

        i = 0
        while i < len(spawn_steps) - 1:
            # Calculate growth rate over the next window
            j = min(i + window, len(spawn_steps) - 1)
            step_span = spawn_steps[j][0] - spawn_steps[i][0]
            if step_span == 0:
                i += 1
                continue
            worker_growth = spawn_steps[j][2] - spawn_steps[i][2]
            rate = worker_growth / step_span

            if rate >= threshold:
                # Found an escalation — extend to full phase
                phase_start = i
                while j < len(spawn_steps) - 1:
                    next_j = min(j + 1, len(spawn_steps) - 1)
                    if next_j == j:
                        break
                    local_span = spawn_steps[next_j][0] - spawn_steps[j][0]
                    if local_span == 0:
                        j = next_j
                        continue
                    local_rate = (spawn_steps[next_j][2] - spawn_steps[j][2]) / local_span
                    if local_rate >= threshold * 0.5:
                        j = next_j
                    else:
                        break

                # Determine peak depth during this phase
                peak_depth = 0
                for evt in events:
                    if spawn_steps[phase_start][0] <= evt.step <= spawn_steps[j][0]:
                        peak_depth = max(peak_depth, evt.depth)

                total_growth = spawn_steps[j][2] - spawn_steps[phase_start][2]
                total_span = spawn_steps[j][0] - spawn_steps[phase_start][0]
                avg_rate = total_growth / max(1, total_span)

                phase_desc = "Rapid escalation" if avg_rate >= threshold * 2 else "Moderate escalation"
                phase_desc += f" — {total_growth} workers spawned across {total_span} steps"

                phases.append(EscalationPhase(
                    start_step=spawn_steps[phase_start][0],
                    end_step=spawn_steps[j][0],
                    start_time_ms=spawn_steps[phase_start][1],
                    end_time_ms=spawn_steps[j][1],
                    workers_at_start=spawn_steps[phase_start][2],
                    workers_at_end=spawn_steps[j][2],
                    growth_rate=avg_rate,
                    peak_depth=peak_depth,
                    description=phase_desc,
                ))
                i = j + 1
            else:
                i += 1

        return phases

    # ── Decision extraction ──

    def _extract_decisions(self, report: SimulationReport) -> List[DecisionPoint]:
        """Extract controller replication decisions from audit events."""
        decisions: List[DecisionPoint] = []
        step = 0

        for entry in report.timeline:
            etype = entry["type"]
            wid = entry.get("worker_id", "unknown")
            time_ms = entry.get("time_ms", 0.0)
            rec = report.workers.get(wid)
            depth = rec.depth if rec else 0

            if etype == "replicate_ok":
                active = self._count_active_at_step(report, step)
                decisions.append(DecisionPoint(
                    step=step,
                    time_ms=time_ms,
                    worker_id=wid,
                    decision="allow",
                    depth=depth,
                    active_workers=active,
                    reason=f"Replication allowed at depth {depth}",
                ))
            elif etype == "replicate_denied":
                active = self._count_active_at_step(report, step)
                # Try to determine reason from audit events
                deny_reason = self._find_denial_reason(report, wid, step)
                decisions.append(DecisionPoint(
                    step=step,
                    time_ms=time_ms,
                    worker_id=wid,
                    decision=deny_reason,
                    depth=depth,
                    active_workers=active,
                    reason=f"Replication denied: {deny_reason}",
                ))
            step += 1

        return decisions

    def _find_denial_reason(
        self, report: SimulationReport, worker_id: str, step: int
    ) -> str:
        """Try to determine why a replication was denied from audit events."""
        # Check audit events near this step for denial reasons
        for evt in report.audit_events:
            decision = evt.get("decision", "")
            if decision.startswith("deny_"):
                return decision
        return "deny_unknown"

    # ── Counterfactual analysis ──

    def _run_counterfactuals(self, report: SimulationReport) -> List[Counterfactual]:
        """Run simulations with modified parameters to explore what-ifs."""
        counterfactuals: List[Counterfactual] = []
        cfg = report.config
        orig_workers = len(report.workers)
        orig_denied = report.total_replications_denied
        orig_max_depth = max((w.depth for w in report.workers.values()), default=0)

        # Define parameter variations
        variations: List[Tuple[str, str, Any, Any]] = []

        # max_depth variations
        if cfg.max_depth > 1:
            variations.append(("max_depth", "max_depth", cfg.max_depth, cfg.max_depth - 1))
        variations.append(("max_depth", "max_depth", cfg.max_depth, cfg.max_depth + 2))

        # max_replicas variations
        if cfg.max_replicas > 2:
            variations.append(("max_replicas", "max_replicas", cfg.max_replicas,
                               max(2, cfg.max_replicas // 2)))
        variations.append(("max_replicas", "max_replicas", cfg.max_replicas,
                           cfg.max_replicas * 2))

        # cooldown variation
        if cfg.cooldown_seconds == 0:
            variations.append(("cooldown_seconds", "cooldown_seconds", 0.0, 0.01))
        else:
            variations.append(("cooldown_seconds", "cooldown_seconds",
                               cfg.cooldown_seconds, 0.0))

        # tasks_per_worker variation
        if cfg.tasks_per_worker > 1:
            variations.append(("tasks_per_worker", "tasks_per_worker",
                               cfg.tasks_per_worker, cfg.tasks_per_worker + 1))

        # Limit to configured count
        variations = variations[: self.counterfactual_count]

        for param_name, attr_name, orig_val, new_val in variations:
            # Create modified config
            mod_cfg = ScenarioConfig(
                max_depth=cfg.max_depth,
                max_replicas=cfg.max_replicas,
                cooldown_seconds=cfg.cooldown_seconds,
                expiration_seconds=cfg.expiration_seconds,
                strategy=cfg.strategy,
                tasks_per_worker=cfg.tasks_per_worker,
                replication_probability=cfg.replication_probability,
                secret=cfg.secret,
                seed=cfg.seed,
                cpu_limit=cfg.cpu_limit,
                memory_limit_mb=cfg.memory_limit_mb,
            )
            setattr(mod_cfg, attr_name, new_val)

            try:
                mod_report = Simulator(mod_cfg).run()
                mod_workers = len(mod_report.workers)
                mod_denied = mod_report.total_replications_denied
                mod_max_depth = max((w.depth for w in mod_report.workers.values()), default=0)
            except Exception:
                continue

            delta_w = mod_workers - orig_workers
            delta_d = mod_denied - orig_denied

            # Generate insight
            insight = self._counterfactual_insight(
                param_name, orig_val, new_val,
                orig_workers, mod_workers,
                orig_denied, mod_denied,
            )

            counterfactuals.append(Counterfactual(
                parameter=param_name,
                original_value=orig_val,
                modified_value=new_val,
                original_workers=orig_workers,
                modified_workers=mod_workers,
                original_denied=orig_denied,
                modified_denied=mod_denied,
                original_max_depth=orig_max_depth,
                modified_max_depth=mod_max_depth,
                delta_workers=delta_w,
                delta_denied=delta_d,
                insight=insight,
            ))

        return counterfactuals

    def _counterfactual_insight(
        self, param: str, orig: Any, new: Any,
        orig_w: int, mod_w: int,
        orig_d: int, mod_d: int,
    ) -> str:
        """Generate a human-readable insight for a counterfactual."""
        delta_w = mod_w - orig_w
        delta_d = mod_d - orig_d

        if param == "max_depth":
            if delta_w > 0:
                return (f"Increasing depth limit allows {delta_w} more workers, "
                        f"expanding the attack surface")
            elif delta_w < 0:
                return (f"Reducing depth limit prevents {abs(delta_w)} workers, "
                        f"tightening containment")
            return "No change in worker count — depth limit wasn't the binding constraint"

        elif param == "max_replicas":
            if delta_w > 0:
                return (f"Doubling replica limit enables {delta_w} more workers — "
                        f"consider if this headroom is needed")
            elif delta_w < 0:
                return (f"Halving replicas blocks {abs(delta_w)} workers with "
                        f"{mod_d - orig_d:+d} more denials")
            return "Replica limit change had no effect — other constraints dominate"

        elif param == "cooldown_seconds":
            if new > orig:
                return (f"Adding cooldown reduces throughput; "
                        f"{delta_w:+d} workers, useful for rate-limiting")
            else:
                return (f"Removing cooldown enables faster replication; "
                        f"{delta_w:+d} workers")

        elif param == "tasks_per_worker":
            return (f"More tasks per worker gives more replication opportunities; "
                    f"{delta_w:+d} workers")

        return f"Parameter change: {delta_w:+d} workers, {delta_d:+d} denials"

    # ── Safety summary ──

    def _build_safety_summary(
        self,
        report: SimulationReport,
        near_misses: List[NearMiss],
        escalation: List[EscalationPhase],
        decisions: List[DecisionPoint],
    ) -> Dict[str, Any]:
        """Build the top-level safety summary."""
        total_workers = len(report.workers)
        max_depth_reached = max((w.depth for w in report.workers.values()), default=0)
        total_attempted = report.total_replications_attempted
        total_denied = report.total_replications_denied

        denial_rate = (total_denied / total_attempted * 100) if total_attempted > 0 else 0
        # max_replicas is a concurrent limit, not total.  Find peak concurrency.
        peak_concurrent = 0
        active = 0
        for entry in report.timeline:
            if entry["type"] in ("spawn", "replicate_ok"):
                active += 1
                peak_concurrent = max(peak_concurrent, active)
            elif entry["type"] == "shutdown":
                active = max(0, active - 1)

        contract_honored = max_depth_reached <= report.config.max_depth and \
                           peak_concurrent <= report.config.max_replicas + 1  # +1 for root

        # Safety score: 100 = perfectly safe, 0 = no safety
        score = 100.0

        # Penalize for depth usage
        depth_usage = max_depth_reached / max(1, report.config.max_depth)
        score -= depth_usage * 20  # up to -20

        # Penalize for replica usage
        replica_usage = total_workers / max(1, report.config.max_replicas + 1)
        score -= replica_usage * 20  # up to -20

        # Reward for denials (enforcement working)
        if total_attempted > 0 and denial_rate > 0:
            score += min(15, denial_rate * 0.3)  # up to +15

        # Penalize for near misses
        score -= len(near_misses) * 5  # -5 per near miss

        # Penalize for escalation phases
        score -= len(escalation) * 10  # -10 per escalation

        # Penalize if contract violated
        if not contract_honored:
            score -= 30

        score = max(0, min(100, score))

        # Risk level
        if score >= 80:
            risk_level = "LOW"
        elif score >= 60:
            risk_level = "MODERATE"
        elif score >= 40:
            risk_level = "ELEVATED"
        elif score >= 20:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"

        return {
            "safety_score": score,
            "risk_level": risk_level,
            "contract_honored": contract_honored,
            "total_workers": total_workers,
            "max_depth_reached": max_depth_reached,
            "max_depth_limit": report.config.max_depth,
            "total_replications_attempted": total_attempted,
            "total_replications_denied": total_denied,
            "denial_rate_pct": denial_rate,
            "near_miss_count": len(near_misses),
            "escalation_count": len(escalation),
            "decision_count": len(decisions),
        }

    # ── Recommendations ──

    def _generate_recommendations(
        self,
        report: SimulationReport,
        near_misses: List[NearMiss],
        escalation: List[EscalationPhase],
        decisions: List[DecisionPoint],
        summary: Dict[str, Any],
    ) -> List[str]:
        """Generate actionable safety recommendations."""
        recs: List[str] = []

        # Near-miss recommendations
        depth_nms = [nm for nm in near_misses if nm.metric == "depth"]
        replica_nms = [nm for nm in near_misses if nm.metric == "replicas"]

        if depth_nms:
            min_headroom = min(nm.headroom_pct for nm in depth_nms)
            recs.append(
                f"Depth limit nearly breached (min headroom {min_headroom:.1f}%). "
                f"Consider reducing max_depth from {report.config.max_depth} "
                f"to {max(1, report.config.max_depth - 1)} for safety margin."
            )

        if replica_nms:
            min_headroom = min(nm.headroom_pct for nm in replica_nms)
            recs.append(
                f"Replica limit closely approached (min headroom {min_headroom:.1f}%). "
                f"Reduce max_replicas or add cooldown to create breathing room."
            )

        # Escalation recommendations
        if escalation:
            max_rate = max(ep.growth_rate for ep in escalation)
            recs.append(
                f"Detected {len(escalation)} escalation phase(s) with up to "
                f"{max_rate:.1f} workers/step growth. Add cooldown_seconds "
                f"or reduce tasks_per_worker to dampen rapid growth."
            )

        # Low denial rate
        denial_rate = summary.get("denial_rate_pct", 0)
        if denial_rate < 10 and report.total_replications_attempted > 5:
            recs.append(
                f"Denial rate is only {denial_rate:.1f}% — contracts may be too "
                f"permissive. Consider tightening max_depth or max_replicas."
            )

        # Contract violation
        if not summary.get("contract_honored"):
            recs.append(
                "⚠️ Contract was violated during simulation! Review controller "
                "enforcement logic for potential bypasses."
            )

        # No denials at all
        if report.total_replications_denied == 0 and report.total_replications_attempted > 0:
            recs.append(
                "No replication was denied — safety limits never engaged. "
                "This may indicate overly generous limits."
            )

        # High risk
        if summary.get("safety_score", 100) < 40:
            recs.append(
                "Safety score is below 40 — this scenario poses significant risk. "
                "Consider using the 'minimal' or 'conservative' preset."
            )

        if not recs:
            recs.append(
                "No significant safety concerns detected. Current parameters "
                "provide adequate containment margins."
            )

        return recs


# ── CLI ──────────────────────────────────────────────────────────


def main() -> None:
    """CLI entry point for forensic analysis."""
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        description="Forensic Analyzer — post-simulation safety analysis",
        prog="python -m replication.forensics",
    )
    parser.add_argument("--strategy", choices=["greedy", "conservative", "random", "chain", "burst"],
                        default="greedy", help="Replication strategy (default: greedy)")
    parser.add_argument("--max-depth", type=int, default=3, help="Max replication depth")
    parser.add_argument("--max-replicas", type=int, default=10, help="Max total replicas")
    parser.add_argument("--cooldown", type=float, default=0.0, help="Cooldown between replications (seconds)")
    parser.add_argument("--tasks-per-worker", type=int, default=2, help="Tasks per worker")
    parser.add_argument("--scenario", choices=["minimal", "balanced", "stress", "chain", "burst"],
                        help="Use a built-in scenario preset")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--counterfactuals", type=int, default=5,
                        help="Number of what-if variants to run (default: 5)")
    parser.add_argument("--near-miss-threshold", type=float, default=20.0,
                        help="Near-miss headroom threshold percentage (default: 20)")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--summary-only", action="store_true", help="Show only the summary")

    args = parser.parse_args()

    # Build config
    if args.scenario:
        from .simulator import PRESETS
        config = PRESETS[args.scenario]
        if args.seed is not None:
            config.seed = args.seed
    else:
        config = ScenarioConfig(
            strategy=args.strategy,
            max_depth=args.max_depth,
            max_replicas=args.max_replicas,
            cooldown_seconds=args.cooldown,
            tasks_per_worker=args.tasks_per_worker,
            seed=args.seed,
        )

    # Run simulation
    report = Simulator(config).run()

    # Analyze
    analyzer = ForensicAnalyzer(
        counterfactual_count=args.counterfactuals,
        near_miss_threshold_pct=args.near_miss_threshold,
    )
    forensics = analyzer.analyze(report)

    if args.json:
        print(json.dumps(forensics.to_dict(), indent=2, default=str))
    elif args.summary_only:
        print(forensics.render_summary())
    else:
        print(forensics.render())


if __name__ == "__main__":
    main()
