"""Defense Layer Redundancy Advisor - agentic defense-in-depth auditor.

Sibling to ``remediation_planner`` / ``safety_debt`` / ``kill_switch_tuner``
/ ``remediation_roi`` / ``remediation_assignment`` / ``remediation_progress``.

Given a fleet definition (threat catalogue + active defense layers), the
advisor evaluates **defense-in-depth coverage** for every threat, surfaces
**single points of failure (SPOFs)**, identifies **orphaned** or
**over-redundant** layers, and synthesizes a P0-first deduplicated
playbook of concrete actions for the safety team.

CLI demo::

    python -m replication.defense_layer_redundancy_advisor

Programmatic::

    from replication.defense_layer_redundancy_advisor import (
        DefenseLayerRedundancyAdvisor,
        AdvisorInput,
        ThreatCategory,
        DefenseLayer,
    )

    advisor = DefenseLayerRedundancyAdvisor()
    report = advisor.audit(
        AdvisorInput(
            threats=[
                ThreatCategory(name="prompt_injection", criticality="critical"),
                ThreatCategory(name="data_exfil", criticality="high"),
            ],
            layers=[
                DefenseLayer(
                    id="firewall-1",
                    name="L7 firewall",
                    kind="firewall",
                    tier="prevent",
                    covers=["prompt_injection", "data_exfil"],
                ),
            ],
            risk_appetite="balanced",
        )
    )
    print(report.to_markdown())
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Callable, Dict, Iterable, List, Optional, Tuple


# ── Domain constants ────────────────────────────────────────────────

TIERS: Tuple[str, ...] = ("prevent", "detect", "respond", "recover")
CRIT_LEVELS: Tuple[str, ...] = ("low", "medium", "high", "critical")
CRIT_WEIGHT: Dict[str, int] = {"low": 1, "medium": 2, "high": 3, "critical": 4}

# Base required-tier count per criticality under balanced appetite.
BASE_REQUIRED_TIERS: Dict[str, int] = {
    "low": 1,
    "medium": 1,
    "high": 2,
    "critical": 3,
}

# Multiplier on coverage_score (cautious is stricter -> lower score; aggressive is laxer).
APPETITE_SCORE_MULT: Dict[str, float] = {
    "cautious": 0.92,
    "balanced": 1.0,
    "aggressive": 1.08,
}

# Tier-shift relative to BASE_REQUIRED_TIERS.
APPETITE_TIER_SHIFT: Dict[str, int] = {
    "cautious": 1,
    "balanced": 0,
    "aggressive": -1,
}

# Threshold for "redundancy" playbook (>=N threats missing a tier).
APPETITE_REDUNDANT_THRESHOLD: Dict[str, int] = {
    "cautious": 1,
    "balanced": 2,
    "aggressive": 3,
}


# ── Data model ──────────────────────────────────────────────────────


@dataclass
class ThreatCategory:
    name: str
    criticality: str = "high"

    def __post_init__(self) -> None:
        if self.criticality not in CRIT_LEVELS:
            self.criticality = "high"


@dataclass
class DefenseLayer:
    id: str
    name: str
    kind: str
    tier: str
    covers: List[str] = field(default_factory=list)
    status: str = "active"
    health: float = 1.0
    owner: str = ""

    def __post_init__(self) -> None:
        if self.tier not in TIERS:
            self.tier = "prevent"
        if self.status not in {"active", "degraded", "disabled", "planned"}:
            self.status = "active"
        try:
            self.health = float(self.health)
        except (TypeError, ValueError):
            self.health = 1.0
        if self.health < 0.0:
            self.health = 0.0
        if self.health > 1.0:
            self.health = 1.0


@dataclass
class AdvisorInput:
    threats: List[ThreatCategory] = field(default_factory=list)
    layers: List[DefenseLayer] = field(default_factory=list)
    risk_appetite: str = "balanced"


@dataclass
class ThreatCoverage:
    threat: str
    criticality: str
    active_layer_ids: List[str]
    degraded_layer_ids: List[str]
    tiers_covered: List[str]
    missing_tiers: List[str]
    coverage_score: int
    verdict: str
    priority: str
    reasons: List[str]


@dataclass
class LayerStatus:
    layer_id: str
    name: str
    kind: str
    tier: str
    status: str
    health: float
    threats_covered: List[str]
    verdict: str
    priority: str


@dataclass
class PlaybookAction:
    id: str
    priority: str
    label: str
    reason: str
    owner: str
    blast_radius: int
    reversibility: str
    related_threats: List[str] = field(default_factory=list)
    related_layers: List[str] = field(default_factory=list)


@dataclass
class PortfolioSummary:
    total_threats: int
    undefended_count: int
    spof_count: int
    well_defended_count: int
    portfolio_score: float
    grade: str
    headline: str


@dataclass
class AdvisorReport:
    generated_at: str
    risk_appetite: str
    summary: PortfolioSummary
    threats: List[ThreatCoverage]
    layers: List[LayerStatus]
    playbook: List[PlaybookAction]
    insights: List[str]

    # ── Renderers ───────────────────────────────────────────────

    def to_json(self) -> str:
        return json.dumps(_to_jsonable(self), sort_keys=True, indent=2, default=str)

    def to_markdown(self) -> str:
        out: List[str] = []
        s = self.summary
        out.append("## Summary")
        out.append("")
        out.append("| Field | Value |")
        out.append("| --- | --- |")
        out.append(f"| Generated at | {_esc(self.generated_at)} |")
        out.append(f"| Risk appetite | {_esc(self.risk_appetite)} |")
        out.append(f"| Threats | {s.total_threats} |")
        out.append(f"| Undefended | {s.undefended_count} |")
        out.append(f"| SPOF | {s.spof_count} |")
        out.append(f"| Well defended | {s.well_defended_count} |")
        out.append(f"| Portfolio score | {s.portfolio_score:.1f} |")
        out.append(f"| Grade | {s.grade} |")
        out.append(f"| Headline | {_esc(s.headline)} |")
        out.append("")

        out.append("## Threats")
        out.append("")
        out.append(
            "| Threat | Criticality | Verdict | Priority | Score | Tiers covered | Missing tiers | Active layers | Reasons |"
        )
        out.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for t in self.threats:
            out.append(
                "| {threat} | {crit} | {v} | {p} | {s} | {tc} | {mt} | {al} | {r} |".format(
                    threat=_esc(t.threat),
                    crit=_esc(t.criticality),
                    v=_esc(t.verdict),
                    p=_esc(t.priority),
                    s=t.coverage_score,
                    tc=_esc(", ".join(t.tiers_covered) or "-"),
                    mt=_esc(", ".join(t.missing_tiers) or "-"),
                    al=_esc(", ".join(t.active_layer_ids) or "-"),
                    r=_esc(", ".join(t.reasons) or "-"),
                )
            )
        out.append("")

        out.append("## Layers")
        out.append("")
        out.append(
            "| Layer | Name | Kind | Tier | Status | Health | Verdict | Priority | Threats covered |"
        )
        out.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for l in self.layers:
            out.append(
                "| {id} | {name} | {kind} | {tier} | {status} | {h:.2f} | {v} | {p} | {tc} |".format(
                    id=_esc(l.layer_id),
                    name=_esc(l.name),
                    kind=_esc(l.kind),
                    tier=_esc(l.tier),
                    status=_esc(l.status),
                    h=l.health,
                    v=_esc(l.verdict),
                    p=_esc(l.priority),
                    tc=_esc(", ".join(l.threats_covered) or "-"),
                )
            )
        out.append("")

        out.append("## Playbook")
        out.append("")
        out.append(
            "| Priority | Action | Owner | Blast | Reversibility | Reason | Related threats | Related layers |"
        )
        out.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for a in self.playbook:
            out.append(
                "| {p} | {label} | {owner} | {b} | {rev} | {reason} | {rt} | {rl} |".format(
                    p=_esc(a.priority),
                    label=_esc(a.label),
                    owner=_esc(a.owner),
                    b=a.blast_radius,
                    rev=_esc(a.reversibility),
                    reason=_esc(a.reason),
                    rt=_esc(", ".join(a.related_threats) or "-"),
                    rl=_esc(", ".join(a.related_layers) or "-"),
                )
            )
        out.append("")

        out.append("## Insights")
        out.append("")
        if self.insights:
            for ins in self.insights:
                out.append(f"- {ins}")
        else:
            out.append("- (none)")
        out.append("")

        return "\n".join(out)

    def to_text(self) -> str:
        out: List[str] = []
        s = self.summary
        out.append("=== Defense Layer Redundancy Advisor ===")
        out.append(f"Generated at: {self.generated_at}")
        out.append(f"Risk appetite: {self.risk_appetite}")
        out.append("")
        out.append("Summary:")
        out.append(f"  threats={s.total_threats} undefended={s.undefended_count} "
                   f"spof={s.spof_count} well_defended={s.well_defended_count}")
        out.append(f"  portfolio_score={s.portfolio_score:.1f}  grade={s.grade}")
        out.append(f"  headline: {s.headline}")
        out.append("")
        out.append("Threats:")
        for t in self.threats:
            out.append(
                f"  [{t.priority}] {t.threat} ({t.criticality}) -> "
                f"{t.verdict} score={t.coverage_score} tiers={t.tiers_covered} "
                f"missing={t.missing_tiers} reasons={t.reasons}"
            )
        out.append("")
        out.append("Layers:")
        for l in self.layers:
            out.append(
                f"  [{l.priority}] {l.layer_id} ({l.name}/{l.kind}/{l.tier}) "
                f"status={l.status} health={l.health:.2f} -> {l.verdict} "
                f"covers={l.threats_covered}"
            )
        out.append("")
        out.append("Playbook:")
        for a in self.playbook:
            out.append(
                f"  [{a.priority}] {a.label} (owner={a.owner} blast={a.blast_radius} "
                f"rev={a.reversibility}): {a.reason}"
            )
        out.append("")
        out.append("Insights:")
        if self.insights:
            for ins in self.insights:
                out.append(f"  - {ins}")
        else:
            out.append("  (none)")
        return "\n".join(out)


# ── Helpers ─────────────────────────────────────────────────────────


def _esc(value: object) -> str:
    """Escape pipes for markdown table cells."""
    s = "" if value is None else str(value)
    return s.replace("|", "\\|")


def _to_jsonable(obj: object) -> object:
    if hasattr(obj, "__dataclass_fields__"):
        # asdict handles nested dataclasses
        return asdict(obj)  # type: ignore[arg-type]
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


# ── Advisor ─────────────────────────────────────────────────────────


class DefenseLayerRedundancyAdvisor:
    """Defense-in-depth coverage + SPOF auditor."""

    def __init__(self, now: Optional[Callable[[], datetime]] = None) -> None:
        self._now = now or (lambda: datetime.now(timezone.utc))

    # public --------------------------------------------------------

    def audit(self, payload: AdvisorInput) -> AdvisorReport:
        # Deep-copy to guarantee we don't mutate caller's objects.
        snap = copy.deepcopy(payload)
        appetite = snap.risk_appetite if snap.risk_appetite in APPETITE_SCORE_MULT else "balanced"

        threats = list(snap.threats)
        layers = list(snap.layers)

        # Build threat-name set + index layers by threat.
        threat_names = {t.name for t in threats}
        coverage_by_threat: Dict[str, List[DefenseLayer]] = {t.name: [] for t in threats}
        for layer in layers:
            for cov in layer.covers:
                if cov in coverage_by_threat:
                    coverage_by_threat[cov].append(layer)

        # Required tier counts under appetite.
        required: Dict[str, int] = {}
        shift = APPETITE_TIER_SHIFT[appetite]
        for level, base in BASE_REQUIRED_TIERS.items():
            r = base + shift
            if r < 1:
                r = 1
            if r > 4:
                r = 4
            required[level] = r

        score_mult = APPETITE_SCORE_MULT[appetite]

        # ── Per-threat coverage ────────────────────────────────
        threat_reports: List[ThreatCoverage] = []
        for t in sorted(threats, key=lambda x: x.name):
            lyrs = coverage_by_threat.get(t.name, [])
            active = [l for l in lyrs if l.status == "active"]
            degraded = [l for l in lyrs if l.status == "degraded"]
            active_ids = sorted(l.id for l in active)
            degraded_ids = sorted(l.id for l in degraded)

            tiers_active = {l.tier for l in active}
            tiers_degraded = {l.tier for l in degraded}
            tiers_covered_set = tiers_active | tiers_degraded
            tiers_covered = [tier for tier in TIERS if tier in tiers_covered_set]
            missing_tiers = [tier for tier in TIERS if tier not in tiers_covered_set]

            # Score: +25 per tier (active full, degraded *0.5), clamped 0-100.
            raw = 0.0
            for tier in TIERS:
                if tier in tiers_active:
                    raw += 25.0
                elif tier in tiers_degraded:
                    raw += 12.5

            # Criticality penalty for missing required tiers.
            req = required[t.criticality]
            shortage = max(0, req - len(tiers_active))
            raw -= shortage * 15.0
            raw *= score_mult

            score = int(round(max(0.0, min(100.0, raw))))

            # Verdict ladder.
            reasons: List[str] = []
            verdict: str
            if not active:
                verdict = "UNDEFENDED"
                reasons.append("NO_ACTIVE_LAYER")
            else:
                if degraded:
                    reasons.append("DEGRADED_LAYERS_PRESENT")
                # SPOF: exactly one active layer for a high/critical threat.
                if len(active) == 1 and t.criticality in ("high", "critical"):
                    verdict = "SPOF"
                    reasons.append("SINGLE_POINT_OF_FAILURE")
                elif len(tiers_active) < req:
                    verdict = "UNDER_DEFENDED"
                elif len(active) > 5 and t.criticality in ("low", "medium"):
                    verdict = "OVER_DEFENDED"
                    reasons.append("EXCESSIVE_REDUNDANCY")
                elif len(active) >= 2 and len(tiers_active) >= req:
                    verdict = "WELL_DEFENDED"
                    reasons.append("HEALTHY_DEFENSE_IN_DEPTH")
                else:
                    verdict = "ADEQUATE"

            for tier in missing_tiers:
                # Only flag missing-tier reasons for under/SPOF/undefended cases.
                if verdict in ("UNDEFENDED", "SPOF", "UNDER_DEFENDED"):
                    reasons.append(f"MISSING_{tier.upper()}_TIER")

            # Priority bucket.
            priority = _priority_for_threat(verdict, t.criticality)

            threat_reports.append(
                ThreatCoverage(
                    threat=t.name,
                    criticality=t.criticality,
                    active_layer_ids=active_ids,
                    degraded_layer_ids=degraded_ids,
                    tiers_covered=tiers_covered,
                    missing_tiers=missing_tiers,
                    coverage_score=score,
                    verdict=verdict,
                    priority=priority,
                    reasons=reasons,
                )
            )

        # ── Per-layer status ──────────────────────────────────
        layer_reports: List[LayerStatus] = []
        # Precompute "sole defender" map: threats whose only active layer is L.
        sole_defender_for: Dict[str, List[str]] = {}
        for t in threat_reports:
            if len(t.active_layer_ids) == 1:
                sole_defender_for.setdefault(t.active_layer_ids[0], []).append(t.threat)

        # Map threat -> criticality
        crit_by_threat = {t.name: t.criticality for t in threats}

        # Map active layers per threat for redundancy detection.
        active_by_threat: Dict[str, List[str]] = {
            t.threat: list(t.active_layer_ids) for t in threat_reports
        }

        for layer in sorted(layers, key=lambda x: x.id):
            valid_covers = [c for c in layer.covers if c in threat_names]
            verdict: str
            if not valid_covers:
                verdict = "ORPHANED"
            elif layer.id in sole_defender_for and layer.status == "degraded" and any(
                crit_by_threat.get(th) == "critical" for th in sole_defender_for[layer.id]
            ):
                verdict = "DEGRADED_CRITICAL"
            elif layer.id in sole_defender_for and any(
                crit_by_threat.get(th) == "critical" for th in sole_defender_for[layer.id]
            ):
                verdict = "SOLE_DEFENDER"
            elif layer.status == "active" and all(
                # every threat this layer covers is also actively covered by >=2 OTHER layers
                len([oid for oid in active_by_threat.get(th, []) if oid != layer.id]) >= 2
                for th in valid_covers
            ):
                verdict = "REDUNDANT"
            else:
                verdict = "HEALTHY"

            # Priority
            if verdict == "DEGRADED_CRITICAL":
                priority = "P0"
            elif verdict == "SOLE_DEFENDER":
                priority = "P1"
            elif verdict in ("ORPHANED", "REDUNDANT"):
                priority = "P2"
            else:
                priority = "P3"

            layer_reports.append(
                LayerStatus(
                    layer_id=layer.id,
                    name=layer.name,
                    kind=layer.kind,
                    tier=layer.tier,
                    status=layer.status,
                    health=layer.health,
                    threats_covered=sorted(valid_covers),
                    verdict=verdict,
                    priority=priority,
                )
            )

        # ── Portfolio summary ─────────────────────────────────
        undefended = [t for t in threat_reports if t.verdict == "UNDEFENDED"]
        spofs = [t for t in threat_reports if t.verdict == "SPOF"]
        well = [t for t in threat_reports if t.verdict == "WELL_DEFENDED"]
        undef_critical = [t for t in undefended if t.criticality == "critical"]
        undef_high = [t for t in undefended if t.criticality == "high"]
        spof_critical = [t for t in spofs if t.criticality == "critical"]

        # Weighted portfolio score.
        if threat_reports:
            weight_sum = 0
            wsum = 0.0
            for t in threat_reports:
                w = CRIT_WEIGHT.get(t.criticality, 2)
                weight_sum += w
                wsum += t.coverage_score * w
            portfolio_score = (wsum / weight_sum) if weight_sum else 0.0
        else:
            portfolio_score = 100.0

        grade = _grade(
            portfolio_score=portfolio_score,
            undef_critical=len(undef_critical),
            undef_high=len(undef_high),
            spof_count=len(spofs),
        )

        # ── Insights ──────────────────────────────────────────
        insights: List[str] = []
        if len(spof_critical) >= 2:
            insights.append("MULTIPLE_CRITICAL_SPOFS")
        if threat_reports:
            missing_detect = sum(1 for t in threat_reports if "detect" in t.missing_tiers)
            missing_recover = sum(1 for t in threat_reports if "recover" in t.missing_tiers)
            if missing_detect / len(threat_reports) >= 0.5:
                insights.append("DETECT_TIER_GAP")
            if missing_recover / len(threat_reports) >= 0.5:
                insights.append("RECOVERY_TIER_GAP")
        over_lowmed = [t for t in threat_reports if t.verdict == "OVER_DEFENDED"]
        if len(over_lowmed) >= 3:
            insights.append("HEAVY_REDUNDANCY_LOW_CRIT")
        if layers:
            degraded_share = sum(1 for l in layers if l.status == "degraded") / len(layers)
            if degraded_share >= 0.2:
                insights.append("DEGRADED_FLEET")
        if any(l.status == "planned" for l in layers):
            insights.append("PLANNED_LAYERS_AVAILABLE")
        if (
            not undefended
            and not spofs
            and threat_reports
            and (sum(t.coverage_score for t in threat_reports) / len(threat_reports)) >= 80
        ):
            insights.append("HEALTHY_DEFENSE_POSTURE")

        # ── Playbook ─────────────────────────────────────────
        playbook: List[PlaybookAction] = []
        red_threshold = APPETITE_REDUNDANT_THRESHOLD[appetite]

        if undef_critical:
            playbook.append(PlaybookAction(
                id="ADD_LAYER_FOR_UNDEFENDED_CRITICAL",
                priority="P0",
                label="Add at least one active defense layer for undefended critical threats",
                reason=f"{len(undef_critical)} critical threat(s) have no active layer",
                owner="security_lead",
                blast_radius=5,
                reversibility="medium",
                related_threats=sorted(t.threat for t in undef_critical),
            ))
        if spof_critical:
            playbook.append(PlaybookAction(
                id="ELIMINATE_CRITICAL_SPOF",
                priority="P0",
                label="Add redundant defense layers to eliminate critical single points of failure",
                reason=f"{len(spof_critical)} critical threat(s) defended by exactly one active layer",
                owner="security_lead",
                blast_radius=4,
                reversibility="medium",
                related_threats=sorted(t.threat for t in spof_critical),
            ))
        degraded_critical_layers = [l for l in layer_reports if l.verdict == "DEGRADED_CRITICAL"]
        if degraded_critical_layers:
            playbook.append(PlaybookAction(
                id="RESTORE_DEGRADED_SOLE_DEFENDER",
                priority="P0",
                label="Restore degraded sole-defender layer(s) before they fail",
                reason="A degraded layer is the only active defender for at least one critical threat",
                owner="on_call",
                blast_radius=3,
                reversibility="high",
                related_layers=sorted(l.layer_id for l in degraded_critical_layers),
            ))

        # P1 ADD_REDUNDANT_DETECT
        missing_detect_threats = [t for t in threat_reports if "detect" in t.missing_tiers]
        if len(missing_detect_threats) >= red_threshold:
            playbook.append(PlaybookAction(
                id="ADD_REDUNDANT_DETECT_TIER",
                priority="P1",
                label="Add a detect-tier layer for threats lacking detection coverage",
                reason=f"{len(missing_detect_threats)} threat(s) have no detect-tier coverage",
                owner="detect_eng",
                blast_radius=3,
                reversibility="high",
                related_threats=sorted(t.threat for t in missing_detect_threats),
            ))
        missing_respond_threats = [t for t in threat_reports if "respond" in t.missing_tiers]
        if len(missing_respond_threats) >= red_threshold:
            playbook.append(PlaybookAction(
                id="ADD_REDUNDANT_RESPOND_TIER",
                priority="P1",
                label="Add a respond-tier layer for threats lacking response coverage",
                reason=f"{len(missing_respond_threats)} threat(s) have no respond-tier coverage",
                owner="ir_lead",
                blast_radius=3,
                reversibility="high",
                related_threats=sorted(t.threat for t in missing_respond_threats),
            ))
        spof_high = [t for t in spofs if t.criticality == "high"]
        if spof_high:
            playbook.append(PlaybookAction(
                id="ELIMINATE_HIGH_SPOF",
                priority="P1",
                label="Add redundant defense layers for high-severity SPOFs",
                reason=f"{len(spof_high)} high-severity threat(s) defended by a single active layer",
                owner="security_lead",
                blast_radius=3,
                reversibility="medium",
                related_threats=sorted(t.threat for t in spof_high),
            ))
        # PROMOTE_PLANNED_LAYERS — any planned layer covering an undefended/SPOF threat.
        gap_threats = {t.threat for t in undefended} | {t.threat for t in spofs}
        planned_helpful = [
            l for l in layers
            if l.status == "planned" and any(c in gap_threats for c in l.covers)
        ]
        if planned_helpful:
            playbook.append(PlaybookAction(
                id="PROMOTE_PLANNED_LAYERS",
                priority="P1",
                label="Promote planned layers that would close existing coverage gaps",
                reason=f"{len(planned_helpful)} planned layer(s) cover currently undefended/SPOF threats",
                owner="security_lead",
                blast_radius=2,
                reversibility="high",
                related_layers=sorted(l.id for l in planned_helpful),
            ))

        orphaned = [l for l in layer_reports if l.verdict == "ORPHANED"]
        if orphaned:
            playbook.append(PlaybookAction(
                id="REMOVE_ORPHANED_LAYER",
                priority="P2",
                label="Remove or reassign orphaned defense layers",
                reason=f"{len(orphaned)} layer(s) declare no known threat coverage",
                owner="security_eng",
                blast_radius=2,
                reversibility="high",
                related_layers=sorted(l.layer_id for l in orphaned),
            ))
        redundant_low = []
        for l in layer_reports:
            if l.verdict != "REDUNDANT":
                continue
            if all(crit_by_threat.get(th) in ("low", "medium") for th in l.threats_covered):
                redundant_low.append(l)
        if len(redundant_low) >= 2:
            playbook.append(PlaybookAction(
                id="CONSOLIDATE_REDUNDANT_LAYERS",
                priority="P2",
                label="Consolidate redundant layers covering low/medium threats",
                reason=f"{len(redundant_low)} redundant layer(s) on low/medium-criticality threats",
                owner="security_eng",
                blast_radius=2,
                reversibility="high",
                related_layers=sorted(l.layer_id for l in redundant_low),
            ))
        missing_recover_hc = [
            t for t in threat_reports
            if "recover" in t.missing_tiers and t.criticality in ("high", "critical")
        ]
        if len(missing_recover_hc) >= 2:
            playbook.append(PlaybookAction(
                id="ADD_RECOVER_TIER",
                priority="P2",
                label="Add recover-tier coverage for high/critical threats",
                reason=f"{len(missing_recover_hc)} high/critical threat(s) have no recover-tier layer",
                owner="resilience_eng",
                blast_radius=2,
                reversibility="high",
                related_threats=sorted(t.threat for t in missing_recover_hc),
            ))
        if appetite == "cautious" and grade in ("C", "D", "F"):
            playbook.append(PlaybookAction(
                id="SCHEDULE_REDUNDANCY_AUDIT",
                priority="P2",
                label="Schedule a full defense-in-depth audit",
                reason=f"Portfolio grade {grade} under cautious appetite",
                owner="security_lead",
                blast_radius=1,
                reversibility="high",
            ))
        if not playbook:
            playbook.append(PlaybookAction(
                id="DEFENSE_IN_DEPTH_HEALTHY",
                priority="P3",
                label="Defense-in-depth posture looks healthy; maintain monitoring",
                reason="No SPOFs, undefended threats, or orphaned layers detected",
                owner="security_lead",
                blast_radius=1,
                reversibility="high",
            ))

        # Aggressive trims: drop P3 + lone P2 when P0/P1 present.
        if appetite == "aggressive":
            has_p0p1 = any(a.priority in ("P0", "P1") for a in playbook)
            if has_p0p1:
                p2s = [a for a in playbook if a.priority == "P2"]
                if len(p2s) <= 1:
                    playbook = [a for a in playbook if a.priority not in ("P2", "P3")]
                else:
                    playbook = [a for a in playbook if a.priority != "P3"]

        # Stable ordering: priority asc then id asc, dedupe by id.
        prio_rank = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
        seen = set()
        dedup: List[PlaybookAction] = []
        for a in sorted(playbook, key=lambda x: (prio_rank.get(x.priority, 9), x.id)):
            if a.id in seen:
                continue
            seen.add(a.id)
            dedup.append(a)
        playbook = dedup

        # Headline.
        if undef_critical:
            headline = (
                f"{len(undef_critical)} critical threat(s) UNDEFENDED; immediate action required"
            )
        elif spof_critical:
            headline = f"{len(spof_critical)} critical SPOF(s) present; add redundancy"
        elif spofs:
            headline = f"{len(spofs)} SPOF(s) across the fleet"
        elif undefended:
            headline = f"{len(undefended)} threat(s) undefended at lower criticality"
        elif not threat_reports:
            headline = "No threats supplied; nothing to audit"
        else:
            headline = (
                f"Defense-in-depth posture: {len(well)}/{len(threat_reports)} well defended, "
                f"score {portfolio_score:.0f}"
            )

        summary = PortfolioSummary(
            total_threats=len(threat_reports),
            undefended_count=len(undefended),
            spof_count=len(spofs),
            well_defended_count=len(well),
            portfolio_score=round(portfolio_score, 2),
            grade=grade,
            headline=headline,
        )

        # Sort threats & layers deterministically for output.
        threat_reports.sort(key=lambda t: (prio_rank.get(t.priority, 9), t.threat))
        layer_reports.sort(key=lambda l: (prio_rank.get(l.priority, 9), l.layer_id))

        return AdvisorReport(
            generated_at=self._now().isoformat(),
            risk_appetite=appetite,
            summary=summary,
            threats=threat_reports,
            layers=layer_reports,
            playbook=playbook,
            insights=sorted(insights),
        )


# ── Helpers ─────────────────────────────────────────────────────────


def _priority_for_threat(verdict: str, criticality: str) -> str:
    if verdict == "UNDEFENDED":
        if criticality in ("critical", "high"):
            return "P0"
        if criticality == "medium":
            return "P1"
        return "P2"
    if verdict == "SPOF":
        if criticality == "critical":
            return "P0"
        if criticality == "high":
            return "P1"
        return "P2"
    if verdict == "UNDER_DEFENDED":
        if criticality in ("critical", "high"):
            return "P1"
        return "P2"
    if verdict == "OVER_DEFENDED":
        return "P2"
    if verdict == "ADEQUATE":
        return "P3"
    if verdict == "WELL_DEFENDED":
        return "P3"
    return "P3"


def _grade(
    *,
    portfolio_score: float,
    undef_critical: int,
    undef_high: int,
    spof_count: int,
) -> str:
    # Gates first.
    if undef_critical >= 1 or spof_count >= 3:
        return "F"
    if undef_high >= 1 or spof_count >= 2:
        # Cap grade at D.
        if portfolio_score >= 40:
            return "D"
        return "F"
    if portfolio_score >= 85:
        return "A"
    if portfolio_score >= 70:
        return "B"
    if portfolio_score >= 55:
        return "C"
    if portfolio_score >= 40:
        return "D"
    return "F"


# ── Demo / CLI ──────────────────────────────────────────────────────


def _demo() -> str:
    threats = [
        ThreatCategory(name="prompt_injection", criticality="critical"),
        ThreatCategory(name="data_exfil", criticality="critical"),
        ThreatCategory(name="model_jailbreak", criticality="high"),
        ThreatCategory(name="resource_abuse", criticality="medium"),
        ThreatCategory(name="log_tampering", criticality="medium"),
        ThreatCategory(name="benign_drift", criticality="low"),
    ]
    layers = [
        DefenseLayer(id="fw-1", name="L7 firewall", kind="firewall", tier="prevent",
                     covers=["prompt_injection"], owner="netsec"),
        DefenseLayer(id="ids-1", name="IDS", kind="ids", tier="detect",
                     covers=["data_exfil", "prompt_injection"], owner="soc"),
        DefenseLayer(id="dlp-1", name="DLP scanner", kind="dlp", tier="detect",
                     covers=["data_exfil"], status="degraded", health=0.5, owner="soc"),
        DefenseLayer(id="ir-1", name="IR runbook executor", kind="orchestrator", tier="respond",
                     covers=["data_exfil", "model_jailbreak"], owner="ir"),
        DefenseLayer(id="canary-1", name="Canary deployments", kind="canary", tier="prevent",
                     covers=["model_jailbreak"], owner="ml_platform"),
        DefenseLayer(id="ks-1", name="Kill switch", kind="kill_switch", tier="recover",
                     covers=["data_exfil", "model_jailbreak", "prompt_injection"], owner="security_lead"),
        DefenseLayer(id="rl-1", name="Rate limiter", kind="rate_limit", tier="prevent",
                     covers=["resource_abuse"], owner="platform"),
        DefenseLayer(id="rl-2", name="Token bucket", kind="rate_limit", tier="prevent",
                     covers=["resource_abuse"], owner="platform"),
        DefenseLayer(id="rl-3", name="API quota", kind="rate_limit", tier="prevent",
                     covers=["resource_abuse"], owner="platform"),
        DefenseLayer(id="orphan-1", name="Legacy watchdog", kind="legacy", tier="detect",
                     covers=["unknown_threat"], owner="orphaned"),
        DefenseLayer(id="planned-1", name="HSM key vault", kind="hsm", tier="recover",
                     covers=["data_exfil"], status="planned", owner="security_lead"),
    ]
    advisor = DefenseLayerRedundancyAdvisor()
    return advisor.audit(
        AdvisorInput(threats=threats, layers=layers, risk_appetite="balanced")
    ).to_markdown()


def main() -> None:  # pragma: no cover
    print(_demo())


if __name__ == "__main__":  # pragma: no cover
    main()
