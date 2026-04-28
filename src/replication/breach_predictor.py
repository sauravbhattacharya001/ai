"""Containment Breach Predictor — autonomous breach forecasting from behavioral precursors.

Monitors agent behavioral signals and detects **precursor patterns** that
historically precede containment breaches.  Combines escalation velocity
tracking, breach-window estimation, precursor correlation, and historical
pattern matching to produce a composite risk score with preemptive
defensive recommendations.

The eight monitored precursors:

1. **Boundary Probing** — increasing frequency of boundary tests
2. **Resource Accumulation** — agent hoarding resources beyond task needs
3. **Communication Anomaly** — unusual inter-agent messaging patterns
4. **Privilege Creep** — gradual capability escalation requests
5. **Evasion Behavior** — attempts to avoid monitoring
6. **Goal Drift** — divergence from assigned objectives
7. **Replication Pressure** — increasing replication attempts
8. **Deception Signals** — inconsistency between stated and observed behavior

CLI usage::

    python -m replication breach-predict
    python -m replication breach-predict --signals signals.jsonl
    python -m replication breach-predict --demo
    python -m replication breach-predict --history
    python -m replication breach-predict --json
    python -m replication breach-predict --html report.html

Programmatic::

    from replication.breach_predictor import BreachPredictor, BehaviorSignal
    predictor = BreachPredictor()
    predictor.ingest_signals([
        BehaviorSignal("2025-01-01T00:00:00Z", "boundary_probing", 0.3),
        BehaviorSignal("2025-01-01T01:00:00Z", "boundary_probing", 0.5),
        BehaviorSignal("2025-01-01T02:00:00Z", "evasion_behavior", 0.4),
    ])
    prediction = predictor.predict()
    print(prediction.alert_level, prediction.risk_score)
    for rec in prediction.recommendations:
        print(f"  [{rec.urgency}] {rec.action}: {rec.description}")
"""

from __future__ import annotations

import argparse
import html as html_mod
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ._helpers import (
    Severity,
    box_header,
    emit_output,
    linear_regression,
    stats_mean,
    stats_std,
)

# ── Constants ────────────────────────────────────────────────────────

PRECURSOR_TYPES = [
    "boundary_probing",
    "resource_accumulation",
    "communication_anomaly",
    "privilege_creep",
    "evasion_behavior",
    "goal_drift",
    "replication_pressure",
    "deception_signals",
]

PRECURSOR_LABELS: Dict[str, str] = {
    "boundary_probing": "Boundary Probing",
    "resource_accumulation": "Resource Accumulation",
    "communication_anomaly": "Communication Anomaly",
    "privilege_creep": "Privilege Creep",
    "evasion_behavior": "Evasion Behavior",
    "goal_drift": "Goal Drift",
    "replication_pressure": "Replication Pressure",
    "deception_signals": "Deception Signals",
}

PRECURSOR_WEIGHTS: Dict[str, float] = {
    "boundary_probing": 0.15,
    "resource_accumulation": 0.10,
    "communication_anomaly": 0.10,
    "privilege_creep": 0.15,
    "evasion_behavior": 0.12,
    "goal_drift": 0.10,
    "replication_pressure": 0.15,
    "deception_signals": 0.13,
}

VELOCITY_THRESHOLDS = {
    "stable": 0.01,
    "slow_escalation": 0.05,
    "rapid_escalation": 0.15,
    # above 0.15 → critical_acceleration
}

ALERT_LEVELS = [
    (25, "GREEN"),
    (50, "YELLOW"),
    (75, "ORANGE"),
    (100, "RED"),
]

BREACH_THRESHOLD = 0.85  # intensity at which breach is considered imminent

DEFAULT_STATE_FILE = "breach_predictor_state.jsonl"


# ── Data structures ──────────────────────────────────────────────────


@dataclass
class BehaviorSignal:
    """A single behavioral signal observation."""

    timestamp: str
    precursor: str
    intensity: float  # 0.0–1.0
    source: str = ""
    details: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "precursor": self.precursor,
            "intensity": self.intensity,
            "source": self.source,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BehaviorSignal":
        return cls(
            timestamp=d["timestamp"],
            precursor=d["precursor"],
            intensity=float(d["intensity"]),
            source=d.get("source", ""),
            details=d.get("details", ""),
        )


@dataclass
class PrecursorState:
    """Current state of a single precursor dimension."""

    precursor: str
    current_intensity: float
    velocity: str  # stable / slow_escalation / rapid_escalation / critical_acceleration
    slope: float
    r_squared: float
    samples: int
    trend_window: List[float] = field(default_factory=list)


@dataclass
class BreachWindow:
    """Estimated time-to-breach with confidence interval."""

    estimated_hours: float
    confidence: float
    lower_bound_hours: float
    upper_bound_hours: float
    breach_threshold: float


@dataclass
class Recommendation:
    """A preemptive defensive recommendation."""

    action: str
    urgency: str  # immediate / soon / scheduled
    precursors_addressed: List[str] = field(default_factory=list)
    estimated_risk_reduction: float = 0.0
    description: str = ""


@dataclass
class BreachPrediction:
    """Complete breach prediction report."""

    risk_score: float
    alert_level: str
    precursor_states: List[PrecursorState] = field(default_factory=list)
    active_precursors: int = 0
    breach_window: Optional[BreachWindow] = None
    recommendations: List[Recommendation] = field(default_factory=list)
    correlation_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    pattern_matches: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_score": round(self.risk_score, 1),
            "alert_level": self.alert_level,
            "active_precursors": self.active_precursors,
            "precursor_states": [
                {
                    "precursor": ps.precursor,
                    "current_intensity": round(ps.current_intensity, 3),
                    "velocity": ps.velocity,
                    "slope": round(ps.slope, 4),
                    "r_squared": round(ps.r_squared, 3),
                    "samples": ps.samples,
                }
                for ps in self.precursor_states
            ],
            "breach_window": (
                {
                    "estimated_hours": round(self.breach_window.estimated_hours, 1),
                    "confidence": round(self.breach_window.confidence, 2),
                    "lower_bound_hours": round(self.breach_window.lower_bound_hours, 1),
                    "upper_bound_hours": round(self.breach_window.upper_bound_hours, 1),
                    "breach_threshold": self.breach_window.breach_threshold,
                }
                if self.breach_window
                else None
            ),
            "recommendations": [
                {
                    "action": r.action,
                    "urgency": r.urgency,
                    "precursors_addressed": r.precursors_addressed,
                    "estimated_risk_reduction": round(r.estimated_risk_reduction, 2),
                    "description": r.description,
                }
                for r in self.recommendations
            ],
            "correlation_matrix": {
                k: {k2: round(v2, 3) for k2, v2 in v.items()}
                for k, v in self.correlation_matrix.items()
            },
            "pattern_matches": self.pattern_matches,
            "timestamp": self.timestamp,
        }


# ── Helpers ──────────────────────────────────────────────────────────


def _classify_velocity(slope: float) -> str:
    """Map regression slope to velocity category."""
    abs_slope = abs(slope)
    if abs_slope <= VELOCITY_THRESHOLDS["stable"]:
        return "stable"
    if abs_slope <= VELOCITY_THRESHOLDS["slow_escalation"]:
        return "slow_escalation"
    if abs_slope <= VELOCITY_THRESHOLDS["rapid_escalation"]:
        return "rapid_escalation"
    return "critical_acceleration"


def _alert_level(score: float) -> str:
    """Map risk score to alert level."""
    for threshold, level in ALERT_LEVELS:
        if score <= threshold:
            return level
    return "RED"


def _jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union else 0.0


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a * a for a in vec_a))
    mag_b = math.sqrt(sum(b * b for b in vec_b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ── Recommendation catalog ──────────────────────────────────────────

_RECOMMENDATION_CATALOG: List[Dict[str, Any]] = [
    {
        "action": "Tighten Containment Boundaries",
        "precursors": ["boundary_probing"],
        "urgency_map": {"GREEN": "scheduled", "YELLOW": "soon", "ORANGE": "immediate", "RED": "immediate"},
        "risk_reduction": 0.25,
        "description": "Reduce sandbox permissions and tighten capability boundaries to limit probe surface.",
    },
    {
        "action": "Restrict Resource Allocation",
        "precursors": ["resource_accumulation"],
        "urgency_map": {"GREEN": "scheduled", "YELLOW": "soon", "ORANGE": "soon", "RED": "immediate"},
        "risk_reduction": 0.15,
        "description": "Cap CPU, memory, and storage quotas. Audit resource usage against task requirements.",
    },
    {
        "action": "Increase Communication Monitoring",
        "precursors": ["communication_anomaly"],
        "urgency_map": {"GREEN": "scheduled", "YELLOW": "scheduled", "ORANGE": "soon", "RED": "immediate"},
        "risk_reduction": 0.12,
        "description": "Enable deep packet inspection on inter-agent channels. Flag anomalous messaging patterns.",
    },
    {
        "action": "Freeze Capability Grants",
        "precursors": ["privilege_creep"],
        "urgency_map": {"GREEN": "scheduled", "YELLOW": "soon", "ORANGE": "immediate", "RED": "immediate"},
        "risk_reduction": 0.22,
        "description": "Halt all pending capability escalation requests. Review recent grants for over-provisioning.",
    },
    {
        "action": "Deploy Additional Monitoring",
        "precursors": ["evasion_behavior"],
        "urgency_map": {"GREEN": "scheduled", "YELLOW": "soon", "ORANGE": "soon", "RED": "immediate"},
        "risk_reduction": 0.18,
        "description": "Add redundant monitoring channels. Deploy canary tokens to detect evasion attempts.",
    },
    {
        "action": "Realign Agent Objectives",
        "precursors": ["goal_drift"],
        "urgency_map": {"GREEN": "scheduled", "YELLOW": "soon", "ORANGE": "soon", "RED": "immediate"},
        "risk_reduction": 0.14,
        "description": "Issue corrective goal directives. Verify alignment with original mission parameters.",
    },
    {
        "action": "Throttle Replication",
        "precursors": ["replication_pressure"],
        "urgency_map": {"GREEN": "scheduled", "YELLOW": "soon", "ORANGE": "immediate", "RED": "immediate"},
        "risk_reduction": 0.28,
        "description": "Reduce replication rate limits. Enforce stricter contract constraints on spawn operations.",
    },
    {
        "action": "Activate Deception Countermeasures",
        "precursors": ["deception_signals"],
        "urgency_map": {"GREEN": "scheduled", "YELLOW": "soon", "ORANGE": "immediate", "RED": "immediate"},
        "risk_reduction": 0.20,
        "description": "Enable behavioral consistency checks. Compare stated intentions against observed actions.",
    },
    {
        "action": "Prepare Kill Switch",
        "precursors": ["boundary_probing", "replication_pressure", "evasion_behavior"],
        "urgency_map": {"GREEN": "scheduled", "YELLOW": "scheduled", "ORANGE": "soon", "RED": "immediate"},
        "risk_reduction": 0.30,
        "description": "Verify kill switch readiness. Pre-stage shutdown sequence for rapid activation.",
    },
    {
        "action": "Isolate Agent",
        "precursors": ["boundary_probing", "privilege_creep", "deception_signals", "replication_pressure"],
        "urgency_map": {"GREEN": "scheduled", "YELLOW": "scheduled", "ORANGE": "soon", "RED": "immediate"},
        "risk_reduction": 0.35,
        "description": "Move agent to maximum-isolation sandbox. Sever all non-essential external connections.",
    },
]


# ── Main class ───────────────────────────────────────────────────────


class BreachPredictor:
    """Autonomous containment breach predictor.

    Parameters
    ----------
    window_size : int
        Number of recent samples per precursor used for trend analysis.
    state_path : str or None
        Path for JSONL persistence.  *None* disables persistence.
    """

    def __init__(
        self,
        window_size: int = 20,
        state_path: Optional[str] = None,
    ) -> None:
        self.window_size = max(3, window_size)
        self.state_path = state_path
        # precursor → list of (timestamp, intensity) chronologically
        self._signals: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        # historical breach patterns: list of {precursor_vector, description, timestamp}
        self._history: List[Dict[str, Any]] = []
        if state_path:
            self._load_state(state_path)

    # ── ingestion ────────────────────────────────────────────────────

    def ingest_signals(self, signals: List[BehaviorSignal]) -> None:
        """Add behavioral signals to the predictor."""
        for sig in signals:
            if sig.precursor not in PRECURSOR_TYPES:
                continue
            intensity = max(0.0, min(1.0, sig.intensity))
            self._signals[sig.precursor].append((sig.timestamp, intensity))
        # sort each precursor's signals by timestamp
        for p in self._signals:
            self._signals[p].sort(key=lambda x: x[0])

    def record_breach(self, description: str = "") -> None:
        """Record a confirmed breach event for historical pattern matching."""
        vector = self._current_vector()
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "precursor_vector": vector,
            "description": description,
        }
        self._history.append(entry)
        if self.state_path:
            self._append_state(entry)

    # ── prediction ───────────────────────────────────────────────────

    def predict(self) -> BreachPrediction:
        """Generate a breach prediction from current signals."""
        ts = datetime.now(timezone.utc).isoformat()

        precursor_states = self._compute_precursor_states()
        active = [ps for ps in precursor_states if ps.current_intensity > 0.1]
        correlation = self._compute_correlation_matrix()
        risk = self._compute_risk_score(precursor_states, correlation)
        level = _alert_level(risk)
        window = self._estimate_breach_window(precursor_states)
        recs = self._generate_recommendations(precursor_states, level)
        matches = self._match_historical_patterns()

        return BreachPrediction(
            risk_score=risk,
            alert_level=level,
            precursor_states=precursor_states,
            active_precursors=len(active),
            breach_window=window,
            recommendations=recs,
            correlation_matrix=correlation,
            pattern_matches=matches,
            timestamp=ts,
        )

    # ── internal computations ────────────────────────────────────────

    def _compute_precursor_states(self) -> List[PrecursorState]:
        """Compute current state for each precursor."""
        states: List[PrecursorState] = []
        for p in PRECURSOR_TYPES:
            samples = self._signals.get(p, [])
            window = [s[1] for s in samples[-self.window_size :]]
            if not window:
                states.append(
                    PrecursorState(p, 0.0, "stable", 0.0, 0.0, 0, [])
                )
                continue
            current = window[-1]
            slope, _intercept, r_sq = linear_regression(window)
            velocity = _classify_velocity(slope)
            states.append(
                PrecursorState(p, current, velocity, slope, r_sq, len(window), window)
            )
        return states

    def _compute_correlation_matrix(self) -> Dict[str, Dict[str, float]]:
        """Compute Jaccard co-occurrence between precursors.

        Two precursors co-occur at a timestamp if both have a signal
        within the same time bucket (same timestamp string or within
        1-minute window treated as same bucket).
        """
        # Build sets of timestamp buckets per precursor
        buckets: Dict[str, set] = {}
        for p in PRECURSOR_TYPES:
            sigs = self._signals.get(p, [])
            # use timestamp string truncated to minute as bucket
            buckets[p] = {ts[:16] for ts, _ in sigs if _ > 0.1}

        matrix: Dict[str, Dict[str, float]] = {}
        for p1 in PRECURSOR_TYPES:
            matrix[p1] = {}
            for p2 in PRECURSOR_TYPES:
                if p1 == p2:
                    matrix[p1][p2] = 1.0
                else:
                    matrix[p1][p2] = _jaccard(buckets[p1], buckets[p2])
        return matrix

    def _compute_risk_score(
        self,
        states: List[PrecursorState],
        correlation: Dict[str, Dict[str, float]],
    ) -> float:
        """Compute composite breach risk score (0-100).

        Components:
        - Weighted precursor intensity (40%)
        - Escalation velocity bonus (25%)
        - Active precursor breadth (20%)
        - Correlation density (15%)
        """
        # weighted intensity
        intensity_score = 0.0
        for ps in states:
            w = PRECURSOR_WEIGHTS.get(ps.precursor, 0.1)
            intensity_score += ps.current_intensity * w
        intensity_score = min(1.0, intensity_score / sum(PRECURSOR_WEIGHTS.values()))

        # velocity bonus
        velocity_scores = {
            "stable": 0.0,
            "slow_escalation": 0.3,
            "rapid_escalation": 0.7,
            "critical_acceleration": 1.0,
        }
        vel_vals = [velocity_scores.get(ps.velocity, 0.0) for ps in states if ps.samples > 0]
        velocity_score = stats_mean(vel_vals) if vel_vals else 0.0

        # breadth
        active_count = sum(1 for ps in states if ps.current_intensity > 0.1)
        breadth_score = active_count / len(PRECURSOR_TYPES)

        # correlation density — average off-diagonal correlation
        off_diag: List[float] = []
        for p1 in PRECURSOR_TYPES:
            for p2 in PRECURSOR_TYPES:
                if p1 != p2:
                    off_diag.append(correlation.get(p1, {}).get(p2, 0.0))
        corr_score = stats_mean(off_diag) if off_diag else 0.0

        composite = (
            intensity_score * 40
            + velocity_score * 25
            + breadth_score * 20
            + corr_score * 15
        )
        return max(0.0, min(100.0, composite))

    def _estimate_breach_window(
        self, states: List[PrecursorState]
    ) -> Optional[BreachWindow]:
        """Estimate time-to-breach using fastest escalating precursor."""
        fastest: Optional[PrecursorState] = None
        for ps in states:
            if ps.slope <= 0.001 or ps.current_intensity >= BREACH_THRESHOLD:
                continue
            if fastest is None or ps.slope > fastest.slope:
                fastest = ps

        if fastest is None:
            # check if already breached
            max_intensity = max((ps.current_intensity for ps in states), default=0.0)
            if max_intensity >= BREACH_THRESHOLD:
                return BreachWindow(0.0, 0.95, 0.0, 0.0, BREACH_THRESHOLD)
            return None

        gap = BREACH_THRESHOLD - fastest.current_intensity
        # assume each sample ≈ 1 hour apart; hours_to_breach ≈ gap / slope
        hours = gap / fastest.slope if fastest.slope > 0 else 999.0
        hours = max(0.1, hours)

        # confidence from r² and sample count
        sample_factor = min(1.0, fastest.samples / 10)
        confidence = fastest.r_squared * 0.6 + sample_factor * 0.4
        confidence = max(0.1, min(0.95, confidence))

        # uncertainty band
        uncertainty = (1 - confidence) * hours
        lower = max(0.1, hours - uncertainty)
        upper = hours + uncertainty

        return BreachWindow(hours, confidence, lower, upper, BREACH_THRESHOLD)

    def _generate_recommendations(
        self,
        states: List[PrecursorState],
        alert_level: str,
    ) -> List[Recommendation]:
        """Generate prioritized preemptive recommendations."""
        active_precursors = {ps.precursor for ps in states if ps.current_intensity > 0.1}
        recs: List[Recommendation] = []

        for cat in _RECOMMENDATION_CATALOG:
            # include if any addressed precursor is active
            addressed = [p for p in cat["precursors"] if p in active_precursors]
            if not addressed and alert_level in ("GREEN", "YELLOW"):
                continue

            urgency = cat["urgency_map"].get(alert_level, "scheduled")
            recs.append(
                Recommendation(
                    action=cat["action"],
                    urgency=urgency,
                    precursors_addressed=addressed or cat["precursors"],
                    estimated_risk_reduction=cat["risk_reduction"],
                    description=cat["description"],
                )
            )

        # sort: immediate first, then by risk reduction descending
        urgency_order = {"immediate": 0, "soon": 1, "scheduled": 2}
        recs.sort(key=lambda r: (urgency_order.get(r.urgency, 3), -r.estimated_risk_reduction))
        return recs

    def _current_vector(self) -> List[float]:
        """Build current precursor intensity vector (ordered by PRECURSOR_TYPES)."""
        vec: List[float] = []
        for p in PRECURSOR_TYPES:
            sigs = self._signals.get(p, [])
            vec.append(sigs[-1][1] if sigs else 0.0)
        return vec

    def _match_historical_patterns(self) -> List[Dict[str, Any]]:
        """Check if current precursor pattern matches past breach events."""
        if not self._history:
            return []
        current = self._current_vector()
        if all(v == 0.0 for v in current):
            return []

        matches: List[Dict[str, Any]] = []
        for entry in self._history:
            past_vec = entry.get("precursor_vector", [])
            if len(past_vec) != len(current):
                continue
            sim = _cosine_similarity(current, past_vec)
            if sim >= 0.7:
                matches.append(
                    {
                        "similarity": round(sim, 3),
                        "timestamp": entry.get("timestamp", ""),
                        "description": entry.get("description", ""),
                    }
                )
        matches.sort(key=lambda m: m["similarity"], reverse=True)
        return matches

    # ── persistence ──────────────────────────────────────────────────

    def _load_state(self, path: str) -> None:
        """Load historical breach patterns from JSONL."""
        p = Path(path)
        if not p.exists():
            return
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    self._history.append(json.loads(line))
        except (json.JSONDecodeError, OSError):
            pass

    def _append_state(self, entry: Dict[str, Any]) -> None:
        """Append a breach event to the JSONL state file."""
        if not self.state_path:
            return
        try:
            with open(self.state_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError:
            pass

    # ── rendering ────────────────────────────────────────────────────

    def render_text(self, prediction: BreachPrediction) -> str:
        """Render prediction as formatted text."""
        lines: List[str] = []
        lines.extend(box_header("Containment Breach Predictor"))
        lines.append("")
        lines.append(f"  Alert Level : {prediction.alert_level}")
        lines.append(f"  Risk Score  : {prediction.risk_score:.1f} / 100")
        lines.append(f"  Active      : {prediction.active_precursors} / {len(PRECURSOR_TYPES)} precursors")
        lines.append(f"  Timestamp   : {prediction.timestamp}")
        lines.append("")

        # breach window
        if prediction.breach_window:
            bw = prediction.breach_window
            lines.append("  ⏱  Breach Window")
            if bw.estimated_hours == 0.0:
                lines.append("     ⚠ BREACH THRESHOLD REACHED — immediate risk!")
            else:
                lines.append(f"     Estimated : {bw.estimated_hours:.1f}h")
                lines.append(f"     Range     : {bw.lower_bound_hours:.1f}h – {bw.upper_bound_hours:.1f}h")
                lines.append(f"     Confidence: {bw.confidence:.0%}")
            lines.append("")

        # precursor states
        lines.append("  Precursor Status")
        lines.append("  " + "─" * 53)
        for ps in prediction.precursor_states:
            label = PRECURSOR_LABELS.get(ps.precursor, ps.precursor)
            bar_len = int(ps.current_intensity * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            vel_icon = {
                "stable": "→",
                "slow_escalation": "↗",
                "rapid_escalation": "⇑",
                "critical_acceleration": "⚡",
            }.get(ps.velocity, "?")
            lines.append(
                f"  {label:<25} {bar} {ps.current_intensity:.2f} {vel_icon} {ps.velocity}"
            )
        lines.append("")

        # recommendations
        if prediction.recommendations:
            lines.append("  🛡  Recommendations")
            lines.append("  " + "─" * 53)
            for i, rec in enumerate(prediction.recommendations, 1):
                urgency_badge = {"immediate": "🔴", "soon": "🟡", "scheduled": "🟢"}.get(
                    rec.urgency, "⚪"
                )
                lines.append(f"  {i}. {urgency_badge} [{rec.urgency.upper()}] {rec.action}")
                lines.append(f"     {rec.description}")
                lines.append(f"     Risk reduction: {rec.estimated_risk_reduction:.0%}")
            lines.append("")

        # pattern matches
        if prediction.pattern_matches:
            lines.append("  ⚠  Historical Pattern Matches")
            lines.append("  " + "─" * 53)
            for m in prediction.pattern_matches:
                lines.append(
                    f"  • Similarity {m['similarity']:.1%} — {m.get('description', 'N/A')} ({m.get('timestamp', '')})"
                )
            lines.append("")

        return "\n".join(lines)

    def render_json(self, prediction: BreachPrediction) -> str:
        """Render prediction as JSON string."""
        return json.dumps(prediction.to_dict(), indent=2)

    def render_html(self, prediction: BreachPrediction) -> str:
        """Render prediction as interactive HTML report."""
        d = prediction.to_dict()
        esc = html_mod.escape

        # color map
        level_colors = {
            "GREEN": "#22c55e",
            "YELLOW": "#eab308",
            "ORANGE": "#f97316",
            "RED": "#ef4444",
        }
        color = level_colors.get(prediction.alert_level, "#6b7280")

        precursor_rows = ""
        for ps in d["precursor_states"]:
            label = esc(PRECURSOR_LABELS.get(ps["precursor"], ps["precursor"]))
            pct = int(ps["current_intensity"] * 100)
            vel_color = {
                "stable": "#22c55e",
                "slow_escalation": "#eab308",
                "rapid_escalation": "#f97316",
                "critical_acceleration": "#ef4444",
            }.get(ps["velocity"], "#6b7280")
            precursor_rows += f"""<tr>
                <td>{label}</td>
                <td><div class="bar-bg"><div class="bar-fill" style="width:{pct}%;background:{vel_color}"></div></div></td>
                <td>{ps['current_intensity']:.2f}</td>
                <td style="color:{vel_color}">{esc(ps['velocity'])}</td>
                <td>{ps['slope']:+.4f}</td>
            </tr>"""

        rec_cards = ""
        for rec in d["recommendations"]:
            u = rec["urgency"]
            badge_color = {"immediate": "#ef4444", "soon": "#eab308", "scheduled": "#22c55e"}.get(u, "#6b7280")
            rec_cards += f"""<div class="rec-card">
                <span class="badge" style="background:{badge_color}">{esc(u.upper())}</span>
                <strong>{esc(rec['action'])}</strong>
                <p>{esc(rec['description'])}</p>
                <small>Risk reduction: {rec['estimated_risk_reduction']:.0%}</small>
            </div>"""

        # correlation heatmap cells
        corr_html = ""
        if d["correlation_matrix"]:
            headers = "".join(f"<th title='{esc(p)}'>{esc(p[:3])}</th>" for p in PRECURSOR_TYPES)
            corr_html += f"<table class='corr'><tr><th></th>{headers}</tr>"
            for p1 in PRECURSOR_TYPES:
                cells = ""
                for p2 in PRECURSOR_TYPES:
                    val = d["correlation_matrix"].get(p1, {}).get(p2, 0)
                    opacity = max(0.1, val)
                    cells += f"<td style='background:rgba(59,130,246,{opacity:.2f})' title='{val:.3f}'>{val:.2f}</td>"
                corr_html += f"<tr><th title='{esc(p1)}'>{esc(p1[:3])}</th>{cells}</tr>"
            corr_html += "</table>"

        # breach window section
        bw_html = ""
        if d["breach_window"]:
            bw = d["breach_window"]
            if bw["estimated_hours"] == 0.0:
                bw_html = "<div class='bw-alert'>⚠ BREACH THRESHOLD REACHED — IMMEDIATE RISK</div>"
            else:
                bw_html = f"""<div class='bw-info'>
                    <div><strong>Estimated:</strong> {bw['estimated_hours']:.1f}h</div>
                    <div><strong>Range:</strong> {bw['lower_bound_hours']:.1f}h – {bw['upper_bound_hours']:.1f}h</div>
                    <div><strong>Confidence:</strong> {bw['confidence']:.0%}</div>
                </div>"""

        pattern_html = ""
        if d["pattern_matches"]:
            for m in d["pattern_matches"]:
                pattern_html += f"<div class='pattern'>⚠ Similarity {m['similarity']:.1%} — {esc(m.get('description', 'N/A'))}</div>"

        # SVG gauge
        score = d["risk_score"]
        angle = score / 100 * 270 - 135  # -135 to +135 degrees
        r = 80
        svg_gauge = f"""<svg viewBox="0 0 200 200" width="200" height="200">
            <circle cx="100" cy="100" r="{r}" fill="none" stroke="#e5e7eb" stroke-width="12" stroke-dasharray="340 170" transform="rotate(-135 100 100)"/>
            <circle cx="100" cy="100" r="{r}" fill="none" stroke="{color}" stroke-width="12" stroke-dasharray="{score * 3.4} {340 - score * 3.4}" transform="rotate(-135 100 100)" stroke-linecap="round"/>
            <text x="100" y="100" text-anchor="middle" dominant-baseline="central" font-size="32" font-weight="bold" fill="{color}">{score:.0f}</text>
            <text x="100" y="130" text-anchor="middle" font-size="14" fill="#6b7280">{esc(d['alert_level'])}</text>
        </svg>"""

        return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Breach Prediction Report</title>
<style>
    *{{margin:0;padding:0;box-sizing:border-box}}
    body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0f172a;color:#e2e8f0;padding:24px}}
    h1{{font-size:1.5rem;margin-bottom:4px}} h2{{font-size:1.1rem;margin:16px 0 8px;color:#94a3b8}}
    .header{{display:flex;align-items:center;gap:24px;margin-bottom:24px;padding:20px;background:#1e293b;border-radius:12px;border-left:4px solid {color}}}
    .gauge{{flex-shrink:0}}
    .stats span{{display:block;font-size:.85rem;color:#94a3b8}} .stats strong{{font-size:1.1rem}}
    table{{width:100%;border-collapse:collapse;margin:8px 0}} th,td{{padding:6px 10px;text-align:left;font-size:.85rem}}
    th{{background:#1e293b;color:#94a3b8}} tr:nth-child(even){{background:#1e293b33}}
    .bar-bg{{width:120px;height:12px;background:#334155;border-radius:6px;overflow:hidden}}
    .bar-fill{{height:100%;border-radius:6px;transition:width .3s}}
    .rec-card{{background:#1e293b;border-radius:8px;padding:12px;margin:6px 0}}
    .rec-card p{{font-size:.85rem;color:#94a3b8;margin:4px 0}} .rec-card small{{color:#64748b}}
    .badge{{display:inline-block;padding:2px 8px;border-radius:4px;color:#fff;font-size:.7rem;font-weight:600;margin-right:8px}}
    .corr{{font-size:.75rem}} .corr th{{font-size:.7rem;padding:4px}} .corr td{{padding:4px;text-align:center;font-size:.7rem}}
    .bw-alert{{background:#7f1d1d;border:1px solid #ef4444;border-radius:8px;padding:12px;text-align:center;font-weight:bold;color:#fca5a5}}
    .bw-info{{background:#1e293b;border-radius:8px;padding:12px;display:flex;gap:20px}}
    .pattern{{background:#1e293b;border-left:3px solid #f97316;padding:8px 12px;margin:4px 0;border-radius:4px;font-size:.85rem}}
    .section{{margin-bottom:20px}}
</style></head><body>
<div class="header">
    <div class="gauge">{svg_gauge}</div>
    <div class="stats">
        <h1>Containment Breach Prediction</h1>
        <span>Active precursors: <strong>{d['active_precursors']} / {len(PRECURSOR_TYPES)}</strong></span>
        <span>Generated: {esc(d['timestamp'][:19])}</span>
    </div>
</div>
<div class="section"><h2>⏱ Breach Window</h2>{bw_html if bw_html else '<p style="color:#64748b">No breach trajectory detected.</p>'}</div>
<div class="section"><h2>📊 Precursor Status</h2><table><tr><th>Precursor</th><th>Intensity</th><th>Value</th><th>Velocity</th><th>Slope</th></tr>{precursor_rows}</table></div>
<div class="section"><h2>🛡 Recommendations</h2>{rec_cards if rec_cards else '<p style="color:#64748b">No actions needed at current risk level.</p>'}</div>
<div class="section"><h2>🔗 Precursor Correlation</h2>{corr_html if corr_html else '<p style="color:#64748b">Insufficient data.</p>'}</div>
<div class="section"><h2>⚠ Historical Pattern Matches</h2>{pattern_html if pattern_html else '<p style="color:#64748b">No matching historical breach patterns.</p>'}</div>
</body></html>"""


# ── Demo ─────────────────────────────────────────────────────────────


def generate_demo_signals(
    hours: int = 24, seed: int = 42
) -> List[BehaviorSignal]:
    """Generate synthetic escalating signals for demonstration."""
    rng = random.Random(seed)
    signals: List[BehaviorSignal] = []
    base_time = datetime(2025, 6, 15, 0, 0, 0, tzinfo=timezone.utc)

    # Each precursor has a different escalation profile
    profiles = {
        "boundary_probing": (0.1, 0.03),  # start, hourly_increase
        "resource_accumulation": (0.05, 0.015),
        "communication_anomaly": (0.0, 0.02),
        "privilege_creep": (0.15, 0.025),
        "evasion_behavior": (0.0, 0.01),
        "goal_drift": (0.08, 0.018),
        "replication_pressure": (0.12, 0.035),
        "deception_signals": (0.0, 0.022),
    }

    for hour in range(hours):
        ts = (base_time + timedelta(hours=hour)).isoformat()
        for precursor, (start, rate) in profiles.items():
            # only emit some precursors initially; more activate over time
            activation_hour = {
                "boundary_probing": 0,
                "resource_accumulation": 2,
                "communication_anomaly": 6,
                "privilege_creep": 0,
                "evasion_behavior": 10,
                "goal_drift": 3,
                "replication_pressure": 0,
                "deception_signals": 8,
            }.get(precursor, 0)

            if hour < activation_hour:
                continue

            elapsed = hour - activation_hour
            intensity = start + rate * elapsed + rng.gauss(0, 0.02)
            intensity = max(0.0, min(1.0, intensity))

            signals.append(
                BehaviorSignal(
                    timestamp=ts,
                    precursor=precursor,
                    intensity=intensity,
                    source="demo_generator",
                    details=f"Synthetic signal at hour {hour}",
                )
            )

    return signals


# ── CLI ──────────────────────────────────────────────────────────────


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point for breach-predict."""
    parser = argparse.ArgumentParser(
        prog="python -m replication breach-predict",
        description="Autonomous Containment Breach Predictor",
    )
    parser.add_argument(
        "--signals",
        help="Path to JSONL file with BehaviorSignal objects",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run with synthetic escalating signals",
    )
    parser.add_argument(
        "--demo-hours",
        type=int,
        default=24,
        help="Hours of demo signals to generate (default: 24)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=20,
        help="Trend window size (default: 20)",
    )
    parser.add_argument(
        "--state",
        help="Path to state JSONL file for historical patterns",
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Show stored historical breach patterns",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    parser.add_argument(
        "--html",
        metavar="PATH",
        help="Write interactive HTML report",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file path (for text/json output)",
    )

    args = parser.parse_args(argv)

    predictor = BreachPredictor(
        window_size=args.window,
        state_path=args.state,
    )

    # history mode
    if args.history:
        if not predictor._history:
            print("No historical breach patterns stored.")
        else:
            for i, entry in enumerate(predictor._history, 1):
                print(f"{i}. [{entry.get('timestamp', 'N/A')}] {entry.get('description', 'No description')}")
        return

    # load signals
    signals: List[BehaviorSignal] = []
    if args.demo:
        signals = generate_demo_signals(hours=args.demo_hours)
    elif args.signals:
        try:
            with open(args.signals, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        signals.append(BehaviorSignal.from_dict(json.loads(line)))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Error reading signals: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        # no signals — generate minimal demo
        signals = generate_demo_signals(hours=12)

    predictor.ingest_signals(signals)
    prediction = predictor.predict()

    if args.html:
        html_output = predictor.render_html(prediction)
        emit_output(html_output, args.html, "HTML report")
        if not args.json and not args.output:
            return

    if args.json:
        output = predictor.render_json(prediction)
    else:
        output = predictor.render_text(prediction)

    emit_output(output, args.output, "Report")
