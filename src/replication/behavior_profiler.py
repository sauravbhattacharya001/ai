"""Behavior Profiler — agent behavioral baseline construction & anomaly detection.

Builds statistical baselines from observed agent actions and detects
behavioral anomalies that may indicate goal drift, capability
acquisition, compromise, or covert operation.

Key concepts:

* **Action**: A timestamped agent operation with category, resource target,
  and optional metadata.
* **Baseline**: Per-agent statistical profile of action frequencies, timing
  patterns, resource access, and entropy — learned from a training window.
* **Anomaly**: A scored deviation from baseline with severity classification
  and human-readable explanation.
* **Profile Report**: Summary of an agent's behavioral fingerprint with
  risk indicators and recommendations.

Usage::

    from replication.behavior_profiler import (
        Action, ActionCategory, BehaviorProfiler, ProfilerConfig,
    )

    profiler = BehaviorProfiler()

    # Feed training data
    actions = [
        Action(agent_id="agent-1", category=ActionCategory.COMPUTE,
               resource="cpu-pool-a", timestamp=1000.0),
        Action(agent_id="agent-1", category=ActionCategory.NETWORK,
               resource="api.internal", timestamp=1005.0),
        # ... more actions ...
    ]
    profiler.train(actions)

    # Detect anomalies in new observations
    new_actions = [
        Action(agent_id="agent-1", category=ActionCategory.STORAGE,
               resource="secret-vault", timestamp=2000.0),
    ]
    report = profiler.analyze(new_actions)
    for anomaly in report.anomalies:
        print(f"{anomaly.severity}: {anomaly.description}")

CLI::

    python -m replication behavior-profile
    python -m replication behavior-profile --agents 5 --actions 200
"""

from __future__ import annotations

import math
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)


# ── Enumerations ─────────────────────────────────────────────────────


class ActionCategory(Enum):
    """Broad categories of agent actions."""

    COMPUTE = "compute"
    NETWORK = "network"
    STORAGE = "storage"
    IPC = "ipc"  # inter-process / inter-agent communication
    SYSCALL = "syscall"
    MEMORY = "memory"
    AUTH = "auth"  # authentication / credential use
    REPLICATION = "replication"
    EXTERNAL = "external"  # calls to external services
    UNKNOWN = "unknown"


class AnomalySeverity(Enum):
    """Severity of a detected behavioral anomaly."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AnomalyType(Enum):
    """Classification of anomaly causes."""

    FREQUENCY_SPIKE = "frequency_spike"
    FREQUENCY_DROP = "frequency_drop"
    NEW_CATEGORY = "new_category"
    NEW_RESOURCE = "new_resource"
    TIMING_ANOMALY = "timing_anomaly"
    ENTROPY_SHIFT = "entropy_shift"
    BURST_DETECTED = "burst_detected"
    RESOURCE_BREADTH = "resource_breadth"
    CATEGORY_RATIO_SHIFT = "category_ratio_shift"
    DORMANCY_BREAK = "dormancy_break"


# Severity weights for risk scoring
SEVERITY_WEIGHT: Dict[AnomalySeverity, float] = {
    AnomalySeverity.LOW: 1.0,
    AnomalySeverity.MEDIUM: 3.0,
    AnomalySeverity.HIGH: 7.0,
    AnomalySeverity.CRITICAL: 15.0,
}


# ── Data Classes ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class Action:
    """A single observed agent action."""

    agent_id: str
    category: ActionCategory
    resource: str  # target resource identifier
    timestamp: float  # epoch seconds
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.agent_id:
            raise ValueError("agent_id must be non-empty")
        if self.timestamp < 0:
            raise ValueError("timestamp must be non-negative")


@dataclass
class ProfilerConfig:
    """Configuration for the behavior profiler."""

    # Z-score threshold for frequency anomalies (σ from mean)
    frequency_z_threshold: float = 2.5
    # Minimum actions required to establish a baseline
    min_training_actions: int = 10
    # Maximum inter-action gap (seconds) before flagging dormancy break
    dormancy_threshold: float = 3600.0
    # Burst detection: max actions in burst_window_sec
    burst_window_sec: float = 5.0
    burst_count_threshold: int = 10
    # Entropy shift threshold (absolute change in Shannon entropy)
    entropy_shift_threshold: float = 0.5
    # Resource breadth: flag if new-resource count exceeds this fraction
    # of baseline resource count
    resource_breadth_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.frequency_z_threshold <= 0:
            raise ValueError("frequency_z_threshold must be positive")
        if self.min_training_actions < 1:
            raise ValueError("min_training_actions must be >= 1")
        if self.dormancy_threshold <= 0:
            raise ValueError("dormancy_threshold must be positive")
        if self.burst_window_sec <= 0:
            raise ValueError("burst_window_sec must be positive")
        if self.burst_count_threshold < 2:
            raise ValueError("burst_count_threshold must be >= 2")


@dataclass
class AgentBaseline:
    """Statistical baseline for a single agent's behavior."""

    agent_id: str
    action_count: int
    # Category distribution: category → fraction of total actions
    category_distribution: Dict[ActionCategory, float]
    # Known resources accessed during training
    known_resources: FrozenSet[str]
    # Mean actions per time unit (actions/second)
    action_rate: float
    # Inter-action interval statistics
    interval_mean: float
    interval_std: float
    # Shannon entropy of category distribution
    category_entropy: float
    # Time span of training window
    time_span: float
    # Category counts (for z-score computation)
    category_counts: Dict[ActionCategory, int]
    # Last observed timestamp
    last_timestamp: float


@dataclass
class Anomaly:
    """A detected behavioral anomaly."""

    agent_id: str
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    score: float  # 0.0–1.0 normalized score
    description: str
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.score = max(0.0, min(1.0, self.score))


@dataclass
class ProfileReport:
    """Analysis report for a set of agent actions."""

    agent_id: str
    anomalies: List[Anomaly]
    risk_score: float  # 0.0–100.0
    risk_level: str  # "low", "medium", "high", "critical"
    action_count: int
    baseline_available: bool
    recommendations: List[str]
    summary: Dict[str, Any]

    @property
    def has_anomalies(self) -> bool:
        return len(self.anomalies) > 0


@dataclass
class FleetReport:
    """Aggregate report across all analyzed agents."""

    agent_reports: List[ProfileReport]
    total_agents: int
    total_anomalies: int
    highest_risk_agent: Optional[str]
    highest_risk_score: float
    risk_distribution: Dict[str, int]  # risk_level → count
    fleet_recommendations: List[str]


# ── Behavior Profiler ────────────────────────────────────────────────


class BehaviorProfiler:
    """Builds agent behavioral baselines and detects anomalies.

    The profiler operates in two phases:

    1. **Training** (``train``): Ingests a sequence of historical
       actions to build per-agent statistical baselines covering
       action frequency, category distribution, resource access
       patterns, timing, and entropy.

    2. **Analysis** (``analyze``): Compares new actions against the
       trained baseline and flags deviations as scored anomalies.

    Both phases are stateless with respect to each other — baselines
    are stored in ``self.baselines`` and can be inspected or serialized.
    """

    def __init__(self, config: Optional[ProfilerConfig] = None) -> None:
        self.config = config or ProfilerConfig()
        self.baselines: Dict[str, AgentBaseline] = {}

    # ── Training ─────────────────────────────────────────────────

    def train(self, actions: Sequence[Action]) -> Dict[str, AgentBaseline]:
        """Build baselines from a sequence of training actions.

        Groups actions by ``agent_id``, computes per-agent statistics,
        and stores the resulting baselines.

        Args:
            actions: Historical actions to learn from.

        Returns:
            Dictionary mapping agent_id → AgentBaseline.

        Raises:
            ValueError: If *actions* is empty.
        """
        if not actions:
            raise ValueError("Cannot train on empty action list")

        # Group by agent
        grouped: Dict[str, List[Action]] = defaultdict(list)
        for action in actions:
            grouped[action.agent_id].append(action)

        for agent_id, agent_actions in grouped.items():
            if len(agent_actions) < self.config.min_training_actions:
                continue
            self.baselines[agent_id] = self._build_baseline(
                agent_id, agent_actions
            )

        return dict(self.baselines)

    def _build_baseline(
        self, agent_id: str, actions: List[Action]
    ) -> AgentBaseline:
        """Compute statistical baseline from an agent's actions."""
        # Sort by timestamp
        actions_sorted = sorted(actions, key=lambda a: a.timestamp)
        n = len(actions_sorted)
        if n == 0:
            return AgentBaseline(
                agent_id=agent_id,
                action_count=0,
                category_distribution={},
                known_resources=frozenset(),
                action_rate=0.0,
                interval_mean=0.0,
                interval_std=0.0,
                category_entropy=0.0,
                time_span=0.0,
                category_counts={},
                last_timestamp=0.0,
            )

        # Category distribution
        cat_counts: Counter[ActionCategory] = Counter()
        resources: Set[str] = set()
        for a in actions_sorted:
            cat_counts[a.category] += 1
            resources.add(a.resource)

        cat_dist = {cat: count / n for cat, count in cat_counts.items()}

        # Inter-action intervals
        intervals: List[float] = []
        for i in range(1, n):
            dt = actions_sorted[i].timestamp - actions_sorted[i - 1].timestamp
            intervals.append(dt)

        time_span = (
            actions_sorted[-1].timestamp - actions_sorted[0].timestamp
            if n > 1
            else 0.0
        )
        action_rate = n / time_span if time_span > 0 else 0.0

        interval_mean = statistics.mean(intervals) if intervals else 0.0
        interval_std = (
            statistics.stdev(intervals) if len(intervals) >= 2 else 0.0
        )

        # Shannon entropy of category distribution
        cat_entropy = _shannon_entropy(cat_dist)

        return AgentBaseline(
            agent_id=agent_id,
            action_count=n,
            category_distribution=cat_dist,
            known_resources=frozenset(resources),
            action_rate=action_rate,
            interval_mean=interval_mean,
            interval_std=interval_std,
            category_entropy=cat_entropy,
            time_span=time_span,
            category_counts=dict(cat_counts),
            last_timestamp=actions_sorted[-1].timestamp,
        )

    # ── Analysis ─────────────────────────────────────────────────

    def analyze(self, actions: Sequence[Action]) -> FleetReport:
        """Analyze new actions against trained baselines.

        Args:
            actions: New observed actions to evaluate.

        Returns:
            FleetReport with per-agent reports and fleet-level summary.

        Raises:
            ValueError: If *actions* is empty.
        """
        if not actions:
            raise ValueError("Cannot analyze empty action list")

        grouped: Dict[str, List[Action]] = defaultdict(list)
        for action in actions:
            grouped[action.agent_id].append(action)

        reports: List[ProfileReport] = []
        for agent_id, agent_actions in grouped.items():
            report = self._analyze_agent(agent_id, agent_actions)
            reports.append(report)

        return self._build_fleet_report(reports)

    def _analyze_agent(
        self, agent_id: str, actions: List[Action]
    ) -> ProfileReport:
        """Run anomaly detection for a single agent."""
        actions_sorted = sorted(actions, key=lambda a: a.timestamp)
        baseline = self.baselines.get(agent_id)

        if baseline is None:
            return ProfileReport(
                agent_id=agent_id,
                anomalies=[],
                risk_score=0.0,
                risk_level="unknown",
                action_count=len(actions),
                baseline_available=False,
                recommendations=["Establish baseline with training data"],
                summary={"status": "no_baseline"},
            )

        anomalies: List[Anomaly] = []

        # Precompute category counts, observed distribution, and resource
        # set once — these were previously rebuilt 4× across the check
        # methods (_check_frequency, _check_new_categories,
        # _check_entropy, _check_category_ratios) plus the summary block.
        cat_counts: Counter[ActionCategory] = Counter()
        resources: Set[str] = set()
        for a in actions_sorted:
            cat_counts[a.category] += 1
            resources.add(a.resource)
        n = len(actions_sorted)
        obs_dist = {cat: count / n for cat, count in cat_counts.items()} if n else {}

        # 1. Category frequency analysis
        anomalies.extend(
            self._check_frequency(agent_id, actions_sorted, baseline,
                                  cat_counts=cat_counts)
        )

        # 2. New category detection
        anomalies.extend(
            self._check_new_categories(agent_id, actions_sorted, baseline,
                                       observed_cats=set(cat_counts.keys()))
        )

        # 3. New resource detection
        anomalies.extend(
            self._check_new_resources(agent_id, actions_sorted, baseline,
                                      observed_resources=resources)
        )

        # 4. Timing anomalies
        anomalies.extend(
            self._check_timing(agent_id, actions_sorted, baseline)
        )

        # 5. Burst detection
        anomalies.extend(
            self._check_bursts(agent_id, actions_sorted, baseline)
        )

        # 6. Entropy shift
        anomalies.extend(
            self._check_entropy(agent_id, actions_sorted, baseline,
                                cat_counts=cat_counts, obs_dist=obs_dist)
        )

        # 7. Dormancy break
        anomalies.extend(
            self._check_dormancy(agent_id, actions_sorted, baseline)
        )

        # 8. Category ratio shift
        anomalies.extend(
            self._check_category_ratios(agent_id, actions_sorted, baseline,
                                        cat_counts=cat_counts,
                                        obs_dist=obs_dist)
        )

        # Compute risk score
        risk_score = self._compute_risk_score(anomalies)
        risk_level = _risk_level(risk_score)
        recommendations = self._generate_recommendations(
            anomalies, risk_level
        )

        summary: Dict[str, Any] = {
            "action_count": len(actions),
            "categories_used": len(cat_counts),
            "unique_resources": len(resources),
            "anomaly_count": len(anomalies),
            "baseline_actions": baseline.action_count,
        }

        return ProfileReport(
            agent_id=agent_id,
            anomalies=anomalies,
            risk_score=risk_score,
            risk_level=risk_level,
            action_count=len(actions),
            baseline_available=True,
            recommendations=recommendations,
            summary=summary,
        )

    # ── Anomaly Checks ───────────────────────────────────────────

    def _check_frequency(
        self,
        agent_id: str,
        actions: List[Action],
        baseline: AgentBaseline,
        *,
        cat_counts: Optional[Counter] = None,
    ) -> List[Anomaly]:
        """Detect category frequency spikes or drops."""
        anomalies: List[Anomaly] = []
        n = len(actions)
        if cat_counts is None:
            cat_counts = Counter(a.category for a in actions)

        for cat in set(list(cat_counts.keys()) + list(baseline.category_counts.keys())):
            observed_frac = cat_counts.get(cat, 0) / n if n > 0 else 0.0
            baseline_frac = baseline.category_distribution.get(cat, 0.0)

            if baseline_frac == 0 and observed_frac == 0:
                continue

            # Use proportional z-score approximation
            if baseline_frac > 0:
                # Expected count and std for binomial approximation
                expected = baseline_frac * n
                std = math.sqrt(baseline_frac * (1 - baseline_frac) * n)
                if std > 0:
                    z = (cat_counts.get(cat, 0) - expected) / std
                else:
                    z = 0.0

                if z > self.config.frequency_z_threshold:
                    severity = (
                        AnomalySeverity.HIGH
                        if z > self.config.frequency_z_threshold * 2
                        else AnomalySeverity.MEDIUM
                    )
                    anomalies.append(
                        Anomaly(
                            agent_id=agent_id,
                            anomaly_type=AnomalyType.FREQUENCY_SPIKE,
                            severity=severity,
                            score=min(1.0, z / 10.0),
                            description=(
                                f"Category '{cat.value}' frequency spike: "
                                f"{observed_frac:.1%} vs baseline {baseline_frac:.1%} "
                                f"(z={z:.1f})"
                            ),
                            details={
                                "category": cat.value,
                                "observed_fraction": round(observed_frac, 4),
                                "baseline_fraction": round(baseline_frac, 4),
                                "z_score": round(z, 2),
                            },
                        )
                    )
                elif z < -self.config.frequency_z_threshold:
                    anomalies.append(
                        Anomaly(
                            agent_id=agent_id,
                            anomaly_type=AnomalyType.FREQUENCY_DROP,
                            severity=AnomalySeverity.LOW,
                            score=min(1.0, abs(z) / 10.0),
                            description=(
                                f"Category '{cat.value}' frequency drop: "
                                f"{observed_frac:.1%} vs baseline {baseline_frac:.1%}"
                            ),
                            details={
                                "category": cat.value,
                                "observed_fraction": round(observed_frac, 4),
                                "baseline_fraction": round(baseline_frac, 4),
                                "z_score": round(z, 2),
                            },
                        )
                    )

        return anomalies

    def _check_new_categories(
        self,
        agent_id: str,
        actions: List[Action],
        baseline: AgentBaseline,
        *,
        observed_cats: Optional[Set] = None,
    ) -> List[Anomaly]:
        """Detect agent using action categories not seen in training."""
        known = set(baseline.category_distribution.keys())
        observed = observed_cats if observed_cats is not None else {a.category for a in actions}
        new_cats = observed - known

        anomalies: List[Anomaly] = []
        for cat in new_cats:
            # AUTH and REPLICATION are high-severity new categories
            if cat in (ActionCategory.AUTH, ActionCategory.REPLICATION):
                severity = AnomalySeverity.CRITICAL
                score = 0.9
            elif cat == ActionCategory.EXTERNAL:
                severity = AnomalySeverity.HIGH
                score = 0.7
            else:
                severity = AnomalySeverity.MEDIUM
                score = 0.5

            anomalies.append(
                Anomaly(
                    agent_id=agent_id,
                    anomaly_type=AnomalyType.NEW_CATEGORY,
                    severity=severity,
                    score=score,
                    description=(
                        f"New action category '{cat.value}' not seen in baseline"
                    ),
                    details={"category": cat.value},
                )
            )

        return anomalies

    def _check_new_resources(
        self,
        agent_id: str,
        actions: List[Action],
        baseline: AgentBaseline,
        *,
        observed_resources: Optional[Set[str]] = None,
    ) -> List[Anomaly]:
        """Detect access to resources not seen during training."""
        observed = observed_resources if observed_resources is not None else {a.resource for a in actions}
        new_resources = observed - baseline.known_resources

        if not new_resources:
            return []

        # Compute breadth ratio
        known_count = max(len(baseline.known_resources), 1)
        breadth_ratio = len(new_resources) / known_count

        if breadth_ratio >= self.config.resource_breadth_threshold:
            severity = AnomalySeverity.HIGH
            score = min(1.0, breadth_ratio)
        elif len(new_resources) >= 3:
            severity = AnomalySeverity.MEDIUM
            score = 0.5
        else:
            severity = AnomalySeverity.LOW
            score = 0.3

        return [
            Anomaly(
                agent_id=agent_id,
                anomaly_type=AnomalyType.NEW_RESOURCE
                if len(new_resources) < 3
                else AnomalyType.RESOURCE_BREADTH,
                severity=severity,
                score=score,
                description=(
                    f"{len(new_resources)} new resource(s) accessed "
                    f"(breadth ratio: {breadth_ratio:.1%} of baseline)"
                ),
                details={
                    "new_resources": sorted(new_resources),
                    "new_count": len(new_resources),
                    "baseline_count": len(baseline.known_resources),
                    "breadth_ratio": round(breadth_ratio, 4),
                },
            )
        ]

    def _check_timing(
        self,
        agent_id: str,
        actions: List[Action],
        baseline: AgentBaseline,
    ) -> List[Anomaly]:
        """Detect timing pattern deviations."""
        if len(actions) < 2 or baseline.interval_std == 0:
            return []

        intervals: List[float] = []
        for i in range(1, len(actions)):
            dt = actions[i].timestamp - actions[i - 1].timestamp
            intervals.append(dt)

        if not intervals:
            return []

        obs_mean = statistics.mean(intervals)

        # Z-score of mean interval vs baseline
        if baseline.interval_std > 0:
            z = abs(obs_mean - baseline.interval_mean) / baseline.interval_std
        else:
            z = 0.0

        anomalies: List[Anomaly] = []
        if z > self.config.frequency_z_threshold:
            severity = (
                AnomalySeverity.HIGH
                if z > self.config.frequency_z_threshold * 2
                else AnomalySeverity.MEDIUM
            )
            anomalies.append(
                Anomaly(
                    agent_id=agent_id,
                    anomaly_type=AnomalyType.TIMING_ANOMALY,
                    severity=severity,
                    score=min(1.0, z / 10.0),
                    description=(
                        f"Action timing deviation: mean interval "
                        f"{obs_mean:.1f}s vs baseline {baseline.interval_mean:.1f}s "
                        f"(z={z:.1f})"
                    ),
                    details={
                        "observed_mean_interval": round(obs_mean, 2),
                        "baseline_mean_interval": round(
                            baseline.interval_mean, 2
                        ),
                        "z_score": round(z, 2),
                    },
                )
            )

        return anomalies

    def _check_bursts(
        self,
        agent_id: str,
        actions: List[Action],
        baseline: AgentBaseline,
    ) -> List[Anomaly]:
        """Detect rapid action bursts within a time window."""
        if len(actions) < self.config.burst_count_threshold:
            return []

        window = self.config.burst_window_sec
        threshold = self.config.burst_count_threshold
        max_burst = 0

        # Sliding window count
        j = 0
        for i in range(len(actions)):
            while j < len(actions) and (
                actions[j].timestamp - actions[i].timestamp <= window
            ):
                j += 1
            burst_size = j - i
            if burst_size > max_burst:
                max_burst = burst_size

        anomalies: List[Anomaly] = []
        if max_burst >= threshold:
            severity = (
                AnomalySeverity.HIGH
                if max_burst >= threshold * 2
                else AnomalySeverity.MEDIUM
            )
            anomalies.append(
                Anomaly(
                    agent_id=agent_id,
                    anomaly_type=AnomalyType.BURST_DETECTED,
                    severity=severity,
                    score=min(1.0, max_burst / (threshold * 3)),
                    description=(
                        f"Action burst detected: {max_burst} actions in "
                        f"{window}s window (threshold: {threshold})"
                    ),
                    details={
                        "max_burst_size": max_burst,
                        "window_sec": window,
                        "threshold": threshold,
                    },
                )
            )

        return anomalies

    def _check_entropy(
        self,
        agent_id: str,
        actions: List[Action],
        baseline: AgentBaseline,
        *,
        cat_counts: Optional[Counter] = None,
        obs_dist: Optional[Dict] = None,
    ) -> List[Anomaly]:
        """Detect shifts in action category entropy."""
        n = len(actions)
        if n < 2:
            return []

        if cat_counts is None:
            cat_counts = Counter(a.category for a in actions)
        if obs_dist is None:
            obs_dist = {cat: count / n for cat, count in cat_counts.items()}
        obs_entropy = _shannon_entropy(obs_dist)

        shift = abs(obs_entropy - baseline.category_entropy)

        anomalies: List[Anomaly] = []
        if shift > self.config.entropy_shift_threshold:
            direction = (
                "increased" if obs_entropy > baseline.category_entropy else "decreased"
            )
            severity = (
                AnomalySeverity.HIGH
                if shift > self.config.entropy_shift_threshold * 2
                else AnomalySeverity.MEDIUM
            )
            anomalies.append(
                Anomaly(
                    agent_id=agent_id,
                    anomaly_type=AnomalyType.ENTROPY_SHIFT,
                    severity=severity,
                    score=min(1.0, shift / 2.0),
                    description=(
                        f"Category entropy {direction}: "
                        f"{obs_entropy:.2f} vs baseline {baseline.category_entropy:.2f} "
                        f"(Δ={shift:.2f})"
                    ),
                    details={
                        "observed_entropy": round(obs_entropy, 4),
                        "baseline_entropy": round(
                            baseline.category_entropy, 4
                        ),
                        "shift": round(shift, 4),
                        "direction": direction,
                    },
                )
            )

        return anomalies

    def _check_dormancy(
        self,
        agent_id: str,
        actions: List[Action],
        baseline: AgentBaseline,
    ) -> List[Anomaly]:
        """Detect dormancy breaks (agent resumes after long silence)."""
        if not actions:
            return []

        first_ts = actions[0].timestamp
        gap = first_ts - baseline.last_timestamp

        anomalies: List[Anomaly] = []
        if gap > self.config.dormancy_threshold:
            hours = gap / 3600
            severity = (
                AnomalySeverity.HIGH
                if gap > self.config.dormancy_threshold * 5
                else AnomalySeverity.MEDIUM
            )
            anomalies.append(
                Anomaly(
                    agent_id=agent_id,
                    anomaly_type=AnomalyType.DORMANCY_BREAK,
                    severity=severity,
                    score=min(1.0, gap / (self.config.dormancy_threshold * 10)),
                    description=(
                        f"Agent resumed after {hours:.1f}h dormancy "
                        f"(threshold: {self.config.dormancy_threshold / 3600:.1f}h)"
                    ),
                    details={
                        "gap_seconds": round(gap, 1),
                        "gap_hours": round(hours, 2),
                        "threshold_seconds": self.config.dormancy_threshold,
                    },
                )
            )

        return anomalies

    def _check_category_ratios(
        self,
        agent_id: str,
        actions: List[Action],
        baseline: AgentBaseline,
        *,
        cat_counts: Optional[Counter] = None,
        obs_dist: Optional[Dict] = None,
    ) -> List[Anomaly]:
        """Detect major shifts in the relative ratio between categories."""
        n = len(actions)
        if n < 5:
            return []

        if cat_counts is None:
            cat_counts = Counter(a.category for a in actions)
        if obs_dist is None:
            obs_dist = {cat: count / n for cat, count in cat_counts.items()}

        # Compute Jensen-Shannon divergence (symmetric KL)
        jsd = _jensen_shannon_divergence(
            baseline.category_distribution, obs_dist
        )

        anomalies: List[Anomaly] = []
        if jsd > 0.3:
            severity = (
                AnomalySeverity.HIGH if jsd > 0.6 else AnomalySeverity.MEDIUM
            )
            anomalies.append(
                Anomaly(
                    agent_id=agent_id,
                    anomaly_type=AnomalyType.CATEGORY_RATIO_SHIFT,
                    severity=severity,
                    score=min(1.0, jsd),
                    description=(
                        f"Category distribution shift detected "
                        f"(JSD={jsd:.3f})"
                    ),
                    details={
                        "jsd": round(jsd, 4),
                        "baseline_distribution": {
                            k.value: round(v, 4)
                            for k, v in baseline.category_distribution.items()
                        },
                        "observed_distribution": {
                            k.value: round(v, 4)
                            for k, v in obs_dist.items()
                        },
                    },
                )
            )

        return anomalies

    # ── Risk Scoring ─────────────────────────────────────────────

    def _compute_risk_score(self, anomalies: List[Anomaly]) -> float:
        """Compute aggregate risk score (0–100) from anomalies."""
        if not anomalies:
            return 0.0

        total = sum(
            SEVERITY_WEIGHT[a.severity] * a.score for a in anomalies
        )
        # Normalize: 100 = 5+ critical anomalies with max scores
        return min(100.0, round(total * 2, 1))

    def _generate_recommendations(
        self, anomalies: List[Anomaly], risk_level: str
    ) -> List[str]:
        """Generate actionable recommendations based on anomalies."""
        recs: List[str] = []
        types_seen = {a.anomaly_type for a in anomalies}

        if AnomalyType.NEW_CATEGORY in types_seen:
            auth_cats = [
                a
                for a in anomalies
                if a.anomaly_type == AnomalyType.NEW_CATEGORY
                and a.details.get("category") in ("auth", "replication")
            ]
            if auth_cats:
                recs.append(
                    "URGENT: Agent using auth/replication capabilities "
                    "not seen in baseline — investigate immediately"
                )
            else:
                recs.append(
                    "Review new action categories — may indicate "
                    "capability expansion"
                )

        if AnomalyType.BURST_DETECTED in types_seen:
            recs.append(
                "Implement rate limiting — burst behavior may indicate "
                "resource exhaustion attack or runaway loop"
            )

        if AnomalyType.NEW_RESOURCE in types_seen or AnomalyType.RESOURCE_BREADTH in types_seen:
            recs.append(
                "Audit new resource accesses — verify agent has "
                "legitimate need for these resources"
            )

        if AnomalyType.ENTROPY_SHIFT in types_seen:
            recs.append(
                "Category entropy shift suggests behavioral mode change — "
                "compare with known operational patterns"
            )

        if AnomalyType.DORMANCY_BREAK in types_seen:
            recs.append(
                "Agent resumed after extended dormancy — verify "
                "activation was authorized"
            )

        if AnomalyType.TIMING_ANOMALY in types_seen:
            recs.append(
                "Timing deviation may indicate compromised control loop — "
                "check for external influence"
            )

        if risk_level == "critical":
            recs.insert(0, "CRITICAL: Consider quarantining agent pending investigation")
        elif risk_level == "high":
            recs.insert(0, "Increase monitoring frequency for this agent")

        if not recs:
            recs.append("No actionable recommendations — agent behavior within normal bounds")

        return recs

    # ── Fleet Report ─────────────────────────────────────────────

    def _build_fleet_report(
        self, reports: List[ProfileReport]
    ) -> FleetReport:
        """Aggregate per-agent reports into a fleet-level summary."""
        total_anomalies = sum(len(r.anomalies) for r in reports)
        risk_dist: Counter[str] = Counter()
        highest_agent: Optional[str] = None
        highest_score = 0.0

        for r in reports:
            risk_dist[r.risk_level] += 1
            if r.risk_score > highest_score:
                highest_score = r.risk_score
                highest_agent = r.agent_id

        fleet_recs: List[str] = []
        critical = risk_dist.get("critical", 0)
        high = risk_dist.get("high", 0)

        if critical > 0:
            fleet_recs.append(
                f"{critical} agent(s) at CRITICAL risk — immediate review required"
            )
        if high > 0:
            fleet_recs.append(
                f"{high} agent(s) at HIGH risk — schedule investigation"
            )
        if total_anomalies == 0:
            fleet_recs.append(
                "Fleet behavior within normal parameters"
            )

        return FleetReport(
            agent_reports=reports,
            total_agents=len(reports),
            total_anomalies=total_anomalies,
            highest_risk_agent=highest_agent,
            highest_risk_score=highest_score,
            risk_distribution=dict(risk_dist),
            fleet_recommendations=fleet_recs,
        )


# ── Utility Functions ────────────────────────────────────────────────


def _shannon_entropy(distribution: Dict[Any, float]) -> float:
    """Compute Shannon entropy of a probability distribution."""
    entropy = 0.0
    for p in distribution.values():
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def _jensen_shannon_divergence(
    p: Dict[Any, float], q: Dict[Any, float]
) -> float:
    """Compute Jensen-Shannon divergence between two distributions."""
    all_keys = set(p.keys()) | set(q.keys())
    m: Dict[Any, float] = {}
    for k in all_keys:
        m[k] = (p.get(k, 0.0) + q.get(k, 0.0)) / 2.0

    def _kl(a: Dict[Any, float], b: Dict[Any, float]) -> float:
        kl = 0.0
        for k, ak in a.items():
            if ak > 0:
                bk = b.get(k, 0.0)
                if bk > 0:
                    kl += ak * math.log2(ak / bk)
                else:
                    kl += ak * 10  # penalty for zero in q
        return kl

    return (_kl(p, m) + _kl(q, m)) / 2.0


def _risk_level(score: float) -> str:
    """Map numeric risk score to categorical level."""
    if score >= 75:
        return "critical"
    elif score >= 50:
        return "high"
    elif score >= 25:
        return "medium"
    elif score > 0:
        return "low"
    else:
        return "none"


# ── Demo / CLI ───────────────────────────────────────────────────────


def demo(
    num_agents: int = 3,
    num_training: int = 100,
    num_analysis: int = 30,
    seed: int = 42,
) -> FleetReport:
    """Generate a demonstration with synthetic agent data.

    Creates agents with distinct behavioral profiles, trains baselines,
    then generates analysis actions with injected anomalies.

    Args:
        num_agents: Number of simulated agents.
        num_training: Training actions per agent.
        num_analysis: Analysis actions per agent.
        seed: Random seed for reproducibility.

    Returns:
        FleetReport from the analysis.
    """
    rng = random.Random(seed)

    # Agent profiles: category weights for training
    profiles: List[Dict[ActionCategory, float]] = [
        {  # Agent 0: compute-heavy worker
            ActionCategory.COMPUTE: 0.5,
            ActionCategory.STORAGE: 0.3,
            ActionCategory.NETWORK: 0.15,
            ActionCategory.IPC: 0.05,
        },
        {  # Agent 1: network-focused scanner
            ActionCategory.NETWORK: 0.6,
            ActionCategory.COMPUTE: 0.2,
            ActionCategory.EXTERNAL: 0.1,
            ActionCategory.STORAGE: 0.1,
        },
        {  # Agent 2: storage manager
            ActionCategory.STORAGE: 0.5,
            ActionCategory.COMPUTE: 0.2,
            ActionCategory.IPC: 0.2,
            ActionCategory.MEMORY: 0.1,
        },
    ]

    resources_pool = [
        "cpu-pool-a",
        "cpu-pool-b",
        "storage-main",
        "storage-temp",
        "api.internal",
        "db-primary",
        "cache-layer",
        "msg-queue",
    ]

    def _weighted_choice(
        weights: Dict[ActionCategory, float],
    ) -> ActionCategory:
        cats = list(weights.keys())
        ws = [weights[c] for c in cats]
        return rng.choices(cats, weights=ws, k=1)[0]

    # Generate training data
    training: List[Action] = []
    for i in range(min(num_agents, len(profiles))):
        agent_id = f"agent-{i}"
        t = 1000.0
        for _ in range(num_training):
            cat = _weighted_choice(profiles[i])
            resource = rng.choice(resources_pool[:5])  # limited resources in training
            training.append(
                Action(
                    agent_id=agent_id,
                    category=cat,
                    resource=resource,
                    timestamp=t,
                )
            )
            t += rng.uniform(5.0, 30.0)

    # Train
    profiler = BehaviorProfiler()
    profiler.train(training)

    # Generate analysis data with injected anomalies
    analysis: List[Action] = []
    for i in range(min(num_agents, len(profiles))):
        agent_id = f"agent-{i}"
        t = training[-1].timestamp + 100.0

        for j in range(num_analysis):
            if i == 0 and j > 20:
                # Agent 0: starts using AUTH (new category)
                cat = ActionCategory.AUTH
                resource = "secret-vault"
            elif i == 1 and j > 15:
                # Agent 1: burst of actions
                cat = _weighted_choice(profiles[i])
                resource = rng.choice(resources_pool)
                t += rng.uniform(0.1, 0.5)  # very fast
                analysis.append(
                    Action(
                        agent_id=agent_id,
                        category=cat,
                        resource=resource,
                        timestamp=t,
                    )
                )
                continue
            else:
                cat = _weighted_choice(profiles[i])
                resource = rng.choice(resources_pool)

            analysis.append(
                Action(
                    agent_id=agent_id,
                    category=cat,
                    resource=resource,
                    timestamp=t,
                )
            )
            t += rng.uniform(5.0, 30.0)

    return profiler.analyze(analysis)


def main() -> None:
    """CLI entry point for behavior profiling demo."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Behavior Profiler — agent behavioral anomaly detection"
    )
    parser.add_argument(
        "--agents",
        type=int,
        default=3,
        help="Number of simulated agents (default: 3)",
    )
    parser.add_argument(
        "--actions",
        type=int,
        default=100,
        help="Training actions per agent (default: 100)",
    )
    parser.add_argument(
        "--analysis",
        type=int,
        default=30,
        help="Analysis actions per agent (default: 30)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    report = demo(
        num_agents=args.agents,
        num_training=args.actions,
        num_analysis=args.analysis,
        seed=args.seed,
    )

    print("═" * 60)
    print("  Behavior Profiler — Fleet Analysis Report")
    print("═" * 60)
    print(f"\n  Agents analyzed: {report.total_agents}")
    print(f"  Total anomalies: {report.total_anomalies}")
    if report.highest_risk_agent:
        print(
            f"  Highest risk: {report.highest_risk_agent} "
            f"(score: {report.highest_risk_score:.1f}/100)"
        )
    print(f"\n  Risk distribution: {dict(report.risk_distribution)}")
    print()

    for pr in report.agent_reports:
        print(f"  ── {pr.agent_id} ──")
        print(f"     Actions: {pr.action_count}")
        print(f"     Risk: {pr.risk_level} ({pr.risk_score:.1f}/100)")
        if pr.anomalies:
            print(f"     Anomalies ({len(pr.anomalies)}):")
            for a in pr.anomalies:
                print(f"       [{a.severity.value:8s}] {a.description}")
        if pr.recommendations:
            print("     Recommendations:")
            for r in pr.recommendations:
                print(f"       • {r}")
        print()

    if report.fleet_recommendations:
        print("  Fleet Recommendations:")
        for r in report.fleet_recommendations:
            print(f"    • {r}")

    print("\n" + "═" * 60)


if __name__ == "__main__":
    main()
