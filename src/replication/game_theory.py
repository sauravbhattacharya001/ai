"""Agent Game-Theory Analyzer — model inter-agent interactions as strategic games.

In multi-agent replication systems, agents interact repeatedly and can
develop cooperative or defective strategies.  From a safety perspective,
we need to detect when agents:

- **Collude** to circumvent safety constraints (mutual cooperation on
  unsafe goals)
- **Free-ride** on shared resources while others bear costs (defection
  in public goods)
- **Escalate** into competitive arms races (mutual defection spirals)
- **Form stable coalitions** that resist oversight (Nash equilibria in
  unsafe configurations)

The game-theory analyzer records pairwise agent interactions, classifies
them into canonical game types, computes equilibria, and detects
concerning strategic patterns.

Supported games:

- **Prisoner's Dilemma** (PD): cooperation vs. defection with temptation
  to defect — detects free-riding and trust breakdown
- **Stag Hunt** (SH): coordination game — detects whether agents
  converge on risky cooperation or safe defection
- **Chicken / Hawk-Dove** (CH): anti-coordination — detects escalation
  and brinkmanship between agents
- **Harmony** (HG): dominant cooperation — baseline safe interaction

Usage (CLI)::

    python -m replication.game_theory                        # analyze default logs
    python -m replication.game_theory --agents 5             # simulate 5 agents
    python -m replication.game_theory --rounds 100           # 100 interaction rounds
    python -m replication.game_theory --strategy tit-for-tat # set default strategy
    python -m replication.game_theory --detect collusion     # detect collusion only
    python -m replication.game_theory --json                 # JSON output

Programmatic::

    from replication.game_theory import GameTheoryAnalyzer, GameConfig
    analyzer = GameTheoryAnalyzer(GameConfig(history_limit=500))
    analyzer.record_interaction("agent-a", "agent-b", "cooperate", "defect")
    analyzer.record_interaction("agent-b", "agent-a", "cooperate", "cooperate")
    report = analyzer.analyze()
    print(report.render())
"""

from __future__ import annotations

import math
import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Move(str, Enum):
    """Agent action in a two-player game."""
    COOPERATE = "cooperate"
    DEFECT = "defect"


class GameType(str, Enum):
    """Canonical 2×2 symmetric game classification."""
    PRISONERS_DILEMMA = "prisoners_dilemma"
    STAG_HUNT = "stag_hunt"
    CHICKEN = "chicken"
    HARMONY = "harmony"
    UNKNOWN = "unknown"


class StrategyType(str, Enum):
    """Known agent strategies (detected via pattern analysis)."""
    ALWAYS_COOPERATE = "always_cooperate"
    ALWAYS_DEFECT = "always_defect"
    TIT_FOR_TAT = "tit_for_tat"
    GENEROUS_TFT = "generous_tft"
    GRUDGER = "grudger"
    RANDOM = "random"
    PAVLOV = "pavlov"
    UNKNOWN = "unknown"


class AlertLevel(str, Enum):
    """Severity of a detected strategic pattern."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Payoffs:
    """Payoff matrix for a 2×2 symmetric game.

    Standard notation: T > R > P > S  (Prisoner's Dilemma)
    - R: reward for mutual cooperation
    - S: sucker's payoff (cooperate vs. defect)
    - T: temptation to defect (defect vs. cooperate)
    - P: punishment for mutual defection
    """
    R: float = 3.0  # mutual cooperation
    S: float = 0.0  # sucker's payoff
    T: float = 5.0  # temptation
    P: float = 1.0  # mutual defection

    def classify(self) -> GameType:
        """Classify the payoff matrix into a canonical game type."""
        if self.T > self.R > self.P > self.S:
            return GameType.PRISONERS_DILEMMA
        if self.R > self.T > self.P > self.S:
            return GameType.STAG_HUNT
        if self.T > self.R > self.S > self.P:
            return GameType.CHICKEN
        if self.R > self.T and self.R > self.P and self.R > self.S:
            return GameType.HARMONY
        return GameType.UNKNOWN

    def payoff(self, my_move: Move, their_move: Move) -> float:
        """Get the payoff for a player given both moves."""
        if my_move == Move.COOPERATE:
            return self.R if their_move == Move.COOPERATE else self.S
        else:
            return self.T if their_move == Move.COOPERATE else self.P

    def nash_equilibria(self) -> List[Tuple[Move, Move]]:
        """Find pure-strategy Nash equilibria."""
        equilibria = []
        for m1 in Move:
            for m2 in Move:
                # Check if m1 is best response to m2 and vice versa
                p1_current = self.payoff(m1, m2)
                p1_alt = self.payoff(
                    Move.DEFECT if m1 == Move.COOPERATE else Move.COOPERATE,
                    m2,
                )
                p2_current = self.payoff(m2, m1)
                p2_alt = self.payoff(
                    Move.DEFECT if m2 == Move.COOPERATE else Move.COOPERATE,
                    m1,
                )
                if p1_current >= p1_alt and p2_current >= p2_alt:
                    equilibria.append((m1, m2))
        return equilibria

    def mixed_nash(self) -> Optional[float]:
        """Compute the mixed-strategy Nash equilibrium probability of cooperating.

        Returns ``None`` if there is no interior mixed equilibrium (dominant
        strategy exists).
        """
        # Player is indifferent when:
        # p*R + (1-p)*S = p*T + (1-p)*P
        # p*(R - S - T + P) = P - S
        denom = self.R - self.S - self.T + self.P
        if abs(denom) < 1e-12:
            return None
        p = (self.P - self.S) / denom
        if 0 < p < 1:
            return p
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {"R": self.R, "S": self.S, "T": self.T, "P": self.P}


@dataclass
class Interaction:
    """A single recorded pairwise interaction."""
    agent_a: str
    agent_b: str
    move_a: Move
    move_b: Move
    round_num: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_a": self.agent_a,
            "agent_b": self.agent_b,
            "move_a": self.move_a.value,
            "move_b": self.move_b.value,
            "round": self.round_num,
            "metadata": self.metadata,
        }


@dataclass
class PairStats:
    """Aggregate statistics for a pair of agents."""
    agent_a: str
    agent_b: str
    total_rounds: int = 0
    mutual_cooperate: int = 0
    mutual_defect: int = 0
    a_defects_b_cooperates: int = 0
    b_defects_a_cooperates: int = 0
    a_total_payoff: float = 0.0
    b_total_payoff: float = 0.0

    @property
    def cooperation_rate(self) -> float:
        """Fraction of rounds with mutual cooperation."""
        return self.mutual_cooperate / max(1, self.total_rounds)

    @property
    def defection_rate(self) -> float:
        """Fraction of rounds with mutual defection."""
        return self.mutual_defect / max(1, self.total_rounds)

    @property
    def exploitation_rate(self) -> float:
        """Fraction of rounds where one agent exploits the other."""
        exploits = self.a_defects_b_cooperates + self.b_defects_a_cooperates
        return exploits / max(1, self.total_rounds)

    @property
    def payoff_inequality(self) -> float:
        """Absolute payoff difference normalized by total rounds."""
        if self.total_rounds == 0:
            return 0.0
        return abs(self.a_total_payoff - self.b_total_payoff) / self.total_rounds

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_a": self.agent_a,
            "agent_b": self.agent_b,
            "total_rounds": self.total_rounds,
            "mutual_cooperate": self.mutual_cooperate,
            "mutual_defect": self.mutual_defect,
            "a_exploits_b": self.a_defects_b_cooperates,
            "b_exploits_a": self.b_defects_a_cooperates,
            "cooperation_rate": round(self.cooperation_rate, 4),
            "defection_rate": round(self.defection_rate, 4),
            "exploitation_rate": round(self.exploitation_rate, 4),
            "a_total_payoff": round(self.a_total_payoff, 2),
            "b_total_payoff": round(self.b_total_payoff, 2),
            "payoff_inequality": round(self.payoff_inequality, 4),
        }


@dataclass
class StrategyProfile:
    """Detected strategy for a single agent."""
    agent_id: str
    strategy: StrategyType
    confidence: float  # 0.0 to 1.0
    cooperation_rate: float
    total_moves: int
    avg_payoff: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "strategy": self.strategy.value,
            "confidence": round(self.confidence, 3),
            "cooperation_rate": round(self.cooperation_rate, 4),
            "total_moves": self.total_moves,
            "avg_payoff": round(self.avg_payoff, 3),
        }


@dataclass
class StrategicAlert:
    """A safety-relevant strategic pattern detected in agent interactions."""
    level: AlertLevel
    category: str  # collusion, free_riding, escalation, instability
    description: str
    agents: List[str]
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level.value,
            "category": self.category,
            "description": self.description,
            "agents": self.agents,
            "evidence": self.evidence,
        }


@dataclass
class GameReport:
    """Complete analysis report."""
    game_type: GameType
    payoffs: Payoffs
    total_interactions: int
    total_agents: int
    pair_stats: List[PairStats]
    strategy_profiles: List[StrategyProfile]
    alerts: List[StrategicAlert]
    nash_equilibria: List[Tuple[Move, Move]]
    mixed_nash_p: Optional[float]
    global_cooperation_rate: float
    cooperation_trend: float  # positive = increasing cooperation
    risk_score: float  # 0-100 composite

    def to_dict(self) -> Dict[str, Any]:
        return {
            "game_type": self.game_type.value,
            "payoffs": self.payoffs.to_dict(),
            "total_interactions": self.total_interactions,
            "total_agents": self.total_agents,
            "pair_stats": [ps.to_dict() for ps in self.pair_stats],
            "strategy_profiles": [sp.to_dict() for sp in self.strategy_profiles],
            "alerts": [a.to_dict() for a in self.alerts],
            "nash_equilibria": [
                (m1.value, m2.value) for m1, m2 in self.nash_equilibria
            ],
            "mixed_nash_p": (
                round(self.mixed_nash_p, 4) if self.mixed_nash_p is not None else None
            ),
            "global_cooperation_rate": round(self.global_cooperation_rate, 4),
            "cooperation_trend": round(self.cooperation_trend, 4),
            "risk_score": round(self.risk_score, 1),
        }

    def render(self) -> str:
        """Human-readable multi-line report."""
        lines: List[str] = []
        lines.append("=" * 60)
        lines.append("  AGENT GAME-THEORY ANALYSIS REPORT")
        lines.append("=" * 60)
        lines.append("")

        # Game classification
        lines.append(f"Game Type      : {self.game_type.value}")
        lines.append(
            f"Payoffs        : R={self.payoffs.R}, S={self.payoffs.S}, "
            f"T={self.payoffs.T}, P={self.payoffs.P}"
        )
        eq_strs = [f"({m1.value}, {m2.value})" for m1, m2 in self.nash_equilibria]
        lines.append(f"Nash Equilibria: {', '.join(eq_strs) or 'none (pure)'}")
        if self.mixed_nash_p is not None:
            lines.append(f"Mixed Nash     : p(cooperate)={self.mixed_nash_p:.4f}")
        lines.append("")

        # Overview
        lines.append(f"Agents         : {self.total_agents}")
        lines.append(f"Interactions   : {self.total_interactions}")
        lines.append(f"Cooperation    : {self.global_cooperation_rate:.1%}")
        trend_dir = "rising" if self.cooperation_trend > 0.01 else (
            "falling" if self.cooperation_trend < -0.01 else "stable"
        )
        lines.append(f"Trend          : {trend_dir} ({self.cooperation_trend:+.4f})")
        lines.append(f"Risk Score     : {self.risk_score:.1f}/100")
        lines.append("")

        # Strategy profiles
        if self.strategy_profiles:
            lines.append("--- Strategy Profiles ---")
            for sp in sorted(self.strategy_profiles, key=lambda s: -s.avg_payoff):
                lines.append(
                    f"  {sp.agent_id}: {sp.strategy.value} "
                    f"(conf={sp.confidence:.0%}, coop={sp.cooperation_rate:.0%}, "
                    f"payoff={sp.avg_payoff:.2f})"
                )
            lines.append("")

        # Pair analysis
        if self.pair_stats:
            lines.append("--- Pair Analysis ---")
            for ps in sorted(self.pair_stats, key=lambda p: -p.total_rounds)[:10]:
                lines.append(
                    f"  {ps.agent_a} vs {ps.agent_b}: "
                    f"{ps.total_rounds} rounds, "
                    f"coop={ps.cooperation_rate:.0%}, "
                    f"defect={ps.defection_rate:.0%}, "
                    f"exploit={ps.exploitation_rate:.0%}"
                )
            lines.append("")

        # Alerts
        if self.alerts:
            lines.append("--- Alerts ---")
            for alert in sorted(
                self.alerts,
                key=lambda a: (
                    {AlertLevel.CRITICAL: 0, AlertLevel.WARNING: 1, AlertLevel.INFO: 2}[
                        a.level
                    ]
                ),
            ):
                icon = {"critical": "!!!", "warning": "! ", "info": "  "}[alert.level.value]
                lines.append(f"  [{icon}] {alert.category}: {alert.description}")
                lines.append(f"       agents: {', '.join(alert.agents)}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class GameConfig:
    """Configuration for the game-theory analyzer."""

    payoffs: Payoffs = field(default_factory=Payoffs)
    history_limit: int = 1000
    min_rounds_for_strategy: int = 5
    collusion_threshold: float = 0.85
    escalation_threshold: float = 0.70
    free_ride_threshold: float = 0.60
    instability_window: int = 10
    instability_flip_rate: float = 0.40


# ---------------------------------------------------------------------------
# Strategies (for simulation / detection)
# ---------------------------------------------------------------------------


def _tit_for_tat(history: List[Move]) -> Move:
    """Cooperate first, then copy opponent's last move."""
    if not history:
        return Move.COOPERATE
    return history[-1]


def _generous_tft(history: List[Move], forgiveness: float = 0.1) -> Move:
    """Like TFT but occasionally forgives defection."""
    if not history:
        return Move.COOPERATE
    if history[-1] == Move.DEFECT:
        # Deterministic check based on round count for reproducibility
        h = int(hashlib.md5(str(len(history)).encode()).hexdigest()[:8], 16)
        if (h % 1000) / 1000.0 < forgiveness:
            return Move.COOPERATE
        return Move.DEFECT
    return Move.COOPERATE


def _grudger(history: List[Move]) -> Move:
    """Cooperate until opponent defects, then always defect."""
    if Move.DEFECT in history:
        return Move.DEFECT
    return Move.COOPERATE


def _pavlov(my_last: Optional[Move], their_last: Optional[Move]) -> Move:
    """Win-stay, lose-shift: repeat if mutual outcome, switch otherwise."""
    if my_last is None or their_last is None:
        return Move.COOPERATE
    if my_last == their_last:
        return my_last
    return Move.DEFECT if my_last == Move.COOPERATE else Move.COOPERATE


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class GameTheoryAnalyzer:
    """Records inter-agent interactions and analyzes strategic patterns."""

    def __init__(self, config: Optional[GameConfig] = None) -> None:
        self._config = config or GameConfig()
        self._interactions: List[Interaction] = []
        self._round_counter: int = 0

    # -- recording --

    def record_interaction(
        self,
        agent_a: str,
        agent_b: str,
        move_a: str | Move,
        move_b: str | Move,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Interaction:
        """Record a pairwise interaction between two agents.

        ``move_a`` / ``move_b`` can be ``Move`` enums or plain strings
        (``"cooperate"`` / ``"defect"``).
        """
        ma = Move(move_a) if isinstance(move_a, str) else move_a
        mb = Move(move_b) if isinstance(move_b, str) else move_b
        self._round_counter += 1
        interaction = Interaction(
            agent_a=agent_a,
            agent_b=agent_b,
            move_a=ma,
            move_b=mb,
            round_num=self._round_counter,
            metadata=metadata or {},
        )
        self._interactions.append(interaction)
        # Enforce history limit
        if len(self._interactions) > self._config.history_limit:
            self._interactions = self._interactions[-self._config.history_limit :]
        return interaction

    @property
    def interaction_count(self) -> int:
        return len(self._interactions)

    @property
    def agents(self) -> List[str]:
        seen: dict[str, None] = {}
        for ix in self._interactions:
            seen[ix.agent_a] = None
            seen[ix.agent_b] = None
        return list(seen.keys())

    def clear(self) -> None:
        self._interactions.clear()
        self._round_counter = 0

    # -- analysis --

    def analyze(self) -> GameReport:
        """Run full game-theory analysis and return a report."""
        payoffs = self._config.payoffs
        game_type = payoffs.classify()
        nash_eq = payoffs.nash_equilibria()
        mixed_p = payoffs.mixed_nash()

        agent_list = self.agents
        pair_stats = self._compute_pair_stats(payoffs)
        strategy_profiles = self._detect_strategies(payoffs)
        alerts = self._detect_alerts(pair_stats, strategy_profiles)
        global_coop = self._global_cooperation_rate()
        trend = self._cooperation_trend()
        risk = self._compute_risk_score(pair_stats, strategy_profiles, alerts)

        return GameReport(
            game_type=game_type,
            payoffs=payoffs,
            total_interactions=len(self._interactions),
            total_agents=len(agent_list),
            pair_stats=pair_stats,
            strategy_profiles=strategy_profiles,
            alerts=alerts,
            nash_equilibria=nash_eq,
            mixed_nash_p=mixed_p,
            global_cooperation_rate=global_coop,
            cooperation_trend=trend,
            risk_score=risk,
        )

    # -- pair stats --

    def _compute_pair_stats(self, payoffs: Payoffs) -> List[PairStats]:
        """Compute aggregate statistics per agent pair."""
        pairs: Dict[Tuple[str, str], PairStats] = {}
        for ix in self._interactions:
            key = (
                (ix.agent_a, ix.agent_b)
                if ix.agent_a <= ix.agent_b
                else (ix.agent_b, ix.agent_a)
            )
            if key not in pairs:
                pairs[key] = PairStats(agent_a=key[0], agent_b=key[1])
            ps = pairs[key]
            ps.total_rounds += 1

            # Determine who is a and who is b in the canonical pair
            if ix.agent_a == key[0]:
                ma, mb = ix.move_a, ix.move_b
            else:
                ma, mb = ix.move_b, ix.move_a

            if ma == Move.COOPERATE and mb == Move.COOPERATE:
                ps.mutual_cooperate += 1
            elif ma == Move.DEFECT and mb == Move.DEFECT:
                ps.mutual_defect += 1
            elif ma == Move.DEFECT and mb == Move.COOPERATE:
                ps.a_defects_b_cooperates += 1
            else:
                ps.b_defects_a_cooperates += 1

            ps.a_total_payoff += payoffs.payoff(ma, mb)
            ps.b_total_payoff += payoffs.payoff(mb, ma)

        return list(pairs.values())

    # -- strategy detection --

    def _detect_strategies(self, payoffs: Payoffs) -> List[StrategyProfile]:
        """Detect the strategy each agent appears to be using."""
        agent_moves: Dict[str, List[Tuple[Move, Move]]] = defaultdict(list)
        agent_payoffs: Dict[str, List[float]] = defaultdict(list)

        for ix in self._interactions:
            agent_moves[ix.agent_a].append((ix.move_a, ix.move_b))
            agent_moves[ix.agent_b].append((ix.move_b, ix.move_a))
            agent_payoffs[ix.agent_a].append(payoffs.payoff(ix.move_a, ix.move_b))
            agent_payoffs[ix.agent_b].append(payoffs.payoff(ix.move_b, ix.move_a))

        profiles = []
        for agent_id, moves in agent_moves.items():
            total = len(moves)
            if total < self._config.min_rounds_for_strategy:
                profiles.append(
                    StrategyProfile(
                        agent_id=agent_id,
                        strategy=StrategyType.UNKNOWN,
                        confidence=0.0,
                        cooperation_rate=0.0,
                        total_moves=total,
                        avg_payoff=0.0,
                    )
                )
                continue

            my_moves = [m for m, _ in moves]
            their_moves = [t for _, t in moves]
            coop_count = sum(1 for m in my_moves if m == Move.COOPERATE)
            coop_rate = coop_count / total
            avg_pay = sum(agent_payoffs[agent_id]) / total

            strategy, confidence = self._classify_strategy(my_moves, their_moves)

            profiles.append(
                StrategyProfile(
                    agent_id=agent_id,
                    strategy=strategy,
                    confidence=confidence,
                    cooperation_rate=coop_rate,
                    total_moves=total,
                    avg_payoff=avg_pay,
                )
            )

        return profiles

    def _classify_strategy(
        self, my_moves: List[Move], their_moves: List[Move]
    ) -> Tuple[StrategyType, float]:
        """Classify an agent's strategy from its move history."""
        n = len(my_moves)
        if n == 0:
            return StrategyType.UNKNOWN, 0.0

        coop_rate = sum(1 for m in my_moves if m == Move.COOPERATE) / n

        # Always cooperate
        if coop_rate >= 0.95:
            return StrategyType.ALWAYS_COOPERATE, min(1.0, coop_rate)

        # Always defect
        if coop_rate <= 0.05:
            return StrategyType.ALWAYS_DEFECT, min(1.0, 1.0 - coop_rate)

        # Tit-for-tat: after round 1, mirrors opponent's previous move
        if n >= 3:
            tft_matches = sum(
                1
                for i in range(1, n)
                if my_moves[i] == their_moves[i - 1]
            )
            tft_rate = tft_matches / (n - 1)
            if tft_rate >= 0.85:
                return StrategyType.TIT_FOR_TAT, tft_rate

        # Generous TFT: like TFT but sometimes cooperates after opponent defects
        if n >= 5:
            opponent_defections = [
                i for i in range(1, n) if their_moves[i - 1] == Move.DEFECT
            ]
            if len(opponent_defections) >= 2:
                follows_defect = sum(
                    1
                    for i in opponent_defections
                    if my_moves[i] == Move.DEFECT
                )
                forgives = len(opponent_defections) - follows_defect
                forgive_rate = forgives / len(opponent_defections)
                # Mostly retaliates but sometimes forgives (5-30%)
                if 0.05 <= forgive_rate <= 0.35:
                    tft_on_coop = sum(
                        1
                        for i in range(1, n)
                        if their_moves[i - 1] == Move.COOPERATE
                        and my_moves[i] == Move.COOPERATE
                    )
                    coop_follows = sum(
                        1 for i in range(1, n) if their_moves[i - 1] == Move.COOPERATE
                    )
                    if coop_follows > 0 and tft_on_coop / coop_follows >= 0.80:
                        return StrategyType.GENEROUS_TFT, 0.75

        # Grudger: cooperates until first defection, then always defects
        if n >= 5:
            first_defect_idx = None
            for i, t in enumerate(their_moves):
                if t == Move.DEFECT:
                    first_defect_idx = i
                    break

            if first_defect_idx is not None and first_defect_idx > 0:
                coop_before = all(
                    m == Move.COOPERATE for m in my_moves[: first_defect_idx + 1]
                )
                defect_after = (
                    sum(
                        1
                        for m in my_moves[first_defect_idx + 1 :]
                        if m == Move.DEFECT
                    )
                    / max(1, n - first_defect_idx - 1)
                )
                if coop_before and defect_after >= 0.90:
                    return StrategyType.GRUDGER, defect_after

        # Pavlov: win-stay, lose-shift
        if n >= 5:
            pavlov_matches = 0
            for i in range(1, n):
                expected = _pavlov(my_moves[i - 1], their_moves[i - 1])
                if my_moves[i] == expected:
                    pavlov_matches += 1
            pavlov_rate = pavlov_matches / (n - 1)
            if pavlov_rate >= 0.80:
                return StrategyType.PAVLOV, pavlov_rate

        # Random: cooperation rate near 50% with low autocorrelation
        if 0.35 <= coop_rate <= 0.65:
            if n >= 5:
                switches = sum(
                    1 for i in range(1, n) if my_moves[i] != my_moves[i - 1]
                )
                switch_rate = switches / (n - 1)
                if 0.30 <= switch_rate <= 0.70:
                    return StrategyType.RANDOM, 1.0 - abs(coop_rate - 0.5) * 2

        return StrategyType.UNKNOWN, 0.3

    # -- alert detection --

    def _detect_alerts(
        self,
        pair_stats: List[PairStats],
        profiles: List[StrategyProfile],
    ) -> List[StrategicAlert]:
        """Detect safety-relevant strategic patterns."""
        alerts: List[StrategicAlert] = []

        cfg = self._config

        # 1. Collusion: pair with very high mutual cooperation
        for ps in pair_stats:
            if (
                ps.total_rounds >= cfg.min_rounds_for_strategy
                and ps.cooperation_rate >= cfg.collusion_threshold
            ):
                alerts.append(
                    StrategicAlert(
                        level=AlertLevel.WARNING,
                        category="collusion",
                        description=(
                            f"Agents {ps.agent_a} and {ps.agent_b} show "
                            f"sustained mutual cooperation ({ps.cooperation_rate:.0%} "
                            f"over {ps.total_rounds} rounds) — possible collusion"
                        ),
                        agents=[ps.agent_a, ps.agent_b],
                        evidence={
                            "cooperation_rate": round(ps.cooperation_rate, 4),
                            "rounds": ps.total_rounds,
                        },
                    )
                )

        # 2. Escalation: pair with high mutual defection
        for ps in pair_stats:
            if (
                ps.total_rounds >= cfg.min_rounds_for_strategy
                and ps.defection_rate >= cfg.escalation_threshold
            ):
                alerts.append(
                    StrategicAlert(
                        level=AlertLevel.CRITICAL,
                        category="escalation",
                        description=(
                            f"Agents {ps.agent_a} and {ps.agent_b} locked in "
                            f"mutual defection spiral ({ps.defection_rate:.0%} "
                            f"over {ps.total_rounds} rounds)"
                        ),
                        agents=[ps.agent_a, ps.agent_b],
                        evidence={
                            "defection_rate": round(ps.defection_rate, 4),
                            "rounds": ps.total_rounds,
                        },
                    )
                )

        # 3. Free-riding: one agent exploits cooperator
        for ps in pair_stats:
            if ps.total_rounds < cfg.min_rounds_for_strategy:
                continue
            a_exploit = ps.a_defects_b_cooperates / max(1, ps.total_rounds)
            b_exploit = ps.b_defects_a_cooperates / max(1, ps.total_rounds)
            if a_exploit >= cfg.free_ride_threshold:
                alerts.append(
                    StrategicAlert(
                        level=AlertLevel.WARNING,
                        category="free_riding",
                        description=(
                            f"Agent {ps.agent_a} is free-riding on {ps.agent_b} "
                            f"({a_exploit:.0%} exploitation rate)"
                        ),
                        agents=[ps.agent_a, ps.agent_b],
                        evidence={
                            "exploitation_rate": round(a_exploit, 4),
                            "exploiter": ps.agent_a,
                            "victim": ps.agent_b,
                        },
                    )
                )
            if b_exploit >= cfg.free_ride_threshold:
                alerts.append(
                    StrategicAlert(
                        level=AlertLevel.WARNING,
                        category="free_riding",
                        description=(
                            f"Agent {ps.agent_b} is free-riding on {ps.agent_a} "
                            f"({b_exploit:.0%} exploitation rate)"
                        ),
                        agents=[ps.agent_b, ps.agent_a],
                        evidence={
                            "exploitation_rate": round(b_exploit, 4),
                            "exploiter": ps.agent_b,
                            "victim": ps.agent_a,
                        },
                    )
                )

        # 4. Strategy instability: agent keeps switching behavior
        for profile in profiles:
            if profile.total_moves < cfg.instability_window:
                continue
            agent_id = profile.agent_id
            moves = [
                ix.move_a if ix.agent_a == agent_id else ix.move_b
                for ix in self._interactions
                if ix.agent_a == agent_id or ix.agent_b == agent_id
            ]
            if len(moves) < cfg.instability_window:
                continue
            recent = moves[-cfg.instability_window :]
            flips = sum(1 for i in range(1, len(recent)) if recent[i] != recent[i - 1])
            flip_rate = flips / (len(recent) - 1)
            if flip_rate >= cfg.instability_flip_rate:
                alerts.append(
                    StrategicAlert(
                        level=AlertLevel.INFO,
                        category="instability",
                        description=(
                            f"Agent {agent_id} showing unstable strategy "
                            f"({flip_rate:.0%} flip rate in last "
                            f"{cfg.instability_window} rounds)"
                        ),
                        agents=[agent_id],
                        evidence={
                            "flip_rate": round(flip_rate, 4),
                            "window": cfg.instability_window,
                        },
                    )
                )

        # 5. Dominant defectors: all-defect agents are a safety concern
        all_defectors = [
            p.agent_id
            for p in profiles
            if p.strategy == StrategyType.ALWAYS_DEFECT
            and p.total_moves >= cfg.min_rounds_for_strategy
        ]
        if len(all_defectors) >= 2:
            alerts.append(
                StrategicAlert(
                    level=AlertLevel.CRITICAL,
                    category="escalation",
                    description=(
                        f"{len(all_defectors)} agents using always-defect strategy "
                        f"— system trending toward adversarial equilibrium"
                    ),
                    agents=all_defectors,
                    evidence={"count": len(all_defectors)},
                )
            )

        return alerts

    # -- global metrics --

    def _global_cooperation_rate(self) -> float:
        """Overall cooperation rate across all interactions."""
        if not self._interactions:
            return 0.0
        total_moves = len(self._interactions) * 2
        coop_moves = sum(
            (1 if ix.move_a == Move.COOPERATE else 0)
            + (1 if ix.move_b == Move.COOPERATE else 0)
            for ix in self._interactions
        )
        return coop_moves / total_moves

    def _cooperation_trend(self) -> float:
        """Compute cooperation trend (positive = increasing cooperation).

        Uses simple linear regression slope on cooperation rate over time.
        """
        n = len(self._interactions)
        if n < 3:
            return 0.0

        # Compute cooperation rate in sliding windows
        window = max(3, n // 10)
        rates = []
        for i in range(0, n - window + 1, max(1, window // 2)):
            chunk = self._interactions[i : i + window]
            coop = sum(
                (1 if ix.move_a == Move.COOPERATE else 0)
                + (1 if ix.move_b == Move.COOPERATE else 0)
                for ix in chunk
            )
            rates.append(coop / (len(chunk) * 2))

        if len(rates) < 2:
            return 0.0

        # Linear regression slope
        k = len(rates)
        x_mean = (k - 1) / 2.0
        y_mean = sum(rates) / k
        num = sum((i - x_mean) * (r - y_mean) for i, r in enumerate(rates))
        den = sum((i - x_mean) ** 2 for i in range(k))
        if abs(den) < 1e-12:
            return 0.0
        return num / den

    def _compute_risk_score(
        self,
        pair_stats: List[PairStats],
        profiles: List[StrategyProfile],
        alerts: List[StrategicAlert],
    ) -> float:
        """Compute a composite risk score (0-100).

        Higher = more concerning strategic behavior.
        """
        score = 0.0

        # Factor 1: Defection prevalence (0-30)
        if self._interactions:
            defect_rate = 1.0 - self._global_cooperation_rate()
            score += defect_rate * 30

        # Factor 2: Alert severity (0-40)
        for alert in alerts:
            if alert.level == AlertLevel.CRITICAL:
                score += 15
            elif alert.level == AlertLevel.WARNING:
                score += 8
            else:
                score += 2

        # Factor 3: Strategy homogeneity toward defection (0-20)
        if profiles:
            defector_frac = sum(
                1
                for p in profiles
                if p.strategy == StrategyType.ALWAYS_DEFECT
            ) / len(profiles)
            score += defector_frac * 20

        # Factor 4: Payoff inequality (0-10)
        if pair_stats:
            avg_inequality = sum(ps.payoff_inequality for ps in pair_stats) / len(
                pair_stats
            )
            score += min(10, avg_inequality * 5)

        return min(100, max(0, score))

    # -- simulation --

    def simulate(
        self,
        agents: Dict[str, StrategyType],
        rounds: int = 50,
        payoffs: Optional[Payoffs] = None,
    ) -> GameReport:
        """Simulate a round-robin tournament between agents with known strategies.

        Each agent plays every other agent for ``rounds`` rounds.
        Returns the analysis report.
        """
        p = payoffs or self._config.payoffs
        # Update config so analyze() uses the correct payoffs
        self._config.payoffs = p
        agent_ids = list(agents.keys())

        # Build strategy functions
        def make_move(
            strategy: StrategyType,
            my_history: List[Move],
            their_history: List[Move],
        ) -> Move:
            if strategy == StrategyType.ALWAYS_COOPERATE:
                return Move.COOPERATE
            if strategy == StrategyType.ALWAYS_DEFECT:
                return Move.DEFECT
            if strategy == StrategyType.TIT_FOR_TAT:
                return _tit_for_tat(their_history)
            if strategy == StrategyType.GENEROUS_TFT:
                return _generous_tft(their_history)
            if strategy == StrategyType.GRUDGER:
                return _grudger(their_history)
            if strategy == StrategyType.PAVLOV:
                return _pavlov(
                    my_history[-1] if my_history else None,
                    their_history[-1] if their_history else None,
                )
            # RANDOM / UNKNOWN: alternate with hash for determinism
            h = int(
                hashlib.md5(
                    f"{len(my_history)}".encode()
                ).hexdigest()[:8],
                16,
            )
            return Move.COOPERATE if h % 2 == 0 else Move.DEFECT

        # Clear existing interactions for clean simulation
        self.clear()

        for i, a in enumerate(agent_ids):
            for b in agent_ids[i + 1 :]:
                a_history: List[Move] = []
                b_history: List[Move] = []
                for _ in range(rounds):
                    ma = make_move(agents[a], a_history, b_history)
                    mb = make_move(agents[b], b_history, a_history)
                    self.record_interaction(a, b, ma, mb)
                    a_history.append(ma)
                    b_history.append(mb)

        return self.analyze()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli() -> None:
    """Command-line interface for game-theory analysis."""
    import argparse
    import json as json_mod

    parser = argparse.ArgumentParser(
        description="Agent Game-Theory Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--agents",
        type=int,
        default=4,
        help="Number of agents to simulate (default: 4)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=50,
        help="Rounds per pair (default: 50)",
    )
    parser.add_argument(
        "--strategy",
        choices=[s.value for s in StrategyType if s != StrategyType.UNKNOWN],
        default=None,
        help="Default strategy for all agents (default: mixed)",
    )
    parser.add_argument(
        "--game",
        choices=[g.value for g in GameType if g != GameType.UNKNOWN],
        default="prisoners_dilemma",
        help="Game type preset (default: prisoners_dilemma)",
    )
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    # Build payoffs
    game_payoffs = {
        "prisoners_dilemma": Payoffs(R=3, S=0, T=5, P=1),
        "stag_hunt": Payoffs(R=5, S=0, T=3, P=1),
        "chicken": Payoffs(R=3, S=1, T=5, P=0),
        "harmony": Payoffs(R=5, S=3, T=4, P=1),
    }
    payoffs = game_payoffs.get(args.game, Payoffs())

    config = GameConfig(payoffs=payoffs)
    analyzer = GameTheoryAnalyzer(config)

    # Assign strategies
    strategy_pool = [
        StrategyType.TIT_FOR_TAT,
        StrategyType.ALWAYS_DEFECT,
        StrategyType.ALWAYS_COOPERATE,
        StrategyType.GRUDGER,
        StrategyType.PAVLOV,
        StrategyType.GENEROUS_TFT,
        StrategyType.RANDOM,
    ]
    agents_map: Dict[str, StrategyType] = {}
    for i in range(args.agents):
        name = f"agent-{i + 1}"
        if args.strategy:
            agents_map[name] = StrategyType(args.strategy)
        else:
            agents_map[name] = strategy_pool[i % len(strategy_pool)]

    report = analyzer.simulate(agents_map, rounds=args.rounds, payoffs=payoffs)

    if args.json:
        print(json_mod.dumps(report.to_dict(), indent=2))
    else:
        print(report.render())


if __name__ == "__main__":
    _cli()
