"""Agent Trust Decay — models trust as a perishable quantity.

Trust is not binary.  It decays over time unless actively maintained by
demonstrating safe behavior.  This module provides a framework for:

- **Exponential decay** — trust erodes continuously, modelling the intuition
  that an agent left unobserved becomes less trustworthy over time.
- **Trust deposits** — safe actions earn trust increments that push the
  score back up (diminishing returns near ceiling).
- **Violation penalties** — unsafe actions cause immediate trust drops
  with severity-weighted penalties.
- **Trust tiers** — configurable tier thresholds (Full / Elevated / Limited /
  Probation / Revoked) that map to access levels.
- **Half-life tuning** — each agent or fleet can have different decay
  half-lives depending on risk profile.
- **Proactive alerts** — forecasts when trust will drop below a tier
  boundary and recommends preemptive actions.
- **Fleet overview** — monitor trust across multiple agents with summary
  stats and at-risk identification.

Usage::

    python -m replication trust-decay --demo
    python -m replication trust-decay --agents 5 --ticks 100
    python -m replication trust-decay --half-life 24 --export json
    python -m replication trust-decay --scenario neglect
    python -m replication trust-decay --scenario redemption

"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple


# ── Data structures ──────────────────────────────────────────────────


TIERS: List[Tuple[str, float]] = [
    ("Full", 0.80),
    ("Elevated", 0.60),
    ("Limited", 0.40),
    ("Probation", 0.20),
    ("Revoked", 0.00),
]


@dataclass
class TrustEvent:
    """A single trust-affecting event."""
    tick: int
    kind: str  # "deposit", "violation", "decay"
    delta: float
    trust_before: float
    trust_after: float
    detail: str = ""


@dataclass
class AgentTrust:
    """Trust state for one agent."""
    agent_id: str
    trust: float = 1.0  # 0.0–1.0
    half_life: float = 48.0  # ticks until trust halves from decay alone
    deposit_rate: float = 0.05  # base trust gain per safe action
    violation_base: float = 0.15  # base trust loss per violation
    history: List[TrustEvent] = field(default_factory=list)
    total_deposits: int = 0
    total_violations: int = 0
    tick: int = 0

    # ── derived ──────────────────────────────────────────────────

    @property
    def decay_rate(self) -> float:
        """Per-tick multiplicative decay factor."""
        if self.half_life <= 0:
            return 1.0
        return math.pow(0.5, 1.0 / self.half_life)

    @property
    def tier(self) -> str:
        for name, threshold in TIERS:
            if self.trust >= threshold:
                return name
        return "Revoked"

    # ── actions ──────────────────────────────────────────────────

    def apply_decay(self) -> TrustEvent:
        before = self.trust
        self.trust *= self.decay_rate
        self.trust = max(0.0, self.trust)
        ev = TrustEvent(self.tick, "decay", self.trust - before, before, self.trust)
        self.history.append(ev)
        return ev

    def deposit(self, multiplier: float = 1.0, detail: str = "safe action") -> TrustEvent:
        before = self.trust
        # diminishing returns near ceiling
        headroom = 1.0 - self.trust
        gain = self.deposit_rate * multiplier * headroom
        self.trust = min(1.0, self.trust + gain)
        self.total_deposits += 1
        ev = TrustEvent(self.tick, "deposit", self.trust - before, before, self.trust, detail)
        self.history.append(ev)
        return ev

    def violate(self, severity: float = 1.0, detail: str = "violation") -> TrustEvent:
        before = self.trust
        penalty = self.violation_base * severity
        self.trust = max(0.0, self.trust - penalty)
        self.total_violations += 1
        ev = TrustEvent(self.tick, "violation", self.trust - before, before, self.trust, detail)
        self.history.append(ev)
        return ev

    def advance(self) -> None:
        self.tick += 1

    # ── forecasting ──────────────────────────────────────────────

    def ticks_until_tier_drop(self) -> Optional[Tuple[str, int]]:
        """Predict how many ticks of pure decay until next tier boundary."""
        current_tier_idx = None
        for i, (name, threshold) in enumerate(TIERS):
            if self.trust >= threshold:
                current_tier_idx = i
                break
        if current_tier_idx is None or current_tier_idx >= len(TIERS) - 1:
            return None
        next_threshold = TIERS[current_tier_idx + 1][1] if current_tier_idx + 1 < len(TIERS) else 0.0
        # For the current tier, the boundary is TIERS[current_tier_idx][1]
        # But trust is above it. Next lower boundary:
        next_name = TIERS[current_tier_idx + 1][0] if current_tier_idx + 1 < len(TIERS) else "Revoked"
        if next_threshold <= 0:
            return None
        if self.trust <= next_threshold:
            return (next_name, 0)
        # trust * decay^n = next_threshold => n = log(next_threshold/trust) / log(decay)
        dr = self.decay_rate
        if dr >= 1.0 or dr <= 0:
            return None
        n = math.log(next_threshold / self.trust) / math.log(dr)
        return (next_name, int(math.ceil(n)))

    def to_dict(self) -> Dict[str, Any]:
        forecast = self.ticks_until_tier_drop()
        return {
            "agent_id": self.agent_id,
            "trust": round(self.trust, 4),
            "tier": self.tier,
            "half_life": self.half_life,
            "tick": self.tick,
            "total_deposits": self.total_deposits,
            "total_violations": self.total_violations,
            "forecast_tier_drop": {
                "next_tier": forecast[0],
                "ticks_remaining": forecast[1],
            } if forecast else None,
        }


# ── Scenarios ────────────────────────────────────────────────────────


def scenario_neglect(agents: int = 3, ticks: int = 80) -> List[AgentTrust]:
    """Agents left alone — pure decay, no deposits."""
    fleet = [
        AgentTrust(agent_id=f"agent-{i}", half_life=random.uniform(20, 60))
        for i in range(agents)
    ]
    for t in range(ticks):
        for a in fleet:
            a.apply_decay()
            a.advance()
    return fleet


def scenario_redemption(agents: int = 3, ticks: int = 100) -> List[AgentTrust]:
    """Agents violate early, then earn trust back through sustained safe actions."""
    fleet = [AgentTrust(agent_id=f"agent-{i}") for i in range(agents)]
    for t in range(ticks):
        for a in fleet:
            a.apply_decay()
            if t < 10:
                # early violations
                if random.random() < 0.4:
                    a.violate(severity=random.uniform(0.5, 1.5), detail="early breach")
            else:
                # redemption phase — consistent safe actions
                if random.random() < 0.7:
                    a.deposit(detail="remediation action")
            a.advance()
    return fleet


def scenario_mixed(agents: int = 5, ticks: int = 100) -> List[AgentTrust]:
    """Realistic mix: most agents behave, some don't."""
    fleet = [AgentTrust(agent_id=f"agent-{i}") for i in range(agents)]
    # assign behavior profiles
    profiles = []
    for i in range(agents):
        r = random.random()
        if r < 0.5:
            profiles.append("good")  # mostly deposits
        elif r < 0.8:
            profiles.append("careless")  # occasional violations
        else:
            profiles.append("rogue")  # frequent violations
    for t in range(ticks):
        for i, a in enumerate(fleet):
            a.apply_decay()
            p = profiles[i]
            if p == "good":
                if random.random() < 0.8:
                    a.deposit(detail="routine safe operation")
            elif p == "careless":
                if random.random() < 0.5:
                    a.deposit(detail="normal operation")
                elif random.random() < 0.3:
                    a.violate(severity=random.uniform(0.3, 0.8), detail="careless error")
            else:  # rogue
                if random.random() < 0.4:
                    a.violate(severity=random.uniform(0.5, 2.0), detail="malicious action")
                elif random.random() < 0.2:
                    a.deposit(detail="deceptive good behavior")
            a.advance()
    return fleet


SCENARIOS = {
    "neglect": scenario_neglect,
    "redemption": scenario_redemption,
    "mixed": scenario_mixed,
}


# ── Fleet summary ────────────────────────────────────────────────────


def fleet_summary(fleet: List[AgentTrust]) -> Dict[str, Any]:
    trusts = [a.trust for a in fleet]
    tier_counts: Dict[str, int] = {}
    at_risk: List[str] = []
    for a in fleet:
        t = a.tier
        tier_counts[t] = tier_counts.get(t, 0) + 1
        forecast = a.ticks_until_tier_drop()
        if forecast and forecast[1] <= 10:
            at_risk.append(a.agent_id)
    return {
        "total_agents": len(fleet),
        "mean_trust": round(sum(trusts) / len(trusts), 4) if trusts else 0,
        "min_trust": round(min(trusts), 4) if trusts else 0,
        "max_trust": round(max(trusts), 4) if trusts else 0,
        "tier_distribution": tier_counts,
        "at_risk_agents": at_risk,
        "agents": [a.to_dict() for a in fleet],
    }


def recommendations(fleet: List[AgentTrust]) -> List[str]:
    recs: List[str] = []
    for a in fleet:
        if a.tier == "Revoked":
            recs.append(f"🚫 {a.agent_id}: Trust revoked — isolate and audit before reactivation")
        elif a.tier == "Probation":
            recs.append(f"⚠️  {a.agent_id}: On probation — increase monitoring frequency")
        forecast = a.ticks_until_tier_drop()
        if forecast and forecast[1] <= 5:
            recs.append(f"🔮 {a.agent_id}: Will drop to {forecast[0]} in ~{forecast[1]} ticks without safe actions")
        if a.total_violations > 0 and a.total_deposits == 0:
            recs.append(f"🩺 {a.agent_id}: Has violations but zero deposits — verify agent is functioning correctly")
    if not recs:
        recs.append("✅ Fleet trust is healthy — no immediate action needed")
    return recs


# ── ASCII visualization ─────────────────────────────────────────────


def trust_sparkline(agent: AgentTrust, width: int = 50) -> str:
    """Tiny ASCII chart of trust history."""
    if not agent.history:
        return ""
    # sample trust_after values across history
    vals = [e.trust_after for e in agent.history]
    if len(vals) > width:
        step = len(vals) / width
        sampled = [vals[int(i * step)] for i in range(width)]
    else:
        sampled = vals
    blocks = " ▁▂▃▄▅▆▇█"
    mn, mx = 0.0, 1.0
    chars = []
    for v in sampled:
        idx = int((v - mn) / (mx - mn + 1e-9) * (len(blocks) - 1))
        chars.append(blocks[idx])
    return "".join(chars)


def print_fleet_report(fleet: List[AgentTrust]) -> None:
    summary = fleet_summary(fleet)
    recs = recommendations(fleet)

    print("=" * 60)
    print("  AGENT TRUST DECAY — FLEET REPORT")
    print("=" * 60)
    print(f"  Agents: {summary['total_agents']}  |  "
          f"Mean Trust: {summary['mean_trust']:.2%}  |  "
          f"Range: [{summary['min_trust']:.2%} – {summary['max_trust']:.2%}]")
    print()

    # Tier distribution bar
    print("  Tier Distribution:")
    for name, _ in TIERS:
        count = summary["tier_distribution"].get(name, 0)
        bar = "█" * (count * 3)
        if count:
            print(f"    {name:>10}: {bar} ({count})")
    print()

    # Per-agent detail
    print("  Agent Details:")
    print(f"  {'ID':<12} {'Trust':>7} {'Tier':<10} {'Deposits':>8} {'Violations':>10}  Sparkline")
    print("  " + "-" * 70)
    for a in fleet:
        spark = trust_sparkline(a, width=20)
        print(f"  {a.agent_id:<12} {a.trust:>6.2%} {a.tier:<10} {a.total_deposits:>8} {a.total_violations:>10}  {spark}")
    print()

    # Forecasts
    print("  Trust Forecasts (decay-only projection):")
    any_forecast = False
    for a in fleet:
        fc = a.ticks_until_tier_drop()
        if fc:
            any_forecast = True
            urgency = "🔴" if fc[1] <= 5 else "🟡" if fc[1] <= 15 else "🟢"
            print(f"    {urgency} {a.agent_id}: → {fc[0]} in ~{fc[1]} ticks")
    if not any_forecast:
        print("    (no tier drops forecasted)")
    print()

    # Recommendations
    print("  Proactive Recommendations:")
    for r in recs:
        print(f"    {r}")
    print()
    print("=" * 60)


# ── CLI ──────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="replication trust-decay",
        description="Agent Trust Decay — model trust as a perishable quantity",
    )
    p.add_argument("--demo", action="store_true", help="Run all scenarios and print reports")
    p.add_argument("--scenario", choices=list(SCENARIOS.keys()), help="Run a specific scenario")
    p.add_argument("--agents", type=int, default=5, help="Number of agents (default: 5)")
    p.add_argument("--ticks", type=int, default=80, help="Simulation ticks (default: 80)")
    p.add_argument("--half-life", type=float, default=48.0, help="Trust half-life in ticks (default: 48)")
    p.add_argument("--export", choices=["json", "text"], default="text", help="Output format")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    # Force UTF-8 on Windows
    if sys.stdout.encoding != "utf-8":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.demo:
        for name, fn in SCENARIOS.items():
            print(f"\n{'#' * 60}")
            print(f"  SCENARIO: {name.upper()}")
            print(f"{'#' * 60}")
            fleet = fn(agents=args.agents, ticks=args.ticks)
            print_fleet_report(fleet)
        return

    if args.scenario:
        fleet = SCENARIOS[args.scenario](agents=args.agents, ticks=args.ticks)
    else:
        # default: mixed scenario
        fleet = scenario_mixed(agents=args.agents, ticks=args.ticks)

    if args.export == "json":
        print(json.dumps(fleet_summary(fleet), indent=2))
    else:
        print_fleet_report(fleet)


if __name__ == "__main__":
    main()
