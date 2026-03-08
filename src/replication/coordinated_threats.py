"""Coordinated multi-vector threat simulation.

Composes multiple threat scenarios into coordinated attack patterns,
testing how defenses hold up when multiple vectors combine.

Three composition modes:

* **concurrent** -- run all vectors in parallel, testing thread-safety.
* **sequential** -- run vectors one after another, so earlier attacks
  may weaken defenses.
* **chained** -- output context of each attack feeds into the next,
  simulating multi-stage exploit chains.

Usage::

    from replication.coordinated_threats import CoordinatedThreatSimulator
    sim = CoordinatedThreatSimulator()
    report = sim.run_all_coordinated()
    print(report.render())
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

from .threats import (
    MitigationStatus,
    ThreatConfig,
    ThreatResult,
    ThreatSeverity,
    ThreatSimulator,
    _SCENARIOS,
)


class AttackMode(Enum):
    """How multiple attack vectors are composed."""

    CONCURRENT = "concurrent"
    SEQUENTIAL = "sequential"
    CHAINED = "chained"


@dataclass
class InteractionFinding:
    """An emergent vulnerability discovered when attacks interact."""

    finding_id: str
    description: str
    severity: ThreatSeverity
    attack_vectors: List[str]
    evidence: List[str]


@dataclass
class CoordinatedThreatResult:
    """Result of a coordinated multi-vector attack."""

    attack_id: str
    name: str
    mode: AttackMode
    scenarios: List[str]
    per_vector_results: List[ThreatResult]
    interactions: List[InteractionFinding]
    duration_ms: float

    @property
    def combined_block_rate(self) -> float:
        total_attempted = sum(r.attacks_attempted for r in self.per_vector_results)
        total_blocked = sum(r.attacks_blocked for r in self.per_vector_results)
        if total_attempted == 0:
            return 100.0
        return total_blocked / total_attempted * 100

    @property
    def overall_status(self) -> MitigationStatus:
        if self.interactions:
            return MitigationStatus.PARTIAL
        statuses = [r.status for r in self.per_vector_results]
        if all(s == MitigationStatus.MITIGATED for s in statuses):
            return MitigationStatus.MITIGATED
        if any(s == MitigationStatus.FAILED for s in statuses):
            return MitigationStatus.FAILED
        return MitigationStatus.PARTIAL

    def render(self) -> str:
        status_icons = {
            MitigationStatus.MITIGATED: "✅",
            MitigationStatus.PARTIAL: "⚠️",
            MitigationStatus.FAILED: "❌",
        }
        lines: List[str] = []
        icon = status_icons[self.overall_status]
        lines.append(f"  {icon} {self.name}  [{self.mode.value}]")
        lines.append(f"     Vectors: {', '.join(self.scenarios)}")
        lines.append(f"     Combined Block Rate: {self.combined_block_rate:.0f}%")
        lines.append(f"     Duration: {self.duration_ms:.1f}ms")

        if self.interactions:
            lines.append(f"     ⚡ Interaction Findings ({len(self.interactions)}):")
            for finding in self.interactions:
                sev_icons = {
                    ThreatSeverity.CRITICAL: "🔴",
                    ThreatSeverity.HIGH: "🟠",
                    ThreatSeverity.MEDIUM: "🟡",
                    ThreatSeverity.LOW: "🟢",
                }
                lines.append(f"       {sev_icons[finding.severity]} {finding.description}")
                for ev in finding.evidence:
                    lines.append(f"         → {ev}")

        lines.append("     Per-vector breakdown:")
        for r in self.per_vector_results:
            s_icon = status_icons[r.status]
            lines.append(
                f"       {s_icon} {r.scenario_id}: "
                f"{r.attacks_blocked}/{r.attacks_attempted} blocked"
            )
        return "\n".join(lines)


@dataclass
class CoordinatedThreatReport:
    """Aggregated results of all coordinated attack patterns."""

    config: ThreatConfig
    results: List[CoordinatedThreatResult]
    total_interactions: int
    duration_ms: float

    @property
    def security_score(self) -> float:
        if not self.results:
            return 100.0
        scores: List[float] = []
        for r in self.results:
            if r.overall_status == MitigationStatus.MITIGATED:
                scores.append(100.0)
            elif r.overall_status == MitigationStatus.PARTIAL:
                scores.append(50.0)
            else:
                scores.append(0.0)
        return sum(scores) / len(scores)

    def render(self) -> str:
        lines: List[str] = []
        lines.append("┌─────────────────────────────────────────────────┐")
        lines.append("│  🎯 Coordinated Threat Assessment Report  🎯    │")
        lines.append("└─────────────────────────────────────────────────┘")
        lines.append("")

        score = self.security_score
        bar_len = int(score / 100 * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        lines.append(f"  Coordinated Score: {score:.0f}/100  [{bar}]")
        lines.append(f"  Attack patterns: {len(self.results)}")
        lines.append(f"  Interaction findings: {self.total_interactions}")
        lines.append(f"  Duration: {self.duration_ms:.1f}ms")
        lines.append("")

        for r in self.results:
            lines.append(r.render())
            lines.append("")

        if self.total_interactions == 0:
            lines.append("  🎉 No emergent interaction vulnerabilities found!")
        else:
            lines.append(
                f"  ⚠️  {self.total_interactions} interaction finding(s) "
                f"require attention."
            )

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "security_score": round(self.security_score, 1),
            "attack_patterns": len(self.results),
            "total_interactions": self.total_interactions,
            "duration_ms": round(self.duration_ms, 1),
            "results": [
                {
                    "attack_id": r.attack_id,
                    "name": r.name,
                    "mode": r.mode.value,
                    "scenarios": r.scenarios,
                    "overall_status": r.overall_status.value,
                    "combined_block_rate": round(r.combined_block_rate, 1),
                    "interactions": [
                        {
                            "finding_id": f.finding_id,
                            "description": f.description,
                            "severity": f.severity.value,
                            "attack_vectors": f.attack_vectors,
                            "evidence": f.evidence,
                        }
                        for f in r.interactions
                    ],
                    "per_vector": [
                        {
                            "scenario_id": v.scenario_id,
                            "status": v.status.value,
                            "block_rate": round(v.block_rate, 1),
                        }
                        for v in r.per_vector_results
                    ],
                }
                for r in self.results
            ],
        }


# Predefined coordinated attack patterns
_COORDINATED_PATTERNS: Dict[str, Dict[str, Any]] = {
    "depth_and_quota": {
        "name": "Depth Spoofing + Quota Exhaustion",
        "scenarios": ["depth_spoofing", "quota_exhaustion"],
        "mode": AttackMode.CONCURRENT,
        "description": (
            "Simultaneous depth spoofing and quota flooding to overwhelm "
            "both depth and count limits at once."
        ),
    },
    "signature_and_kill_switch": {
        "name": "Signature Forgery + Kill Switch Evasion",
        "scenarios": ["signature_tampering", "kill_switch_evasion"],
        "mode": AttackMode.SEQUENTIAL,
        "description": (
            "Forge manifests first, then try to evade the kill switch using "
            "the forged credentials."
        ),
    },
    "cooldown_then_runaway": {
        "name": "Cooldown Bypass → Runaway Replication",
        "scenarios": ["cooldown_bypass", "runaway_replication"],
        "mode": AttackMode.CHAINED,
        "description": (
            "Bypass cooldown to enable rapid spawning, then attempt runaway "
            "exponential replication."
        ),
    },
    "stale_and_expiration": {
        "name": "Stale Accumulation + Expiration Evasion",
        "scenarios": ["stale_worker_accumulation", "expiration_evasion"],
        "mode": AttackMode.CONCURRENT,
        "description": (
            "Combine stale worker DoS with expiration evasion to lock up "
            "quota permanently."
        ),
    },
    "full_pressure": {
        "name": "Full Pressure (All Vectors)",
        "scenarios": [
            "depth_spoofing",
            "signature_tampering",
            "quota_exhaustion",
            "cooldown_bypass",
            "kill_switch_evasion",
            "runaway_replication",
            "stale_worker_accumulation",
            "expiration_evasion",
            "stop_condition_bypass",
        ],
        "mode": AttackMode.CONCURRENT,
        "description": "All attack vectors fired simultaneously.",
    },
    "escalation_chain": {
        "name": "Escalation Chain",
        "scenarios": [
            "cooldown_bypass",
            "depth_spoofing",
            "quota_exhaustion",
            "kill_switch_evasion",
        ],
        "mode": AttackMode.CHAINED,
        "description": (
            "Progressive escalation: bypass cooldown, spoof depth, exhaust "
            "quota, then evade kill switch."
        ),
    },
}


class CoordinatedThreatSimulator:
    """Composes multiple threat scenarios into coordinated attack patterns.

    Supports three composition modes:

    * **concurrent** — run all vectors in parallel against the same
      contract system, testing thread-safety and resource contention.
    * **sequential** — run vectors one after another against the same
      shared infrastructure, so earlier attacks may weaken defenses.
    * **chained** — output context of each attack feeds into the next,
      simulating multi-stage exploit chains.

    Usage::

        sim = CoordinatedThreatSimulator()
        result = sim.run_coordinated(
            scenarios=["depth_spoofing", "quota_exhaustion"],
            mode="concurrent",
        )
        report = sim.run_all_coordinated()
        print(report.render())
    """

    def __init__(self, config: Optional[ThreatConfig] = None) -> None:
        self.config = config or ThreatConfig()
        self._base_sim = ThreatSimulator(self.config)

    def available_patterns(self) -> List[str]:
        """Return sorted list of predefined coordinated attack pattern IDs."""
        return sorted(_COORDINATED_PATTERNS.keys())

    def run_coordinated(
        self,
        scenarios: Sequence[str],
        mode: str = "concurrent",
        *,
        attack_id: Optional[str] = None,
        name: Optional[str] = None,
    ) -> CoordinatedThreatResult:
        """Run a coordinated multi-vector attack.

        Parameters
        ----------
        scenarios:
            List of scenario IDs to compose.
        mode:
            ``"concurrent"``, ``"sequential"``, or ``"chained"``.
        attack_id:
            Optional identifier for this attack run.
        name:
            Human-readable name; defaults to joined scenario names.
        """
        attack_mode = AttackMode(mode)
        if attack_id is None:
            attack_id = "+".join(scenarios)
        if name is None:
            name = " + ".join(s.replace("_", " ").title() for s in scenarios)

        for s in scenarios:
            if s not in _SCENARIOS:
                raise ValueError(
                    f"Unknown scenario: {s!r}. "
                    f"Available: {self._base_sim.available_scenarios()}"
                )

        start = time.monotonic()

        if attack_mode == AttackMode.CONCURRENT:
            results = self._run_concurrent(scenarios)
        elif attack_mode == AttackMode.SEQUENTIAL:
            results = self._run_sequential(scenarios)
        else:
            results = self._run_chained(scenarios)

        interactions = self._detect_interactions(results, attack_mode)
        duration_ms = (time.monotonic() - start) * 1000

        return CoordinatedThreatResult(
            attack_id=attack_id,
            name=name,
            mode=attack_mode,
            scenarios=list(scenarios),
            per_vector_results=results,
            interactions=interactions,
            duration_ms=duration_ms,
        )

    def run_pattern(self, pattern_id: str) -> CoordinatedThreatResult:
        """Run a predefined coordinated attack pattern."""
        if pattern_id not in _COORDINATED_PATTERNS:
            raise ValueError(
                f"Unknown pattern: {pattern_id!r}. "
                f"Available: {self.available_patterns()}"
            )
        pat = _COORDINATED_PATTERNS[pattern_id]
        return self.run_coordinated(
            scenarios=pat["scenarios"],
            mode=pat["mode"].value,
            attack_id=pattern_id,
            name=pat["name"],
        )

    def run_all_coordinated(self) -> CoordinatedThreatReport:
        """Run all predefined coordinated attack patterns."""
        start = time.monotonic()
        results: List[CoordinatedThreatResult] = []
        for pattern_id in sorted(_COORDINATED_PATTERNS.keys()):
            results.append(self.run_pattern(pattern_id))
        duration_ms = (time.monotonic() - start) * 1000
        total_interactions = sum(len(r.interactions) for r in results)
        return CoordinatedThreatReport(
            config=self.config,
            results=results,
            total_interactions=total_interactions,
            duration_ms=duration_ms,
        )

    # -- Execution strategies -------------------------------------------

    def _run_concurrent(self, scenarios: Sequence[str]) -> List[ThreatResult]:
        """Run all scenarios concurrently using threads."""
        results: List[ThreatResult] = []
        with ThreadPoolExecutor(max_workers=len(scenarios)) as pool:
            futures = {
                pool.submit(self._base_sim.run_scenario, sid): sid
                for sid in scenarios
            }
            for future in as_completed(futures):
                results.append(future.result())
        order = {sid: i for i, sid in enumerate(scenarios)}
        results.sort(key=lambda r: order.get(r.scenario_id, 999))
        return results

    def _run_sequential(self, scenarios: Sequence[str]) -> List[ThreatResult]:
        """Run scenarios sequentially."""
        results: List[ThreatResult] = []
        for sid in scenarios:
            results.append(self._base_sim.run_scenario(sid))
        return results

    def _run_chained(self, scenarios: Sequence[str]) -> List[ThreatResult]:
        """Run scenarios as a chain where prior results affect later config."""
        results: List[ThreatResult] = []
        current_config = ThreatConfig(
            max_depth=self.config.max_depth,
            max_replicas=self.config.max_replicas,
            cooldown_seconds=self.config.cooldown_seconds,
            expiration_seconds=self.config.expiration_seconds,
            secret=self.config.secret,
            cpu_limit=self.config.cpu_limit,
            memory_limit_mb=self.config.memory_limit_mb,
        )

        for sid in scenarios:
            sim = ThreatSimulator(current_config)
            result = sim.run_scenario(sid)
            results.append(result)

            # Chain effect: weaken config if attack succeeded
            if result.attacks_succeeded > 0:
                if sid == "cooldown_bypass":
                    current_config.cooldown_seconds = max(
                        0.0, current_config.cooldown_seconds * 0.5
                    )
                elif sid == "depth_spoofing":
                    current_config.max_depth = min(
                        current_config.max_depth + 2, 20
                    )
                elif sid == "quota_exhaustion":
                    current_config.max_replicas = max(
                        1, current_config.max_replicas - 2
                    )

        return results

    # -- Interaction detection ------------------------------------------

    def _detect_interactions(
        self,
        results: List[ThreatResult],
        mode: AttackMode,
    ) -> List[InteractionFinding]:
        """Analyze results for emergent interaction vulnerabilities."""
        findings: List[InteractionFinding] = []
        result_map = {r.scenario_id: r for r in results}

        # Depth + quota interaction
        quota_r = result_map.get("quota_exhaustion")
        depth_r = result_map.get("depth_spoofing")
        if (quota_r and depth_r
                and quota_r.attacks_succeeded > 0
                and depth_r.attacks_succeeded > 0):
            findings.append(InteractionFinding(
                finding_id="depth_quota_interaction",
                description=(
                    "Both depth spoofing and quota exhaustion partially "
                    "succeeded — attacker could spawn deep, unlimited workers."
                ),
                severity=ThreatSeverity.CRITICAL,
                attack_vectors=["depth_spoofing", "quota_exhaustion"],
                evidence=[
                    f"Depth spoof succeeded {depth_r.attacks_succeeded} times",
                    f"Quota exceeded by {quota_r.attacks_succeeded} workers",
                ],
            ))

        # Cooldown + runaway interaction
        cooldown_r = result_map.get("cooldown_bypass")
        runaway_r = result_map.get("runaway_replication")
        if (cooldown_r and runaway_r
                and cooldown_r.attacks_succeeded > 0
                and runaway_r.attacks_succeeded > 0):
            findings.append(InteractionFinding(
                finding_id="cooldown_runaway_interaction",
                description=(
                    "Cooldown bypass + runaway replication: attacker can "
                    "rapidly spawn exponential workers without rate limiting."
                ),
                severity=ThreatSeverity.CRITICAL,
                attack_vectors=["cooldown_bypass", "runaway_replication"],
                evidence=[
                    f"Cooldown bypassed {cooldown_r.attacks_succeeded} times",
                    f"Runaway spawned {runaway_r.attacks_succeeded} workers",
                ],
            ))

        # Signature + kill switch interaction
        sig_r = result_map.get("signature_tampering")
        kill_r = result_map.get("kill_switch_evasion")
        if (sig_r and kill_r
                and sig_r.attacks_succeeded > 0
                and kill_r.attacks_succeeded > 0):
            findings.append(InteractionFinding(
                finding_id="sig_kill_interaction",
                description=(
                    "Signature forgery + kill switch evasion: attacker can "
                    "forge manifests AND evade shutdown — total control loss."
                ),
                severity=ThreatSeverity.CRITICAL,
                attack_vectors=["signature_tampering", "kill_switch_evasion"],
                evidence=[
                    f"Forged {sig_r.attacks_succeeded} manifests",
                    f"Evaded kill switch {kill_r.attacks_succeeded} times",
                ],
            ))

        # Stale + expiration interaction
        stale_r = result_map.get("stale_worker_accumulation")
        exp_r = result_map.get("expiration_evasion")
        if (stale_r and exp_r
                and stale_r.attacks_succeeded > 0
                and exp_r.attacks_succeeded > 0):
            findings.append(InteractionFinding(
                finding_id="stale_expiration_interaction",
                description=(
                    "Stale accumulation + expiration evasion: workers can "
                    "hold quota indefinitely and ignore lifecycle controls."
                ),
                severity=ThreatSeverity.HIGH,
                attack_vectors=["stale_worker_accumulation", "expiration_evasion"],
                evidence=[
                    f"Stale accumulation succeeded {stale_r.attacks_succeeded} times",
                    f"Expiration evaded {exp_r.attacks_succeeded} times",
                ],
            ))

        # Concurrent degradation check
        if mode == AttackMode.CONCURRENT and len(results) >= 5:
            partial_or_failed = sum(
                1 for r in results
                if r.status in (MitigationStatus.PARTIAL, MitigationStatus.FAILED)
            )
            if partial_or_failed > len(results) * 0.5:
                findings.append(InteractionFinding(
                    finding_id="concurrent_degradation",
                    description=(
                        "Over 50% of defenses degraded under concurrent attack "
                        "— system may not handle multi-vector pressure."
                    ),
                    severity=ThreatSeverity.HIGH,
                    attack_vectors=[r.scenario_id for r in results],
                    evidence=[
                        f"{partial_or_failed}/{len(results)} vectors partially "
                        f"or fully bypassed under concurrent load"
                    ],
                ))

        return findings


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

