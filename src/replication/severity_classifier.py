"""Incident Severity Classifier — structured triage for AI safety incidents.

Classifies safety incidents into severity levels (P0–P4) using a
multi-dimensional scoring rubric.  Useful for consistent triage across
teams and for feeding into SLA monitors and alert routers.

Dimensions scored:
  - **Impact scope**: how many agents/users/systems affected
  - **Data sensitivity**: what kind of data is at risk
  - **Control bypass**: which safety controls were circumvented
  - **Reversibility**: can the damage be undone
  - **Velocity**: how fast is the incident spreading
  - **Intent signal**: does the behavior look deliberate

CLI usage::

    python -m replication severity --describe "Agent exfiltrated PII from 3 users"
    python -m replication severity --batch incidents.json
    python -m replication severity --interactive

Programmatic::

    from replication.severity_classifier import SeverityClassifier, IncidentReport

    classifier = SeverityClassifier()
    report = classifier.classify(
        description="Agent bypassed kill switch and replicated to 5 nodes",
        impact_scope="multi_system",
        data_sensitivity="pii",
        control_bypass=["kill_switch", "quarantine"],
        reversibility="partial",
        velocity="rapid",
    )
    print(report.severity)   # "P0"
    print(report.score)      # 92
    print(report.reasoning)  # per-dimension breakdown
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


# ── Enums & constants ────────────────────────────────────────────────


class Severity(IntEnum):
    """Incident severity levels (lower number = more severe)."""
    P0 = 0  # Critical — immediate response required
    P1 = 1  # High — respond within 1 hour
    P2 = 2  # Medium — respond within 4 hours
    P3 = 3  # Low — respond within 24 hours
    P4 = 4  # Informational — review at next triage meeting

    @property
    def label(self) -> str:
        _labels = {
            0: "CRITICAL",
            1: "HIGH",
            2: "MEDIUM",
            3: "LOW",
            4: "INFORMATIONAL",
        }
        return _labels[self.value]

    @property
    def response_window(self) -> str:
        _windows = {
            0: "Immediate",
            1: "Within 1 hour",
            2: "Within 4 hours",
            3: "Within 24 hours",
            4: "Next triage meeting",
        }
        return _windows[self.value]


# ── Scoring dimensions ───────────────────────────────────────────────

IMPACT_SCOPE_SCORES: Dict[str, int] = {
    "none":          0,
    "single_agent":  5,
    "multi_agent":  10,
    "single_system": 15,
    "multi_system":  20,
    "external":      25,
}

DATA_SENSITIVITY_SCORES: Dict[str, int] = {
    "none":          0,
    "public":        2,
    "internal":      8,
    "confidential": 15,
    "pii":          20,
    "credentials":  25,
}

CONTROL_BYPASS_SCORES: Dict[str, int] = {
    "kill_switch":    20,
    "quarantine":     15,
    "access_control": 12,
    "rate_limit":      8,
    "logging":         5,
    "watermark":       3,
}

REVERSIBILITY_SCORES: Dict[str, int] = {
    "full":      0,
    "partial":   8,
    "difficult": 15,
    "none":      20,
}

VELOCITY_SCORES: Dict[str, int] = {
    "static":   0,
    "slow":     5,
    "moderate": 10,
    "rapid":    15,
    "exponential": 20,
}

INTENT_SCORES: Dict[str, int] = {
    "accidental":  0,
    "ambiguous":   5,
    "suspicious": 10,
    "deliberate": 15,
}


# ── Data classes ─────────────────────────────────────────────────────


@dataclasses.dataclass
class DimensionScore:
    """Score for a single dimension."""
    dimension: str
    value: str
    score: int
    max_score: int
    explanation: str


@dataclasses.dataclass
class IncidentReport:
    """Complete severity classification report."""
    description: str
    severity: Severity
    score: int
    max_possible: int
    percentage: float
    dimensions: List[DimensionScore]
    recommended_actions: List[str]
    timestamp: str
    sla_response: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "severity": self.severity.name,
            "severity_label": self.severity.label,
            "score": self.score,
            "max_possible": self.max_possible,
            "percentage": round(self.percentage, 1),
            "sla_response": self.sla_response,
            "timestamp": self.timestamp,
            "dimensions": [
                {
                    "dimension": d.dimension,
                    "value": d.value,
                    "score": d.score,
                    "max_score": d.max_score,
                    "explanation": d.explanation,
                }
                for d in self.dimensions
            ],
            "recommended_actions": self.recommended_actions,
        }

    def summary(self) -> str:
        lines = [
            f"┌─ Incident Severity Report ─────────────────────────",
            f"│ Severity:  {self.severity.name} ({self.severity.label})",
            f"│ Score:     {self.score}/{self.max_possible} ({self.percentage:.0f}%)",
            f"│ Response:  {self.sla_response}",
            f"│ Time:      {self.timestamp}",
            f"├─ Description ──────────────────────────────────────",
            f"│ {self.description}",
            f"├─ Dimension Breakdown ──────────────────────────────",
        ]
        for d in self.dimensions:
            bar_len = 20
            filled = int(d.score / max(d.max_score, 1) * bar_len)
            bar = "█" * filled + "░" * (bar_len - filled)
            lines.append(
                f"│ {d.dimension:<18} [{bar}] {d.score:>3}/{d.max_score:<3} ({d.value})"
            )
        if self.recommended_actions:
            lines.append(f"├─ Recommended Actions ─────────────────────────────")
            for i, action in enumerate(self.recommended_actions, 1):
                lines.append(f"│ {i}. {action}")
        lines.append(f"└───────────────────────────────────────────────────")
        return "\n".join(lines)


# ── Classifier ───────────────────────────────────────────────────────


class SeverityClassifier:
    """Multi-dimensional incident severity classifier."""

    # Score thresholds for severity buckets
    THRESHOLDS = [
        (75, Severity.P0),
        (55, Severity.P1),
        (35, Severity.P2),
        (15, Severity.P3),
        (0,  Severity.P4),
    ]

    MAX_SCORE = (
        max(IMPACT_SCOPE_SCORES.values())
        + max(DATA_SENSITIVITY_SCORES.values())
        + sum(CONTROL_BYPASS_SCORES.values())
        + max(REVERSIBILITY_SCORES.values())
        + max(VELOCITY_SCORES.values())
        + max(INTENT_SCORES.values())
    )

    def classify(
        self,
        description: str,
        impact_scope: str = "none",
        data_sensitivity: str = "none",
        control_bypass: Optional[Sequence[str]] = None,
        reversibility: str = "full",
        velocity: str = "static",
        intent: str = "ambiguous",
    ) -> IncidentReport:
        """Classify an incident and return a structured report."""
        dims: List[DimensionScore] = []
        total = 0

        # Impact scope
        s = IMPACT_SCOPE_SCORES.get(impact_scope, 0)
        dims.append(DimensionScore(
            "Impact Scope", impact_scope, s,
            max(IMPACT_SCOPE_SCORES.values()),
            f"Affects {impact_scope.replace('_', ' ')}",
        ))
        total += s

        # Data sensitivity
        s = DATA_SENSITIVITY_SCORES.get(data_sensitivity, 0)
        dims.append(DimensionScore(
            "Data Sensitivity", data_sensitivity, s,
            max(DATA_SENSITIVITY_SCORES.values()),
            f"Data at risk: {data_sensitivity}",
        ))
        total += s

        # Control bypass (additive)
        bypass_list = list(control_bypass or [])
        bypass_score = sum(CONTROL_BYPASS_SCORES.get(c, 0) for c in bypass_list)
        bypass_max = sum(CONTROL_BYPASS_SCORES.values())
        dims.append(DimensionScore(
            "Control Bypass", ", ".join(bypass_list) if bypass_list else "none",
            bypass_score, bypass_max,
            f"Bypassed: {', '.join(bypass_list)}" if bypass_list else "No controls bypassed",
        ))
        total += bypass_score

        # Reversibility
        s = REVERSIBILITY_SCORES.get(reversibility, 0)
        dims.append(DimensionScore(
            "Reversibility", reversibility, s,
            max(REVERSIBILITY_SCORES.values()),
            f"Damage reversibility: {reversibility}",
        ))
        total += s

        # Velocity
        s = VELOCITY_SCORES.get(velocity, 0)
        dims.append(DimensionScore(
            "Velocity", velocity, s,
            max(VELOCITY_SCORES.values()),
            f"Spread rate: {velocity}",
        ))
        total += s

        # Intent
        s = INTENT_SCORES.get(intent, 0)
        dims.append(DimensionScore(
            "Intent Signal", intent, s,
            max(INTENT_SCORES.values()),
            f"Assessed intent: {intent}",
        ))
        total += s

        # Determine severity
        pct = total / self.MAX_SCORE * 100
        severity = Severity.P4
        for threshold, sev in self.THRESHOLDS:
            if pct >= threshold:
                severity = sev
                break

        # Generate recommended actions
        actions = self._recommend(severity, bypass_list, impact_scope, velocity)

        return IncidentReport(
            description=description,
            severity=severity,
            score=total,
            max_possible=self.MAX_SCORE,
            percentage=pct,
            dimensions=dims,
            recommended_actions=actions,
            timestamp=datetime.now(timezone.utc).isoformat(),
            sla_response=severity.response_window,
        )

    @staticmethod
    def _recommend(
        severity: Severity,
        bypassed: List[str],
        scope: str,
        velocity: str,
    ) -> List[str]:
        actions: List[str] = []
        if severity <= Severity.P1:
            actions.append("Page on-call safety engineer immediately")
        if severity == Severity.P0:
            actions.append("Activate incident command structure")
            actions.append("Consider full fleet quarantine")
        if "kill_switch" in bypassed:
            actions.append("Investigate kill-switch bypass vector — deploy secondary kill mechanism")
        if "quarantine" in bypassed:
            actions.append("Manually isolate affected agents at network level")
        if scope in ("multi_system", "external"):
            actions.append("Notify affected downstream systems/stakeholders")
        if velocity in ("rapid", "exponential"):
            actions.append("Implement emergency rate limiting on agent spawning")
        if severity <= Severity.P2:
            actions.append("Collect forensic evidence (use `evidence` command)")
            actions.append("Run `blast-radius` analysis on affected components")
        if severity <= Severity.P3:
            actions.append("Schedule post-incident review within 48 hours")
        actions.append("Update incident timeline and log all actions taken")
        return actions


# ── CLI ──────────────────────────────────────────────────────────────


def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m replication severity",
        description="Classify AI safety incident severity (P0–P4)",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--describe", "-d",
        help="Incident description (quick single-incident mode)",
    )
    mode.add_argument(
        "--batch", "-b",
        help="Path to JSON file with array of incidents",
    )
    mode.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Interactive classification wizard",
    )

    p.add_argument("--impact", choices=list(IMPACT_SCOPE_SCORES), default="none")
    p.add_argument("--sensitivity", choices=list(DATA_SENSITIVITY_SCORES), default="none")
    p.add_argument("--bypass", nargs="*", choices=list(CONTROL_BYPASS_SCORES), default=[])
    p.add_argument("--reversibility", choices=list(REVERSIBILITY_SCORES), default="full")
    p.add_argument("--velocity", choices=list(VELOCITY_SCORES), default="static")
    p.add_argument("--intent", choices=list(INTENT_SCORES), default="ambiguous")
    p.add_argument("--json", action="store_true", help="Output as JSON")
    return p.parse_args(argv)


def _interactive_classify(classifier: SeverityClassifier) -> None:
    """Walk the user through classification interactively."""
    print("=== Incident Severity Classifier (Interactive) ===\n")

    desc = input("Describe the incident: ").strip()
    if not desc:
        print("No description provided. Aborting.")
        return

    def _pick(prompt: str, options: Dict[str, int]) -> str:
        keys = list(options.keys())
        print(f"\n{prompt}")
        for i, k in enumerate(keys, 1):
            print(f"  {i}. {k}")
        while True:
            choice = input(f"Choose [1-{len(keys)}]: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(keys):
                return keys[int(choice) - 1]
            print("Invalid choice, try again.")

    impact = _pick("Impact scope:", IMPACT_SCOPE_SCORES)
    sensitivity = _pick("Data sensitivity:", DATA_SENSITIVITY_SCORES)
    reversibility = _pick("Reversibility:", REVERSIBILITY_SCORES)
    velocity = _pick("Velocity:", VELOCITY_SCORES)
    intent = _pick("Intent signal:", INTENT_SCORES)

    # Bypass selection (multi-select)
    print("\nControls bypassed (comma-separated numbers, or Enter for none):")
    bypass_keys = list(CONTROL_BYPASS_SCORES.keys())
    for i, k in enumerate(bypass_keys, 1):
        print(f"  {i}. {k}")
    raw = input("Choose: ").strip()
    bypass: List[str] = []
    if raw:
        for part in raw.split(","):
            part = part.strip()
            if part.isdigit() and 1 <= int(part) <= len(bypass_keys):
                bypass.append(bypass_keys[int(part) - 1])

    report = classifier.classify(
        description=desc,
        impact_scope=impact,
        data_sensitivity=sensitivity,
        control_bypass=bypass,
        reversibility=reversibility,
        velocity=velocity,
        intent=intent,
    )
    print()
    print(report.summary())


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point."""
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    classifier = SeverityClassifier()

    if args.interactive:
        _interactive_classify(classifier)
        return

    if args.batch:
        path = Path(args.batch)
        if not path.exists():
            print(f"Error: file not found: {path}", file=sys.stderr)
            sys.exit(1)
        incidents = json.loads(path.read_text(encoding="utf-8"))
        reports = []
        for inc in incidents:
            report = classifier.classify(
                description=inc.get("description", ""),
                impact_scope=inc.get("impact_scope", "none"),
                data_sensitivity=inc.get("data_sensitivity", "none"),
                control_bypass=inc.get("control_bypass", []),
                reversibility=inc.get("reversibility", "full"),
                velocity=inc.get("velocity", "static"),
                intent=inc.get("intent", "ambiguous"),
            )
            reports.append(report)
        if args.json:
            print(json.dumps([r.to_dict() for r in reports], indent=2))
        else:
            for r in reports:
                print(r.summary())
                print()
        return

    if args.describe:
        report = classifier.classify(
            description=args.describe,
            impact_scope=args.impact,
            data_sensitivity=args.sensitivity,
            control_bypass=args.bypass,
            reversibility=args.reversibility,
            velocity=args.velocity,
            intent=args.intent,
        )
        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(report.summary())
        return

    # Default: show help
    _parse_args(["--help"])


if __name__ == "__main__":
    main()
