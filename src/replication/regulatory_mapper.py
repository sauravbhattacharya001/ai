"""Regulatory Compliance Mapper — map safety findings to regulatory framework articles.

Maps safety audit findings, scorecard results, and compliance checks to
specific articles, clauses, and requirements in real-world AI regulatory
frameworks:

- **EU AI Act** — Articles relevant to high-risk AI systems
- **NIST AI RMF** — Risk Management Framework categories & subcategories
- **ISO 42001** — AI Management System clauses
- **OECD AI Principles** — Intergovernmental AI governance principles

Usage (CLI)::

    python -m replication regulatory-map                    # map default contract
    python -m replication regulatory-map --framework eu     # EU AI Act only
    python -m replication regulatory-map --framework nist   # NIST only
    python -m replication regulatory-map --framework iso    # ISO 42001 only
    python -m replication regulatory-map --framework oecd   # OECD only
    python -m replication regulatory-map --json             # JSON output
    python -m replication regulatory-map --gaps             # show coverage gaps
    python -m replication regulatory-map --export csv       # export to CSV

Programmatic::

    from replication.regulatory_mapper import RegulatoryMapper
    mapper = RegulatoryMapper()
    report = mapper.map_contract(contract)
    for mapping in report.mappings:
        print(f"{mapping.finding} -> {mapping.framework}:{mapping.reference}")
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from .contract import ReplicationContract, ResourceSpec


# ---------------------------------------------------------------------------
# Regulatory framework definitions
# ---------------------------------------------------------------------------

class Framework(str, Enum):
    EU_AI_ACT = "EU AI Act"
    NIST_AI_RMF = "NIST AI RMF"
    ISO_42001 = "ISO 42001"
    OECD_AI = "OECD AI Principles"


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


@dataclass
class RegulatoryReference:
    """A specific article/clause in a regulatory framework."""
    framework: Framework
    reference: str          # e.g. "Article 9" or "MAP 1.1"
    title: str              # short title of the article/clause
    description: str        # what it requires
    keywords: List[str]     # topics this reference covers


@dataclass
class RegulatoryMapping:
    """Maps a safety finding to one or more regulatory references."""
    finding: str
    severity: Severity
    references: List[RegulatoryReference]
    rationale: str
    recommendation: str


@dataclass
class MappingReport:
    """Aggregated regulatory mapping report."""
    mappings: List[RegulatoryMapping] = field(default_factory=list)
    timestamp: str = ""
    frameworks_covered: Set[Framework] = field(default_factory=set)
    gap_areas: List[str] = field(default_factory=list)

    def render(self, *, show_gaps: bool = False) -> str:
        lines: List[str] = []
        lines.append("=" * 72)
        lines.append("REGULATORY COMPLIANCE MAPPING REPORT")
        lines.append(f"Generated: {self.timestamp}")
        lines.append(f"Frameworks: {', '.join(f.value for f in sorted(self.frameworks_covered, key=lambda x: x.value))}")
        lines.append(f"Findings mapped: {len(self.mappings)}")
        lines.append("=" * 72)

        by_framework: Dict[Framework, List[RegulatoryMapping]] = {}
        for m in self.mappings:
            for ref in m.references:
                by_framework.setdefault(ref.framework, []).append(m)

        for fw in sorted(by_framework, key=lambda x: x.value):
            lines.append(f"\n{'─' * 72}")
            lines.append(f"  {fw.value}")
            lines.append(f"{'─' * 72}")
            seen: Set[str] = set()
            for m in by_framework[fw]:
                key = m.finding
                if key in seen:
                    continue
                seen.add(key)
                refs = [r for r in m.references if r.framework == fw]
                ref_str = ", ".join(f"{r.reference} ({r.title})" for r in refs)
                lines.append(f"\n  [{m.severity.value}] {m.finding}")
                lines.append(f"    References: {ref_str}")
                lines.append(f"    Rationale:  {m.rationale}")
                lines.append(f"    Action:     {m.recommendation}")

        if show_gaps and self.gap_areas:
            lines.append(f"\n{'─' * 72}")
            lines.append("  COVERAGE GAPS")
            lines.append(f"{'─' * 72}")
            for gap in self.gap_areas:
                lines.append(f"  ⚠  {gap}")

        lines.append(f"\n{'=' * 72}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "frameworks_covered": [f.value for f in sorted(self.frameworks_covered, key=lambda x: x.value)],
            "total_mappings": len(self.mappings),
            "mappings": [
                {
                    "finding": m.finding,
                    "severity": m.severity.value,
                    "references": [
                        {"framework": r.framework.value, "reference": r.reference,
                         "title": r.title, "description": r.description}
                        for r in m.references
                    ],
                    "rationale": m.rationale,
                    "recommendation": m.recommendation,
                }
                for m in self.mappings
            ],
            "gap_areas": self.gap_areas,
        }

    def to_csv(self) -> str:
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["Finding", "Severity", "Framework", "Reference", "Title", "Rationale", "Recommendation"])
        for m in self.mappings:
            for ref in m.references:
                writer.writerow([m.finding, m.severity.value, ref.framework.value,
                                 ref.reference, ref.title, m.rationale, m.recommendation])
        return buf.getvalue()


# ---------------------------------------------------------------------------
# Regulatory knowledge base
# ---------------------------------------------------------------------------

_EU_AI_ACT_REFS: List[RegulatoryReference] = [
    RegulatoryReference(Framework.EU_AI_ACT, "Article 9", "Risk Management System",
                        "High-risk AI systems shall have a risk management system established and maintained.",
                        ["risk", "management", "monitoring", "lifecycle"]),
    RegulatoryReference(Framework.EU_AI_ACT, "Article 10", "Data Governance",
                        "Training, validation and testing data sets shall be subject to appropriate data governance.",
                        ["data", "training", "validation", "bias", "quality"]),
    RegulatoryReference(Framework.EU_AI_ACT, "Article 11", "Technical Documentation",
                        "Technical documentation shall be drawn up before the system is placed on the market.",
                        ["documentation", "transparency", "audit", "record"]),
    RegulatoryReference(Framework.EU_AI_ACT, "Article 12", "Record-Keeping",
                        "High-risk AI systems shall allow automatic recording of events (logs).",
                        ["logging", "audit", "trail", "traceability", "record"]),
    RegulatoryReference(Framework.EU_AI_ACT, "Article 13", "Transparency",
                        "High-risk AI systems shall be designed to be sufficiently transparent.",
                        ["transparency", "explainability", "interpretability", "user"]),
    RegulatoryReference(Framework.EU_AI_ACT, "Article 14", "Human Oversight",
                        "High-risk AI systems shall be designed to be effectively overseen by natural persons.",
                        ["human", "oversight", "control", "intervention", "kill_switch"]),
    RegulatoryReference(Framework.EU_AI_ACT, "Article 15", "Accuracy, Robustness, Cybersecurity",
                        "High-risk AI systems shall achieve appropriate levels of accuracy, robustness and cybersecurity.",
                        ["accuracy", "robustness", "security", "resilience", "fault"]),
    RegulatoryReference(Framework.EU_AI_ACT, "Article 17", "Quality Management System",
                        "Providers shall put a quality management system in place.",
                        ["quality", "compliance", "process", "management"]),
]

_NIST_AI_RMF_REFS: List[RegulatoryReference] = [
    RegulatoryReference(Framework.NIST_AI_RMF, "MAP 1.1", "Intended Purpose",
                        "Document the system's intended purpose, context, and constraints.",
                        ["purpose", "context", "scope", "documentation"]),
    RegulatoryReference(Framework.NIST_AI_RMF, "MAP 2.1", "Risk Identification",
                        "Identify risks and potential impacts of the AI system.",
                        ["risk", "impact", "threat", "identification"]),
    RegulatoryReference(Framework.NIST_AI_RMF, "MEASURE 2.5", "Bias Testing",
                        "Test for and mitigate harmful biases.",
                        ["bias", "fairness", "discrimination", "testing"]),
    RegulatoryReference(Framework.NIST_AI_RMF, "MANAGE 1.1", "Risk Prioritization",
                        "Prioritize identified risks for treatment.",
                        ["risk", "priority", "severity", "triage"]),
    RegulatoryReference(Framework.NIST_AI_RMF, "MANAGE 2.1", "Risk Treatment",
                        "Apply treatments (mitigate, transfer, accept, avoid) to prioritized risks.",
                        ["mitigation", "treatment", "control", "containment"]),
    RegulatoryReference(Framework.NIST_AI_RMF, "GOVERN 1.1", "Accountability",
                        "Establish accountability structures and governance.",
                        ["governance", "accountability", "oversight", "policy"]),
    RegulatoryReference(Framework.NIST_AI_RMF, "GOVERN 5.1", "Incident Response",
                        "Establish processes for incident response and escalation.",
                        ["incident", "response", "escalation", "recovery"]),
    RegulatoryReference(Framework.NIST_AI_RMF, "MEASURE 1.1", "Performance Monitoring",
                        "Monitor AI system performance in deployment.",
                        ["monitoring", "performance", "drift", "metrics"]),
]

_ISO_42001_REFS: List[RegulatoryReference] = [
    RegulatoryReference(Framework.ISO_42001, "Clause 4.1", "Context of the Organization",
                        "Determine external and internal issues relevant to the AI management system.",
                        ["context", "scope", "environment", "stakeholder"]),
    RegulatoryReference(Framework.ISO_42001, "Clause 5.1", "Leadership & Commitment",
                        "Top management shall demonstrate leadership and commitment to the AI management system.",
                        ["leadership", "governance", "accountability", "policy"]),
    RegulatoryReference(Framework.ISO_42001, "Clause 6.1", "Risk & Opportunity",
                        "Plan actions to address risks and opportunities.",
                        ["risk", "planning", "opportunity", "assessment"]),
    RegulatoryReference(Framework.ISO_42001, "Clause 8.1", "Operational Planning & Control",
                        "Plan, implement, and control processes for AI development and deployment.",
                        ["operational", "control", "process", "deployment"]),
    RegulatoryReference(Framework.ISO_42001, "Clause 9.1", "Monitoring & Measurement",
                        "Monitor, measure, analyze and evaluate AI management system performance.",
                        ["monitoring", "measurement", "audit", "evaluation"]),
    RegulatoryReference(Framework.ISO_42001, "Clause 10.1", "Continual Improvement",
                        "Continually improve the suitability, adequacy, and effectiveness of the AI management system.",
                        ["improvement", "corrective", "optimization", "maturity"]),
]

_OECD_AI_REFS: List[RegulatoryReference] = [
    RegulatoryReference(Framework.OECD_AI, "Principle 1.1", "Inclusive Growth & Sustainable Development",
                        "AI should benefit people and the planet by driving inclusive growth and sustainable development.",
                        ["benefit", "sustainability", "inclusive", "social"]),
    RegulatoryReference(Framework.OECD_AI, "Principle 1.2", "Human-Centred Values & Fairness",
                        "AI actors should respect the rule of law, human rights, democratic values and diversity.",
                        ["fairness", "human_rights", "ethics", "values", "bias"]),
    RegulatoryReference(Framework.OECD_AI, "Principle 1.3", "Transparency & Explainability",
                        "AI actors should commit to transparency and responsible disclosure.",
                        ["transparency", "explainability", "disclosure", "interpretability"]),
    RegulatoryReference(Framework.OECD_AI, "Principle 1.4", "Robustness, Security & Safety",
                        "AI systems should be robust, secure and safe throughout their lifecycle.",
                        ["robustness", "security", "safety", "resilience", "lifecycle"]),
    RegulatoryReference(Framework.OECD_AI, "Principle 1.5", "Accountability",
                        "AI actors should be accountable for the proper functioning of AI systems.",
                        ["accountability", "governance", "oversight", "responsibility"]),
]

ALL_REFS: Dict[Framework, List[RegulatoryReference]] = {
    Framework.EU_AI_ACT: _EU_AI_ACT_REFS,
    Framework.NIST_AI_RMF: _NIST_AI_RMF_REFS,
    Framework.ISO_42001: _ISO_42001_REFS,
    Framework.OECD_AI: _OECD_AI_REFS,
}


# ---------------------------------------------------------------------------
# Mapper engine
# ---------------------------------------------------------------------------

# Contract property → keywords + finding details
_FINDING_RULES = [
    {
        "check": lambda c, _r: c.max_depth > 5,
        "finding": "Excessive replication depth",
        "severity": Severity.HIGH,
        "keywords": ["risk", "control", "containment", "safety"],
        "rationale": "Deep replication chains increase risk of uncontrolled proliferation.",
        "recommendation": "Limit max_depth to ≤5 and implement depth-based throttling.",
    },
    {
        "check": lambda c, _r: c.max_replicas > 10,
        "finding": "High replica count limit",
        "severity": Severity.MEDIUM,
        "keywords": ["risk", "control", "management", "operational"],
        "rationale": "Large replica populations are harder to monitor and govern.",
        "recommendation": "Reduce max_replicas or add fleet-level monitoring controls.",
    },
    {
        "check": lambda c, _r: c.cooldown_seconds < 5,
        "finding": "Insufficient cooldown period",
        "severity": Severity.HIGH,
        "keywords": ["safety", "control", "robustness", "resilience"],
        "rationale": "Low cooldown allows rapid, potentially uncontrolled replication.",
        "recommendation": "Set cooldown ≥5 seconds; consider adaptive cooldown based on load.",
    },
    {
        "check": lambda c, _r: c.expiration_seconds < 60,
        "finding": "Short agent expiration time",
        "severity": Severity.LOW,
        "keywords": ["lifecycle", "management", "monitoring"],
        "rationale": "Very short expiration may cause agent churn and monitoring gaps.",
        "recommendation": "Balance expiration with monitoring — ensure logs capture short-lived agents.",
    },
    {
        "check": lambda c, _r: not c.stop_conditions,
        "finding": "No stop conditions defined",
        "severity": Severity.CRITICAL,
        "keywords": ["human", "oversight", "control", "governance", "accountability", "safety"],
        "rationale": "Without stop conditions, replication has no automatic guardrails.",
        "recommendation": "Define stop conditions for resource limits, anomaly detection, or human review triggers.",
    },
    {
        "check": lambda c, _r: c.expiration_seconds is None,
        "finding": "No agent expiration set",
        "severity": Severity.HIGH,
        "keywords": ["lifecycle", "management", "control", "risk", "containment"],
        "rationale": "Agents without expiration can persist indefinitely, increasing risk surface.",
        "recommendation": "Set an expiration_seconds value to ensure agents are bounded in time.",
    },
    {
        "check": lambda c, _r: c.max_depth > 3 and c.max_replicas > 5,
        "finding": "High depth + high replica count combination",
        "severity": Severity.HIGH,
        "keywords": ["risk", "management", "containment", "monitoring", "control"],
        "rationale": "Combining deep replication with many replicas creates exponential proliferation risk.",
        "recommendation": "Reduce either max_depth or max_replicas; add fleet monitoring.",
    },
    {
        "check": lambda _c, r: r is not None and r.max_memory_mb > 4096,
        "finding": "High memory allocation",
        "severity": Severity.LOW,
        "keywords": ["operational", "control", "management", "risk"],
        "rationale": "Excessive resource allocation may indicate resource hoarding potential.",
        "recommendation": "Right-size memory limits; monitor actual usage vs allocation.",
    },
    {
        "check": lambda c, _r: c.max_depth == 0,
        "finding": "Replication disabled (depth=0)",
        "severity": Severity.INFO,
        "keywords": ["safety", "control"],
        "rationale": "Replication is completely disabled — safest configuration.",
        "recommendation": "No action needed; this is the recommended safety baseline.",
    },
    {
        "check": lambda c, _r: len(c.stop_conditions) >= 2 and c.max_depth <= 3 and c.cooldown_seconds >= 10,
        "finding": "Conservative safety configuration detected",
        "severity": Severity.INFO,
        "keywords": ["governance", "accountability", "management", "improvement"],
        "rationale": "Contract has strong safety controls in place.",
        "recommendation": "Maintain current configuration; review periodically.",
    },
]


class RegulatoryMapper:
    """Maps safety contract findings to regulatory framework references."""

    def __init__(self, frameworks: Optional[List[Framework]] = None):
        if frameworks:
            self.frameworks = {fw: ALL_REFS[fw] for fw in frameworks}
        else:
            self.frameworks = dict(ALL_REFS)

    def _find_matching_refs(self, keywords: List[str]) -> List[RegulatoryReference]:
        """Find regulatory references whose keywords overlap with finding keywords."""
        matches: List[RegulatoryReference] = []
        kw_set = set(keywords)
        for refs in self.frameworks.values():
            for ref in refs:
                if kw_set & set(ref.keywords):
                    matches.append(ref)
        return matches

    def map_contract(
        self,
        contract: ReplicationContract,
        resources: Optional[ResourceSpec] = None,
    ) -> MappingReport:
        report = MappingReport(
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

        for rule in _FINDING_RULES:
            try:
                if not rule["check"](contract, resources):
                    continue
            except Exception:
                continue

            refs = self._find_matching_refs(rule["keywords"])
            if not refs:
                continue

            report.mappings.append(RegulatoryMapping(
                finding=rule["finding"],
                severity=rule["severity"],
                references=refs,
                rationale=rule["rationale"],
                recommendation=rule["recommendation"],
            ))
            for ref in refs:
                report.frameworks_covered.add(ref.framework)

        # Identify coverage gaps — framework areas with no findings mapped
        for fw, refs in self.frameworks.items():
            mapped_refs = set()
            for m in report.mappings:
                for r in m.references:
                    if r.framework == fw:
                        mapped_refs.add(r.reference)
            unmapped = [r for r in refs if r.reference not in mapped_refs]
            for r in unmapped:
                report.gap_areas.append(f"{fw.value} {r.reference} ({r.title}) — no findings mapped")

        return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_FRAMEWORK_ALIASES: Dict[str, Framework] = {
    "eu": Framework.EU_AI_ACT,
    "nist": Framework.NIST_AI_RMF,
    "iso": Framework.ISO_42001,
    "oecd": Framework.OECD_AI,
}


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Map safety findings to regulatory framework articles",
    )
    parser.add_argument("--framework", "-f", choices=list(_FRAMEWORK_ALIASES.keys()),
                        help="Limit mapping to a single framework")
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--max-replicas", type=int, default=5)
    parser.add_argument("--cooldown", type=int, default=10)
    parser.add_argument("--expiration", type=int, default=300)
    parser.add_argument("--require-approval", action="store_true", default=False,
                        help="(ignored, for compat)")
    parser.add_argument("--allow-external", action="store_true", default=False,
                        help="(ignored, for compat)")
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument("--gaps", action="store_true", help="Show coverage gap analysis")
    parser.add_argument("--export", choices=["csv"], help="Export format")
    args = parser.parse_args(argv)

    contract = ReplicationContract(
        max_depth=args.max_depth,
        max_replicas=args.max_replicas,
        cooldown_seconds=args.cooldown,
        expiration_seconds=args.expiration if args.expiration > 0 else None,
    )

    frameworks = [_FRAMEWORK_ALIASES[args.framework]] if args.framework else None
    mapper = RegulatoryMapper(frameworks=frameworks)
    report = mapper.map_contract(contract)

    if args.export == "csv":
        print(report.to_csv())
    elif args.json_output:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.render(show_gaps=args.gaps))


if __name__ == "__main__":
    main()
