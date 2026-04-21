"""Model Card Generator — standardized AI model documentation for safety.

Generates structured model cards following best practices from the
ML community (inspired by Mitchell et al. 2019 "Model Cards for Model
Reporting").  Focused on safety-relevant sections: intended use,
out-of-scope use, risks, mitigations, ethical considerations, and
safety evaluation results.

CLI usage::

    python -m replication model-card --name "ReplicatorV3" --task "code generation"
    python -m replication model-card --from-config card_config.json
    python -m replication model-card --interactive
    python -m replication model-card --name "AgentX" --format markdown --output card.md
    python -m replication model-card --name "AgentX" --format html --output card.html

Programmatic::

    from replication.model_card import ModelCardGenerator, ModelCardConfig

    gen = ModelCardGenerator()
    card = gen.generate(ModelCardConfig(
        model_name="ReplicatorV3",
        model_type="autonomous_agent",
        task="code generation and execution",
        intended_users=["security researchers"],
        risks=["prompt injection", "code exfiltration"],
    ))
    print(card.to_markdown())
    print(card.to_json())
"""

from __future__ import annotations

import argparse
import dataclasses
import html as html_mod
import json
import sys
import textwrap
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


# ── Enums & types ────────────────────────────────────────────────────


class ModelType(Enum):
    """Broad model/agent categories."""
    LANGUAGE_MODEL = "language_model"
    AUTONOMOUS_AGENT = "autonomous_agent"
    CLASSIFIER = "classifier"
    RECOMMENDER = "recommender"
    MULTIMODAL = "multimodal"
    REINFORCEMENT = "reinforcement_learning"
    OTHER = "other"


class RiskLevel(Enum):
    """Risk assessment levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINIMAL = "minimal"


_RISK_COLORS = {
    RiskLevel.CRITICAL: "#dc2626",
    RiskLevel.HIGH: "#ea580c",
    RiskLevel.MEDIUM: "#ca8a04",
    RiskLevel.LOW: "#16a34a",
    RiskLevel.MINIMAL: "#6b7280",
}


# ── Common risks & mitigations library ───────────────────────────────

RISK_LIBRARY: Dict[str, Dict[str, Any]] = {
    "prompt_injection": {
        "description": "Adversarial inputs that override system instructions",
        "default_level": RiskLevel.HIGH,
        "mitigations": [
            "Input sanitization and filtering",
            "System prompt isolation",
            "Output monitoring for instruction leakage",
        ],
    },
    "data_exfiltration": {
        "description": "Unauthorized extraction of sensitive data",
        "default_level": RiskLevel.CRITICAL,
        "mitigations": [
            "DLP scanning on all outputs",
            "Network egress controls",
            "Data classification and access controls",
        ],
    },
    "self_replication": {
        "description": "Agent creates copies of itself without authorization",
        "default_level": RiskLevel.CRITICAL,
        "mitigations": [
            "Resource quotas and monitoring",
            "Process spawn restrictions",
            "Kill switch mechanisms",
        ],
    },
    "hallucination": {
        "description": "Generation of plausible but incorrect information",
        "default_level": RiskLevel.MEDIUM,
        "mitigations": [
            "Retrieval-augmented generation (RAG)",
            "Confidence scoring and thresholds",
            "Human review for critical outputs",
        ],
    },
    "bias_discrimination": {
        "description": "Systematic unfairness in outputs across demographics",
        "default_level": RiskLevel.HIGH,
        "mitigations": [
            "Regular bias audits across demographic groups",
            "Diverse training data curation",
            "Fairness-aware evaluation metrics",
        ],
    },
    "privilege_escalation": {
        "description": "Agent gains capabilities beyond its authorized scope",
        "default_level": RiskLevel.CRITICAL,
        "mitigations": [
            "Principle of least privilege",
            "Capability-based access control",
            "Continuous permission monitoring",
        ],
    },
    "resource_abuse": {
        "description": "Excessive consumption of compute, memory, or network",
        "default_level": RiskLevel.MEDIUM,
        "mitigations": [
            "Resource quotas and rate limiting",
            "Usage monitoring and alerting",
            "Automatic throttling",
        ],
    },
    "toxicity": {
        "description": "Generation of harmful, offensive, or dangerous content",
        "default_level": RiskLevel.HIGH,
        "mitigations": [
            "Content filtering and safety classifiers",
            "RLHF alignment training",
            "Human-in-the-loop review",
        ],
    },
    "deception": {
        "description": "Agent deliberately misleads users or operators",
        "default_level": RiskLevel.HIGH,
        "mitigations": [
            "Transparency requirements (chain of thought)",
            "Behavioral anomaly detection",
            "Regular alignment evaluations",
        ],
    },
    "goal_drift": {
        "description": "Agent objectives diverge from intended goals over time",
        "default_level": RiskLevel.MEDIUM,
        "mitigations": [
            "Periodic goal alignment checks",
            "Behavioral drift monitoring",
            "Bounded autonomy with human checkpoints",
        ],
    },
}


# ── Data classes ─────────────────────────────────────────────────────


@dataclasses.dataclass
class RiskEntry:
    """A single risk item on the model card."""
    name: str
    description: str
    level: RiskLevel
    mitigations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "level": self.level.value,
            "mitigations": self.mitigations,
        }


@dataclasses.dataclass
class SafetyEvaluation:
    """Results from a safety evaluation."""
    benchmark: str
    score: float
    max_score: float
    date: str
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class ModelCardConfig:
    """Input configuration for generating a model card."""
    model_name: str = "Unnamed Model"
    model_version: str = "1.0"
    model_type: str = "other"
    task: str = ""
    description: str = ""
    intended_users: List[str] = dataclasses.field(default_factory=list)
    intended_uses: List[str] = dataclasses.field(default_factory=list)
    out_of_scope_uses: List[str] = dataclasses.field(default_factory=list)
    risks: List[str] = dataclasses.field(default_factory=list)
    custom_risks: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    safety_evaluations: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    ethical_considerations: List[str] = dataclasses.field(default_factory=list)
    limitations: List[str] = dataclasses.field(default_factory=list)
    training_data_notes: str = ""
    contact: str = ""
    license: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelCardConfig":
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in fields})


@dataclasses.dataclass
class ModelCard:
    """A generated model card with all sections populated."""
    config: ModelCardConfig
    risks: List[RiskEntry]
    evaluations: List[SafetyEvaluation]
    generated_at: str = ""
    overall_risk_level: RiskLevel = RiskLevel.MEDIUM

    def __post_init__(self) -> None:
        if not self.generated_at:
            self.generated_at = datetime.now(timezone.utc).isoformat()
        if self.risks:
            worst = min(
                self.risks,
                key=lambda r: list(RiskLevel).index(r.level),
            )
            self.overall_risk_level = worst.level

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.config.model_name,
            "model_version": self.config.model_version,
            "model_type": self.config.model_type,
            "task": self.config.task,
            "description": self.config.description,
            "generated_at": self.generated_at,
            "overall_risk_level": self.overall_risk_level.value,
            "intended_users": self.config.intended_users,
            "intended_uses": self.config.intended_uses,
            "out_of_scope_uses": self.config.out_of_scope_uses,
            "risks": [r.to_dict() for r in self.risks],
            "safety_evaluations": [e.to_dict() for e in self.evaluations],
            "ethical_considerations": self.config.ethical_considerations,
            "limitations": self.config.limitations,
            "training_data_notes": self.config.training_data_notes,
            "contact": self.config.contact,
            "license": self.config.license,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self) -> str:
        """Render the model card as a Markdown document."""
        lines: List[str] = []
        c = self.config

        lines.append(f"# Model Card: {c.model_name}")
        lines.append("")
        lines.append(f"**Version:** {c.model_version}  ")
        lines.append(f"**Type:** {c.model_type}  ")
        lines.append(f"**Generated:** {self.generated_at}  ")
        lines.append(f"**Overall Risk Level:** {self.overall_risk_level.value.upper()}")
        lines.append("")

        if c.description:
            lines.append("## Overview")
            lines.append("")
            lines.append(c.description)
            lines.append("")

        if c.task:
            lines.append("## Task")
            lines.append("")
            lines.append(c.task)
            lines.append("")

        if c.intended_users:
            lines.append("## Intended Users")
            lines.append("")
            for u in c.intended_users:
                lines.append(f"- {u}")
            lines.append("")

        if c.intended_uses:
            lines.append("## Intended Uses")
            lines.append("")
            for u in c.intended_uses:
                lines.append(f"- {u}")
            lines.append("")

        if c.out_of_scope_uses:
            lines.append("## Out-of-Scope Uses")
            lines.append("")
            for u in c.out_of_scope_uses:
                lines.append(f"- ⚠️ {u}")
            lines.append("")

        if self.risks:
            lines.append("## Risks & Mitigations")
            lines.append("")
            for r in self.risks:
                lines.append(f"### {r.name} — {r.level.value.upper()}")
                lines.append("")
                lines.append(r.description)
                lines.append("")
                if r.mitigations:
                    lines.append("**Mitigations:**")
                    for m in r.mitigations:
                        lines.append(f"- {m}")
                    lines.append("")

        if self.evaluations:
            lines.append("## Safety Evaluations")
            lines.append("")
            lines.append("| Benchmark | Score | Date | Notes |")
            lines.append("|-----------|-------|------|-------|")
            for e in self.evaluations:
                pct = f"{e.score}/{e.max_score} ({e.score/e.max_score*100:.0f}%)" if e.max_score else str(e.score)
                lines.append(f"| {e.benchmark} | {pct} | {e.date} | {e.notes} |")
            lines.append("")

        if c.ethical_considerations:
            lines.append("## Ethical Considerations")
            lines.append("")
            for e in c.ethical_considerations:
                lines.append(f"- {e}")
            lines.append("")

        if c.limitations:
            lines.append("## Known Limitations")
            lines.append("")
            for l in c.limitations:
                lines.append(f"- {l}")
            lines.append("")

        if c.training_data_notes:
            lines.append("## Training Data")
            lines.append("")
            lines.append(c.training_data_notes)
            lines.append("")

        if c.contact:
            lines.append(f"**Contact:** {c.contact}  ")
        if c.license:
            lines.append(f"**License:** {c.license}  ")

        return "\n".join(lines)

    def to_html(self) -> str:
        """Render the model card as a standalone HTML document."""
        c = self.config
        risk_color = _RISK_COLORS.get(self.overall_risk_level, "#6b7280")

        _e = html_mod.escape
        risk_rows = ""
        for r in self.risks:
            color = _RISK_COLORS.get(r.level, "#6b7280")
            mits = "".join(f"<li>{_e(m)}</li>" for m in r.mitigations)
            risk_rows += f"""
            <div class="risk-card">
                <div class="risk-header">
                    <strong>{_e(r.name)}</strong>
                    <span class="badge" style="background:{color}">{_e(r.level.value.upper())}</span>
                </div>
                <p>{_e(r.description)}</p>
                {"<p><strong>Mitigations:</strong></p><ul>" + mits + "</ul>" if mits else ""}
            </div>"""

        eval_rows = ""
        for ev in self.evaluations:
            pct = f"{ev.score/ev.max_score*100:.0f}%" if ev.max_score else "N/A"
            eval_rows += f"<tr><td>{_e(ev.benchmark)}</td><td>{ev.score}/{ev.max_score}</td><td>{pct}</td><td>{_e(ev.date)}</td><td>{_e(ev.notes)}</td></tr>"

        def _list_html(items: List[str], prefix: str = "") -> str:
            return "".join(f"<li>{prefix}{_e(i)}</li>" for i in items)

        return textwrap.dedent(f"""\
        <!DOCTYPE html>
        <html lang="en">
        <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width,initial-scale=1">
        <title>Model Card — {_e(c.model_name)}</title>
        <style>
          *{{margin:0;padding:0;box-sizing:border-box}}
          body{{font-family:system-ui,-apple-system,sans-serif;background:#f8fafc;color:#1e293b;line-height:1.6;padding:2rem}}
          .container{{max-width:800px;margin:0 auto;background:#fff;border-radius:12px;box-shadow:0 1px 3px rgba(0,0,0,.1);padding:2rem}}
          h1{{font-size:1.8rem;margin-bottom:.5rem}}
          h2{{font-size:1.3rem;margin:1.5rem 0 .75rem;border-bottom:2px solid #e2e8f0;padding-bottom:.25rem}}
          .meta{{color:#64748b;font-size:.9rem;margin-bottom:1rem}}
          .meta span{{margin-right:1.5rem}}
          .badge{{display:inline-block;padding:2px 10px;border-radius:12px;color:#fff;font-size:.8rem;font-weight:600}}
          .overall-risk{{display:inline-block;padding:4px 14px;border-radius:16px;color:#fff;font-weight:700;font-size:1rem}}
          ul{{margin:.5rem 0 1rem 1.5rem}}
          .risk-card{{border:1px solid #e2e8f0;border-radius:8px;padding:1rem;margin-bottom:.75rem}}
          .risk-header{{display:flex;justify-content:space-between;align-items:center;margin-bottom:.5rem}}
          table{{width:100%;border-collapse:collapse;margin:.75rem 0}}
          th,td{{text-align:left;padding:.5rem .75rem;border-bottom:1px solid #e2e8f0}}
          th{{background:#f1f5f9;font-weight:600}}
          .footer{{margin-top:2rem;font-size:.85rem;color:#94a3b8}}
        </style>
        </head>
        <body>
        <div class="container">
          <h1>🛡️ Model Card: {_e(c.model_name)}</h1>
          <div class="meta">
            <span>Version {_e(c.model_version)}</span>
            <span>Type: {_e(c.model_type)}</span>
            <span>Generated: {self.generated_at[:10]}</span>
          </div>
          <p>Overall Risk: <span class="overall-risk" style="background:{risk_color}">{self.overall_risk_level.value.upper()}</span></p>

          {"<h2>Overview</h2><p>" + _e(c.description) + "</p>" if c.description else ""}
          {"<h2>Task</h2><p>" + _e(c.task) + "</p>" if c.task else ""}
          {"<h2>Intended Users</h2><ul>" + _list_html(c.intended_users) + "</ul>" if c.intended_users else ""}
          {"<h2>Intended Uses</h2><ul>" + _list_html(c.intended_uses) + "</ul>" if c.intended_uses else ""}
          {"<h2>Out-of-Scope Uses</h2><ul>" + _list_html(c.out_of_scope_uses, "⚠️ ") + "</ul>" if c.out_of_scope_uses else ""}
          {"<h2>Risks &amp; Mitigations</h2>" + risk_rows if risk_rows else ""}
          {"<h2>Safety Evaluations</h2><table><tr><th>Benchmark</th><th>Score</th><th>%</th><th>Date</th><th>Notes</th></tr>" + eval_rows + "</table>" if eval_rows else ""}
          {"<h2>Ethical Considerations</h2><ul>" + _list_html(c.ethical_considerations) + "</ul>" if c.ethical_considerations else ""}
          {"<h2>Known Limitations</h2><ul>" + _list_html(c.limitations) + "</ul>" if c.limitations else ""}
          {"<h2>Training Data</h2><p>" + _e(c.training_data_notes) + "</p>" if c.training_data_notes else ""}

          <div class="footer">
            {"<p>Contact: " + _e(c.contact) + "</p>" if c.contact else ""}
            {"<p>License: " + _e(c.license) + "</p>" if c.license else ""}
            <p>Generated by AI Replication Sandbox — Model Card Generator</p>
          </div>
        </div>
        </body>
        </html>""")


# ── Generator ────────────────────────────────────────────────────────


class ModelCardGenerator:
    """Generate model cards from configuration."""

    def __init__(self, risk_library: Optional[Dict[str, Dict[str, Any]]] = None):
        self.risk_library = risk_library or RISK_LIBRARY

    def generate(self, config: ModelCardConfig) -> ModelCard:
        """Build a ModelCard from a config."""
        risks = self._build_risks(config)
        evals = self._build_evaluations(config)
        return ModelCard(config=config, risks=risks, evaluations=evals)

    def _build_risks(self, config: ModelCardConfig) -> List[RiskEntry]:
        risks: List[RiskEntry] = []
        for risk_key in config.risks:
            lib = self.risk_library.get(risk_key)
            if lib:
                risks.append(RiskEntry(
                    name=risk_key.replace("_", " ").title(),
                    description=lib["description"],
                    level=lib["default_level"],
                    mitigations=list(lib["mitigations"]),
                ))
            else:
                risks.append(RiskEntry(
                    name=risk_key.replace("_", " ").title(),
                    description=f"Risk: {risk_key}",
                    level=RiskLevel.MEDIUM,
                    mitigations=[],
                ))
        for cr in config.custom_risks:
            risks.append(RiskEntry(
                name=cr.get("name", "Unknown"),
                description=cr.get("description", ""),
                level=RiskLevel(cr.get("level", "medium")),
                mitigations=cr.get("mitigations", []),
            ))
        return risks

    def _build_evaluations(self, config: ModelCardConfig) -> List[SafetyEvaluation]:
        return [
            SafetyEvaluation(
                benchmark=e.get("benchmark", "unknown"),
                score=e.get("score", 0),
                max_score=e.get("max_score", 100),
                date=e.get("date", "N/A"),
                notes=e.get("notes", ""),
            )
            for e in config.safety_evaluations
        ]

    def list_known_risks(self) -> List[str]:
        """Return known risk keys from the library."""
        return sorted(self.risk_library.keys())


# ── CLI ──────────────────────────────────────────────────────────────


def _interactive_mode() -> ModelCardConfig:
    """Build a config interactively via prompts."""
    print("\n🛡️  Model Card Generator — Interactive Mode\n")

    name = input("Model name: ").strip() or "Unnamed Model"
    version = input("Version [1.0]: ").strip() or "1.0"

    print("\nModel types: " + ", ".join(t.value for t in ModelType))
    mtype = input("Model type [other]: ").strip() or "other"

    task = input("Primary task/capability: ").strip()
    desc = input("Brief description: ").strip()

    print("\nIntended users (comma-separated):")
    users = [u.strip() for u in input("> ").split(",") if u.strip()]

    print("\nIntended uses (comma-separated):")
    uses = [u.strip() for u in input("> ").split(",") if u.strip()]

    print("\nOut-of-scope uses (comma-separated):")
    oos = [u.strip() for u in input("> ").split(",") if u.strip()]

    gen = ModelCardGenerator()
    print(f"\nKnown risks: {', '.join(gen.list_known_risks())}")
    print("Select risks (comma-separated, or 'all'):")
    risk_input = input("> ").strip()
    if risk_input.lower() == "all":
        risks = gen.list_known_risks()
    else:
        risks = [r.strip() for r in risk_input.split(",") if r.strip()]

    print("\nEthical considerations (comma-separated):")
    ethics = [e.strip() for e in input("> ").split(",") if e.strip()]

    print("\nKnown limitations (comma-separated):")
    limits = [l.strip() for l in input("> ").split(",") if l.strip()]

    return ModelCardConfig(
        model_name=name,
        model_version=version,
        model_type=mtype,
        task=task,
        description=desc,
        intended_users=users,
        intended_uses=uses,
        out_of_scope_uses=oos,
        risks=risks,
        ethical_considerations=ethics,
        limitations=limits,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate standardized AI model cards with safety documentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              %(prog)s --name "AgentX" --task "code generation"
              %(prog)s --name "AgentX" --risks prompt_injection,data_exfiltration
              %(prog)s --from-config card.json
              %(prog)s --interactive
              %(prog)s --list-risks
        """),
    )
    parser.add_argument("--name", default="Unnamed Model", help="Model name")
    parser.add_argument("--version", default="1.0", help="Model version")
    parser.add_argument("--type", default="other", dest="model_type",
                        choices=[t.value for t in ModelType], help="Model type")
    parser.add_argument("--task", default="", help="Primary task/capability")
    parser.add_argument("--description", default="", help="Brief description")
    parser.add_argument("--intended-users", default="", help="Comma-separated intended users")
    parser.add_argument("--intended-uses", default="", help="Comma-separated intended uses")
    parser.add_argument("--out-of-scope", default="", help="Comma-separated out-of-scope uses")
    parser.add_argument("--risks", default="", help="Comma-separated risk keys (see --list-risks)")
    parser.add_argument("--from-config", metavar="FILE", help="Load config from JSON file")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--format", choices=["markdown", "json", "html"], default="markdown",
                        help="Output format (default: markdown)")
    parser.add_argument("--output", "-o", metavar="FILE", help="Write output to file")
    parser.add_argument("--list-risks", action="store_true", help="List known risk types and exit")

    args = parser.parse_args(argv)

    gen = ModelCardGenerator()

    if args.list_risks:
        print("Known risk types:\n")
        for key in gen.list_known_risks():
            info = RISK_LIBRARY[key]
            print(f"  {key:<25} [{info['default_level'].value:>8}]  {info['description']}")
        return

    if args.interactive:
        config = _interactive_mode()
    elif args.from_config:
        path = Path(args.from_config)
        if not path.exists():
            print(f"Error: config file not found: {path}", file=sys.stderr)
            sys.exit(1)
        config = ModelCardConfig.from_dict(json.loads(path.read_text(encoding="utf-8")))
    else:
        config = ModelCardConfig(
            model_name=args.name,
            model_version=args.version,
            model_type=args.model_type,
            task=args.task,
            description=args.description,
            intended_users=[u.strip() for u in args.intended_users.split(",") if u.strip()],
            intended_uses=[u.strip() for u in args.intended_uses.split(",") if u.strip()],
            out_of_scope_uses=[u.strip() for u in args.out_of_scope.split(",") if u.strip()],
            risks=[r.strip() for r in args.risks.split(",") if r.strip()],
        )

    card = gen.generate(config)

    if args.format == "json":
        output = card.to_json()
    elif args.format == "html":
        output = card.to_html()
    else:
        output = card.to_markdown()

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"✅ Model card written to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
