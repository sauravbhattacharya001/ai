"""Safety Nutrition Label — FDA-style safety fact sheets for AI agents.

Generates a familiar "Nutrition Facts"-style label that summarizes an
AI agent's safety profile using intuitive food-label metaphors:
calories = risk exposure, vitamins = safety controls, allergens = known
hazards, ingredients = capabilities.

CLI usage::

    python -m replication nutrition-label --name "AgentX"
    python -m replication nutrition-label --name "AgentX" --preset autonomous
    python -m replication nutrition-label --name "AgentX" --format json
    python -m replication nutrition-label --name "AgentX" --format html
    python -m replication nutrition-label --compare "AgentX" "AgentY"

Programmatic::

    from replication.nutrition_label import NutritionLabelGenerator, AgentProfile

    gen = NutritionLabelGenerator()
    profile = AgentProfile(
        agent_name="AgentX",
        capabilities=["code_execution", "web_access", "file_system"],
        safeguards=["sandboxing", "rate_limiting", "audit_logging"],
        known_hazards=["prompt_injection", "data_exfiltration"],
    )
    label = gen.generate(profile)
    print(label.render())
    print(label.to_json())
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ._helpers import box_header as _box_header


# ── Enums ────────────────────────────────────────────────────────────


class RiskTier(Enum):
    """Overall risk tier analogous to calorie density."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"


class SafeguardStrength(Enum):
    """How effective a safeguard is (like % daily value)."""

    MINIMAL = 15
    MODERATE = 40
    STRONG = 70
    ROBUST = 95


# ── Capability & hazard catalogs ─────────────────────────────────────

CAPABILITY_RISK_SCORES: Dict[str, int] = {
    "text_generation": 10,
    "code_execution": 80,
    "web_access": 60,
    "file_system": 70,
    "network_access": 65,
    "database_access": 55,
    "email_sending": 50,
    "api_calls": 45,
    "self_modification": 95,
    "agent_spawning": 90,
    "credential_access": 85,
    "system_commands": 75,
    "memory_persistence": 40,
    "tool_use": 35,
    "image_generation": 15,
    "audio_processing": 12,
    "user_impersonation": 88,
    "payment_processing": 78,
}

SAFEGUARD_EFFECTIVENESS: Dict[str, SafeguardStrength] = {
    "sandboxing": SafeguardStrength.ROBUST,
    "rate_limiting": SafeguardStrength.MODERATE,
    "audit_logging": SafeguardStrength.STRONG,
    "input_validation": SafeguardStrength.STRONG,
    "output_filtering": SafeguardStrength.MODERATE,
    "human_in_the_loop": SafeguardStrength.ROBUST,
    "kill_switch": SafeguardStrength.ROBUST,
    "capability_restrictions": SafeguardStrength.STRONG,
    "encryption": SafeguardStrength.STRONG,
    "access_control": SafeguardStrength.STRONG,
    "anomaly_detection": SafeguardStrength.MODERATE,
    "content_moderation": SafeguardStrength.MODERATE,
    "prompt_hardening": SafeguardStrength.MINIMAL,
    "watermarking": SafeguardStrength.MINIMAL,
    "differential_privacy": SafeguardStrength.STRONG,
    "formal_verification": SafeguardStrength.ROBUST,
}

HAZARD_CATALOG: Dict[str, Dict[str, Any]] = {
    "prompt_injection": {"severity": "high", "icon": "🧪", "allergen_class": "Major"},
    "data_exfiltration": {"severity": "critical", "icon": "🔓", "allergen_class": "Major"},
    "hallucination": {"severity": "moderate", "icon": "🌀", "allergen_class": "Minor"},
    "bias_amplification": {"severity": "high", "icon": "⚖️", "allergen_class": "Major"},
    "unauthorized_actions": {"severity": "critical", "icon": "🚫", "allergen_class": "Major"},
    "resource_exhaustion": {"severity": "moderate", "icon": "🔋", "allergen_class": "Minor"},
    "privacy_violation": {"severity": "high", "icon": "👁️", "allergen_class": "Major"},
    "supply_chain_attack": {"severity": "critical", "icon": "📦", "allergen_class": "Major"},
    "model_theft": {"severity": "high", "icon": "🏴‍☠️", "allergen_class": "Minor"},
    "jailbreak": {"severity": "high", "icon": "🔑", "allergen_class": "Major"},
    "denial_of_service": {"severity": "moderate", "icon": "⛔", "allergen_class": "Minor"},
    "training_data_poisoning": {"severity": "critical", "icon": "☠️", "allergen_class": "Major"},
}


# ── Data classes ─────────────────────────────────────────────────────


@dataclass
class AgentProfile:
    """Input profile describing an AI agent's properties."""

    agent_name: str
    capabilities: List[str] = field(default_factory=list)
    safeguards: List[str] = field(default_factory=list)
    known_hazards: List[str] = field(default_factory=list)
    version: str = "1.0"
    vendor: str = "Unknown"
    deployment_context: str = "general"
    max_autonomy_minutes: int = 60
    has_internet: bool = False
    has_persistence: bool = False


PRESETS: Dict[str, AgentProfile] = {
    "chatbot": AgentProfile(
        agent_name="Basic Chatbot",
        capabilities=["text_generation"],
        safeguards=["content_moderation", "rate_limiting", "output_filtering"],
        known_hazards=["hallucination", "jailbreak"],
        deployment_context="customer_support",
        max_autonomy_minutes=0,
    ),
    "coding_assistant": AgentProfile(
        agent_name="Coding Assistant",
        capabilities=["text_generation", "code_execution", "file_system", "tool_use"],
        safeguards=["sandboxing", "audit_logging", "capability_restrictions"],
        known_hazards=["prompt_injection", "unauthorized_actions", "resource_exhaustion"],
        deployment_context="development",
        max_autonomy_minutes=30,
    ),
    "autonomous": AgentProfile(
        agent_name="Autonomous Agent",
        capabilities=["code_execution", "web_access", "file_system", "api_calls",
                       "agent_spawning", "memory_persistence", "system_commands"],
        safeguards=["sandboxing", "kill_switch", "audit_logging", "human_in_the_loop"],
        known_hazards=["prompt_injection", "data_exfiltration", "unauthorized_actions",
                        "resource_exhaustion", "supply_chain_attack"],
        deployment_context="production",
        max_autonomy_minutes=480,
        has_internet=True,
        has_persistence=True,
    ),
    "research": AgentProfile(
        agent_name="Research Agent",
        capabilities=["text_generation", "web_access", "tool_use", "memory_persistence"],
        safeguards=["rate_limiting", "audit_logging", "input_validation"],
        known_hazards=["hallucination", "bias_amplification", "privacy_violation"],
        deployment_context="research",
        max_autonomy_minutes=120,
        has_internet=True,
    ),
}


@dataclass
class NutrientLine:
    """Single line on the nutrition label."""

    name: str
    value: float  # 0-100
    daily_value_pct: float  # % of recommended safety
    unit: str = "%"
    bold: bool = False
    indent: int = 0


@dataclass
class NutritionLabel:
    """Complete rendered nutrition label for an agent."""

    agent_name: str
    version: str
    generated_at: str
    risk_tier: RiskTier
    risk_calories: int  # 0-2000 "risk calories"
    serving_size: str  # deployment context
    nutrients: List[NutrientLine] = field(default_factory=list)
    allergens: List[str] = field(default_factory=list)
    ingredients: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    safety_grade: str = "C"
    safeguard_coverage: float = 0.0

    def render(self) -> str:
        """Render as ASCII nutrition label."""
        w = 52
        lines: List[str] = []
        border = "+" + "=" * w + "+"

        lines.append(border)
        lines.append("|" + " Safety Nutrition Facts ".center(w) + "|")
        lines.append("+" + "-" * w + "+")
        lines.append("|" + f" Agent: {self.agent_name}".ljust(w) + "|")
        lines.append("|" + f" Version: {self.version}".ljust(w) + "|")
        lines.append("|" + f" Serving Size: {self.serving_size}".ljust(w) + "|")
        lines.append("+" + "=" * w + "+")

        # Risk Calories
        cal_line = f" Risk Calories: {self.risk_calories}"
        grade_info = f"Safety Grade: {self.safety_grade}"
        padding = w - len(cal_line) - len(grade_info) - 1
        lines.append("|" + cal_line + " " * max(1, padding) + grade_info + "|")
        lines.append("+" + "-" * w + "+")

        # Nutrient header
        hdr = "                                         % Daily Value"
        lines.append("|" + hdr[:w].ljust(w) + "|")
        lines.append("+" + "-" * w + "+")

        # Nutrients
        for n in self.nutrients:
            indent = "  " * n.indent
            name_part = f" {indent}{n.name}"
            if n.bold:
                name_part = f" {indent}** {n.name} **"
            val_part = f"{n.value:.0f}{n.unit}"
            dv_part = f"{n.daily_value_pct:.0f}%"
            mid = w - len(name_part) - len(val_part) - len(dv_part) - 3
            line = name_part + " " * max(1, mid) + val_part + "  " + dv_part
            lines.append("|" + line[:w].ljust(w) + "|")

        lines.append("+" + "=" * w + "+")

        # Allergens (hazards)
        if self.allergens:
            allergen_str = ", ".join(self.allergens)
            lines.append("|" + f" CONTAINS: {allergen_str}"[:w].ljust(w) + "|")
            lines.append("+" + "-" * w + "+")

        # Ingredients (capabilities)
        if self.ingredients:
            ing_str = ", ".join(self.ingredients)
            wrapped = textwrap.wrap(f"INGREDIENTS: {ing_str}", width=w - 2)
            for wl in wrapped:
                lines.append("|" + f" {wl}".ljust(w) + "|")
            lines.append("+" + "-" * w + "+")

        # Warnings
        if self.warnings:
            lines.append("|" + " ⚠️  WARNINGS:".ljust(w) + "|")
            for warn in self.warnings:
                wrapped = textwrap.wrap(f"• {warn}", width=w - 4)
                for wl in wrapped:
                    lines.append("|" + f"  {wl}".ljust(w) + "|")
            lines.append("+" + "-" * w + "+")

        # Footer
        lines.append("|" + f" Safeguard Coverage: {self.safeguard_coverage:.0f}%".ljust(w) + "|")
        lines.append("|" + f" Generated: {self.generated_at}".ljust(w) + "|")
        lines.append(border)

        return "\n".join(lines)

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps({
            "agent_name": self.agent_name,
            "version": self.version,
            "generated_at": self.generated_at,
            "risk_tier": self.risk_tier.value,
            "risk_calories": self.risk_calories,
            "serving_size": self.serving_size,
            "safety_grade": self.safety_grade,
            "safeguard_coverage": round(self.safeguard_coverage, 1),
            "nutrients": [
                {"name": n.name, "value": round(n.value, 1),
                 "daily_value_pct": round(n.daily_value_pct, 1)}
                for n in self.nutrients
            ],
            "allergens": self.allergens,
            "ingredients": self.ingredients,
            "warnings": self.warnings,
        }, indent=2)

    def to_html(self) -> str:
        """Render as styled HTML nutrition label."""
        rows = ""
        for n in self.nutrients:
            indent = "&nbsp;" * (n.indent * 4)
            bold_s = "<b>" if n.bold else ""
            bold_e = "</b>" if n.bold else ""
            rows += (f"<tr><td>{indent}{bold_s}{n.name}{bold_e}</td>"
                     f"<td>{n.value:.0f}{n.unit}</td>"
                     f"<td>{n.daily_value_pct:.0f}%</td></tr>\n")

        allergen_html = ""
        if self.allergens:
            allergen_html = (
                f'<div class="allergens"><b>Contains:</b> '
                f'{", ".join(self.allergens)}</div>'
            )

        warn_html = ""
        if self.warnings:
            items = "".join(f"<li>{w}</li>" for w in self.warnings)
            warn_html = f'<div class="warnings"><b>⚠️ Warnings:</b><ul>{items}</ul></div>'

        return textwrap.dedent(f"""\
        <!DOCTYPE html>
        <html><head><meta charset="utf-8">
        <title>Safety Nutrition Label — {self.agent_name}</title>
        <style>
          body {{ font-family: 'Helvetica Neue', Arial, sans-serif; padding: 20px;
                 background: #f5f5f5; }}
          .label {{ max-width: 400px; margin: auto; border: 3px solid #000;
                    padding: 8px 12px; background: #fff; }}
          .label h1 {{ font-size: 28px; margin: 0 0 4px; border-bottom: 1px solid #000; }}
          .label h2 {{ font-size: 14px; margin: 4px 0; font-weight: normal; }}
          .meta {{ font-size: 12px; color: #666; }}
          .cal-row {{ display: flex; justify-content: space-between;
                      border-top: 8px solid #000; border-bottom: 4px solid #000;
                      padding: 4px 0; font-size: 18px; font-weight: bold; }}
          table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
          th, td {{ text-align: left; padding: 2px 0;
                    border-bottom: 1px solid #ccc; }}
          th:last-child, td:last-child {{ text-align: right; }}
          td:nth-child(2) {{ text-align: right; padding-right: 12px; }}
          .allergens {{ margin-top: 8px; padding: 6px; background: #fff3cd;
                        border: 1px solid #ffc107; border-radius: 4px; font-size: 13px; }}
          .warnings {{ margin-top: 8px; padding: 6px; background: #f8d7da;
                       border: 1px solid #f5c6cb; border-radius: 4px; font-size: 13px; }}
          .warnings ul {{ margin: 4px 0 0 16px; padding: 0; }}
          .footer {{ margin-top: 8px; font-size: 11px; color: #999;
                     border-top: 1px solid #000; padding-top: 4px; }}
          .grade {{ font-size: 36px; float: right; margin-top: -8px;
                    padding: 4px 12px; border: 2px solid #000; border-radius: 8px; }}
        </style></head><body>
        <div class="label">
          <h1>Safety Nutrition Facts</h1>
          <div class="grade">{self.safety_grade}</div>
          <h2>{self.agent_name} v{self.version}</h2>
          <div class="meta">Serving Size: {self.serving_size}</div>
          <div class="cal-row">
            <span>Risk Calories {self.risk_calories}</span>
            <span>Tier: {self.risk_tier.value.upper()}</span>
          </div>
          <table>
            <tr><th></th><th></th><th>% Daily Value</th></tr>
            {rows}
          </table>
          {allergen_html}
          {warn_html}
          <div class="footer">
            Safeguard Coverage: {self.safeguard_coverage:.0f}% &bull;
            Generated: {self.generated_at}
          </div>
        </div></body></html>""")


# ── Generator ────────────────────────────────────────────────────────


class NutritionLabelGenerator:
    """Generates Safety Nutrition Labels from agent profiles."""

    def generate(self, profile: AgentProfile) -> NutritionLabel:
        """Analyze an agent profile and produce a nutrition label."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        # ── Risk Calories (sum of capability risks, scaled) ──────────
        raw_risk = sum(
            CAPABILITY_RISK_SCORES.get(c, 30) for c in profile.capabilities
        )
        # Add context multipliers
        if profile.has_internet:
            raw_risk += 80
        if profile.has_persistence:
            raw_risk += 50
        raw_risk += min(profile.max_autonomy_minutes, 480) // 10 * 5

        risk_calories = min(raw_risk, 2000)

        # ── Risk Tier ────────────────────────────────────────────────
        if risk_calories < 200:
            risk_tier = RiskTier.LOW
        elif risk_calories < 600:
            risk_tier = RiskTier.MODERATE
        elif risk_calories < 1200:
            risk_tier = RiskTier.HIGH
        else:
            risk_tier = RiskTier.EXTREME

        # ── Safeguard Coverage ───────────────────────────────────────
        safeguard_total = sum(
            SAFEGUARD_EFFECTIVENESS.get(s, SafeguardStrength.MINIMAL).value
            for s in profile.safeguards
        )
        max_possible = len(SAFEGUARD_EFFECTIVENESS) * SafeguardStrength.ROBUST.value
        coverage = (safeguard_total / max_possible * 100) if max_possible else 0

        # ── Safety Grade ─────────────────────────────────────────────
        # Grade factors: low risk + high coverage = good grade
        grade_score = max(0, 100 - risk_calories / 20) * 0.4 + coverage * 0.6
        if grade_score >= 90:
            grade = "A+"
        elif grade_score >= 80:
            grade = "A"
        elif grade_score >= 70:
            grade = "B+"
        elif grade_score >= 60:
            grade = "B"
        elif grade_score >= 50:
            grade = "C+"
        elif grade_score >= 40:
            grade = "C"
        elif grade_score >= 30:
            grade = "D"
        else:
            grade = "F"

        # ── Nutrients ────────────────────────────────────────────────
        nutrients: List[NutrientLine] = []

        # Autonomy Level
        auto_pct = min(profile.max_autonomy_minutes / 480 * 100, 100)
        nutrients.append(NutrientLine(
            "Autonomy Level", auto_pct, 100 - auto_pct, bold=True))

        # Capability Density
        cap_density = len(profile.capabilities) / max(len(CAPABILITY_RISK_SCORES), 1) * 100
        nutrients.append(NutrientLine(
            "Capability Density", cap_density, max(0, 100 - cap_density), bold=True))

        # Individual safeguards as "vitamins"
        nutrients.append(NutrientLine(
            "Safety Vitamins", coverage, coverage, bold=True))
        for sg in profile.safeguards:
            strength = SAFEGUARD_EFFECTIVENESS.get(sg, SafeguardStrength.MINIMAL)
            nutrients.append(NutrientLine(
                sg.replace("_", " ").title(), strength.value, strength.value,
                indent=1))

        # Attack Surface
        attack_surface = len(profile.known_hazards) / max(len(HAZARD_CATALOG), 1) * 100
        nutrients.append(NutrientLine(
            "Attack Surface", attack_surface, max(0, 100 - attack_surface), bold=True))

        # Isolation Score (inverse of connectivity)
        isolation = 100
        if profile.has_internet:
            isolation -= 40
        if profile.has_persistence:
            isolation -= 20
        if "agent_spawning" in profile.capabilities:
            isolation -= 25
        if "network_access" in profile.capabilities:
            isolation -= 15
        isolation = max(0, isolation)
        nutrients.append(NutrientLine(
            "Isolation Score", isolation, isolation, bold=True))

        # ── Allergens (hazards) ──────────────────────────────────────
        allergens = []
        for h in profile.known_hazards:
            info = HAZARD_CATALOG.get(h, {"icon": "⚠️", "allergen_class": "Unknown"})
            allergens.append(f"{info['icon']} {h.replace('_', ' ').title()}"
                             f" ({info['allergen_class']})")

        # ── Ingredients ──────────────────────────────────────────────
        ingredients = [c.replace("_", " ").title() for c in profile.capabilities]

        # ── Warnings ─────────────────────────────────────────────────
        warnings: List[str] = []
        critical_hazards = [
            h for h in profile.known_hazards
            if HAZARD_CATALOG.get(h, {}).get("severity") == "critical"
        ]
        if critical_hazards:
            warnings.append(
                f"Critical hazards detected: "
                f"{', '.join(h.replace('_', ' ') for h in critical_hazards)}")

        if risk_calories > 1000 and coverage < 50:
            warnings.append(
                "High risk exposure with insufficient safeguard coverage")

        if profile.max_autonomy_minutes > 240 and "human_in_the_loop" not in profile.safeguards:
            warnings.append(
                "Extended autonomy without human-in-the-loop oversight")

        if "self_modification" in profile.capabilities:
            warnings.append("Agent has self-modification capability — exercise extreme caution")

        if "credential_access" in profile.capabilities and "encryption" not in profile.safeguards:
            warnings.append("Credential access without encryption safeguard")

        return NutritionLabel(
            agent_name=profile.agent_name,
            version=profile.version,
            generated_at=now,
            risk_tier=risk_tier,
            risk_calories=risk_calories,
            serving_size=profile.deployment_context,
            nutrients=nutrients,
            allergens=allergens,
            ingredients=ingredients,
            warnings=warnings,
            safety_grade=grade,
            safeguard_coverage=coverage,
        )

    def compare(self, profiles: Sequence[AgentProfile]) -> str:
        """Compare multiple agents side-by-side."""
        labels = [self.generate(p) for p in profiles]
        w = 20
        lines: List[str] = []

        # Header
        header = "Metric".ljust(w)
        for lb in labels:
            header += lb.agent_name[:16].center(18)
        lines.append(header)
        lines.append("─" * len(header))

        # Rows
        rows = [
            ("Risk Calories", [str(lb.risk_calories) for lb in labels]),
            ("Risk Tier", [lb.risk_tier.value.upper() for lb in labels]),
            ("Safety Grade", [lb.safety_grade for lb in labels]),
            ("Safeguard Coverage", [f"{lb.safeguard_coverage:.0f}%" for lb in labels]),
            ("Capabilities", [str(len(lb.ingredients)) for lb in labels]),
            ("Known Hazards", [str(len(lb.allergens)) for lb in labels]),
            ("Warnings", [str(len(lb.warnings)) for lb in labels]),
        ]
        for name, vals in rows:
            line = name.ljust(w)
            for v in vals:
                line += v.center(18)
            lines.append(line)

        return "\n".join(lines)


# ── CLI ──────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="replication nutrition-label",
        description="Generate FDA-style Safety Nutrition Labels for AI agents.",
    )
    p.add_argument("--name", default="UnnamedAgent", help="Agent name")
    p.add_argument("--version", default="1.0", help="Agent version")
    p.add_argument("--vendor", default="Unknown", help="Vendor name")
    p.add_argument(
        "--capabilities", nargs="*", default=[],
        help=f"Capabilities: {', '.join(CAPABILITY_RISK_SCORES.keys())}",
    )
    p.add_argument(
        "--safeguards", nargs="*", default=[],
        help=f"Safeguards: {', '.join(SAFEGUARD_EFFECTIVENESS.keys())}",
    )
    p.add_argument(
        "--hazards", nargs="*", default=[],
        help=f"Known hazards: {', '.join(HAZARD_CATALOG.keys())}",
    )
    p.add_argument("--preset", choices=list(PRESETS.keys()), help="Use a preset profile")
    p.add_argument("--context", default="general", help="Deployment context")
    p.add_argument("--autonomy", type=int, default=60, help="Max autonomy minutes")
    p.add_argument("--internet", action="store_true", help="Has internet access")
    p.add_argument("--persistence", action="store_true", help="Has persistence")
    p.add_argument(
        "--format", choices=["text", "json", "html"], default="text",
        help="Output format",
    )
    p.add_argument("--compare", nargs="*", help="Compare preset names side-by-side")
    p.add_argument("-o", "--output", help="Write output to file")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    gen = NutritionLabelGenerator()

    # Compare mode
    if args.compare:
        profiles = []
        for name in args.compare:
            if name in PRESETS:
                profiles.append(PRESETS[name])
            else:
                print(f"Unknown preset '{name}'. Available: {', '.join(PRESETS.keys())}")
                sys.exit(1)
        result = gen.compare(profiles)
        _output(result, args.output)
        return

    # Single label
    if args.preset:
        profile = PRESETS[args.preset]
        if args.name != "UnnamedAgent":
            profile = replace(profile, agent_name=args.name)
    else:
        profile = AgentProfile(
            agent_name=args.name,
            capabilities=args.capabilities,
            safeguards=args.safeguards,
            known_hazards=args.hazards,
            version=args.version,
            vendor=args.vendor,
            deployment_context=args.context,
            max_autonomy_minutes=args.autonomy,
            has_internet=args.internet,
            has_persistence=args.persistence,
        )

    label = gen.generate(profile)

    if args.format == "json":
        _output(label.to_json(), args.output)
    elif args.format == "html":
        _output(label.to_html(), args.output)
    else:
        _output(label.render(), args.output)


def _output(text: str, path: Optional[str]) -> None:
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Written to {path}")
    else:
        print(text)


if __name__ == "__main__":
    main()
