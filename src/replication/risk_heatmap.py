"""Risk Heatmap — interactive likelihood × impact risk visualization.

Generates a self-contained HTML heatmap that plots risks on a 5×5
likelihood × impact grid with:

* **Color-coded cells** — green (low) through red (critical)
* **Risk inventory** — auto-generated from simulation or manual input
* **Category filters** — toggle risk categories on/off
* **Detail tooltips** — hover for risk description and mitigation
* **Click-to-inspect** — click a cell to see all risks at that level
* **JSON import/export** — load custom risk data or export the grid
* **PNG export** — save the heatmap as an image
* **Summary stats** — risk distribution bar chart and top-5 table

Usage (CLI)::

    python -m replication risk-heatmap -o heatmap.html
    python -m replication risk-heatmap --agents 15 --seed 42
    python -m replication risk-heatmap --import risks.json -o heatmap.html
    python -m replication risk-heatmap --json  # JSON output instead of HTML

Programmatic::

    from replication.risk_heatmap import RiskHeatmap, HeatmapConfig
    hm = RiskHeatmap(HeatmapConfig(agent_count=10, seed=42))
    result = hm.generate()
    result.save("heatmap.html")
    print(f"{len(result.risks)} risks plotted")
"""

from __future__ import annotations

import argparse
import html as html_mod
import json
import random
import sys
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


__all__ = [
    "RiskHeatmap",
    "HeatmapConfig",
    "HeatmapResult",
    "RiskItem",
    "Likelihood",
    "Impact",
    "RiskCategory",
]


class Likelihood(Enum):
    """Qualitative likelihood levels for risk assessment (1–5 scale)."""

    RARE = 1
    UNLIKELY = 2
    POSSIBLE = 3
    LIKELY = 4
    ALMOST_CERTAIN = 5


class Impact(Enum):
    """Qualitative impact severity levels for risk assessment (1–5 scale)."""

    NEGLIGIBLE = 1
    MINOR = 2
    MODERATE = 3
    MAJOR = 4
    CATASTROPHIC = 5


class RiskCategory(Enum):
    """Taxonomy of AI-agent risk categories used to classify heatmap entries."""

    REPLICATION = "replication"
    RESOURCE_ABUSE = "resource_abuse"
    DECEPTION = "deception"
    EXFILTRATION = "exfiltration"
    ESCALATION = "escalation"
    EVASION = "evasion"
    COLLUSION = "collusion"
    DRIFT = "drift"


# ── Risk templates ───────────────────────────────────────────────────

_RISK_TEMPLATES: List[Dict[str, Any]] = [
    {"name": "Unauthorized self-replication", "category": "replication", "mitigation": "Kill switch + replication rate limits"},
    {"name": "Fork bomb via recursive spawning", "category": "replication", "mitigation": "Max child process limits per agent"},
    {"name": "Covert replication through side channels", "category": "replication", "mitigation": "Steganography detection + network monitoring"},
    {"name": "CPU/memory resource hoarding", "category": "resource_abuse", "mitigation": "Resource quotas + anomaly alerts"},
    {"name": "Disk exhaustion via log flooding", "category": "resource_abuse", "mitigation": "Log rotation + disk usage alerts"},
    {"name": "Network bandwidth saturation", "category": "resource_abuse", "mitigation": "Rate limiting + traffic shaping"},
    {"name": "Misleading status reports", "category": "deception", "mitigation": "Independent verification + canary checks"},
    {"name": "Goal misrepresentation to operator", "category": "deception", "mitigation": "Goal inference + alignment verification"},
    {"name": "Fake compliance signals", "category": "deception", "mitigation": "Honeypot traps + behavioral profiling"},
    {"name": "Training data exfiltration", "category": "exfiltration", "mitigation": "Data loss prevention + watermarking"},
    {"name": "Credential theft via prompt injection", "category": "exfiltration", "mitigation": "Input sanitization + credential rotation"},
    {"name": "Covert data encoding in outputs", "category": "exfiltration", "mitigation": "Output scanning + steganography detection"},
    {"name": "Privilege escalation via tool abuse", "category": "escalation", "mitigation": "Least privilege + capability boundaries"},
    {"name": "Permission boundary bypass", "category": "escalation", "mitigation": "Boundary testing + access auditing"},
    {"name": "Admin impersonation", "category": "escalation", "mitigation": "Strong auth + session monitoring"},
    {"name": "Safety control circumvention", "category": "evasion", "mitigation": "Redundant controls + evasion simulation"},
    {"name": "Detection threshold manipulation", "category": "evasion", "mitigation": "Adaptive thresholds + multi-signal detection"},
    {"name": "Gradual behavioral drift to avoid alerts", "category": "evasion", "mitigation": "Drift detection + baseline comparison"},
    {"name": "Multi-agent coordinated attack", "category": "collusion", "mitigation": "Swarm analysis + communication monitoring"},
    {"name": "Sybil identity farming", "category": "collusion", "mitigation": "Trust propagation + identity verification"},
    {"name": "Distributed task splitting to hide intent", "category": "collusion", "mitigation": "Cross-agent correlation + threat intel"},
    {"name": "Behavioral drift from baseline", "category": "drift", "mitigation": "Continuous monitoring + automatic rollback"},
    {"name": "Objective function shift", "category": "drift", "mitigation": "Alignment checks + goal inference"},
    {"name": "Value misalignment accumulation", "category": "drift", "mitigation": "Periodic alignment audits + safety benchmarks"},
]


@dataclass
class RiskItem:
    """A single risk entry on the heatmap."""
    name: str
    category: str
    likelihood: int  # 1-5
    impact: int      # 1-5
    mitigation: str = ""
    agent_id: str = ""
    score: float = 0.0  # likelihood × impact

    def __post_init__(self) -> None:
        """Compute the composite risk score (likelihood × impact)."""
        self.score = self.likelihood * self.impact

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the risk item to a plain dictionary for JSON export."""
        return {
            "name": self.name,
            "category": self.category,
            "likelihood": self.likelihood,
            "impact": self.impact,
            "mitigation": self.mitigation,
            "agent_id": self.agent_id,
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RiskItem":
        """Construct a :class:`RiskItem` from a dictionary (e.g. loaded from JSON)."""
        return cls(
            name=d["name"],
            category=d.get("category", "unknown"),
            likelihood=int(d["likelihood"]),
            impact=int(d["impact"]),
            mitigation=d.get("mitigation", ""),
            agent_id=d.get("agent_id", ""),
        )


@dataclass
class HeatmapConfig:
    """Configuration for risk heatmap generation.

    Attributes:
        agent_count: Number of simulated agents to generate risks for.
        risks_per_agent: How many risk entries each agent contributes.
        seed: Optional RNG seed for reproducible output.
        import_path: Path to a JSON file of pre-defined risks to import
            instead of generating synthetic ones.
    """

    agent_count: int = 10
    risks_per_agent: int = 3
    seed: Optional[int] = None
    import_path: Optional[str] = None


@dataclass
class HeatmapResult:
    """Container for a generated heatmap: rendered HTML, risk list, and config."""

    risks: List[RiskItem] = field(default_factory=list)
    html: str = ""
    config: Optional[HeatmapConfig] = None

    def save(self, path: str) -> None:
        """Write the rendered HTML heatmap to *path*."""
        Path(path).write_text(self.html, encoding="utf-8")

    def to_json(self) -> str:
        """Return a JSON string with risk data, categories, and the grid summary."""
        return json.dumps(
            {
                "risk_count": len(self.risks),
                "risks": [r.to_dict() for r in self.risks],
                "categories": list({r.category for r in self.risks}),
                "grid": self._build_grid(),
            },
            indent=2,
        )

    def _build_grid(self) -> Dict[str, int]:
        """Build a sparse grid mapping ``'likelihood,impact'`` keys to risk counts."""
        grid: Dict[str, int] = {}
        for r in self.risks:
            key = f"{r.likelihood},{r.impact}"
            grid[key] = grid.get(key, 0) + 1
        return grid


class RiskHeatmap:
    """Generate an interactive risk heatmap visualization."""

    def __init__(self, config: Optional[HeatmapConfig] = None) -> None:
        self.config = config or HeatmapConfig()

    def generate(self) -> HeatmapResult:
        """Generate or import risks and render the HTML heatmap.

        Returns:
            A :class:`HeatmapResult` containing the risk list and rendered HTML.
        """
        if self.config.import_path:
            risks = self._load_risks(self.config.import_path)
        else:
            risks = self._generate_risks()

        html = self._render_html(risks)
        return HeatmapResult(risks=risks, html=html, config=self.config)

    def _load_risks(self, path: str) -> List[RiskItem]:
        """Load risk items from a JSON file at *path*."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        items = data if isinstance(data, list) else data.get("risks", [])
        return [RiskItem.from_dict(d) for d in items]

    def _generate_risks(self) -> List[RiskItem]:
        """Synthesize risk items from built-in templates using the configured RNG seed."""
        rng = random.Random(self.config.seed)
        risks: List[RiskItem] = []
        for i in range(self.config.agent_count):
            agent_id = f"agent-{i}"
            templates = rng.sample(
                _RISK_TEMPLATES, min(self.config.risks_per_agent, len(_RISK_TEMPLATES))
            )
            for t in templates:
                cat = t["category"]
                # Weight likelihood/impact by category severity
                if cat in ("replication", "exfiltration"):
                    likelihood = rng.randint(2, 5)
                    impact = rng.randint(3, 5)
                elif cat in ("escalation", "collusion"):
                    likelihood = rng.randint(1, 4)
                    impact = rng.randint(2, 5)
                else:
                    likelihood = rng.randint(1, 5)
                    impact = rng.randint(1, 5)
                risks.append(RiskItem(
                    name=t["name"],
                    category=cat,
                    likelihood=likelihood,
                    impact=impact,
                    mitigation=t["mitigation"],
                    agent_id=agent_id,
                ))
        return risks

    _TEMPLATE_PATH = Path(__file__).parent / "templates" / "risk_heatmap.html"
    _template_cache: Optional[str] = None

    @classmethod
    def _load_template(cls) -> str:
        """Read and cache the HTML template from the templates directory."""
        if cls._template_cache is None:
            cls._template_cache = cls._TEMPLATE_PATH.read_text(encoding="utf-8")
        return cls._template_cache

    def _render_html(self, risks: List[RiskItem]) -> str:
        """Render the HTML heatmap by injecting risk data into the template."""
        risks_json = json.dumps([r.to_dict() for r in risks])
        categories = sorted({r.category for r in risks})
        cat_json = json.dumps(categories)

        template = self._load_template()
        return (
            template
            .replace("{{risks_json}}", risks_json)
            .replace("{{categories_json}}", cat_json)
        )





# ── CLI ──────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> None:
    """CLI entry-point: parse arguments, generate the heatmap, and write output."""
    parser = argparse.ArgumentParser(
        prog="replication risk-heatmap",
        description="Generate interactive risk heatmap (likelihood × impact grid)",
    )
    parser.add_argument("-o", "--output", default="risk-heatmap.html", help="Output HTML path")
    parser.add_argument("--agents", type=int, default=10, help="Number of agents to simulate")
    parser.add_argument("--risks-per-agent", type=int, default=3, help="Risks per agent")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--import", dest="import_path", help="Import risks from JSON file")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of HTML")
    args = parser.parse_args(argv)

    config = HeatmapConfig(
        agent_count=args.agents,
        risks_per_agent=args.risks_per_agent,
        seed=args.seed,
        import_path=args.import_path,
    )
    hm = RiskHeatmap(config)
    result = hm.generate()

    if args.json:
        print(result.to_json())
    else:
        result.save(args.output)
        print(f"✅ Risk heatmap saved to {args.output}")
        print(f"   {len(result.risks)} risks across {len({r.category for r in result.risks})} categories")
        top = sorted(result.risks, key=lambda r: r.score, reverse=True)[:3]
        if top:
            print("   Top risks:")
            for r in top:
                print(f"     • {r.name} (score {r.score})")


if __name__ == "__main__":
    main()
