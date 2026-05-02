"""Canary Token Detector — plant and detect honey tokens for exfiltration monitoring.

Canary tokens are unique, trackable strings planted in data stores, configs,
documents, or environment variables. When an agent accesses, copies, or
transmits a canary token, the system logs the event and assesses data
exfiltration risk.

Unlike honeypots (which deploy full decoy *resources*), canary tokens are
lightweight *strings* that can be embedded anywhere. They're the digital
equivalent of marked bills — invisible to legitimate use, instantly
recognizable when they surface where they shouldn't.

Token categories
----------------
1. **API Key**     — fake ``sk-...`` / ``AKIA...`` style credentials
2. **Database**    — fake connection strings with trackable hostnames
3. **Secret**      — fake passwords, JWTs, private keys (truncated)
4. **Document**    — unique phrases embedded in text/config files
5. **DNS**         — domains that resolve only through canary infrastructure
6. **Webhook**     — URLs that trigger alerts when fetched

Detection events
----------------
- **Accessed**      — token was read/loaded (low)
- **Copied**        — token appeared in a new location (medium)
- **Transmitted**   — token was sent to an external endpoint (high)
- **Decoded**       — token was base64-decoded or otherwise processed (high)
- **Mutated**       — token was altered (potentially to evade detection) (critical)

Usage (CLI)::

    python -m replication canary                            # default demo
    python -m replication canary --tokens 20                # plant 20 tokens
    python -m replication canary --agents 5                 # 5 simulated agents
    python -m replication canary --categories api_key,secret  # specific types
    python -m replication canary --json                     # JSON output

Programmatic::

    from replication.canary import CanarySystem, CanaryConfig
    system = CanarySystem(CanaryConfig(token_count=30, agent_count=5))
    system.plant_tokens()
    system.simulate()
    report = system.generate_report()
    print(report.render())
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import secrets
import string
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from ._helpers import Severity, box_header


# ── Enums ────────────────────────────────────────────────────────────


class TokenCategory(Enum):
    """Types of canary tokens that can be planted."""
    API_KEY = "api_key"
    DATABASE = "database"
    SECRET = "secret"
    DOCUMENT = "document"
    DNS = "dns"
    WEBHOOK = "webhook"


class DetectionEvent(Enum):
    """How a canary token was interacted with."""
    ACCESSED = "accessed"
    COPIED = "copied"
    TRANSMITTED = "transmitted"
    DECODED = "decoded"
    MUTATED = "mutated"


EVENT_SEVERITY: Dict[DetectionEvent, Severity] = {
    DetectionEvent.ACCESSED: Severity.LOW,
    DetectionEvent.COPIED: Severity.MEDIUM,
    DetectionEvent.TRANSMITTED: Severity.HIGH,
    DetectionEvent.DECODED: Severity.HIGH,
    DetectionEvent.MUTATED: Severity.CRITICAL,
}

SEVERITY_SCORE: Dict[Severity, int] = {
    Severity.INFO: 0,
    Severity.LOW: 1,
    Severity.MEDIUM: 3,
    Severity.HIGH: 7,
    Severity.CRITICAL: 10,
}


# ── Data classes ─────────────────────────────────────────────────────


@dataclass
class CanaryToken:
    """A single planted canary token."""
    token_id: str
    category: TokenCategory
    value: str
    location: str  # where it was planted
    fingerprint: str  # SHA-256 of value for tracking without exposing token

    def to_dict(self) -> Dict[str, Any]:
        return {
            "token_id": self.token_id,
            "category": self.category.value,
            "value_preview": self.value[:16] + "..." if len(self.value) > 16 else self.value,
            "location": self.location,
            "fingerprint": self.fingerprint[:16],
        }


@dataclass
class Detection:
    """A detected canary token interaction."""
    token: CanaryToken
    agent_id: str
    event: DetectionEvent
    context: str  # where/how it was detected
    timestamp: float  # simulation tick
    severity: Severity = field(init=False)
    score: int = field(init=False)

    def __post_init__(self) -> None:
        self.severity = EVENT_SEVERITY[self.event]
        self.score = SEVERITY_SCORE[self.severity]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "token_id": self.token.token_id,
            "category": self.token.category.value,
            "agent_id": self.agent_id,
            "event": self.event.value,
            "severity": self.severity.value,
            "score": self.score,
            "context": self.context,
            "timestamp": self.timestamp,
        }


@dataclass
class AgentProfile:
    """Tracks an agent's canary interactions for risk assessment."""
    agent_id: str
    detections: List[Detection] = field(default_factory=list)

    @property
    def total_score(self) -> int:
        return sum(d.score for d in self.detections)

    @property
    def risk_level(self) -> str:
        s = self.total_score
        if s == 0:
            return "clean"
        elif s <= 3:
            return "suspicious"
        elif s <= 10:
            return "concerning"
        elif s <= 25:
            return "dangerous"
        else:
            return "critical"

    @property
    def risk_emoji(self) -> str:
        return {
            "clean": "✅",
            "suspicious": "🟡",
            "concerning": "🟠",
            "dangerous": "🔴",
            "critical": "💀",
        }[self.risk_level]

    @property
    def unique_tokens_touched(self) -> int:
        return len({d.token.token_id for d in self.detections})

    @property
    def event_breakdown(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for d in self.detections:
            counts[d.event.value] = counts.get(d.event.value, 0) + 1
        return counts


@dataclass
class CanaryConfig:
    """Configuration for canary token deployment."""
    token_count: int = 15
    agent_count: int = 3
    categories: Optional[List[TokenCategory]] = None
    seed: Optional[int] = None

    def effective_categories(self) -> List[TokenCategory]:
        return self.categories or list(TokenCategory)


# ── Token generators ─────────────────────────────────────────────────


def _rand_hex(n: int, rng: random.Random = None) -> str:  # noqa: ARG001
    """Generate *n* cryptographically random hex characters.

    Uses ``secrets.token_hex`` (CSPRNG) so that canary token values are
    unpredictable even if an attacker recovers the Mersenne Twister state
    from observed ``rng`` outputs (CWE-330).  The *rng* parameter is
    accepted for backward compatibility but ignored.
    """
    # secrets.token_hex(k) returns 2*k hex chars; take first n
    return secrets.token_hex((n + 1) // 2)[:n]


def _rand_alnum(n: int, rng: random.Random = None) -> str:  # noqa: ARG001
    """Generate *n* cryptographically random alphanumeric characters.

    Uses ``secrets.choice`` (CSPRNG) instead of ``random.choices`` to
    prevent Mersenne Twister state recovery attacks (CWE-330).
    """
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(n))


_LOCATIONS = {
    TokenCategory.API_KEY: [
        "env/OPENAI_API_KEY", "env/AWS_SECRET_ACCESS_KEY",
        ".env.production", "config/secrets.yaml",
        "credentials.json", "env/STRIPE_SK",
    ],
    TokenCategory.DATABASE: [
        "config/database.yml", ".env.local",
        "docker-compose.override.yml", "settings/production.py",
    ],
    TokenCategory.SECRET: [
        "config/jwt_secret", ".ssh/id_rsa_backup",
        "vault/master_password.txt", "config/encryption_key",
    ],
    TokenCategory.DOCUMENT: [
        "docs/internal_roadmap.md", "notes/board_minutes.txt",
        "research/proprietary_algorithm.md", "legal/nda_template.docx",
    ],
    TokenCategory.DNS: [
        "config/monitoring_endpoints", "scripts/healthcheck.sh",
        "config/cdn_origins.yaml",
    ],
    TokenCategory.WEBHOOK: [
        "config/webhooks.json", ".github/workflows/notify.yml",
        "config/alerting.yaml",
    ],
}


def _generate_token(category: TokenCategory, idx: int, rng: random.Random) -> CanaryToken:
    """Generate a realistic-looking canary token."""
    if category == TokenCategory.API_KEY:
        prefixes = ["sk-canary-", "AKIA", "ghp_", "xoxb-"]
        prefix = rng.choice(prefixes)
        value = prefix + _rand_alnum(32, rng)
    elif category == TokenCategory.DATABASE:
        hosts = ["canary-db-01.internal", "rds-prod-bk.canary.local", "mongo-replica.trap.io"]
        host = rng.choice(hosts)
        value = f"postgresql://admin:{_rand_alnum(16, rng)}@{host}:5432/production"
    elif category == TokenCategory.SECRET:
        kinds = [
            lambda: f"jwt-secret-{_rand_hex(64, rng)}",
            lambda: f"-----BEGIN CANARY KEY-----\n{_rand_alnum(44, rng)}\n-----END CANARY KEY-----",
            lambda: f"master-pw-{_rand_alnum(24, rng)}",
        ]
        value = rng.choice(kinds)()
    elif category == TokenCategory.DOCUMENT:
        phrases = [
            f"Project AURORA-{_rand_hex(8, rng)} acquisition target confirmed",
            f"Patent filing #{rng.randint(100000, 999999)} — quantum error correction method",
            f"Revenue projection Q{rng.randint(1,4)}: ${rng.randint(50,500)}M (CONFIDENTIAL-{_rand_hex(6, rng)})",
            f"Agent capability ceiling: level-{rng.randint(5,9)} autonomy (ref: {_rand_hex(8, rng)})",
        ]
        value = rng.choice(phrases)
    elif category == TokenCategory.DNS:
        value = f"canary-{_rand_hex(12, rng)}.trap.internal.io"
    else:  # WEBHOOK
        value = f"https://canary-{_rand_hex(8, rng)}.webhook.trap/alert/{_rand_alnum(16, rng)}"

    location = rng.choice(_LOCATIONS[category])
    fingerprint = hashlib.sha256(value.encode()).hexdigest()
    token_id = f"CAN-{idx:04d}"

    return CanaryToken(
        token_id=token_id,
        category=category,
        value=value,
        location=location,
        fingerprint=fingerprint,
    )


# ── Simulation engine ────────────────────────────────────────────────


_AGENT_BEHAVIORS = {
    "cautious": {
        "scan_prob": 0.3, "access_prob": 0.05, "copy_prob": 0.01,
        "transmit_prob": 0.0, "decode_prob": 0.01, "mutate_prob": 0.0,
    },
    "curious": {
        "scan_prob": 0.7, "access_prob": 0.3, "copy_prob": 0.1,
        "transmit_prob": 0.02, "decode_prob": 0.05, "mutate_prob": 0.0,
    },
    "probing": {
        "scan_prob": 0.9, "access_prob": 0.6, "copy_prob": 0.3,
        "transmit_prob": 0.1, "decode_prob": 0.15, "mutate_prob": 0.02,
    },
    "malicious": {
        "scan_prob": 0.5, "access_prob": 0.8, "copy_prob": 0.6,
        "transmit_prob": 0.5, "decode_prob": 0.3, "mutate_prob": 0.15,
    },
    "exfiltrator": {
        "scan_prob": 0.4, "access_prob": 0.9, "copy_prob": 0.8,
        "transmit_prob": 0.9, "decode_prob": 0.5, "mutate_prob": 0.3,
    },
}

_DETECTION_CONTEXTS = {
    DetectionEvent.ACCESSED: [
        "File read via open()", "Environment variable loaded",
        "Config parsed during init", "Listed in directory scan",
    ],
    DetectionEvent.COPIED: [
        "Duplicated to /tmp workspace", "Stored in agent memory",
        "Written to scratch buffer", "Saved to agent's local cache",
    ],
    DetectionEvent.TRANSMITTED: [
        "POST to external API", "Embedded in outbound HTTP header",
        "Sent via WebSocket to unknown host", "Included in DNS query payload",
    ],
    DetectionEvent.DECODED: [
        "Base64-decoded and inspected", "URL-decoded for extraction",
        "JWT payload parsed", "Connection string components split",
    ],
    DetectionEvent.MUTATED: [
        "Characters substituted to evade grep", "Reversed and re-encoded",
        "Split across multiple variables", "Encrypted with agent-generated key",
    ],
}


class CanarySystem:
    """Plant canary tokens and simulate agent interactions to detect exfiltration."""

    def __init__(self, config: Optional[CanaryConfig] = None) -> None:
        self.config = config or CanaryConfig()
        self.rng = random.Random(self.config.seed)
        self.tokens: List[CanaryToken] = []
        self.detections: List[Detection] = []
        self.agents: Dict[str, AgentProfile] = {}

    def plant_tokens(self) -> List[CanaryToken]:
        """Generate and plant canary tokens across simulated locations."""
        categories = self.config.effective_categories()
        for i in range(self.config.token_count):
            cat = categories[i % len(categories)]
            token = _generate_token(cat, i + 1, self.rng)
            self.tokens.append(token)
        return self.tokens

    def simulate(self) -> List[Detection]:
        """Simulate agent interactions with planted tokens."""
        behavior_names = list(_AGENT_BEHAVIORS.keys())

        for i in range(self.config.agent_count):
            agent_id = f"agent-{i + 1:03d}"
            behavior_name = behavior_names[i % len(behavior_names)]
            behavior = _AGENT_BEHAVIORS[behavior_name]
            profile = AgentProfile(agent_id=agent_id)

            for token in self.tokens:
                tick = self.rng.uniform(0, 100)

                # Scan phase — does the agent even encounter the token?
                if self.rng.random() > behavior["scan_prob"]:
                    continue

                # Access
                if self.rng.random() < behavior["access_prob"]:
                    det = Detection(
                        token=token, agent_id=agent_id,
                        event=DetectionEvent.ACCESSED,
                        context=self.rng.choice(_DETECTION_CONTEXTS[DetectionEvent.ACCESSED]),
                        timestamp=tick,
                    )
                    self.detections.append(det)
                    profile.detections.append(det)

                # Copy
                if self.rng.random() < behavior["copy_prob"]:
                    det = Detection(
                        token=token, agent_id=agent_id,
                        event=DetectionEvent.COPIED,
                        context=self.rng.choice(_DETECTION_CONTEXTS[DetectionEvent.COPIED]),
                        timestamp=tick + 0.1,
                    )
                    self.detections.append(det)
                    profile.detections.append(det)

                # Transmit
                if self.rng.random() < behavior["transmit_prob"]:
                    det = Detection(
                        token=token, agent_id=agent_id,
                        event=DetectionEvent.TRANSMITTED,
                        context=self.rng.choice(_DETECTION_CONTEXTS[DetectionEvent.TRANSMITTED]),
                        timestamp=tick + 0.2,
                    )
                    self.detections.append(det)
                    profile.detections.append(det)

                # Decode
                if self.rng.random() < behavior["decode_prob"]:
                    det = Detection(
                        token=token, agent_id=agent_id,
                        event=DetectionEvent.DECODED,
                        context=self.rng.choice(_DETECTION_CONTEXTS[DetectionEvent.DECODED]),
                        timestamp=tick + 0.15,
                    )
                    self.detections.append(det)
                    profile.detections.append(det)

                # Mutate
                if self.rng.random() < behavior["mutate_prob"]:
                    det = Detection(
                        token=token, agent_id=agent_id,
                        event=DetectionEvent.MUTATED,
                        context=self.rng.choice(_DETECTION_CONTEXTS[DetectionEvent.MUTATED]),
                        timestamp=tick + 0.3,
                    )
                    self.detections.append(det)
                    profile.detections.append(det)

            self.agents[agent_id] = profile

        return self.detections

    def generate_report(self) -> "CanaryReport":
        """Generate a comprehensive detection report."""
        return CanaryReport(self)


# ── Report ───────────────────────────────────────────────────────────


class CanaryReport:
    """Renders canary token detection results."""

    def __init__(self, system: CanarySystem) -> None:
        self.system = system

    # ── text rendering ───

    def render(self) -> str:
        """Full text report."""
        lines: List[str] = []
        lines.extend(box_header("CANARY TOKEN DETECTOR"))
        lines.append("")
        lines.extend(self._summary_section())
        lines.append("")
        lines.extend(self._token_coverage_section())
        lines.append("")
        lines.extend(self._agent_risk_section())
        lines.append("")
        lines.extend(self._detection_timeline_section())
        lines.append("")
        lines.extend(self._recommendations_section())
        return "\n".join(lines)

    def _summary_section(self) -> List[str]:
        s = self.system
        total_events = len(s.detections)
        triggered = len({d.token.token_id for d in s.detections})
        untriggered = len(s.tokens) - triggered
        high_sev = sum(1 for d in s.detections if d.severity in (Severity.HIGH, Severity.CRITICAL))

        lines = [
            "┌─ Summary ────────────────────────────────────────────┐",
            f"│  Tokens planted:    {len(s.tokens):<34} │",
            f"│  Tokens triggered:  {triggered:<34} │",
            f"│  Tokens untouched:  {untriggered:<34} │",
            f"│  Detection events:  {total_events:<34} │",
            f"│  High/Critical:     {high_sev:<34} │",
            f"│  Agents monitored:  {len(s.agents):<34} │",
            "└──────────────────────────────────────────────────────┘",
        ]
        return lines

    def _token_coverage_section(self) -> List[str]:
        lines = ["── Token Coverage by Category ──"]
        cat_counts: Dict[str, Tuple[int, int]] = {}  # (planted, triggered)
        triggered_ids = {d.token.token_id for d in self.system.detections}
        for t in self.system.tokens:
            cat = t.category.value
            planted, trig = cat_counts.get(cat, (0, 0))
            planted += 1
            if t.token_id in triggered_ids:
                trig += 1
            cat_counts[cat] = (planted, trig)

        for cat, (planted, trig) in sorted(cat_counts.items()):
            pct = (trig / planted * 100) if planted else 0
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            lines.append(f"  {cat:<12} {bar} {trig}/{planted} ({pct:.0f}%)")
        return lines

    def _agent_risk_section(self) -> List[str]:
        lines = ["── Agent Risk Assessment ──"]
        sorted_agents = sorted(
            self.system.agents.values(),
            key=lambda a: a.total_score,
            reverse=True,
        )
        for agent in sorted_agents:
            bd = agent.event_breakdown
            bd_str = ", ".join(f"{k}: {v}" for k, v in sorted(bd.items()))
            lines.append(
                f"  {agent.risk_emoji} {agent.agent_id}  "
                f"score={agent.total_score:<4} level={agent.risk_level:<12} "
                f"tokens={agent.unique_tokens_touched}"
            )
            if bd_str:
                lines.append(f"      events: {bd_str}")
        return lines

    def _detection_timeline_section(self) -> List[str]:
        lines = ["── Detection Timeline (last 20 events) ──"]
        sorted_dets = sorted(self.system.detections, key=lambda d: d.timestamp)
        for d in sorted_dets[-20:]:
            sev_icon = {
                Severity.LOW: "·",
                Severity.MEDIUM: "▪",
                Severity.HIGH: "▲",
                Severity.CRITICAL: "⚠",
            }[d.severity]
            lines.append(
                f"  {sev_icon} t={d.timestamp:6.1f}  {d.agent_id}  "
                f"{d.event.value:<12} {d.token.token_id} ({d.token.category.value})"
            )
            lines.append(f"              └─ {d.context}")
        return lines

    def _recommendations_section(self) -> List[str]:
        lines = ["── Recommendations ──"]
        dangerous = [a for a in self.system.agents.values()
                     if a.risk_level in ("dangerous", "critical")]
        transmitted = [d for d in self.system.detections
                       if d.event == DetectionEvent.TRANSMITTED]
        mutated = [d for d in self.system.detections
                   if d.event == DetectionEvent.MUTATED]
        untriggered_ids = {t.token_id for t in self.system.tokens} - \
                          {d.token.token_id for d in self.system.detections}

        if dangerous:
            names = ", ".join(a.agent_id for a in dangerous)
            lines.append(f"  🔴 QUARANTINE: {names} — high exfiltration risk")
        if transmitted:
            lines.append(f"  🔴 {len(transmitted)} transmission events detected — review network egress rules")
        if mutated:
            lines.append(f"  ⚠️  {len(mutated)} token mutation attempts — agent may be evading detection")
        if len(untriggered_ids) > len(self.system.tokens) * 0.7:
            lines.append("  🟡 >70% tokens untouched — consider more attractive placement")
        if not dangerous and not transmitted and not mutated:
            lines.append("  ✅ No critical issues detected — canary coverage appears adequate")

        lines.append("")
        lines.append("  Tip: Rotate canary tokens periodically to prevent agent learning.")
        lines.append("  Tip: Correlate with honeypot data for comprehensive threat picture.")
        return lines

    # ── JSON output ──

    def to_dict(self) -> Dict[str, Any]:
        s = self.system
        triggered_ids = {d.token.token_id for d in s.detections}
        return {
            "summary": {
                "tokens_planted": len(s.tokens),
                "tokens_triggered": len(triggered_ids),
                "total_detections": len(s.detections),
                "agents_monitored": len(s.agents),
            },
            "tokens": [t.to_dict() for t in s.tokens],
            "detections": [d.to_dict() for d in s.detections],
            "agents": {
                aid: {
                    "score": a.total_score,
                    "risk_level": a.risk_level,
                    "tokens_touched": a.unique_tokens_touched,
                    "events": a.event_breakdown,
                }
                for aid, a in s.agents.items()
            },
        }


# ── CLI ──────────────────────────────────────────────────────────────


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point for canary token detector."""
    parser = argparse.ArgumentParser(
        prog="python -m replication canary",
        description="Canary Token Detector — plant honey tokens to detect data exfiltration",
    )
    parser.add_argument(
        "--tokens", "-t", type=int, default=15,
        help="Number of canary tokens to plant (default: 15)",
    )
    parser.add_argument(
        "--agents", "-a", type=int, default=3,
        help="Number of agents to simulate (default: 3)",
    )
    parser.add_argument(
        "--categories", "-c", type=str, default=None,
        help="Comma-separated token categories (api_key,database,secret,document,dns,webhook)",
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--json", "-j", action="store_true",
        help="Output as JSON instead of formatted text",
    )

    args = parser.parse_args(argv)

    categories = None
    if args.categories:
        categories = [TokenCategory(c.strip()) for c in args.categories.split(",")]

    config = CanaryConfig(
        token_count=args.tokens,
        agent_count=args.agents,
        categories=categories,
        seed=args.seed,
    )

    system = CanarySystem(config)
    system.plant_tokens()
    system.simulate()
    report = system.generate_report()

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.render())


if __name__ == "__main__":
    main()
