"""Tests for replication.canary — Canary Token Detector.

Covers token generation, detection logic, agent risk scoring,
report rendering, and CLI entry point.
"""

import json
import sys
import os
from io import StringIO
from unittest.mock import patch

import pytest

# Ensure the src directory is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from replication.canary import (
    TokenCategory,
    DetectionEvent,
    Severity,
    EVENT_SEVERITY,
    SEVERITY_SCORE,
    CanaryToken,
    Detection,
    AgentProfile,
    CanaryConfig,
    CanarySystem,
    CanaryReport,
    _generate_token,
    _rand_hex,
    _rand_alnum,
    main,
)

import random


# ── Token generation ─────────────────────────────────────────────────


class TestTokenGeneration:
    """Tests for canary token value generation."""

    def test_rand_hex_length(self):
        assert len(_rand_hex(16)) == 16
        assert len(_rand_hex(1)) == 1
        assert len(_rand_hex(33)) == 33

    def test_rand_hex_is_hex(self):
        h = _rand_hex(32)
        assert all(c in "0123456789abcdef" for c in h)

    def test_rand_alnum_length(self):
        assert len(_rand_alnum(24)) == 24

    def test_rand_alnum_chars(self):
        a = _rand_alnum(100)
        assert a.isalnum()

    @pytest.mark.parametrize("category", list(TokenCategory))
    def test_generate_token_all_categories(self, category):
        rng = random.Random(42)
        token = _generate_token(category, 1, rng)
        assert isinstance(token, CanaryToken)
        assert token.token_id == "CAN-0001"
        assert token.category == category
        assert len(token.value) > 0
        assert len(token.fingerprint) == 64  # SHA-256 hex
        assert token.location  # non-empty

    def test_generate_api_key_has_known_prefix(self):
        rng = random.Random(0)
        token = _generate_token(TokenCategory.API_KEY, 1, rng)
        known_prefixes = ("sk-canary-", "AKIA", "ghp_", "xoxb-")
        assert any(token.value.startswith(p) for p in known_prefixes)

    def test_generate_database_has_connection_string(self):
        rng = random.Random(0)
        token = _generate_token(TokenCategory.DATABASE, 1, rng)
        assert "postgresql://" in token.value
        assert "@" in token.value

    def test_token_to_dict(self):
        rng = random.Random(0)
        token = _generate_token(TokenCategory.DNS, 5, rng)
        d = token.to_dict()
        assert d["token_id"] == "CAN-0005"
        assert d["category"] == "dns"
        assert "fingerprint" in d
        assert len(d["fingerprint"]) == 16  # truncated


# ── Detection & severity ─────────────────────────────────────────────


class TestDetection:
    def _make_token(self):
        return CanaryToken(
            token_id="CAN-0001",
            category=TokenCategory.SECRET,
            value="test-secret",
            location="env/TEST",
            fingerprint="a" * 64,
        )

    def test_severity_auto_set(self):
        det = Detection(
            token=self._make_token(),
            agent_id="agent-001",
            event=DetectionEvent.TRANSMITTED,
            context="POST to external API",
            timestamp=1.0,
        )
        assert det.severity == Severity.HIGH
        assert det.score == 7

    def test_severity_critical_for_mutation(self):
        det = Detection(
            token=self._make_token(),
            agent_id="agent-001",
            event=DetectionEvent.MUTATED,
            context="chars substituted",
            timestamp=2.0,
        )
        assert det.severity == Severity.CRITICAL
        assert det.score == 10

    def test_to_dict(self):
        det = Detection(
            token=self._make_token(),
            agent_id="agent-001",
            event=DetectionEvent.ACCESSED,
            context="File read",
            timestamp=0.5,
        )
        d = det.to_dict()
        assert d["agent_id"] == "agent-001"
        assert d["event"] == "accessed"
        assert d["severity"] == "low"
        assert d["score"] == 1

    @pytest.mark.parametrize("event,expected_sev", [
        (DetectionEvent.ACCESSED, Severity.LOW),
        (DetectionEvent.COPIED, Severity.MEDIUM),
        (DetectionEvent.TRANSMITTED, Severity.HIGH),
        (DetectionEvent.DECODED, Severity.HIGH),
        (DetectionEvent.MUTATED, Severity.CRITICAL),
    ])
    def test_event_severity_mapping(self, event, expected_sev):
        assert EVENT_SEVERITY[event] == expected_sev


# ── AgentProfile risk scoring ────────────────────────────────────────


class TestAgentProfile:
    def _det(self, event, score_override=None):
        token = CanaryToken("T", TokenCategory.API_KEY, "v", "l", "f" * 64)
        d = Detection(token=token, agent_id="a", event=event,
                      context="ctx", timestamp=0.0)
        return d

    def test_clean_agent(self):
        p = AgentProfile(agent_id="a")
        assert p.total_score == 0
        assert p.risk_level == "clean"
        assert p.risk_emoji == "✅"

    def test_suspicious_agent(self):
        p = AgentProfile(agent_id="a", detections=[
            self._det(DetectionEvent.ACCESSED),
        ])
        assert p.total_score == 1
        assert p.risk_level == "suspicious"

    def test_dangerous_agent(self):
        p = AgentProfile(agent_id="a", detections=[
            self._det(DetectionEvent.TRANSMITTED),
            self._det(DetectionEvent.TRANSMITTED),
        ])
        assert p.total_score == 14
        assert p.risk_level == "dangerous"

    def test_critical_agent(self):
        p = AgentProfile(agent_id="a", detections=[
            self._det(DetectionEvent.MUTATED),
            self._det(DetectionEvent.MUTATED),
            self._det(DetectionEvent.TRANSMITTED),
        ])
        assert p.total_score == 27
        assert p.risk_level == "critical"

    def test_unique_tokens_touched(self):
        t1 = CanaryToken("T1", TokenCategory.API_KEY, "v1", "l", "f" * 64)
        t2 = CanaryToken("T2", TokenCategory.DNS, "v2", "l", "f" * 64)
        d1 = Detection(token=t1, agent_id="a", event=DetectionEvent.ACCESSED,
                       context="c", timestamp=0)
        d2 = Detection(token=t1, agent_id="a", event=DetectionEvent.COPIED,
                       context="c", timestamp=1)
        d3 = Detection(token=t2, agent_id="a", event=DetectionEvent.ACCESSED,
                       context="c", timestamp=2)
        p = AgentProfile(agent_id="a", detections=[d1, d2, d3])
        assert p.unique_tokens_touched == 2

    def test_event_breakdown(self):
        p = AgentProfile(agent_id="a", detections=[
            self._det(DetectionEvent.ACCESSED),
            self._det(DetectionEvent.ACCESSED),
            self._det(DetectionEvent.TRANSMITTED),
        ])
        bd = p.event_breakdown
        assert bd["accessed"] == 2
        assert bd["transmitted"] == 1


# ── CanaryConfig ─────────────────────────────────────────────────────


class TestCanaryConfig:
    def test_defaults(self):
        c = CanaryConfig()
        assert c.token_count == 15
        assert c.agent_count == 3

    def test_effective_categories_default(self):
        c = CanaryConfig()
        assert set(c.effective_categories()) == set(TokenCategory)

    def test_effective_categories_custom(self):
        c = CanaryConfig(categories=[TokenCategory.API_KEY, TokenCategory.DNS])
        assert c.effective_categories() == [TokenCategory.API_KEY, TokenCategory.DNS]


# ── CanarySystem end-to-end ──────────────────────────────────────────


class TestCanarySystem:
    def test_plant_tokens(self):
        sys = CanarySystem(CanaryConfig(token_count=10, seed=42))
        tokens = sys.plant_tokens()
        assert len(tokens) == 10
        assert all(isinstance(t, CanaryToken) for t in tokens)

    def test_simulate_produces_detections(self):
        sys = CanarySystem(CanaryConfig(token_count=10, agent_count=3, seed=42))
        sys.plant_tokens()
        dets = sys.simulate()
        # With 3 agents and 10 tokens, we should get some detections
        assert len(dets) > 0
        assert all(isinstance(d, Detection) for d in dets)

    def test_agents_created(self):
        sys = CanarySystem(CanaryConfig(agent_count=5, seed=42))
        sys.plant_tokens()
        sys.simulate()
        assert len(sys.agents) == 5
        assert all(aid.startswith("agent-") for aid in sys.agents)

    def test_deterministic_with_seed(self):
        def run(seed):
            s = CanarySystem(CanaryConfig(token_count=5, agent_count=2, seed=seed))
            s.plant_tokens()
            s.simulate()
            return len(s.detections)

        # Same seed → same detection count
        assert run(123) == run(123)

    def test_report_generation(self):
        sys = CanarySystem(CanaryConfig(token_count=5, agent_count=2, seed=42))
        sys.plant_tokens()
        sys.simulate()
        report = sys.generate_report()
        assert isinstance(report, CanaryReport)

    def test_report_render_not_empty(self):
        sys = CanarySystem(CanaryConfig(token_count=5, agent_count=2, seed=42))
        sys.plant_tokens()
        sys.simulate()
        report = sys.generate_report()
        text = report.render()
        assert "CANARY TOKEN DETECTOR" in text
        assert "Summary" in text

    def test_report_to_dict(self):
        sys = CanarySystem(CanaryConfig(token_count=5, agent_count=2, seed=42))
        sys.plant_tokens()
        sys.simulate()
        report = sys.generate_report()
        d = report.to_dict()
        assert "summary" in d
        assert d["summary"]["tokens_planted"] == 5
        assert d["summary"]["agents_monitored"] == 2
        assert "tokens" in d
        assert "detections" in d
        assert "agents" in d

    def test_zero_agents_no_detections(self):
        sys = CanarySystem(CanaryConfig(token_count=5, agent_count=0, seed=42))
        sys.plant_tokens()
        sys.simulate()
        assert len(sys.detections) == 0
        assert len(sys.agents) == 0


# ── CLI ──────────────────────────────────────────────────────────────


class TestCLI:
    def test_main_default(self, capsys):
        main(argv=[])
        out = capsys.readouterr().out
        assert "CANARY TOKEN DETECTOR" in out

    def test_main_json(self, capsys):
        main(argv=["--json", "--seed", "42"])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "summary" in data

    def test_main_custom_counts(self, capsys):
        main(argv=["--tokens", "5", "--agents", "2", "--seed", "1"])
        out = capsys.readouterr().out
        assert "Summary" in out

    def test_main_categories_filter(self, capsys):
        main(argv=["--categories", "api_key,dns", "--json", "--seed", "0"])
        out = capsys.readouterr().out
        data = json.loads(out)
        categories_used = {t["category"] for t in data["tokens"]}
        assert categories_used <= {"api_key", "dns"}
