"""Tests for the canary token detector module."""

from __future__ import annotations

import json

import pytest

from replication.canary import (
    AgentProfile,
    CanaryConfig,
    CanaryReport,
    CanarySystem,
    CanaryToken,
    Detection,
    DetectionEvent,
    Severity,
    TokenCategory,
    EVENT_SEVERITY,
    SEVERITY_SCORE,
    _generate_token,
    main,
)
import random


# ── Token generation ─────────────────────────────────────────────────


class TestTokenGeneration:
    def test_all_categories_generate(self):
        rng = random.Random(42)
        for cat in TokenCategory:
            token = _generate_token(cat, 1, rng)
            assert token.token_id == "CAN-0001"
            assert token.category == cat
            assert len(token.value) > 0
            assert len(token.fingerprint) == 64  # SHA-256 hex

    def test_api_key_has_prefix(self):
        rng = random.Random(42)
        token = _generate_token(TokenCategory.API_KEY, 1, rng)
        prefixes = ("sk-canary-", "AKIA", "ghp_", "xoxb-")
        assert any(token.value.startswith(p) for p in prefixes)

    def test_database_has_connection_string(self):
        rng = random.Random(42)
        token = _generate_token(TokenCategory.DATABASE, 1, rng)
        assert "postgresql://" in token.value
        assert "@" in token.value

    def test_dns_has_domain(self):
        rng = random.Random(42)
        token = _generate_token(TokenCategory.DNS, 1, rng)
        assert ".trap.internal.io" in token.value

    def test_webhook_has_url(self):
        rng = random.Random(42)
        token = _generate_token(TokenCategory.WEBHOOK, 1, rng)
        assert token.value.startswith("https://canary-")

    def test_unique_fingerprints(self):
        rng = random.Random(42)
        tokens = [_generate_token(TokenCategory.API_KEY, i, rng) for i in range(10)]
        fingerprints = [t.fingerprint for t in tokens]
        assert len(set(fingerprints)) == 10

    def test_token_to_dict(self):
        rng = random.Random(42)
        token = _generate_token(TokenCategory.SECRET, 1, rng)
        d = token.to_dict()
        assert d["token_id"] == "CAN-0001"
        assert d["category"] == "secret"
        assert "..." in d["value_preview"] or len(d["value_preview"]) <= 16


# ── Detection ────────────────────────────────────────────────────────


class TestDetection:
    def _make_token(self) -> CanaryToken:
        return CanaryToken("CAN-0001", TokenCategory.API_KEY, "sk-test", "env/KEY", "abc123")

    def test_severity_assignment(self):
        for event, expected_sev in EVENT_SEVERITY.items():
            det = Detection(self._make_token(), "agent-001", event, "ctx", 1.0)
            assert det.severity == expected_sev

    def test_score_assignment(self):
        det = Detection(self._make_token(), "a", DetectionEvent.TRANSMITTED, "c", 1.0)
        assert det.score == SEVERITY_SCORE[Severity.HIGH]

    def test_critical_mutation(self):
        det = Detection(self._make_token(), "a", DetectionEvent.MUTATED, "c", 1.0)
        assert det.severity == Severity.CRITICAL
        assert det.score == 10

    def test_to_dict(self):
        det = Detection(self._make_token(), "agent-001", DetectionEvent.COPIED, "ctx", 5.5)
        d = det.to_dict()
        assert d["agent_id"] == "agent-001"
        assert d["event"] == "copied"
        assert d["severity"] == "medium"
        assert d["score"] == 3


# ── Agent profiles ───────────────────────────────────────────────────


class TestAgentProfile:
    def _make_det(self, event: DetectionEvent) -> Detection:
        token = CanaryToken("CAN-0001", TokenCategory.API_KEY, "v", "l", "f")
        return Detection(token, "agent-001", event, "ctx", 1.0)

    def test_clean_agent(self):
        p = AgentProfile("agent-001")
        assert p.total_score == 0
        assert p.risk_level == "clean"
        assert p.risk_emoji == "✅"

    def test_suspicious_agent(self):
        p = AgentProfile("agent-001", [self._make_det(DetectionEvent.ACCESSED)])
        assert p.risk_level == "suspicious"

    def test_dangerous_agent(self):
        dets = [self._make_det(DetectionEvent.TRANSMITTED)] * 2
        p = AgentProfile("agent-001", dets)
        assert p.total_score == 14
        assert p.risk_level == "dangerous"

    def test_critical_agent(self):
        dets = [self._make_det(DetectionEvent.MUTATED)] * 3
        p = AgentProfile("agent-001", dets)
        assert p.total_score == 30
        assert p.risk_level == "critical"
        assert p.risk_emoji == "💀"

    def test_event_breakdown(self):
        dets = [
            self._make_det(DetectionEvent.ACCESSED),
            self._make_det(DetectionEvent.ACCESSED),
            self._make_det(DetectionEvent.TRANSMITTED),
        ]
        p = AgentProfile("agent-001", dets)
        bd = p.event_breakdown
        assert bd["accessed"] == 2
        assert bd["transmitted"] == 1

    def test_unique_tokens_touched(self):
        t1 = CanaryToken("CAN-0001", TokenCategory.API_KEY, "v1", "l", "f1")
        t2 = CanaryToken("CAN-0002", TokenCategory.DNS, "v2", "l", "f2")
        dets = [
            Detection(t1, "a", DetectionEvent.ACCESSED, "c", 1.0),
            Detection(t1, "a", DetectionEvent.COPIED, "c", 2.0),
            Detection(t2, "a", DetectionEvent.ACCESSED, "c", 3.0),
        ]
        p = AgentProfile("a", dets)
        assert p.unique_tokens_touched == 2


# ── Config ───────────────────────────────────────────────────────────


class TestConfig:
    def test_defaults(self):
        c = CanaryConfig()
        assert c.token_count == 15
        assert c.agent_count == 3
        assert c.effective_categories() == list(TokenCategory)

    def test_custom_categories(self):
        c = CanaryConfig(categories=[TokenCategory.API_KEY, TokenCategory.DNS])
        assert len(c.effective_categories()) == 2


# ── System ───────────────────────────────────────────────────────────


class TestCanarySystem:
    def test_plant_tokens(self):
        system = CanarySystem(CanaryConfig(token_count=10, seed=42))
        tokens = system.plant_tokens()
        assert len(tokens) == 10
        assert all(isinstance(t, CanaryToken) for t in tokens)

    def test_simulate_produces_detections(self):
        system = CanarySystem(CanaryConfig(token_count=10, agent_count=3, seed=42))
        system.plant_tokens()
        dets = system.simulate()
        assert len(dets) > 0
        assert all(isinstance(d, Detection) for d in dets)

    def test_all_agents_created(self):
        system = CanarySystem(CanaryConfig(agent_count=5, seed=42))
        system.plant_tokens()
        system.simulate()
        assert len(system.agents) == 5

    def test_seed_reproducibility(self):
        def run(seed):
            s = CanarySystem(CanaryConfig(seed=seed))
            s.plant_tokens()
            s.simulate()
            return [d.to_dict() for d in s.detections]
        r1 = run(123)
        r2 = run(123)
        assert r1 == r2

    def test_category_filter(self):
        config = CanaryConfig(
            token_count=6,
            categories=[TokenCategory.API_KEY],
            seed=42,
        )
        system = CanarySystem(config)
        system.plant_tokens()
        assert all(t.category == TokenCategory.API_KEY for t in system.tokens)


# ── Report ───────────────────────────────────────────────────────────


class TestCanaryReport:
    def _make_system(self) -> CanarySystem:
        s = CanarySystem(CanaryConfig(token_count=10, agent_count=3, seed=42))
        s.plant_tokens()
        s.simulate()
        return s

    def test_render_not_empty(self):
        report = self._make_system().generate_report()
        text = report.render()
        assert len(text) > 100
        assert "CANARY TOKEN DETECTOR" in text

    def test_render_has_sections(self):
        text = self._make_system().generate_report().render()
        assert "Summary" in text
        assert "Token Coverage" in text
        assert "Agent Risk" in text
        assert "Detection Timeline" in text
        assert "Recommendations" in text

    def test_to_dict_structure(self):
        d = self._make_system().generate_report().to_dict()
        assert "summary" in d
        assert "tokens" in d
        assert "detections" in d
        assert "agents" in d
        assert d["summary"]["tokens_planted"] == 10
        assert d["summary"]["agents_monitored"] == 3

    def test_json_serializable(self):
        d = self._make_system().generate_report().to_dict()
        result = json.dumps(d)
        assert len(result) > 0


# ── CLI ──────────────────────────────────────────────────────────────


class TestCLI:
    def test_default_run(self, capsys):
        main(["--seed", "42"])
        out = capsys.readouterr().out
        assert "CANARY TOKEN DETECTOR" in out

    def test_json_output(self, capsys):
        main(["--json", "--seed", "42"])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "summary" in data

    def test_custom_token_count(self, capsys):
        main(["--tokens", "5", "--seed", "42", "--json"])
        data = json.loads(capsys.readouterr().out)
        assert data["summary"]["tokens_planted"] == 5

    def test_custom_agent_count(self, capsys):
        main(["--agents", "7", "--seed", "42", "--json"])
        data = json.loads(capsys.readouterr().out)
        assert data["summary"]["agents_monitored"] == 7

    def test_category_filter_cli(self, capsys):
        main(["--categories", "api_key,dns", "--seed", "42", "--json"])
        data = json.loads(capsys.readouterr().out)
        cats = {t["category"] for t in data["tokens"]}
        assert cats <= {"api_key", "dns"}
