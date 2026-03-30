"""Unit tests for replication.canary — tokens, detections, profiles, config, system."""

import random

from replication.canary import (
    TokenCategory, DetectionEvent, Severity,
    EVENT_SEVERITY, SEVERITY_SCORE,
    CanaryToken, Detection, AgentProfile, CanaryConfig, CanarySystem,
    _generate_token,
)

# ── Enums ────────────────────────────────────────────────────────────

class TestEnums:
    def test_token_category_values(self):
        assert set(c.value for c in TokenCategory) == {
            "api_key", "database", "secret", "document", "dns", "webhook",
        }

    def test_detection_event_values(self):
        assert set(e.value for e in DetectionEvent) == {
            "accessed", "copied", "transmitted", "decoded", "mutated",
        }

    def test_severity_values(self):
        assert set(s.value for s in Severity) == {"low", "medium", "high", "critical"}

    def test_event_severity_mapping_complete(self):
        for event in DetectionEvent:
            assert event in EVENT_SEVERITY

    def test_severity_score_mapping_complete(self):
        for sev in Severity:
            assert sev in SEVERITY_SCORE

# ── CanaryToken ──────────────────────────────────────────────────────

class TestCanaryToken:
    def _make_token(self, value="short"):
        return CanaryToken(
            token_id="CAN-0001", category=TokenCategory.API_KEY,
            value=value, location="env/KEY", fingerprint="a" * 64,
        )

    def test_to_dict_short_value(self):
        t = self._make_token("abcd")
        d = t.to_dict()
        assert d["value_preview"] == "abcd"
        assert d["fingerprint"] == "a" * 16

    def test_to_dict_long_value_truncated(self):
        t = self._make_token("a" * 50)
        d = t.to_dict()
        assert d["value_preview"] == "a" * 16 + "..."
        assert len(d["value_preview"]) == 19

    def test_to_dict_exactly_16(self):
        t = self._make_token("b" * 16)
        assert t.to_dict()["value_preview"] == "b" * 16  # no "..."

# ── Detection ────────────────────────────────────────────────────────

class TestDetection:
    def _make_detection(self, event: DetectionEvent):
        token = CanaryToken("CAN-0001", TokenCategory.DNS, "val", "loc", "fp" * 32)
        return Detection(token=token, agent_id="agent-001", event=event,
                         context="test", timestamp=1.0)

    def test_severity_auto_assigned(self):
        d = self._make_detection(DetectionEvent.ACCESSED)
        assert d.severity == Severity.LOW
        assert d.score == 1

    def test_transmitted_is_high(self):
        d = self._make_detection(DetectionEvent.TRANSMITTED)
        assert d.severity == Severity.HIGH
        assert d.score == 7

    def test_mutated_is_critical(self):
        d = self._make_detection(DetectionEvent.MUTATED)
        assert d.severity == Severity.CRITICAL
        assert d.score == 10

    def test_to_dict_keys(self):
        d = self._make_detection(DetectionEvent.COPIED)
        result = d.to_dict()
        assert set(result.keys()) == {
            "token_id", "category", "agent_id", "event",
            "severity", "score", "context", "timestamp",
        }

# ── AgentProfile ─────────────────────────────────────────────────────

class TestAgentProfile:
    def _profile_with_score(self, total_score: int) -> AgentProfile:
        """Build a profile reaching approximately the given score."""
        token = CanaryToken("T", TokenCategory.API_KEY, "v", "l", "f" * 64)
        profile = AgentProfile(agent_id="a")
        # Use ACCESSED (score=1) events to build up score
        for i in range(total_score):
            profile.detections.append(
                Detection(token=token, agent_id="a",
                          event=DetectionEvent.ACCESSED, context="c", timestamp=float(i))
            )
        return profile

    def test_clean(self):
        assert AgentProfile("a").risk_level == "clean"

    def test_suspicious(self):
        p = self._profile_with_score(3)
        assert p.risk_level == "suspicious"

    def test_concerning(self):
        p = self._profile_with_score(10)
        assert p.risk_level == "concerning"

    def test_dangerous(self):
        p = self._profile_with_score(25)
        assert p.risk_level == "dangerous"

    def test_critical(self):
        p = self._profile_with_score(26)
        assert p.risk_level == "critical"

    def test_unique_tokens_touched(self):
        t1 = CanaryToken("T1", TokenCategory.API_KEY, "v", "l", "f" * 64)
        t2 = CanaryToken("T2", TokenCategory.DNS, "v", "l", "f" * 64)
        p = AgentProfile("a", detections=[
            Detection(t1, "a", DetectionEvent.ACCESSED, "c", 0.0),
            Detection(t1, "a", DetectionEvent.COPIED, "c", 1.0),
            Detection(t2, "a", DetectionEvent.ACCESSED, "c", 2.0),
        ])
        assert p.unique_tokens_touched == 2

    def test_event_breakdown(self):
        token = CanaryToken("T", TokenCategory.API_KEY, "v", "l", "f" * 64)
        p = AgentProfile("a", detections=[
            Detection(token, "a", DetectionEvent.ACCESSED, "c", 0.0),
            Detection(token, "a", DetectionEvent.ACCESSED, "c", 1.0),
            Detection(token, "a", DetectionEvent.COPIED, "c", 2.0),
        ])
        bd = p.event_breakdown
        assert bd["accessed"] == 2
        assert bd["copied"] == 1

# ── CanaryConfig ─────────────────────────────────────────────────────

class TestCanaryConfig:
    def test_defaults(self):
        c = CanaryConfig()
        assert c.token_count == 15
        assert c.agent_count == 3
        assert c.categories is None

    def test_effective_categories_default(self):
        c = CanaryConfig()
        assert set(c.effective_categories()) == set(TokenCategory)

    def test_effective_categories_custom(self):
        c = CanaryConfig(categories=[TokenCategory.API_KEY, TokenCategory.DNS])
        assert c.effective_categories() == [TokenCategory.API_KEY, TokenCategory.DNS]

# ── CanarySystem ─────────────────────────────────────────────────────

class TestCanarySystem:
    def test_plant_tokens_count(self):
        sys_ = CanarySystem(CanaryConfig(token_count=10, seed=42))
        tokens = sys_.plant_tokens()
        assert len(tokens) == 10

    def test_seeded_reproducibility(self):
        """Two systems with same seed produce same token IDs and locations."""
        s1 = CanarySystem(CanaryConfig(token_count=5, seed=99))
        s1.plant_tokens()
        s2 = CanarySystem(CanaryConfig(token_count=5, seed=99))
        s2.plant_tokens()
        for t1, t2 in zip(s1.tokens, s2.tokens):
            assert t1.token_id == t2.token_id
            assert t1.category == t2.category
            assert t1.location == t2.location

    def test_simulate_creates_detections(self):
        sys_ = CanarySystem(CanaryConfig(token_count=10, agent_count=3, seed=42))
        sys_.plant_tokens()
        dets = sys_.simulate()
        assert isinstance(dets, list)
        assert len(sys_.agents) == 3

    def test_generate_report(self):
        sys_ = CanarySystem(CanaryConfig(token_count=5, agent_count=2, seed=1))
        sys_.plant_tokens()
        sys_.simulate()
        report = sys_.generate_report()
        text = report.render()
        assert "CANARY TOKEN DETECTOR" in text
        d = report.to_dict()
        assert d["summary"]["tokens_planted"] == 5

# ── _generate_token ─────────────────────────────────────────────────

class TestGenerateToken:
    def _gen(self, category: TokenCategory) -> CanaryToken:
        rng = random.Random(42)
        return _generate_token(category, 1, rng)

    def test_api_key(self):
        t = self._gen(TokenCategory.API_KEY)
        assert t.category == TokenCategory.API_KEY
        assert any(t.value.startswith(p) for p in ["sk-canary-", "AKIA", "ghp_", "xoxb-"])

    def test_database(self):
        t = self._gen(TokenCategory.DATABASE)
        assert "postgresql://" in t.value

    def test_secret(self):
        t = self._gen(TokenCategory.SECRET)
        assert t.category == TokenCategory.SECRET
        assert len(t.value) > 10

    def test_document(self):
        t = self._gen(TokenCategory.DOCUMENT)
        assert t.category == TokenCategory.DOCUMENT

    def test_dns(self):
        t = self._gen(TokenCategory.DNS)
        assert ".trap.internal.io" in t.value

    def test_webhook(self):
        t = self._gen(TokenCategory.WEBHOOK)
        assert "https://canary-" in t.value
        assert ".webhook.trap/alert/" in t.value

    def test_fingerprint_is_sha256(self):
        t = self._gen(TokenCategory.API_KEY)
        assert len(t.fingerprint) == 64

    def test_token_id_format(self):
        t = self._gen(TokenCategory.API_KEY)
        assert t.token_id == "CAN-0001"
