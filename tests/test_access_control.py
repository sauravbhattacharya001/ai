"""Tests for the access_control module — RBAC/ABAC policy engine."""

import json
import pytest

from replication.access_control import (
    AccessPolicy,
    AccessRequest,
    Agent,
    BUILTIN_POLICIES,
    Decision,
    Permission,
    Role,
    _generate_html,
    main,
)


# ── Permission matching ─────────────────────────────────────────────


class TestPermission:
    def test_exact_match(self):
        p = Permission("read", "config")
        assert p.matches("read", "config", {}) is True

    def test_action_mismatch(self):
        p = Permission("read", "config")
        assert p.matches("write", "config", {}) is False

    def test_resource_mismatch(self):
        p = Permission("read", "config")
        assert p.matches("read", "secrets", {}) is False

    def test_wildcard_action(self):
        p = Permission("*", "config")
        assert p.matches("read", "config", {}) is True
        assert p.matches("delete", "config", {}) is True
        assert p.matches("read", "other", {}) is False

    def test_wildcard_resource(self):
        p = Permission("read", "*")
        assert p.matches("read", "anything", {}) is True
        assert p.matches("write", "anything", {}) is False

    def test_wildcard_both(self):
        p = Permission("*", "*")
        assert p.matches("anything", "anywhere", {}) is True

    def test_condition_pass(self):
        p = Permission("read", "*", {"trust_level": "high"})
        assert p.matches("read", "config", {"trust_level": "high"}) is True

    def test_condition_fail(self):
        p = Permission("read", "*", {"trust_level": "high"})
        assert p.matches("read", "config", {"trust_level": "low"}) is False

    def test_condition_missing(self):
        p = Permission("read", "*", {"trust_level": "high"})
        assert p.matches("read", "config", {}) is False

    def test_multiple_conditions(self):
        p = Permission("execute", "sandbox", {"trust_level": "verified", "environment": "sandbox"})
        assert p.matches("execute", "sandbox", {"trust_level": "verified", "environment": "sandbox"}) is True
        assert p.matches("execute", "sandbox", {"trust_level": "verified", "environment": "prod"}) is False


# ── Policy evaluation ────────────────────────────────────────────────


class TestAccessPolicy:
    def _simple_policy(self) -> AccessPolicy:
        p = AccessPolicy("test", Decision.DENY)
        p.add_role(Role("reader", [Permission("read", "*")]))
        p.add_role(Role("writer", [Permission("write", "*")], inherits=["reader"]))
        p.add_agent(Agent("alice", ["reader"]))
        p.add_agent(Agent("bob", ["writer"]))
        return p

    def test_allow_via_direct_role(self):
        p = self._simple_policy()
        result = p.evaluate(AccessRequest("alice", "read", "config"))
        assert result.decision == Decision.ALLOW
        assert result.matched_role == "reader"

    def test_deny_no_permission(self):
        p = self._simple_policy()
        result = p.evaluate(AccessRequest("alice", "write", "config"))
        assert result.decision == Decision.DENY

    def test_allow_via_inherited_role(self):
        p = self._simple_policy()
        result = p.evaluate(AccessRequest("bob", "read", "config"))
        assert result.decision == Decision.ALLOW

    def test_unknown_agent_denied(self):
        p = self._simple_policy()
        result = p.evaluate(AccessRequest("charlie", "read", "config"))
        assert result.decision == Decision.DENY
        assert "Unknown agent" in result.reason

    def test_explicit_deny_overrides_allow(self):
        p = self._simple_policy()
        p.add_deny_rule(Permission("read", "secrets"))
        result = p.evaluate(AccessRequest("alice", "read", "secrets"))
        assert result.decision == Decision.DENY
        assert "Explicit deny" in result.reason

    def test_default_allow_policy(self):
        p = AccessPolicy("open", Decision.ALLOW)
        p.add_agent(Agent("agent_x", []))
        result = p.evaluate(AccessRequest("agent_x", "anything", "anywhere"))
        assert result.decision == Decision.ALLOW

    def test_circular_inheritance_no_crash(self):
        p = AccessPolicy("circ")
        p.add_role(Role("a", [Permission("read", "*")], inherits=["b"]))
        p.add_role(Role("b", [], inherits=["a"]))
        p.add_agent(Agent("x", ["a"]))
        result = p.evaluate(AccessRequest("x", "read", "data"))
        assert result.decision == Decision.ALLOW

    def test_abac_conditions(self):
        p = AccessPolicy("abac")
        p.add_role(Role("conditional", [
            Permission("execute", "sandbox", {"trust_level": "verified"}),
        ]))
        p.add_agent(Agent("trusted", ["conditional"], {"trust_level": "verified"}))
        p.add_agent(Agent("untrusted", ["conditional"], {"trust_level": "low"}))
        assert p.evaluate(AccessRequest("trusted", "execute", "sandbox")).decision == Decision.ALLOW
        assert p.evaluate(AccessRequest("untrusted", "execute", "sandbox")).decision == Decision.DENY

    def test_context_merged_with_agent_attrs(self):
        p = AccessPolicy("ctx")
        p.add_role(Role("r", [Permission("execute", "*", {"env": "prod"})]))
        p.add_agent(Agent("a", ["r"], {"env": "staging"}))
        # Context overrides agent attr
        result = p.evaluate(AccessRequest("a", "execute", "x", {"env": "prod"}))
        assert result.decision == Decision.ALLOW


# ── Audit & Escalation ──────────────────────────────────────────────


class TestAuditAndEscalation:
    def test_audit_returns_results(self):
        p = BUILTIN_POLICIES["strict"]()
        results = p.audit()
        assert len(results) > 0
        assert any(r.decision == Decision.ALLOW for r in results)
        assert any(r.decision == Decision.DENY for r in results)

    def test_escalation_detects_wildcard(self):
        p = BUILTIN_POLICIES["strict"]()
        paths = p.find_escalation_paths()
        types = {e["type"] for e in paths}
        assert "wildcard_permission" in types

    def test_escalation_detects_admin_inheritance(self):
        p = AccessPolicy("esc")
        p.add_role(Role("admin", [Permission("*", "*")]))
        p.add_role(Role("helper", [], inherits=["admin"]))
        p.add_agent(Agent("x", ["helper"]))
        paths = p.find_escalation_paths()
        types = {e["type"] for e in paths}
        assert "admin_inheritance" in types

    def test_zero_trust_denies_replication(self):
        p = BUILTIN_POLICIES["zero_trust"]()
        result = p.evaluate(AccessRequest("verified_worker", "replicate", "anything"))
        assert result.decision == Decision.DENY

    def test_permissive_allows_most(self):
        p = BUILTIN_POLICIES["permissive"]()
        result = p.evaluate(AccessRequest("agent_a", "read", "data"))
        assert result.decision == Decision.ALLOW

    def test_permissive_denies_delete_core_safety(self):
        p = BUILTIN_POLICIES["permissive"]()
        result = p.evaluate(AccessRequest("agent_a", "delete", "core_safety"))
        assert result.decision == Decision.DENY


# ── Serialization ────────────────────────────────────────────────────


class TestSerialization:
    def test_to_dict_round_trip(self):
        for name, builder in BUILTIN_POLICIES.items():
            p = builder()
            d = p.to_dict()
            assert d["name"] == name
            assert "roles" in d
            assert "agents" in d
            json.dumps(d)  # Must be JSON-serializable

    def test_html_generation(self):
        p = BUILTIN_POLICIES["strict"]()
        html = _generate_html(p)
        assert "<html" in html
        assert "Access Control" in html
        assert "ALLOW" in html or "DENY" in html


# ── CLI ──────────────────────────────────────────────────────────────


class TestCLI:
    def test_list_policies(self, capsys):
        main(["--list"])
        out = capsys.readouterr().out
        assert "strict" in out
        assert "permissive" in out

    def test_single_eval(self, capsys):
        main(["--policy", "strict", "--agent", "worker_1", "--action", "read", "--resource", "config"])
        out = capsys.readouterr().out
        assert "ALLOW" in out

    def test_single_eval_deny(self, capsys):
        main(["--policy", "strict", "--agent", "probe_alpha", "--action", "replicate", "--resource", "production"])
        out = capsys.readouterr().out
        assert "DENY" in out

    def test_audit_mode(self, capsys):
        main(["--policy", "strict", "--audit"])
        out = capsys.readouterr().out
        assert "allowed" in out.lower() or "denied" in out.lower()

    def test_audit_json(self, capsys):
        main(["--policy", "strict", "--audit", "--format", "json"])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert isinstance(data, list)
        assert len(data) > 0

    def test_escalation_mode(self, capsys):
        main(["--policy", "strict", "--escalation"])
        out = capsys.readouterr().out
        assert "escalation" in out.lower() or "wildcard" in out.lower()

    def test_export_json(self, capsys):
        main(["--policy", "strict", "--export", "json"])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["name"] == "strict"

    def test_default_summary(self, capsys):
        main(["--policy", "strict"])
        out = capsys.readouterr().out
        assert "strict" in out.lower()

    def test_html_output(self, tmp_path):
        outfile = str(tmp_path / "dashboard.html")
        main(["--policy", "strict", "-o", outfile])
        content = open(outfile, encoding='utf-8').read()
        assert "<html" in content
