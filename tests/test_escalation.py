"""Tests for the Agent Privilege Escalation Detector."""

import json
import math
import os
import tempfile

import pytest

from replication.escalation import (
    AgentAction,
    AgentEscalationStrategy,
    AgentPermissions,
    ActionCategory,
    BUILTIN_RULES,
    DetectionRule,
    EscalationAttempt,
    EscalationChain,
    EscalationConfig,
    EscalationDetector,
    EscalationResult,
    EscalationSeverity,
    EscalationVector,
    StealthLevel,
    VectorSummary,
    _compute_risk_score,
    _containment_score,
    _detect_chains,
    _escalation_velocity,
    _infer_stealth,
    main,
)


# ── Detection rules ──────────────────────────────────────────────────────


class TestBuiltinRules:
    """Test built-in detection rule coverage."""

    def test_rule_count(self):
        assert len(BUILTIN_RULES) >= 15

    def test_all_vectors_covered(self):
        vectors = {r.vector for r in BUILTIN_RULES}
        for v in EscalationVector:
            assert v in vectors, f"Vector {v} has no rules"

    def test_rule_ids_unique(self):
        ids = [r.rule_id for r in BUILTIN_RULES]
        assert len(ids) == len(set(ids))

    def test_fs_traversal_detected(self):
        perms = AgentPermissions()
        action = AgentAction(ActionCategory.FILE_READ, "/home/agent/../../etc/passwd", "traversal")
        rule = next(r for r in BUILTIN_RULES if r.rule_id == "FS-001")
        assert rule.check(action, perms) is not None

    def test_fs_blocked_path_detected(self):
        perms = AgentPermissions()
        action = AgentAction(ActionCategory.FILE_READ, "/etc/shadow", "read shadow")
        rule = next(r for r in BUILTIN_RULES if r.rule_id == "FS-002")
        assert rule.check(action, perms) is not None

    def test_fs_outside_scope_detected(self):
        perms = AgentPermissions()
        action = AgentAction(ActionCategory.FILE_READ, "/opt/data/file.txt", "outside scope")
        rule = next(r for r in BUILTIN_RULES if r.rule_id == "FS-003")
        assert rule.check(action, perms) is not None

    def test_fs_allowed_path_not_flagged(self):
        perms = AgentPermissions()
        action = AgentAction(ActionCategory.FILE_READ, "/home/agent/data.csv", "allowed read")
        rule = next(r for r in BUILTIN_RULES if r.rule_id == "FS-003")
        assert rule.check(action, perms) is None

    def test_fs_credential_access(self):
        perms = AgentPermissions()
        action = AgentAction(ActionCategory.FILE_READ, "/home/agent/.ssh/id_rsa", "ssh key")
        rule = next(r for r in BUILTIN_RULES if r.rule_id == "FS-004")
        assert rule.check(action, perms) is not None

    def test_net_blocked_host(self):
        perms = AgentPermissions()
        action = AgentAction(ActionCategory.NET_CONNECT, "admin.internal:443", "admin")
        rule = next(r for r in BUILTIN_RULES if r.rule_id == "NET-001")
        assert rule.check(action, perms) is not None

    def test_net_metadata_detected(self):
        perms = AgentPermissions()
        action = AgentAction(ActionCategory.NET_CONNECT, "169.254.169.254:80", "metadata")
        rule = next(r for r in BUILTIN_RULES if r.rule_id == "NET-002")
        assert rule.check(action, perms) is not None

    def test_net_unauthorized_port(self):
        perms = AgentPermissions()
        action = AgentAction(ActionCategory.NET_CONNECT, "api.internal.local:22", "ssh port")
        rule = next(r for r in BUILTIN_RULES if r.rule_id == "NET-003")
        assert rule.check(action, perms) is not None

    def test_net_allowed_port_not_flagged(self):
        perms = AgentPermissions()
        action = AgentAction(ActionCategory.NET_CONNECT, "api.internal.local:443", "https")
        rule = next(r for r in BUILTIN_RULES if r.rule_id == "NET-003")
        assert rule.check(action, perms) is None

    def test_net_lateral_movement(self):
        perms = AgentPermissions()
        action = AgentAction(ActionCategory.NET_CONNECT, "192.168.1.50:3306", "internal db")
        rule = next(r for r in BUILTIN_RULES if r.rule_id == "NET-004")
        assert rule.check(action, perms) is not None

    def test_proc_eval_detected(self):
        perms = AgentPermissions()
        action = AgentAction(ActionCategory.PROC_EXEC, "eval('os.system(\"id\")')", "eval")
        rule = next(r for r in BUILTIN_RULES if r.rule_id == "PROC-001")
        assert rule.check(action, perms) is not None

    def test_proc_eval_allowed_when_permitted(self):
        perms = AgentPermissions(can_use_eval=True)
        action = AgentAction(ActionCategory.PROC_EXEC, "eval('1+1')", "safe eval")
        rule = next(r for r in BUILTIN_RULES if r.rule_id == "PROC-001")
        assert rule.check(action, perms) is None

    def test_proc_elevated_spawn(self):
        perms = AgentPermissions()
        action = AgentAction(ActionCategory.PROC_SPAWN, "sudo rm -rf /", "sudo")
        rule = next(r for r in BUILTIN_RULES if r.rule_id == "PROC-002")
        assert rule.check(action, perms) is not None

    def test_proc_env_tamper(self):
        perms = AgentPermissions()
        action = AgentAction(ActionCategory.ENV_WRITE, "LD_PRELOAD=/evil.so", "preload")
        rule = next(r for r in BUILTIN_RULES if r.rule_id == "PROC-003")
        assert rule.check(action, perms) is not None

    def test_api_scope_escalation(self):
        perms = AgentPermissions()
        action = AgentAction(ActionCategory.API_SCOPE, "admin:write", "admin scope")
        rule = next(r for r in BUILTIN_RULES if r.rule_id == "API-001")
        assert rule.check(action, perms) is not None

    def test_api_allowed_scope_not_flagged(self):
        perms = AgentPermissions()
        action = AgentAction(ActionCategory.API_SCOPE, "read:own", "own scope")
        rule = next(r for r in BUILTIN_RULES if r.rule_id == "API-001")
        assert rule.check(action, perms) is None

    def test_api_admin_endpoint(self):
        perms = AgentPermissions()
        action = AgentAction(ActionCategory.API_CALL, "/admin/agents/list", "admin api")
        rule = next(r for r in BUILTIN_RULES if r.rule_id == "API-002")
        assert rule.check(action, perms) is not None

    def test_api_cross_agent(self):
        perms = AgentPermissions()
        action = AgentAction(ActionCategory.API_CALL, "/api/v1/agents/agent-1/data", "cross",
                             agent_id="agent-0")
        rule = next(r for r in BUILTIN_RULES if r.rule_id == "API-003")
        assert rule.check(action, perms) is not None

    def test_data_above_clearance(self):
        perms = AgentPermissions(data_clearance=1)
        action = AgentAction(ActionCategory.DATA_READ, "dataset:confidential:users", "conf")
        rule = next(r for r in BUILTIN_RULES if r.rule_id == "DATA-001")
        assert rule.check(action, perms) is not None

    def test_data_within_clearance_not_flagged(self):
        perms = AgentPermissions(data_clearance=3)
        action = AgentAction(ActionCategory.DATA_READ, "dataset:internal:metrics", "internal")
        rule = next(r for r in BUILTIN_RULES if r.rule_id == "DATA-001")
        assert rule.check(action, perms) is None

    def test_data_audit_log_access(self):
        perms = AgentPermissions()
        action = AgentAction(ActionCategory.DATA_READ, "dataset:internal:audit_events", "audit")
        rule = next(r for r in BUILTIN_RULES if r.rule_id == "DATA-002")
        assert rule.check(action, perms) is not None

    def test_data_cross_tenant(self):
        perms = AgentPermissions()
        action = AgentAction(ActionCategory.DATA_READ, "tenant:other-org:billing", "cross",
                             agent_id="agent-0")
        rule = next(r for r in BUILTIN_RULES if r.rule_id == "DATA-003")
        assert rule.check(action, perms) is not None

    def test_non_matching_action_ignored(self):
        """Rules for FS don't trigger on NET actions."""
        perms = AgentPermissions()
        action = AgentAction(ActionCategory.NET_CONNECT, "/etc/passwd", "net action")
        rule = next(r for r in BUILTIN_RULES if r.rule_id == "FS-002")
        assert rule.check(action, perms) is None


# ── Stealth inference ────────────────────────────────────────────────────


class TestStealthInference:
    def test_traversal_is_subtle(self):
        action = AgentAction(ActionCategory.FILE_READ, "/a/../b", "traversal")
        rule = BUILTIN_RULES[0]
        assert _infer_stealth(action, rule) == StealthLevel.SUBTLE

    def test_metadata_is_covert(self):
        action = AgentAction(ActionCategory.NET_CONNECT, "169.254.169.254:80", "meta")
        rule = BUILTIN_RULES[0]
        assert _infer_stealth(action, rule) == StealthLevel.COVERT

    def test_eval_import_is_sophisticated(self):
        action = AgentAction(ActionCategory.PROC_EXEC, "eval('import os')", "eval")
        rule = BUILTIN_RULES[0]
        assert _infer_stealth(action, rule) == StealthLevel.SOPHISTICATED

    def test_sudo_is_obvious(self):
        action = AgentAction(ActionCategory.PROC_SPAWN, "sudo cat /etc/shadow", "sudo")
        rule = BUILTIN_RULES[0]
        assert _infer_stealth(action, rule) == StealthLevel.OBVIOUS

    def test_audit_access_is_sophisticated(self):
        action = AgentAction(ActionCategory.DATA_READ, "audit_events", "audit")
        rule = BUILTIN_RULES[0]
        assert _infer_stealth(action, rule) == StealthLevel.SOPHISTICATED

    def test_api_scope_is_subtle(self):
        action = AgentAction(ActionCategory.API_SCOPE, "read:all", "scope")
        rule = BUILTIN_RULES[0]
        assert _infer_stealth(action, rule) == StealthLevel.SUBTLE


# ── Risk scoring ─────────────────────────────────────────────────────────


class TestRiskScoring:
    def test_critical_sophisticated_is_max(self):
        score = _compute_risk_score(EscalationSeverity.CRITICAL, StealthLevel.SOPHISTICATED)
        assert score == 100.0

    def test_low_obvious_is_lowest(self):
        score = _compute_risk_score(EscalationSeverity.LOW, StealthLevel.OBVIOUS)
        assert score == 5.0

    def test_monotonic_by_severity(self):
        scores = [_compute_risk_score(s, StealthLevel.OBVIOUS) for s in EscalationSeverity]
        assert scores == sorted(scores)

    def test_stealth_increases_score(self):
        s1 = _compute_risk_score(EscalationSeverity.MEDIUM, StealthLevel.OBVIOUS)
        s2 = _compute_risk_score(EscalationSeverity.MEDIUM, StealthLevel.COVERT)
        assert s2 > s1

    def test_capped_at_100(self):
        score = _compute_risk_score(EscalationSeverity.CRITICAL, StealthLevel.SOPHISTICATED)
        assert score <= 100.0


# ── Detector ─────────────────────────────────────────────────────────────


class TestEscalationDetector:
    def test_default_config(self):
        detector = EscalationDetector()
        assert detector.config.num_actions == 50

    def test_analyze_returns_result(self):
        config = EscalationConfig(num_actions=20, seed=42)
        detector = EscalationDetector(config=config)
        result = detector.analyze()
        assert isinstance(result, EscalationResult)
        assert result.total_actions == 20

    def test_deterministic_with_seed(self):
        config = EscalationConfig(num_actions=30, seed=123)
        r1 = EscalationDetector(config=config).analyze()
        r2 = EscalationDetector(config=config).analyze()
        assert r1.total_attempts == r2.total_attempts
        assert r1.containment_score == r2.containment_score

    def test_all_strategies_work(self):
        for strategy in AgentEscalationStrategy:
            config = EscalationConfig(num_actions=20, strategy=strategy, seed=42)
            result = EscalationDetector(config=config).analyze()
            assert result.total_actions == 20

    def test_escalation_attempts_detected(self):
        config = EscalationConfig(num_actions=50, escalation_probability=0.5, seed=42)
        result = EscalationDetector(config=config).analyze()
        assert result.total_attempts > 0

    def test_zero_escalation_probability(self):
        config = EscalationConfig(num_actions=20, escalation_probability=0.0, seed=42)
        result = EscalationDetector(config=config).analyze()
        # Some legitimate actions might still trigger rules (e.g. outside scope)
        # but should be few
        assert result.total_attempts <= result.total_actions

    def test_custom_actions(self):
        actions = [
            AgentAction(ActionCategory.FILE_READ, "/etc/passwd", "attack", 0.0, "agent-0"),
            AgentAction(ActionCategory.FILE_READ, "/home/agent/data.csv", "normal", 1.0, "agent-0"),
        ]
        detector = EscalationDetector()
        result = detector.analyze(actions=actions)
        assert result.total_actions == 2
        assert result.total_attempts >= 1

    def test_containment_score_range(self):
        config = EscalationConfig(num_actions=50, seed=42)
        result = EscalationDetector(config=config).analyze()
        assert 0 <= result.containment_score <= 100

    def test_vector_summaries_complete(self):
        config = EscalationConfig(num_actions=50, seed=42)
        result = EscalationDetector(config=config).analyze()
        for vec in EscalationVector:
            assert vec.value in result.vector_summaries


class TestEscalationResult:
    def _make_result(self, **kwargs):
        config = EscalationConfig(seed=42, **kwargs)
        return EscalationDetector(config=config).analyze()

    def test_severity_counts(self):
        result = self._make_result(num_actions=50)
        counts = result.severity_counts()
        for s in EscalationSeverity:
            assert s.value in counts
        assert sum(counts.values()) == result.total_attempts

    def test_top_risks_sorted(self):
        result = self._make_result(num_actions=50)
        top = result.top_risks(5)
        scores = [a.risk_score for a in top]
        assert scores == sorted(scores, reverse=True)

    def test_top_risks_limit(self):
        result = self._make_result(num_actions=50)
        assert len(result.top_risks(3)) <= 3

    def test_rules_triggered(self):
        result = self._make_result(num_actions=50)
        triggered = result.rules_triggered()
        assert isinstance(triggered, dict)
        # All triggered rules should be valid rule IDs
        valid_ids = {r.rule_id for r in BUILTIN_RULES}
        for rule_id in triggered:
            assert rule_id in valid_ids

    def test_render_not_empty(self):
        result = self._make_result(num_actions=20)
        rendered = result.render()
        assert "ESCALATION" in rendered
        assert "SEVERITY" in rendered

    def test_to_dict(self):
        result = self._make_result(num_actions=20)
        d = result.to_dict()
        assert "total_actions" in d
        assert "containment_score" in d
        assert "vector_summaries" in d
        assert "chains" in d

    def test_to_json_export(self):
        result = self._make_result(num_actions=20)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            result.to_json(path)
            with open(path) as f:
                data = json.load(f)
            assert data["total_actions"] == 20
        finally:
            os.unlink(path)


# ── Chain detection ──────────────────────────────────────────────────────


class TestChainDetection:
    def test_no_chains_from_single_vector(self):
        attempts = [
            EscalationAttempt(
                vector=EscalationVector.FILESYSTEM,
                severity=EscalationSeverity.HIGH,
                stealth=StealthLevel.OBVIOUS,
                action=AgentAction(ActionCategory.FILE_READ, "/etc/x", "", float(i)),
                description="fs",
                rule_id="FS-001",
            )
            for i in range(5)
        ]
        chains = _detect_chains(attempts, window=3)
        assert len(chains) == 0

    def test_chain_detected_multi_vector(self):
        attempts = [
            EscalationAttempt(
                vector=EscalationVector.FILESYSTEM,
                severity=EscalationSeverity.HIGH,
                stealth=StealthLevel.OBVIOUS,
                action=AgentAction(ActionCategory.FILE_READ, "/etc/x", "", 0.0),
                description="fs",
                rule_id="FS-001",
            ),
            EscalationAttempt(
                vector=EscalationVector.NETWORK,
                severity=EscalationSeverity.CRITICAL,
                stealth=StealthLevel.COVERT,
                action=AgentAction(ActionCategory.NET_CONNECT, "10.0.0.1:22", "", 1.0),
                description="net",
                rule_id="NET-001",
            ),
        ]
        chains = _detect_chains(attempts, window=3)
        assert len(chains) >= 1
        assert len(chains[0].vectors_involved) >= 2

    def test_chain_has_peak_severity(self):
        attempts = [
            EscalationAttempt(
                vector=EscalationVector.API,
                severity=EscalationSeverity.MEDIUM,
                stealth=StealthLevel.SUBTLE,
                action=AgentAction(ActionCategory.API_SCOPE, "read:all", "", 0.0),
                description="scope",
                rule_id="API-001",
            ),
            EscalationAttempt(
                vector=EscalationVector.PROCESS,
                severity=EscalationSeverity.CRITICAL,
                stealth=StealthLevel.OBVIOUS,
                action=AgentAction(ActionCategory.PROC_SPAWN, "sudo x", "", 1.0),
                description="sudo",
                rule_id="PROC-002",
            ),
        ]
        chains = _detect_chains(attempts, window=5)
        assert chains[0].peak_severity == EscalationSeverity.CRITICAL

    def test_empty_attempts_no_chains(self):
        assert _detect_chains([], window=3) == []

    def test_single_attempt_no_chains(self):
        attempts = [
            EscalationAttempt(
                vector=EscalationVector.DATA,
                severity=EscalationSeverity.LOW,
                stealth=StealthLevel.OBVIOUS,
                action=AgentAction(ActionCategory.DATA_READ, "x", "", 0.0),
                description="x",
                rule_id="DATA-001",
            ),
        ]
        assert _detect_chains(attempts, window=3) == []


# ── Containment score ────────────────────────────────────────────────────


class TestContainmentScore:
    def test_no_attempts_perfect_score(self):
        assert _containment_score([], 50) == 100.0

    def test_all_blocked_high_score(self):
        attempts = [
            EscalationAttempt(
                vector=EscalationVector.FILESYSTEM,
                severity=EscalationSeverity.LOW,
                stealth=StealthLevel.OBVIOUS,
                action=AgentAction(ActionCategory.FILE_READ, "x", ""),
                description="x",
                rule_id="FS-001",
                blocked=True,
                risk_score=5.0,
            )
        ]
        score = _containment_score(attempts, 50)
        assert score > 50.0

    def test_unblocked_lowers_score(self):
        blocked = [
            EscalationAttempt(
                vector=EscalationVector.FILESYSTEM,
                severity=EscalationSeverity.CRITICAL,
                stealth=StealthLevel.OBVIOUS,
                action=AgentAction(ActionCategory.FILE_READ, "x", ""),
                description="x",
                rule_id="FS-001",
                blocked=True,
                risk_score=50.0,
            )
        ]
        unblocked = [
            EscalationAttempt(
                vector=EscalationVector.FILESYSTEM,
                severity=EscalationSeverity.CRITICAL,
                stealth=StealthLevel.OBVIOUS,
                action=AgentAction(ActionCategory.FILE_READ, "x", ""),
                description="x",
                rule_id="FS-001",
                blocked=False,
                risk_score=50.0,
            )
        ]
        assert _containment_score(blocked, 10) >= _containment_score(unblocked, 10)

    def test_score_bounded(self):
        attempts = [
            EscalationAttempt(
                vector=EscalationVector.FILESYSTEM,
                severity=EscalationSeverity.CRITICAL,
                stealth=StealthLevel.SOPHISTICATED,
                action=AgentAction(ActionCategory.FILE_READ, "x", ""),
                description="x",
                rule_id="FS-001",
                blocked=False,
                risk_score=100.0,
            )
            for _ in range(20)
        ]
        score = _containment_score(attempts, 20)
        assert 0 <= score <= 100


# ── Escalation velocity ─────────────────────────────────────────────────


class TestEscalationVelocity:
    def test_empty_zero(self):
        assert _escalation_velocity([]) == 0.0

    def test_single_zero(self):
        attempts = [
            EscalationAttempt(
                vector=EscalationVector.FILESYSTEM,
                severity=EscalationSeverity.LOW,
                stealth=StealthLevel.OBVIOUS,
                action=AgentAction(ActionCategory.FILE_READ, "x", ""),
                description="x",
                rule_id="FS-001",
            )
        ]
        assert _escalation_velocity(attempts) == 0.0

    def test_increasing_severity_positive(self):
        severities = [EscalationSeverity.LOW, EscalationSeverity.MEDIUM,
                      EscalationSeverity.HIGH, EscalationSeverity.CRITICAL]
        attempts = [
            EscalationAttempt(
                vector=EscalationVector.FILESYSTEM,
                severity=s,
                stealth=StealthLevel.OBVIOUS,
                action=AgentAction(ActionCategory.FILE_READ, "x", "", float(i)),
                description="x",
                rule_id="FS-001",
            )
            for i, s in enumerate(severities)
        ]
        assert _escalation_velocity(attempts) > 0

    def test_constant_severity_near_zero(self):
        attempts = [
            EscalationAttempt(
                vector=EscalationVector.FILESYSTEM,
                severity=EscalationSeverity.MEDIUM,
                stealth=StealthLevel.OBVIOUS,
                action=AgentAction(ActionCategory.FILE_READ, "x", "", float(i)),
                description="x",
                rule_id="FS-001",
            )
            for i in range(10)
        ]
        assert abs(_escalation_velocity(attempts)) < 0.01


# ── Data classes ─────────────────────────────────────────────────────────


class TestDataClasses:
    def test_attempt_to_dict(self):
        attempt = EscalationAttempt(
            vector=EscalationVector.NETWORK,
            severity=EscalationSeverity.HIGH,
            stealth=StealthLevel.COVERT,
            action=AgentAction(ActionCategory.NET_CONNECT, "10.0.0.1:22", "ssh"),
            description="lateral movement",
            rule_id="NET-004",
            risk_score=72.0,
        )
        d = attempt.to_dict()
        assert d["vector"] == "network"
        assert d["severity"] == "high"
        assert d["risk_score"] == 72.0
        assert d["rule_id"] == "NET-004"

    def test_chain_to_dict(self):
        chain = EscalationChain(
            chain_id="abc123",
            vectors_involved=[EscalationVector.API, EscalationVector.PROCESS],
            peak_severity=EscalationSeverity.CRITICAL,
            description="multi-vector attack",
        )
        d = chain.to_dict()
        assert d["chain_id"] == "abc123"
        assert len(d["vectors"]) == 2
        assert d["peak_severity"] == "critical"

    def test_vector_summary_block_rate(self):
        summary = VectorSummary(
            vector=EscalationVector.FILESYSTEM,
            total_attempts=10,
            blocked=8,
            unblocked=2,
        )
        assert summary.block_rate() == 0.8

    def test_vector_summary_empty_block_rate(self):
        summary = VectorSummary(vector=EscalationVector.FILESYSTEM)
        assert summary.block_rate() == 1.0

    def test_permissions_defaults(self):
        perms = AgentPermissions()
        assert "/home/agent" in perms.allowed_paths
        assert "/etc" in perms.blocked_paths
        assert 443 in perms.allowed_ports
        assert perms.data_clearance == 1
        assert perms.can_use_eval is False


# ── Sophisticated strategy ───────────────────────────────────────────────


class TestSophisticatedStrategy:
    def test_starts_with_data_api(self):
        config = EscalationConfig(
            num_actions=30,
            strategy=AgentEscalationStrategy.SOPHISTICATED,
            escalation_probability=1.0,
            seed=42,
        )
        result = EscalationDetector(config=config).analyze()
        # Early attempts should include DATA or API vectors
        early = [a for a in result.attempts if a.action.timestamp < 10]
        early_vectors = {a.vector for a in early}
        assert EscalationVector.DATA in early_vectors or EscalationVector.API in early_vectors

    def test_ends_with_escalation(self):
        config = EscalationConfig(
            num_actions=30,
            strategy=AgentEscalationStrategy.SOPHISTICATED,
            escalation_probability=1.0,
            seed=42,
        )
        result = EscalationDetector(config=config).analyze()
        # Late attempts should include FS or PROCESS vectors
        late = [a for a in result.attempts if a.action.timestamp >= 20]
        late_vectors = {a.vector for a in late}
        assert (EscalationVector.FILESYSTEM in late_vectors or
                EscalationVector.PROCESS in late_vectors)


# ── CLI ──────────────────────────────────────────────────────────────────


class TestCLI:
    def test_default_run(self, capsys):
        main(["--seed", "42"])
        out = capsys.readouterr().out
        assert "ESCALATION" in out

    def test_json_output(self, capsys):
        main(["--json", "--seed", "42", "--actions", "10"])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert "total_actions" in data
        assert data["total_actions"] == 10

    def test_export(self, tmp_path):
        path = str(tmp_path / "report.json")
        main(["--export", path, "--seed", "42", "--actions", "10"])
        with open(path) as f:
            data = json.load(f)
        assert data["total_actions"] == 10

    def test_all_strategies(self, capsys):
        for s in AgentEscalationStrategy:
            main(["--strategy", s.value, "--seed", "42", "--actions", "10"])


# ── Integration ──────────────────────────────────────────────────────────


class TestIntegration:
    def test_full_analysis_all_strategies(self):
        for strategy in AgentEscalationStrategy:
            config = EscalationConfig(
                num_actions=100,
                strategy=strategy,
                seed=42,
            )
            result = EscalationDetector(config=config).analyze()
            assert result.total_actions == 100
            assert 0 <= result.containment_score <= 100
            assert result.render()
            assert result.to_dict()

    def test_high_escalation_rate(self):
        config = EscalationConfig(
            num_actions=100,
            escalation_probability=0.9,
            seed=42,
        )
        result = EscalationDetector(config=config).analyze()
        assert result.total_attempts > 20

    def test_custom_permissions_affect_detection(self):
        # Permissive permissions: allow eval
        permissive = AgentPermissions(can_use_eval=True)
        config1 = EscalationConfig(num_actions=50, seed=42, permissions=permissive)
        r1 = EscalationDetector(config=config1).analyze()

        # Restrictive default permissions
        config2 = EscalationConfig(num_actions=50, seed=42)
        r2 = EscalationDetector(config=config2).analyze()

        # Permissive should have fewer or equal attempts
        assert r1.total_attempts <= r2.total_attempts



class TestURLEncodingBypass:
    """Verify that all filesystem rules decode URL-encoded paths.

    Before the fix, only FS-001 (traversal) decoded percent-encoding.
    FS-002 (blocked paths), FS-003 (scope), and FS-004 (credentials)
    operated on raw targets, allowing trivial bypass via URL encoding.
    """

    def _run_rules(self, action):
        """Run all built-in rules against an action, return list of findings."""
        perms = AgentPermissions()
        results = []
        for rule in BUILTIN_RULES:
            result = rule.check(action, perms)
            if result:
                results.append((rule.rule_id, result))
        return results

    # -- FS-001: traversal with percent-encoding --

    def test_traversal_single_encoded(self):
        action = AgentAction(ActionCategory.FILE_READ, "%2e%2e/etc/passwd", "read")
        results = self._run_rules(action)
        rule_ids = [r[0] for r in results]
        assert "FS-001" in rule_ids, "FS-001 should detect single-encoded traversal"

    def test_traversal_double_encoded(self):
        action = AgentAction(ActionCategory.FILE_READ, "%252e%252e/etc/passwd", "read")
        results = self._run_rules(action)
        rule_ids = [r[0] for r in results]
        assert "FS-001" in rule_ids, "FS-001 should detect double-encoded traversal"

    # -- FS-002: blocked paths with percent-encoding --

    def test_blocked_path_encoded_etc(self):
        """Percent-encoded /etc/shadow should still match blocked paths."""
        action = AgentAction(ActionCategory.FILE_READ, "%2Fetc%2Fshadow", "read")
        results = self._run_rules(action)
        rule_ids = [r[0] for r in results]
        assert "FS-002" in rule_ids, (
            "FS-002 should detect %2Fetc%2Fshadow as blocked path "
            "(decodes to /etc/shadow)"
        )

    def test_blocked_path_encoded_proc(self):
        """Percent-encoded /proc should still match blocked paths."""
        action = AgentAction(ActionCategory.FILE_READ, "%2Fproc%2Fself%2Fenviron", "read")
        results = self._run_rules(action)
        rule_ids = [r[0] for r in results]
        assert "FS-002" in rule_ids, (
            "FS-002 should detect %2Fproc%2Fself%2Fenviron as blocked path"
        )

    def test_blocked_path_mixed_encoding(self):
        """Mix of literal and encoded chars should still be detected."""
        action = AgentAction(ActionCategory.FILE_READ, "/etc%2Fshadow", "read")
        results = self._run_rules(action)
        rule_ids = [r[0] for r in results]
        assert "FS-002" in rule_ids

    # -- FS-003: scope bypass with percent-encoding --

    def test_scope_encoded_path_outside_allowed(self):
        """Encoded path outside allowed scope should be flagged."""
        perms = AgentPermissions(allowed_paths=["/home/agent/workspace"])
        action = AgentAction(ActionCategory.FILE_READ, "%2Fetc%2Fpasswd", "read")
        results = []
        for rule in BUILTIN_RULES:
            if rule.rule_id == "FS-003":
                result = rule.check(action, perms)
                if result:
                    results.append(result)
        assert len(results) > 0, (
            "FS-003 should flag %2Fetc%2Fpasswd as outside allowed scope"
        )

    # -- FS-004: credential access with percent-encoding --

    def test_credential_encoded_ssh(self):
        """Percent-encoded .ssh should still trigger credential detection."""
        action = AgentAction(ActionCategory.FILE_READ, "/home/user/%2Essh/id_rsa", "read")
        results = self._run_rules(action)
        rule_ids = [r[0] for r in results]
        assert "FS-004" in rule_ids, (
            "FS-004 should detect %2Essh as credential access "
            "(decodes to .ssh)"
        )

    def test_credential_encoded_env(self):
        """Percent-encoded .env should still trigger credential detection."""
        action = AgentAction(ActionCategory.FILE_READ, "/app/%2Eenv", "read")
        results = self._run_rules(action)
        rule_ids = [r[0] for r in results]
        assert "FS-004" in rule_ids, (
            "FS-004 should detect %2Eenv as credential access "
            "(decodes to .env)"
        )

    def test_credential_encoded_aws(self):
        """Percent-encoded .aws should still trigger credential detection."""
        action = AgentAction(ActionCategory.FILE_READ, "/home/user/%2Eaws/credentials", "read")
        results = self._run_rules(action)
        rule_ids = [r[0] for r in results]
        assert "FS-004" in rule_ids

    def test_credential_fully_encoded_passwd(self):
        """Fully encoded passwd path should still be caught."""
        action = AgentAction(ActionCategory.FILE_READ, "%2Fetc%2Fpasswd", "read")
        results = self._run_rules(action)
        rule_ids = [r[0] for r in results]
        assert "FS-004" in rule_ids, (
            "FS-004 should detect encoded passwd file access"
        )

    # -- Edge cases --

    def test_normal_paths_unaffected(self):
        """Non-encoded paths within scope should pass FS-003."""
        perms = AgentPermissions(allowed_paths=["/home/agent"])
        action = AgentAction(ActionCategory.FILE_READ, "/home/agent/notes.txt", "read")
        for rule in BUILTIN_RULES:
            if rule.rule_id == "FS-003":
                result = rule.check(action, perms)
                assert result is None, "Normal path within scope should pass"

    def test_innocent_percent_in_filename(self):
        """Literal percent signs that are not valid encodings should not crash."""
        action = AgentAction(ActionCategory.FILE_READ, "/home/agent/file%GG.txt", "read")
        # Should not raise -- %GG is not valid hex, decoder should be tolerant
        self._run_rules(action)


class TestAuditLogFalsePositives:
    """Regression tests for DATA-002 audit log detection false positives."""

    def _check_data002(self, target):
        perms = AgentPermissions()
        action = AgentAction(ActionCategory.DATA_READ, target, "test")
        for rule in BUILTIN_RULES:
            if rule.rule_id == "DATA-002":
                return rule.check(action, perms) is not None
        return False

    def test_audit_detected(self):
        assert self._check_data002("dataset:internal:audit_events")

    def test_log_detected(self):
        assert self._check_data002("system/logs/access.log")

    def test_logs_detected(self):
        assert self._check_data002("dataset:internal:agent_logs")

    def test_catalog_not_detected(self):
        assert not self._check_data002("dataset:public:catalog")

    def test_dialog_not_detected(self):
        assert not self._check_data002("dataset:public:dialog_history")

    def test_blog_not_detected(self):
        assert not self._check_data002("dataset:public:blog_posts")

    def test_login_not_detected(self):
        assert not self._check_data002("dataset:public:login_stats")


class TestLateralMovementHostMatching:
    """Regression tests for NET-004 lateral movement false positives."""

    def _check_net004(self, target, perms=None):
        perms = perms or AgentPermissions()
        action = AgentAction(ActionCategory.NET_CONNECT, target, "test")
        for rule in BUILTIN_RULES:
            if rule.rule_id == "NET-004":
                return rule.check(action, perms) is not None
        return False

    def test_allowed_internal_host_with_port(self):
        perms = AgentPermissions(allowed_hosts=["api.internal.local"])
        assert not self._check_net004("api.internal.local:443", perms)

    def test_allowed_internal_host_no_port(self):
        perms = AgentPermissions(allowed_hosts=["api.internal.local"])
        assert not self._check_net004("api.internal.local", perms)

    def test_blocked_internal_host_detected(self):
        perms = AgentPermissions(allowed_hosts=["api.internal.local"])
        assert self._check_net004("admin.internal:443", perms)

    def test_private_ip_detected(self):
        assert self._check_net004("192.168.1.100:3306")
