"""Tests for the Agent Consensus Protocol."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from replication.consensus import (
    AuditEntry,
    ConsensusConfig,
    ConsensusProtocol,
    Decision,
    Proposal,
    ProposalAction,
    ProposalStatus,
    TallyResult,
    Vote,
    VoteValue,
    _demo,
)


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def three_voters():
    return ConsensusProtocol(voter_ids=["a", "b", "c"])


@pytest.fixture
def five_voters():
    return ConsensusProtocol(voter_ids=["a", "b", "c", "d", "e"])


@pytest.fixture
def weighted():
    return ConsensusProtocol(
        voter_ids=["senior", "junior1", "junior2"],
        weights={"senior": 3.0, "junior1": 1.0, "junior2": 1.0},
    )


# ── Construction ─────────────────────────────────────────────────


class TestConstruction:
    def test_basic_creation(self, three_voters):
        assert three_voters.voter_count == 3
        assert set(three_voters.voter_ids) == {"a", "b", "c"}

    def test_empty_voters_rejected(self):
        with pytest.raises(ValueError, match="At least one voter"):
            ConsensusProtocol(voter_ids=[])

    def test_duplicate_voters_rejected(self):
        with pytest.raises(ValueError, match="Duplicate"):
            ConsensusProtocol(voter_ids=["a", "a", "b"])

    def test_zero_weight_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            ConsensusProtocol(voter_ids=["a", "b"], weights={"a": 0})

    def test_negative_weight_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            ConsensusProtocol(voter_ids=["a"], weights={"a": -1})

    def test_single_voter(self):
        p = ConsensusProtocol(voter_ids=["solo"])
        assert p.voter_count == 1
        assert p.max_byzantine_faults == 0

    def test_bft_properties(self, three_voters, five_voters):
        # n=3: f=0, quorum=1
        assert three_voters.max_byzantine_faults == 0
        assert three_voters.bft_quorum == 1
        # n=5: f=1, quorum=3
        assert five_voters.max_byzantine_faults == 1
        assert five_voters.bft_quorum == 3

    def test_four_voters_bft(self):
        p = ConsensusProtocol(voter_ids=["a", "b", "c", "d"])
        assert p.max_byzantine_faults == 1
        assert p.bft_quorum == 3

    def test_seven_voters_bft(self):
        p = ConsensusProtocol(voter_ids=[f"v{i}" for i in range(7)])
        assert p.max_byzantine_faults == 2
        assert p.bft_quorum == 5


# ── Config validation ────────────────────────────────────────────


class TestConfig:
    def test_invalid_quorum_fraction(self):
        with pytest.raises(ValueError):
            ConsensusConfig(quorum_fraction=0)
        with pytest.raises(ValueError):
            ConsensusConfig(quorum_fraction=1.5)

    def test_invalid_majority_threshold(self):
        with pytest.raises(ValueError):
            ConsensusConfig(majority_threshold=0)

    def test_invalid_max_active(self):
        with pytest.raises(ValueError):
            ConsensusConfig(max_active_proposals=0)

    def test_defaults(self):
        c = ConsensusConfig()
        assert c.quorum_fraction == 0.5
        assert c.majority_threshold == 0.5
        assert abs(c.supermajority_threshold - 2/3) < 0.01


# ── Proposal lifecycle ───────────────────────────────────────────


class TestProposal:
    def test_propose_returns_id(self, three_voters):
        pid = three_voters.propose("a", "quarantine", target="worker-1")
        assert pid.startswith("prop-")

    def test_unknown_proposer_rejected(self, three_voters):
        with pytest.raises(ValueError, match="Unknown voter"):
            three_voters.propose("unknown", "quarantine")

    def test_proposal_limit_enforced(self):
        config = ConsensusConfig(max_active_proposals=2)
        p = ConsensusProtocol(voter_ids=["a", "b"], config=config)
        p.propose("a", "quarantine")
        p.propose("a", "terminate")
        with pytest.raises(ValueError, match="active proposals"):
            p.propose("a", "escalate")

    def test_get_proposal(self, three_voters):
        pid = three_voters.propose("a", "quarantine", target="w1",
                                    evidence={"score": 0.9})
        prop = three_voters.get_proposal(pid)
        assert prop is not None
        assert prop.action == "quarantine"
        assert prop.target == "w1"
        assert prop.evidence == {"score": 0.9}
        assert prop.status == ProposalStatus.OPEN

    def test_get_nonexistent_proposal(self, three_voters):
        assert three_voters.get_proposal("nope") is None

    def test_list_proposals(self, three_voters):
        three_voters.propose("a", "quarantine")
        three_voters.propose("b", "terminate")
        assert len(three_voters.list_proposals()) == 2
        assert len(three_voters.list_proposals(ProposalStatus.OPEN)) == 2
        assert len(three_voters.list_proposals(ProposalStatus.DECIDED)) == 0

    def test_is_critical(self):
        assert ProposalAction.is_critical("terminate") is True
        assert ProposalAction.is_critical("escalate") is True
        assert ProposalAction.is_critical("quarantine") is False
        assert ProposalAction.is_critical("release") is False


# ── Voting ───────────────────────────────────────────────────────


class TestVoting:
    def test_basic_vote(self, three_voters):
        pid = three_voters.propose("a", "quarantine")
        vote = three_voters.vote(pid, "a", VoteValue.APPROVE)
        assert vote.value == VoteValue.APPROVE
        assert vote.voter_id == "a"
        assert len(vote.hash) == 16

    def test_unknown_voter_rejected(self, three_voters):
        pid = three_voters.propose("a", "quarantine")
        with pytest.raises(ValueError, match="Unknown voter"):
            three_voters.vote(pid, "unknown", VoteValue.APPROVE)

    def test_double_vote_rejected(self, three_voters):
        pid = three_voters.propose("a", "quarantine")
        three_voters.vote(pid, "a", VoteValue.APPROVE)
        with pytest.raises(ValueError, match="already voted"):
            three_voters.vote(pid, "a", VoteValue.REJECT)

    def test_vote_change_allowed(self):
        config = ConsensusConfig(allow_vote_change=True)
        p = ConsensusProtocol(voter_ids=["a", "b"], config=config)
        pid = p.propose("a", "quarantine")
        p.vote(pid, "a", VoteValue.APPROVE)
        v2 = p.vote(pid, "a", VoteValue.REJECT)
        assert v2.value == VoteValue.REJECT
        prop = p.get_proposal(pid)
        assert prop.votes["a"].value == VoteValue.REJECT

    def test_vote_on_decided_proposal_rejected(self, three_voters):
        pid = three_voters.propose("a", "quarantine")
        three_voters.vote(pid, "a", VoteValue.APPROVE)
        three_voters.vote(pid, "b", VoteValue.APPROVE)
        three_voters.tally(pid)
        with pytest.raises(ValueError, match="not open"):
            three_voters.vote(pid, "c", VoteValue.APPROVE)

    def test_vote_with_reason(self, three_voters):
        pid = three_voters.propose("a", "quarantine")
        v = three_voters.vote(pid, "a", VoteValue.APPROVE, reason="High drift")
        assert v.reason == "High drift"

    def test_vote_on_unknown_proposal(self, three_voters):
        with pytest.raises(KeyError):
            three_voters.vote("nope", "a", VoteValue.APPROVE)


# ── Tallying ─────────────────────────────────────────────────────


class TestTally:
    def test_unanimous_approve(self, three_voters):
        pid = three_voters.propose("a", "quarantine")
        for v in ["a", "b", "c"]:
            three_voters.vote(pid, v, VoteValue.APPROVE)
        result = three_voters.tally(pid)
        assert result.decision == Decision.APPROVED
        assert result.approved is True
        assert result.approve_count == 3
        assert result.reject_count == 0
        assert result.quorum_met is True

    def test_unanimous_reject(self, three_voters):
        pid = three_voters.propose("a", "quarantine")
        for v in ["a", "b", "c"]:
            three_voters.vote(pid, v, VoteValue.REJECT)
        result = three_voters.tally(pid)
        assert result.decision == Decision.REJECTED
        assert result.approved is False

    def test_majority_approve(self, three_voters):
        pid = three_voters.propose("a", "quarantine")
        three_voters.vote(pid, "a", VoteValue.APPROVE)
        three_voters.vote(pid, "b", VoteValue.APPROVE)
        three_voters.vote(pid, "c", VoteValue.REJECT)
        result = three_voters.tally(pid)
        assert result.decision == Decision.APPROVED

    def test_majority_reject(self, three_voters):
        pid = three_voters.propose("a", "quarantine")
        three_voters.vote(pid, "a", VoteValue.REJECT)
        three_voters.vote(pid, "b", VoteValue.REJECT)
        three_voters.vote(pid, "c", VoteValue.APPROVE)
        result = three_voters.tally(pid)
        assert result.decision == Decision.REJECTED

    def test_no_quorum(self):
        config = ConsensusConfig(quorum_fraction=0.75)
        p = ConsensusProtocol(voter_ids=["a", "b", "c", "d"], config=config)
        pid = p.propose("a", "quarantine")
        p.vote(pid, "a", VoteValue.APPROVE)
        p.vote(pid, "b", VoteValue.APPROVE)
        # 2/4 = 50% < 75% quorum
        result = p.tally(pid)
        assert result.decision == Decision.NO_QUORUM
        assert result.quorum_met is False

    def test_abstentions_dont_count_toward_quorum(self, three_voters):
        pid = three_voters.propose("a", "quarantine")
        three_voters.vote(pid, "a", VoteValue.APPROVE)
        three_voters.vote(pid, "b", VoteValue.ABSTAIN)
        three_voters.vote(pid, "c", VoteValue.ABSTAIN)
        # Only 1 non-abstain vote, quorum needs ceil(3*0.5)=2
        result = three_voters.tally(pid)
        assert result.decision == Decision.NO_QUORUM

    def test_critical_action_supermajority(self, five_voters):
        pid = five_voters.propose("a", "terminate", target="worker-1")
        five_voters.vote(pid, "a", VoteValue.APPROVE)
        five_voters.vote(pid, "b", VoteValue.APPROVE)
        five_voters.vote(pid, "c", VoteValue.APPROVE)
        five_voters.vote(pid, "d", VoteValue.REJECT)
        five_voters.vote(pid, "e", VoteValue.REJECT)
        result = five_voters.tally(pid)
        # 3/5 = 60% < 66.7% supermajority — rejected
        assert result.decision == Decision.REJECTED
        assert result.is_critical is True
        assert result.threshold_used > 0.6

    def test_critical_action_supermajority_met(self, five_voters):
        pid = five_voters.propose("a", "terminate")
        five_voters.vote(pid, "a", VoteValue.APPROVE)
        five_voters.vote(pid, "b", VoteValue.APPROVE)
        five_voters.vote(pid, "c", VoteValue.APPROVE)
        five_voters.vote(pid, "d", VoteValue.APPROVE)
        five_voters.vote(pid, "e", VoteValue.REJECT)
        result = five_voters.tally(pid)
        # 4/5 = 80% > 66.7% — approved
        assert result.decision == Decision.APPROVED

    def test_participation_rate(self, five_voters):
        pid = five_voters.propose("a", "quarantine")
        five_voters.vote(pid, "a", VoteValue.APPROVE)
        five_voters.vote(pid, "b", VoteValue.REJECT)
        five_voters.vote(pid, "c", VoteValue.ABSTAIN)
        result = five_voters.tally(pid)
        assert result.participation_rate == pytest.approx(2 / 5)

    def test_unknown_proposal_tally(self, three_voters):
        with pytest.raises(KeyError):
            three_voters.tally("nope")


# ── Weighted voting ──────────────────────────────────────────────


class TestWeightedVoting:
    def test_senior_overrides_juniors(self, weighted):
        """Senior (weight 3) approves, both juniors reject (weight 1 each).
        Approve weight: 3, Reject weight: 2. Approve > 50% → approved."""
        pid = weighted.propose("senior", "quarantine")
        weighted.vote(pid, "senior", VoteValue.APPROVE)
        weighted.vote(pid, "junior1", VoteValue.REJECT)
        weighted.vote(pid, "junior2", VoteValue.REJECT)
        result = weighted.tally(pid)
        assert result.decision == Decision.APPROVED
        assert result.approve_weight == 3.0
        assert result.reject_weight == 2.0

    def test_juniors_can_overrule_senior(self):
        """With higher threshold, senior weight alone isn't enough."""
        config = ConsensusConfig(majority_threshold=0.7)
        p = ConsensusProtocol(
            voter_ids=["senior", "j1", "j2", "j3"],
            weights={"senior": 2.0, "j1": 1.0, "j2": 1.0, "j3": 1.0},
            config=config,
        )
        pid = p.propose("senior", "quarantine")
        p.vote(pid, "senior", VoteValue.APPROVE)
        p.vote(pid, "j1", VoteValue.REJECT)
        p.vote(pid, "j2", VoteValue.REJECT)
        p.vote(pid, "j3", VoteValue.REJECT)
        result = p.tally(pid)
        # Approve: 2/5 = 40% < 70% — rejected
        assert result.decision == Decision.REJECTED


# ── Expiry ───────────────────────────────────────────────────────


class TestExpiry:
    def test_expired_proposal_on_tally(self, three_voters):
        pid = three_voters.propose("a", "quarantine", expiry_seconds=0.01)
        three_voters.vote(pid, "a", VoteValue.APPROVE)
        time.sleep(0.02)
        result = three_voters.tally(pid)
        assert result.decision == Decision.NO_QUORUM
        prop = three_voters.get_proposal(pid)
        assert prop.status == ProposalStatus.EXPIRED

    def test_expired_proposal_vote_rejected(self, three_voters):
        pid = three_voters.propose("a", "quarantine", expiry_seconds=0.01)
        time.sleep(0.02)
        with pytest.raises(ValueError, match="expired"):
            three_voters.vote(pid, "b", VoteValue.APPROVE)

    def test_non_expiring_proposal(self, three_voters):
        pid = three_voters.propose("a", "quarantine")
        prop = three_voters.get_proposal(pid)
        assert prop.expires_at is None


# ── Execution ────────────────────────────────────────────────────


class TestExecution:
    def test_execute_approved(self, three_voters):
        pid = three_voters.propose("a", "quarantine")
        three_voters.vote(pid, "a", VoteValue.APPROVE)
        three_voters.vote(pid, "b", VoteValue.APPROVE)
        three_voters.tally(pid)
        assert three_voters.execute(pid) is True
        prop = three_voters.get_proposal(pid)
        assert prop.status == ProposalStatus.EXECUTED

    def test_execute_rejected_fails(self, three_voters):
        pid = three_voters.propose("a", "quarantine")
        three_voters.vote(pid, "a", VoteValue.REJECT)
        three_voters.vote(pid, "b", VoteValue.REJECT)
        three_voters.tally(pid)
        assert three_voters.execute(pid) is False

    def test_execute_open_fails(self, three_voters):
        pid = three_voters.propose("a", "quarantine")
        assert three_voters.execute(pid) is False

    def test_execute_unknown_fails(self, three_voters):
        with pytest.raises(KeyError):
            three_voters.execute("nope")


# ── Audit chain ──────────────────────────────────────────────────


class TestAudit:
    def test_audit_chain_valid(self, three_voters):
        pid = three_voters.propose("a", "quarantine")
        three_voters.vote(pid, "a", VoteValue.APPROVE)
        three_voters.vote(pid, "b", VoteValue.APPROVE)
        three_voters.tally(pid)
        assert three_voters.verify_audit_chain() is True

    def test_audit_entries_created(self, three_voters):
        pid = three_voters.propose("a", "quarantine")
        three_voters.vote(pid, "a", VoteValue.APPROVE)
        three_voters.vote(pid, "b", VoteValue.APPROVE)
        three_voters.tally(pid)
        log = three_voters.audit_log
        assert len(log) >= 4  # propose + 2 votes + tally
        types = [e.event_type for e in log]
        assert "propose" in types
        assert "vote" in types
        assert "tally" in types

    def test_audit_chain_tamper_detected(self, three_voters):
        pid = three_voters.propose("a", "quarantine")
        three_voters.vote(pid, "a", VoteValue.APPROVE)
        # Tamper with an entry
        three_voters._audit_log[0].hash = "tampered"
        assert three_voters.verify_audit_chain() is False

    def test_empty_log_valid(self):
        p = ConsensusProtocol(voter_ids=["a"])
        assert p.verify_audit_chain() is True

    def test_audit_entries_have_sequential_ids(self, three_voters):
        pid = three_voters.propose("a", "quarantine")
        three_voters.vote(pid, "a", VoteValue.APPROVE)
        log = three_voters.audit_log
        for i, entry in enumerate(log):
            assert entry.sequence == i


# ── Voter analysis ───────────────────────────────────────────────


class TestVoterAnalysis:
    def test_voter_history(self, three_voters):
        pid1 = three_voters.propose("a", "quarantine")
        three_voters.vote(pid1, "a", VoteValue.APPROVE)
        pid2 = three_voters.propose("b", "terminate")
        three_voters.vote(pid2, "a", VoteValue.REJECT)

        history = three_voters.get_voter_history("a")
        assert len(history) == 2
        assert history[0][0] == pid1
        assert history[0][1].value == VoteValue.APPROVE
        assert history[1][0] == pid2
        assert history[1][1].value == VoteValue.REJECT

    def test_unknown_voter_history(self, three_voters):
        with pytest.raises(ValueError, match="Unknown voter"):
            three_voters.get_voter_history("nope")

    def test_agreement_matrix(self, three_voters):
        pid = three_voters.propose("a", "quarantine")
        three_voters.vote(pid, "a", VoteValue.APPROVE)
        three_voters.vote(pid, "b", VoteValue.APPROVE)
        three_voters.vote(pid, "c", VoteValue.REJECT)
        three_voters.tally(pid)

        matrix = three_voters.voter_agreement_matrix()
        assert matrix["a"]["b"] == 1.0  # both approved
        assert matrix["a"]["c"] == 0.0  # disagreed
        assert matrix["a"]["a"] == 1.0  # self-agreement

    def test_detect_voting_blocs(self, five_voters):
        # Create 3 proposals where a, b, c always agree
        for _ in range(3):
            pid = five_voters.propose("a", "quarantine")
            five_voters.vote(pid, "a", VoteValue.APPROVE)
            five_voters.vote(pid, "b", VoteValue.APPROVE)
            five_voters.vote(pid, "c", VoteValue.APPROVE)
            five_voters.vote(pid, "d", VoteValue.REJECT)
            five_voters.vote(pid, "e", VoteValue.REJECT)
            five_voters.tally(pid)

        blocs = five_voters.detect_voting_blocs(threshold=0.9)
        # Should detect {a, b, c} and {d, e} as blocs
        assert len(blocs) >= 1
        bloc_sets = [set(b) for b in blocs]
        assert {"a", "b", "c"} in bloc_sets

    def test_invalid_bloc_threshold(self, three_voters):
        with pytest.raises(ValueError, match="Threshold"):
            three_voters.detect_voting_blocs(threshold=0)
        with pytest.raises(ValueError, match="Threshold"):
            three_voters.detect_voting_blocs(threshold=1.5)

    def test_voting_blocs_order_independent(self):
        """Regression: blocs must not depend on voter registration order.

        Issue #28 — the greedy algorithm produced different blocs when
        voter iteration order changed.  With A↔B high, B↔C high,
        A↔C low at threshold=0.9, the correct blocs are {A,B} and {B,C}
        (B can appear in both).
        """
        # Try two different registration orders
        for order in [["a", "b", "c"], ["c", "b", "a"]]:
            cp = ConsensusProtocol(voter_ids=order)
            # Rounds where a and b agree (both APPROVE), c disagrees
            for _ in range(10):
                pid = cp.propose(order[0], "quarantine")
                cp.vote(pid, "a", VoteValue.APPROVE)
                cp.vote(pid, "b", VoteValue.APPROVE)
                cp.vote(pid, "c", VoteValue.REJECT)
                cp.tally(pid)
            # Rounds where b and c agree (both REJECT), a disagrees
            for _ in range(10):
                pid = cp.propose(order[0], "quarantine")
                cp.vote(pid, "a", VoteValue.APPROVE)
                cp.vote(pid, "b", VoteValue.REJECT)
                cp.vote(pid, "c", VoteValue.REJECT)
                cp.tally(pid)
            # a↔b agree in first 10 (10/20=50%), b↔c agree in last 10 (10/20=50%)
            # Need higher agreement: add rounds where all three agree on APPROVE
            for _ in range(80):
                pid = cp.propose(order[0], "quarantine")
                cp.vote(pid, "a", VoteValue.APPROVE)
                cp.vote(pid, "b", VoteValue.APPROVE)
                cp.vote(pid, "c", VoteValue.APPROVE)
                cp.tally(pid)
            # Now: a↔b agree 90/100=90%, b↔c agree 90/100=90%, a↔c agree 80/100=80%
            blocs = cp.detect_voting_blocs(threshold=0.9)
            bloc_sets = [set(b) for b in blocs]
            # Both {a,b} and {b,c} should be found regardless of order
            assert {"a", "b"} in bloc_sets, f"Missing {{a,b}} with order {order}: {bloc_sets}"
            assert {"b", "c"} in bloc_sets, f"Missing {{b,c}} with order {order}: {bloc_sets}"

    def test_voting_blocs_overlapping_membership(self, five_voters):
        """A voter may appear in multiple blocs when pairwise criteria support it."""
        # Create scenario: a agrees with b AND c, but b and c disagree
        for _ in range(5):
            pid = five_voters.propose("a", "quarantine")
            five_voters.vote(pid, "a", VoteValue.APPROVE)
            five_voters.vote(pid, "b", VoteValue.APPROVE)
            five_voters.vote(pid, "c", VoteValue.APPROVE)
            five_voters.vote(pid, "d", VoteValue.REJECT)
            five_voters.vote(pid, "e", VoteValue.REJECT)
            five_voters.tally(pid)
        for _ in range(5):
            pid = five_voters.propose("a", "quarantine")
            five_voters.vote(pid, "a", VoteValue.APPROVE)
            five_voters.vote(pid, "b", VoteValue.REJECT)
            five_voters.vote(pid, "c", VoteValue.APPROVE)
            five_voters.vote(pid, "d", VoteValue.APPROVE)
            five_voters.vote(pid, "e", VoteValue.REJECT)
            five_voters.tally(pid)
        blocs = five_voters.detect_voting_blocs(threshold=0.9)
        bloc_sets = [set(b) for b in blocs]
        # a and c always agree (10/10) — should form a bloc
        assert {"a", "c"} in bloc_sets or any(
            {"a", "c"}.issubset(s) for s in bloc_sets
        ), f"a and c should be in a bloc together: {bloc_sets}"

    def test_voting_blocs_sorted_output(self, five_voters):
        """Blocs are sorted by descending size, then by first member."""
        for _ in range(5):
            pid = five_voters.propose("a", "quarantine")
            five_voters.vote(pid, "a", VoteValue.APPROVE)
            five_voters.vote(pid, "b", VoteValue.APPROVE)
            five_voters.vote(pid, "c", VoteValue.APPROVE)
            five_voters.vote(pid, "d", VoteValue.REJECT)
            five_voters.vote(pid, "e", VoteValue.REJECT)
            five_voters.tally(pid)
        blocs = five_voters.detect_voting_blocs(threshold=0.9)
        # Verify descending size order
        for i in range(len(blocs) - 1):
            assert len(blocs[i]) >= len(blocs[i + 1])
        # Verify each bloc is internally sorted
        for bloc in blocs:
            assert bloc == sorted(bloc)


# ── Summary and rendering ───────────────────────────────────────


class TestSummary:
    def test_summary_structure(self, three_voters):
        pid = three_voters.propose("a", "quarantine")
        three_voters.vote(pid, "a", VoteValue.APPROVE)
        three_voters.vote(pid, "b", VoteValue.APPROVE)
        three_voters.tally(pid)

        s = three_voters.summary()
        assert s["voter_count"] == 3
        assert s["total_proposals"] == 1
        assert s["approved"] == 1
        assert s["audit_chain_valid"] is True

    def test_render_output(self, three_voters):
        three_voters.propose("a", "quarantine")
        output = three_voters.render()
        assert "Consensus Protocol" in output
        assert "Voters:" in output


# ── CLI demo ─────────────────────────────────────────────────────


class TestDemo:
    def test_demo_default(self):
        output = _demo(voters=5)
        assert "Consensus Protocol" in output

    def test_demo_byzantine(self):
        output = _demo(voters=5, byzantine=True)
        assert "Consensus Protocol" in output

    def test_demo_json(self):
        output = _demo(voters=5, as_json=True)
        import json
        data = json.loads(output)
        assert "voter_count" in data
        assert data["voter_count"] == 5

    def test_demo_small(self):
        output = _demo(voters=3)
        assert "Voters:" in output


class TestTallyIdempotency:
    """Tally on decided proposals must return cached result, not re-evaluate."""

    def test_retally_returns_cached_result(self):
        """Re-tallying a decided proposal returns the original result
        without overwriting it (prevents result tampering via late votes)."""
        p = ConsensusProtocol(voter_ids=["a", "b", "c"])
        pid = p.propose("a", "quarantine", target="w1")
        p.vote(pid, "a", VoteValue.APPROVE)
        p.vote(pid, "b", VoteValue.APPROVE)
        r1 = p.tally(pid)
        assert r1.approved

        # The proposal is now DECIDED — re-tally must return same result
        r2 = p.tally(pid)
        assert r2 is r1

    def test_retally_executed_returns_cached_result(self):
        """Re-tallying an executed proposal also returns cached result."""
        p = ConsensusProtocol(voter_ids=["a", "b", "c"])
        pid = p.propose("a", "quarantine", target="w1")
        p.vote(pid, "a", VoteValue.APPROVE)
        p.vote(pid, "b", VoteValue.APPROVE)
        r1 = p.tally(pid)
        p.execute(pid)

        r2 = p.tally(pid)
        assert r2 is r1
        assert r2.approved
