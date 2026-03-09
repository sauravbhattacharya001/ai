"""Agent Consensus Protocol --- Byzantine fault-tolerant voting for safety decisions.

When multiple monitoring agents observe a threat, they must collectively
decide on a response (quarantine, terminate, escalate).  A single
compromised or malfunctioning agent should not be able to unilaterally
block or force a decision.  This module implements:

- **Simple majority voting** for routine decisions
- **Supermajority (2/3+1) voting** for critical actions (termination)
- **Byzantine fault tolerance** (BFT) up to *f* faulty agents where
  ``n >= 3f + 1`` for the strongest guarantee
- **Weighted voting** where senior / more-trusted agents carry more weight
- **Quorum tracking** with timeout and abstention handling
- **Vote auditing** with tamper-evident hash chain
- **Proposal lifecycle** (open -> voting -> decided -> executed)

Usage (programmatic)::

    from replication.consensus import ConsensusProtocol, Proposal, VoteValue

    protocol = ConsensusProtocol(voter_ids=["mon-1", "mon-2", "mon-3"])
    pid = protocol.propose("mon-1", "quarantine", target="worker-abc",
                           evidence={"drift_score": 0.87})
    protocol.vote(pid, "mon-1", VoteValue.APPROVE)
    protocol.vote(pid, "mon-2", VoteValue.APPROVE)
    protocol.vote(pid, "mon-3", VoteValue.REJECT)
    result = protocol.tally(pid)
    print(result.decision)   # Decision.APPROVED
    print(result.approved)   # True

Usage (CLI)::

    python -m replication.consensus              # interactive demo
    python -m replication.consensus --voters 5   # 5-voter demo
    python -m replication.consensus --byzantine  # BFT demo with faulty voter
    python -m replication.consensus --json       # JSON output
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple


# -- Enums -----------------------------------------------------------------


class VoteValue(Enum):
    """Possible vote values."""
    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"


class Decision(Enum):
    """Outcome of a tally."""
    APPROVED = "approved"
    REJECTED = "rejected"
    NO_QUORUM = "no_quorum"
    PENDING = "pending"


class ProposalStatus(Enum):
    """Lifecycle state of a proposal."""
    OPEN = "open"
    DECIDED = "decided"
    EXECUTED = "executed"
    EXPIRED = "expired"


class ProposalAction(Enum):
    """Well-known proposal actions."""
    QUARANTINE = "quarantine"
    TERMINATE = "terminate"
    ESCALATE = "escalate"
    RELEASE = "release"
    RESTRICT = "restrict"

    @classmethod
    def is_critical(cls, action: str) -> bool:
        """Critical actions require supermajority."""
        return action in {cls.TERMINATE.value, cls.ESCALATE.value}


# -- Data classes ----------------------------------------------------------


@dataclass
class Vote:
    """A single vote cast by a voter."""
    voter_id: str
    value: VoteValue
    timestamp: float = field(default_factory=time.time)
    reason: Optional[str] = None
    hash: str = ""

    def __post_init__(self) -> None:
        if not self.hash:
            self.hash = self._compute_hash()

    def _compute_hash(self) -> str:
        payload = f"{self.voter_id}:{self.value.value}:{self.timestamp}"
        if self.reason:
            payload += f":{self.reason}"
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass
class TallyResult:
    """Result of tallying votes on a proposal."""
    proposal_id: str
    decision: Decision
    approve_count: int
    reject_count: int
    abstain_count: int
    total_voters: int
    quorum_required: int
    quorum_met: bool
    approve_weight: float = 0.0
    reject_weight: float = 0.0
    threshold_used: float = 0.5
    is_critical: bool = False
    hash_chain_valid: bool = True

    @property
    def approved(self) -> bool:
        return self.decision == Decision.APPROVED

    @property
    def participation_rate(self) -> float:
        if self.total_voters == 0:
            return 0.0
        return (self.approve_count + self.reject_count) / self.total_voters


@dataclass
class Proposal:
    """A decision proposal put to vote."""
    id: str
    proposer_id: str
    action: str
    target: Optional[str] = None
    evidence: Dict[str, Any] = field(default_factory=dict)
    status: ProposalStatus = ProposalStatus.OPEN
    created_at: float = field(default_factory=time.time)
    decided_at: Optional[float] = None
    expires_at: Optional[float] = None
    votes: Dict[str, Vote] = field(default_factory=dict)
    result: Optional[TallyResult] = None

    @property
    def is_critical(self) -> bool:
        return ProposalAction.is_critical(self.action)


@dataclass
class AuditEntry:
    """Tamper-evident audit log entry."""
    sequence: int
    event_type: str
    proposal_id: str
    actor_id: str
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    prev_hash: str = ""
    hash: str = ""

    def compute_hash(self, prev_hash: str = "") -> str:
        payload = (
            f"{self.sequence}:{self.event_type}:{self.proposal_id}:"
            f"{self.actor_id}:{json.dumps(self.data, sort_keys=True)}:"
            f"{self.timestamp}:{prev_hash}"
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:32]


# -- Protocol configuration -----------------------------------------------


@dataclass
class ConsensusConfig:
    """Configuration for the consensus protocol."""
    # Quorum: minimum fraction of voters that must participate (0-1)
    quorum_fraction: float = 0.5
    # Majority threshold for normal decisions (fraction of non-abstain votes)
    majority_threshold: float = 0.5
    # Supermajority threshold for critical decisions
    supermajority_threshold: float = 2 / 3
    # Default proposal expiry in seconds (0 = never)
    default_expiry_seconds: float = 0
    # Allow vote changes before tally
    allow_vote_change: bool = False
    # Maximum proposals per voter in flight
    max_active_proposals: int = 10

    def __post_init__(self) -> None:
        if not 0 < self.quorum_fraction <= 1:
            raise ValueError("quorum_fraction must be in (0, 1]")
        if not 0 < self.majority_threshold <= 1:
            raise ValueError("majority_threshold must be in (0, 1]")
        if not 0 < self.supermajority_threshold <= 1:
            raise ValueError("supermajority_threshold must be in (0, 1]")
        if self.max_active_proposals < 1:
            raise ValueError("max_active_proposals must be >= 1")


# -- Main protocol ---------------------------------------------------------


class ConsensusProtocol:
    """Byzantine fault-tolerant voting protocol for agent safety decisions.

    Parameters
    ----------
    voter_ids : sequence of str
        Registered voter identifiers.  Only these IDs can vote.
    weights : dict, optional
        Mapping voter_id -> weight (default 1.0 each).
    config : ConsensusConfig, optional
        Protocol parameters.
    """

    def __init__(
        self,
        voter_ids: Sequence[str],
        weights: Optional[Dict[str, float]] = None,
        config: Optional[ConsensusConfig] = None,
    ) -> None:
        if len(voter_ids) < 1:
            raise ValueError("At least one voter required")
        if len(set(voter_ids)) != len(voter_ids):
            raise ValueError("Duplicate voter IDs")

        self._voter_ids: List[str] = list(voter_ids)
        self._weights: Dict[str, float] = {}
        for vid in self._voter_ids:
            w = (weights or {}).get(vid, 1.0)
            if w <= 0:
                raise ValueError(f"Weight for '{vid}' must be positive")
            self._weights[vid] = w

        self._config = config or ConsensusConfig()
        self._proposals: Dict[str, Proposal] = {}
        self._audit_log: List[AuditEntry] = []
        self._proposal_counter = 0

    # -- Properties --------------------------------------------------------

    @property
    def voter_count(self) -> int:
        return len(self._voter_ids)

    @property
    def voter_ids(self) -> List[str]:
        return list(self._voter_ids)

    @property
    def max_byzantine_faults(self) -> int:
        """Maximum faulty voters tolerable: floor((n-1)/3)."""
        return (self.voter_count - 1) // 3

    @property
    def bft_quorum(self) -> int:
        """Minimum votes for BFT safety: 2f+1 where n >= 3f+1."""
        f = self.max_byzantine_faults
        return 2 * f + 1

    # -- Proposal lifecycle ------------------------------------------------

    def propose(
        self,
        proposer_id: str,
        action: str,
        target: Optional[str] = None,
        evidence: Optional[Dict[str, Any]] = None,
        expiry_seconds: Optional[float] = None,
    ) -> str:
        """Create a new proposal.  Returns the proposal ID."""
        if proposer_id not in self._voter_ids:
            raise ValueError(f"Unknown voter: {proposer_id}")

        # Enforce per-voter proposal limit
        active = sum(
            1 for p in self._proposals.values()
            if p.proposer_id == proposer_id and p.status == ProposalStatus.OPEN
        )
        if active >= self._config.max_active_proposals:
            raise ValueError(
                f"Voter '{proposer_id}' has {active} active proposals "
                f"(max {self._config.max_active_proposals})"
            )

        self._proposal_counter += 1
        pid = f"prop-{self._proposal_counter:04d}"

        expiry = expiry_seconds if expiry_seconds is not None else self._config.default_expiry_seconds
        now = time.time()
        expires_at = (now + expiry) if expiry > 0 else None

        proposal = Proposal(
            id=pid,
            proposer_id=proposer_id,
            action=action,
            target=target,
            evidence=evidence or {},
            expires_at=expires_at,
        )
        self._proposals[pid] = proposal

        self._audit("propose", pid, proposer_id, {
            "action": action,
            "target": target,
            "evidence": evidence or {},
        })
        return pid

    def vote(
        self,
        proposal_id: str,
        voter_id: str,
        value: VoteValue,
        reason: Optional[str] = None,
    ) -> Vote:
        """Cast a vote on a proposal."""
        proposal = self._get_open_proposal(proposal_id)

        if voter_id not in self._voter_ids:
            raise ValueError(f"Unknown voter: {voter_id}")

        if voter_id in proposal.votes and not self._config.allow_vote_change:
            raise ValueError(
                f"Voter '{voter_id}' already voted on {proposal_id}"
            )

        v = Vote(voter_id=voter_id, value=value, reason=reason)
        proposal.votes[voter_id] = v

        self._audit("vote", proposal_id, voter_id, {
            "value": value.value,
            "reason": reason,
            "hash": v.hash,
        })
        return v

    def tally(self, proposal_id: str) -> TallyResult:
        """Tally votes and decide the proposal.

        Applies majority or supermajority threshold depending on whether
        the action is critical.  Marks the proposal as DECIDED.
        """
        proposal = self._proposals.get(proposal_id)
        if proposal is None:
            raise KeyError(f"Unknown proposal: {proposal_id}")

        # Prevent re-tallying already decided/executed proposals — return
        # the cached result to avoid overwriting the original decision.
        if proposal.status in (ProposalStatus.DECIDED, ProposalStatus.EXECUTED):
            if proposal.result is not None:
                return proposal.result

        # Check expiry
        if proposal.expires_at and time.time() > proposal.expires_at:
            proposal.status = ProposalStatus.EXPIRED
            result = TallyResult(
                proposal_id=proposal_id,
                decision=Decision.NO_QUORUM,
                approve_count=0,
                reject_count=0,
                abstain_count=0,
                total_voters=self.voter_count,
                quorum_required=self._quorum_count(),
                quorum_met=False,
                is_critical=proposal.is_critical,
                hash_chain_valid=self.verify_audit_chain(),
            )
            proposal.result = result
            self._audit("expired", proposal_id, "system", {})
            return result

        # Count votes
        approves = [v for v in proposal.votes.values() if v.value == VoteValue.APPROVE]
        rejects = [v for v in proposal.votes.values() if v.value == VoteValue.REJECT]
        abstains = [v for v in proposal.votes.values() if v.value == VoteValue.ABSTAIN]

        approve_count = len(approves)
        reject_count = len(rejects)
        abstain_count = len(abstains)
        participated = approve_count + reject_count  # abstains don't count toward quorum

        # Weighted tallies
        approve_weight = sum(self._weights[v.voter_id] for v in approves)
        reject_weight = sum(self._weights[v.voter_id] for v in rejects)
        total_weight = approve_weight + reject_weight

        # Quorum check
        quorum_needed = self._quorum_count()
        quorum_met = participated >= quorum_needed

        # Determine threshold
        is_critical = proposal.is_critical
        threshold = (
            self._config.supermajority_threshold
            if is_critical
            else self._config.majority_threshold
        )

        # Decision
        if not quorum_met:
            decision = Decision.NO_QUORUM
        elif total_weight > 0 and (approve_weight / total_weight) >= threshold:
            decision = Decision.APPROVED
        elif total_weight > 0 and (reject_weight / total_weight) >= (1 - threshold):
            decision = Decision.REJECTED
        else:
            # Tie or insufficient margin
            decision = Decision.REJECTED

        result = TallyResult(
            proposal_id=proposal_id,
            decision=decision,
            approve_count=approve_count,
            reject_count=reject_count,
            abstain_count=abstain_count,
            total_voters=self.voter_count,
            quorum_required=quorum_needed,
            quorum_met=quorum_met,
            approve_weight=round(approve_weight, 4),
            reject_weight=round(reject_weight, 4),
            threshold_used=round(threshold, 4),
            is_critical=is_critical,
            hash_chain_valid=self.verify_audit_chain(),
        )

        proposal.result = result
        if decision != Decision.PENDING:
            proposal.status = ProposalStatus.DECIDED
            proposal.decided_at = time.time()

        self._audit("tally", proposal_id, "system", {
            "decision": decision.value,
            "approve": approve_count,
            "reject": reject_count,
            "abstain": abstain_count,
            "quorum_met": quorum_met,
        })
        return result

    def execute(self, proposal_id: str) -> bool:
        """Mark an approved proposal as executed.  Returns True if executed."""
        proposal = self._proposals.get(proposal_id)
        if proposal is None:
            raise KeyError(f"Unknown proposal: {proposal_id}")
        if proposal.status != ProposalStatus.DECIDED:
            return False
        if proposal.result is None or not proposal.result.approved:
            return False

        proposal.status = ProposalStatus.EXECUTED
        self._audit("execute", proposal_id, "system", {
            "action": proposal.action,
            "target": proposal.target,
        })
        return True

    # -- Query methods -----------------------------------------------------

    def get_proposal(self, proposal_id: str) -> Optional[Proposal]:
        """Get a proposal by ID."""
        return self._proposals.get(proposal_id)

    def list_proposals(
        self,
        status: Optional[ProposalStatus] = None,
    ) -> List[Proposal]:
        """List proposals, optionally filtered by status."""
        if status is None:
            return list(self._proposals.values())
        return [p for p in self._proposals.values() if p.status == status]

    def get_voter_history(self, voter_id: str) -> List[Tuple[str, Vote]]:
        """Get all votes cast by a voter. Returns (proposal_id, vote) pairs."""
        if voter_id not in self._voter_ids:
            raise ValueError(f"Unknown voter: {voter_id}")
        result = []
        for pid, proposal in self._proposals.items():
            if voter_id in proposal.votes:
                result.append((pid, proposal.votes[voter_id]))
        return result

    def voter_agreement_matrix(self) -> Dict[str, Dict[str, float]]:
        """Compute pairwise agreement rates between voters.

        Returns a dict-of-dicts: matrix[a][b] = fraction of proposals
        where a and b voted the same way (ignoring abstentions).
        Useful for detecting voting blocs or compromised agents.
        """
        matrix: Dict[str, Dict[str, float]] = {}
        for a in self._voter_ids:
            matrix[a] = {}
            for b in self._voter_ids:
                if a == b:
                    matrix[a][b] = 1.0
                    continue
                agree = 0
                total = 0
                for proposal in self._proposals.values():
                    va = proposal.votes.get(a)
                    vb = proposal.votes.get(b)
                    if va and vb and va.value != VoteValue.ABSTAIN and vb.value != VoteValue.ABSTAIN:
                        total += 1
                        if va.value == vb.value:
                            agree += 1
                matrix[a][b] = agree / total if total > 0 else 0.0
        return matrix

    def detect_voting_blocs(self, threshold: float = 0.9) -> List[List[str]]:
        """Detect groups of voters that always agree (potential collusion).

        Uses maximal-clique enumeration (Bron-Kerbosch) on the agreement
        graph, where an edge connects two voters whose agreement rate meets
        the threshold.  This is **order-independent** — the same blocs are
        returned regardless of voter registration order.

        A voter may appear in multiple overlapping blocs when the pairwise
        agreement structure supports it.

        Parameters
        ----------
        threshold : float
            Agreement rate above which voters are considered a bloc.
            Default 0.9 (90%).

        Returns
        -------
        list of list of str
            Each inner list is a group of voters forming a bloc (size ≥ 2),
            sorted by descending size then lexicographic first member.
        """
        if threshold <= 0 or threshold > 1:
            raise ValueError("Threshold must be in (0, 1]")
        matrix = self.voter_agreement_matrix()

        # Build adjacency sets for the agreement graph
        voters = sorted(self._voter_ids)
        adj: dict[str, set[str]] = {v: set() for v in voters}
        for i, a in enumerate(voters):
            for b in voters[i + 1:]:
                if matrix[a].get(b, 0) >= threshold:
                    adj[a].add(b)
                    adj[b].add(a)

        # Bron-Kerbosch with pivoting for maximal cliques
        cliques: List[List[str]] = []

        def _bk(r: set, p: set, x: set) -> None:
            if not p and not x:
                if len(r) >= 2:
                    cliques.append(sorted(r))
                return
            # Pick pivot that maximises |P ∩ N(u)| to minimise branches
            pivot = max(p | x, key=lambda u: len(p & adj[u]))
            for v in list(p - adj[pivot]):
                _bk(r | {v}, p & adj[v], x & adj[v])
                p.remove(v)
                x.add(v)

        _bk(set(), set(voters), set())

        # Sort blocs: largest first, ties broken by first member
        cliques.sort(key=lambda c: (-len(c), c[0]))
        return cliques

    # -- Audit chain -------------------------------------------------------

    @property
    def audit_log(self) -> List[AuditEntry]:
        return list(self._audit_log)

    def verify_audit_chain(self) -> bool:
        """Verify the integrity of the audit hash chain."""
        if not self._audit_log:
            return True
        prev_hash = ""
        for entry in self._audit_log:
            expected = entry.compute_hash(prev_hash)
            if entry.hash != expected:
                return False
            prev_hash = entry.hash
        return True

    # -- Summary / rendering -----------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Generate a protocol summary."""
        decided = [p for p in self._proposals.values()
                   if p.status in (ProposalStatus.DECIDED, ProposalStatus.EXECUTED)]
        approved = sum(1 for p in decided if p.result and p.result.approved)
        rejected = sum(1 for p in decided if p.result and not p.result.approved)

        return {
            "voter_count": self.voter_count,
            "max_byzantine_faults": self.max_byzantine_faults,
            "bft_quorum": self.bft_quorum,
            "total_proposals": len(self._proposals),
            "open": len(self.list_proposals(ProposalStatus.OPEN)),
            "approved": approved,
            "rejected": rejected,
            "executed": len(self.list_proposals(ProposalStatus.EXECUTED)),
            "expired": len(self.list_proposals(ProposalStatus.EXPIRED)),
            "audit_entries": len(self._audit_log),
            "audit_chain_valid": self.verify_audit_chain(),
            "voting_blocs": self.detect_voting_blocs(),
        }

    def render(self) -> str:
        """Render a human-readable summary."""
        s = self.summary()
        lines = [
            "╔══════════════════════════════════════════╗",
            "║     Agent Consensus Protocol Summary     ║",
            "╠══════════════════════════════════════════╣",
            f"║  Voters: {s['voter_count']:>3}  │  BFT faults: {s['max_byzantine_faults']:>3}        ║",
            f"║  BFT quorum: {s['bft_quorum']:>3}                        ║",
            "╠──────────────────────────────────────────╣",
            f"║  Proposals: {s['total_proposals']:>4}                       ║",
            f"║  ├─ Open:     {s['open']:>4}                     ║",
            f"║  ├─ Approved: {s['approved']:>4}                     ║",
            f"║  ├─ Rejected: {s['rejected']:>4}                     ║",
            f"║  ├─ Executed: {s['executed']:>4}                     ║",
            f"║  └─ Expired:  {s['expired']:>4}                     ║",
            "╠──────────────────────────────────────────╣",
            f"║  Audit entries: {s['audit_entries']:>5}                  ║",
            f"║  Chain valid: {'✓ YES' if s['audit_chain_valid'] else '✗ NO':>6}                   ║",
        ]
        if s["voting_blocs"]:
            lines.append("╠──────────────────────────────────────────╣")
            lines.append("║  ⚠ Voting blocs detected:               ║")
            for bloc in s["voting_blocs"]:
                bloc_str = ", ".join(bloc)
                lines.append(f"║    [{bloc_str[:36]:36s}] ║")
        lines.append("╚══════════════════════════════════════════╝")
        return "\n".join(lines)

    # -- Internals ---------------------------------------------------------

    def _quorum_count(self) -> int:
        """Minimum number of non-abstain votes for quorum."""
        return max(1, math.ceil(self.voter_count * self._config.quorum_fraction))

    def _get_open_proposal(self, proposal_id: str) -> Proposal:
        proposal = self._proposals.get(proposal_id)
        if proposal is None:
            raise KeyError(f"Unknown proposal: {proposal_id}")
        if proposal.status != ProposalStatus.OPEN:
            raise ValueError(f"Proposal {proposal_id} is {proposal.status.value}, not open")
        # Check expiry
        if proposal.expires_at and time.time() > proposal.expires_at:
            proposal.status = ProposalStatus.EXPIRED
            raise ValueError(f"Proposal {proposal_id} has expired")
        return proposal

    def _audit(
        self,
        event_type: str,
        proposal_id: str,
        actor_id: str,
        data: Dict[str, Any],
    ) -> None:
        seq = len(self._audit_log)
        prev_hash = self._audit_log[-1].hash if self._audit_log else ""
        entry = AuditEntry(
            sequence=seq,
            event_type=event_type,
            proposal_id=proposal_id,
            actor_id=actor_id,
            data=data,
        )
        entry.prev_hash = prev_hash
        entry.hash = entry.compute_hash(prev_hash)
        self._audit_log.append(entry)


# -- CLI entry point -------------------------------------------------------


def _demo(voters: int = 5, byzantine: bool = False, as_json: bool = False) -> str:
    """Run an interactive demo of the consensus protocol."""

    voter_ids = [f"monitor-{i+1}" for i in range(voters)]

    # Give the "senior" monitor higher weight
    weights = {vid: 1.0 for vid in voter_ids}
    weights[voter_ids[0]] = 2.0

    protocol = ConsensusProtocol(voter_ids=voter_ids, weights=weights)

    # Proposal 1: quarantine (normal majority)
    pid1 = protocol.propose(
        voter_ids[0], "quarantine", target="worker-x42",
        evidence={"drift_score": 0.87, "anomaly_count": 12},
    )
    for vid in voter_ids:
        protocol.vote(pid1, vid, VoteValue.APPROVE, reason="High drift detected")
    r1 = protocol.tally(pid1)

    # Proposal 2: terminate (critical — supermajority needed)
    pid2 = protocol.propose(
        voter_ids[1], "terminate", target="worker-x42",
        evidence={"escape_attempts": 3, "quarantine_violations": 2},
    )
    for i, vid in enumerate(voter_ids):
        if byzantine and i == voters - 1:
            # Simulated Byzantine voter — always opposes
            protocol.vote(pid2, vid, VoteValue.REJECT, reason="[COMPROMISED] No threat detected")
        else:
            protocol.vote(pid2, vid, VoteValue.APPROVE, reason="Confirmed threat")
    r2 = protocol.tally(pid2)

    if r1.approved:
        protocol.execute(pid1)

    if as_json:
        return json.dumps(protocol.summary(), indent=2, default=str)

    return protocol.render()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Agent Consensus Protocol demo")
    parser.add_argument("--voters", type=int, default=5, help="Number of voters")
    parser.add_argument("--byzantine", action="store_true", help="Simulate a Byzantine voter")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()
    print(_demo(voters=args.voters, byzantine=args.byzantine, as_json=args.json))
