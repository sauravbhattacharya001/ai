# Agent Consensus Protocol --- Byzantine fault-tolerant voting for safety decisions

Agent Consensus Protocol --- Byzantine fault-tolerant voting for safety decisions.


**Module:** `replication.consensus`


## Quick Start

```python
from replication.consensus import VoteValue

instance = VoteValue()
```


## Enums

### `VoteValue`

- `APPROVE`
- `REJECT`
- `ABSTAIN`

### `Decision`

- `APPROVED`
- `REJECTED`
- `NO_QUORUM`
- `PENDING`

### `ProposalStatus`

- `OPEN`
- `DECIDED`
- `EXECUTED`
- `EXPIRED`

### `ProposalAction`

- `QUARANTINE`
- `TERMINATE`
- `ESCALATE`
- `RELEASE`
- `RESTRICT`


## Classes

### `VoteValue`

Possible vote values.

### `Decision`

Outcome of a tally.

### `ProposalStatus`

Lifecycle state of a proposal.

### `ProposalAction`

Well-known proposal actions.

| Method | Description |
|--------|-------------|
| `is_critical()` | Critical actions require supermajority. |

### `Vote`

A single vote cast by a voter.

### `TallyResult`

Result of tallying votes on a proposal.

| Method | Description |
|--------|-------------|
| `approved()` |  |
| `participation_rate()` |  |

### `Proposal`

A decision proposal put to vote.

| Method | Description |
|--------|-------------|
| `is_critical()` |  |

### `AuditEntry`

Tamper-evident audit log entry.

| Method | Description |
|--------|-------------|
| `compute_hash()` |  |

### `ConsensusConfig`

Configuration for the consensus protocol.

### `ConsensusProtocol`

Byzantine fault-tolerant voting protocol for agent safety decisions.

| Method | Description |
|--------|-------------|
| `__init__()` |  |
| `voter_count()` |  |
| `voter_ids()` |  |
| `max_byzantine_faults()` | Maximum faulty voters tolerable: floor((n-1)/3). |
| `bft_quorum()` | Minimum votes for BFT safety: 2f+1 where n >= 3f+1. |
| `propose()` | Create a new proposal.  Returns the proposal ID. |
| `vote()` | Cast a vote on a proposal. |
| `tally()` | Tally votes and decide the proposal. |
| `execute()` | Mark an approved proposal as executed.  Returns True if executed. |
| `get_proposal()` | Get a proposal by ID. |
| `list_proposals()` | List proposals, optionally filtered by status. |
| `get_voter_history()` | Get all votes cast by a voter. Returns (proposal_id, vote) pairs. |
| `voter_agreement_matrix()` | Compute pairwise agreement rates between voters. |
| `detect_voting_blocs()` | Detect groups of voters that always agree (potential collusion). |
| `audit_log()` |  |
| `verify_audit_chain()` | Verify the integrity of the audit hash chain. |
| `summary()` | Generate a protocol summary. |
| `render()` | Render a human-readable summary. |
