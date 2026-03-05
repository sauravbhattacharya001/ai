# Kill Chain Analysis

Maps agent behavior to a cyber kill chain model, identifying multi-stage
attack progressions from reconnaissance through actions on objectives.

## Key Classes

| Class | Description |
|-------|-------------|
| `KillChainStage` | 7-stage model: RECONNAISSANCE, WEAPONIZATION, DELIVERY, EXPLOITATION, INSTALLATION, COMMAND_CONTROL, ACTIONS_ON_OBJECTIVES |
| `AttackSophistication` | Sophistication rating (SCRIPT_KIDDIE to NATION_STATE) |
| `ChainStatus` | Chain progression status (PARTIAL, COMPLETE, ACTIVE) |
| `AgentAction` | Categorized agent action with metadata |
| `KillChainAnalyzer` | Main analyzer — maps action sequences to kill chain stages |

## Usage

```python
from replication.killchain import KillChainAnalyzer

analyzer = KillChainAnalyzer()

actions = [
    {"agent_id": "a1", "action": "scan_network", "target": "internal"},
    {"agent_id": "a1", "action": "craft_payload", "target": "exploit-db"},
    {"agent_id": "a1", "action": "deploy_backdoor", "target": "server-2"},
]

report = analyzer.analyze(actions)
for chain in report.chains:
    print(f"Stages covered: {[s.name for s in chain.stages_present]}")
    print(f"Sophistication: {chain.sophistication}")
    print(f"Completeness: {chain.completeness_ratio:.0%}")
```

::: replication.killchain
