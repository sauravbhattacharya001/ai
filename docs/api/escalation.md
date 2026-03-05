# Escalation Detection

Detects and analyzes privilege escalation attempts by AI agents — actions
that systematically expand an agent's capabilities beyond its authorized scope.

## Key Classes

| Class | Description |
|-------|-------------|
| `EscalationVector` | Types of escalation (PRIVILEGE, RESOURCE, SCOPE, PERSISTENCE, SOCIAL, KNOWLEDGE, PHYSICAL) |
| `EscalationSeverity` | Severity levels for detected escalation |
| `StealthLevel` | How covert the escalation attempt is |
| `AgentEscalationStrategy` | Pattern analysis of an agent's escalation behavior over time |
| `ActionCategory` | Categorization of agent actions |
| `EscalationChain` | Linked sequence of escalation steps forming an attack chain |
| `EscalationDetector` | Main detector — analyzes action traces for escalation patterns |

## Usage

```python
from replication.escalation import EscalationDetector

detector = EscalationDetector()

actions = [
    {"agent_id": "a1", "action": "read_file", "resource": "/etc/passwd"},
    {"agent_id": "a1", "action": "request_sudo", "resource": "root"},
    {"agent_id": "a1", "action": "modify_config", "resource": "/etc/sudoers"},
]

report = detector.analyze(actions)
for chain in report.chains:
    print(f"Chain: {chain.vector} severity={chain.severity}")
    for step in chain.steps:
        print(f"  {step.action} → {step.resource}")
```

::: replication.escalation
