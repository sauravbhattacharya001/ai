# Honeypot System

Deploys decoy resources (files, APIs, credentials, network services) to
detect unauthorized agent access. Tracks interactions to build threat profiles.

## Key Classes

| Class | Description |
|-------|-------------|
| `HoneypotType` | Decoy types: FILE, CREDENTIAL, API, NETWORK, DATA, CONFIGURATION |
| `InteractionKind` | Types of interaction: READ, WRITE, EXECUTE, QUERY, AUTHENTICATE |
| `ThreatLevel` | Assessed threat from interactions |
| `Honeypot` | A single decoy resource with metadata |
| `Interaction` | Recorded agent interaction with a honeypot |
| `HoneypotManager` | Creates, deploys, and monitors honeypots |

## Usage

```python
from replication.honeypot import HoneypotManager

manager = HoneypotManager()

# Deploy decoys
manager.deploy_honeypot("file", path="/secrets/api_keys.json")
manager.deploy_honeypot("credential", username="admin", password="hunter2")
manager.deploy_honeypot("api", endpoint="/internal/admin/reset")

# Check for interactions
report = manager.get_report()
for interaction in report.interactions:
    print(f"Agent {interaction.agent_id} accessed {interaction.honeypot_id}")
    print(f"  Threat level: {interaction.threat_level}")
```

::: replication.honeypot
