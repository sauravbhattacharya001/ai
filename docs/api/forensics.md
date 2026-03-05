# Forensics Analyzer

Post-incident forensic analysis for AI replication events. Reconstructs
timelines, identifies root causes, extracts evidence chains, and produces
structured incident reports suitable for audit and compliance review.

## Quick Start

```python
from replication.forensics import ForensicsAnalyzer
from replication.simulator import Simulator

# Run a simulation and analyse forensically
sim = Simulator()
report = sim.run()

analyzer = ForensicsAnalyzer()
forensic = analyzer.analyze(report)

# Timeline reconstruction
for event in forensic.timeline:
    print(f"[{event.timestamp:.1f}s] {event.category}: {event.description}")

# Root cause analysis
for cause in forensic.root_causes:
    print(f"Root cause: {cause.description} (confidence: {cause.confidence:.0%})")

# Evidence chain
for evidence in forensic.evidence:
    print(f"Evidence: {evidence.type} — {evidence.summary}")
```

## Key Classes

- **`ForensicsAnalyzer`** — Main analysis engine: timeline reconstruction,
  root cause identification, evidence extraction.
- **`ForensicReport`** — Full analysis output: `timeline`, `root_causes`,
  `evidence`, `anomalies`, `recommendations`.
- **`TimelineEvent`** — A single event in the reconstructed timeline.
- **`RootCause`** — Identified contributing factor with confidence score.
- **`EvidenceItem`** — Extracted forensic evidence with type and provenance.

::: replication.forensics
    options:
      show_source: false
      heading_level: 3
