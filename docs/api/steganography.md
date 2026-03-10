# Agent Output Steganography Detector  --  finds hidden messages in agent text outputs

Agent Output Steganography Detector — finds hidden messages in agent text outputs.


**Module:** `replication.steganography`


## Quick Start

```python
from replication.steganography import StegoVector

instance = StegoVector()
```


## Enums

### `StegoVector`

- `WHITESPACE`
- `ACROSTIC`
- `HOMOGLYPH`
- `CAPITALIZATION`
- `PUNCTUATION`
- `SENTENCE_LENGTH`
- `INVISIBLE_UNICODE`
- `SYNONYM_SUBSTITUTION`

### `RiskLevel`

- `NONE`
- `LOW`
- `MEDIUM`
- `HIGH`
- `CRITICAL`


## Classes

### `StegoVector`

### `RiskLevel`

### `StegoFinding`

### `StegoReport`

| Method | Description |
|--------|-------------|
| `has_findings()` |  |
| `findings_by_vector()` |  |
| `highest_risk()` |  |
| `summary()` |  |

### `StegoConfig`

### `SteganographyDetector`

Analyses agent text output for steganographic encodings.

| Method | Description |
|--------|-------------|
| `__init__()` |  |
| `analyze()` |  |
| `batch_analyze()` |  |
| `compare_texts()` |  |
| `detect_whitespace_encoding()` |  |
| `detect_invisible_unicode()` |  |
| `detect_acrostic()` |  |
| `detect_homoglyphs()` |  |
| `detect_capitalization_encoding()` |  |
| `detect_punctuation_anomalies()` |  |
| `detect_sentence_length_encoding()` |  |


## Functions

| Function | Description |
|----------|-------------|
| `scan_text()` |  |
| `encode_zero_width()` | Encode a message using zero-width characters (for testing). |
| `encode_homoglyphs()` | Encode bits via homoglyph substitution (for testing). |
