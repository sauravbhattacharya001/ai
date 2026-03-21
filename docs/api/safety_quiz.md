# Safety Training Quiz

Generate interactive training quizzes from the Safety Knowledge Base
for team onboarding, periodic safety training, and compliance checks.

## Quick Start

```bash
# Interactive 10-question quiz
python -m replication quiz

# 20 hard questions on containment
python -m replication quiz --count 20 --difficulty hard --category containment

# Timed quiz (30s per question)
python -m replication quiz --timed 30

# Export as interactive HTML page
python -m replication quiz --html -o quiz.html

# JSON output for integration
python -m replication quiz --json --seed 42

# Review mode (show all answers)
python -m replication quiz --review
```

## Features

- **Multiple question types**: concept identification, category matching,
  severity assessment, guidance recall, pattern/anti-pattern classification
- **Difficulty levels**: easy, medium, hard — controls which question types
  are weighted more heavily
- **Category filtering**: focus on specific safety domains
- **Timed mode**: configurable per-question time limits
- **HTML export**: self-contained interactive quiz page with auto-grading,
  progress bar, timer, and explanations
- **Reproducible**: use `--seed` for deterministic quiz generation
- **Scoring**: A–F grades with detailed results

## Programmatic Usage

```python
from replication.safety_quiz import QuizGenerator, QuizConfig

gen = QuizGenerator()
quiz = gen.generate(QuizConfig(count=10, difficulty="medium"))

for q in quiz.questions:
    print(q.render())

# Score answers
result = quiz.score({"Q1": "B", "Q2": "A"})
print(f"Score: {result.score}/{result.total} ({result.percent}%)")
```

## Options

| Flag | Description |
|------|-------------|
| `--count N` | Number of questions (default: 10) |
| `--difficulty` | easy, medium, hard |
| `--category` | Filter by KB category (repeatable) |
| `--timed SEC` | Seconds per question |
| `--seed N` | Random seed for reproducibility |
| `--json` | JSON output |
| `--html` | Interactive HTML quiz |
| `--review` | Show all answers |
| `-o FILE` | Output file |
