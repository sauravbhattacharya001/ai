"""Safety Training Quiz Generator — team training quizzes from the knowledge base.

Generates interactive multiple-choice quizzes from the Safety Knowledge Base
for team onboarding, periodic training, and compliance verification. Supports
timed quizzes, difficulty levels, category filtering, scoring with explanations,
and exportable results.

Usage (CLI)::

    python -m replication quiz                          # 10-question random quiz
    python -m replication quiz --count 20               # 20 questions
    python -m replication quiz --category containment   # filter by category
    python -m replication quiz --difficulty hard         # easy/medium/hard
    python -m replication quiz --timed 30               # 30s per question
    python -m replication quiz --json                   # JSON output (non-interactive)
    python -m replication quiz --html -o quiz.html      # self-contained HTML quiz
    python -m replication quiz --review                 # review mode (show answers)
    python -m replication quiz --seed 42                # reproducible quiz

Programmatic::

    from replication.safety_quiz import QuizGenerator, QuizConfig
    gen = QuizGenerator()
    quiz = gen.generate(QuizConfig(count=10, difficulty="medium"))
    for q in quiz.questions:
        print(q.render())
    # Score answers
    result = quiz.score({"Q1": "B", "Q2": "A", ...})
    print(f"Score: {result.score}/{result.total} ({result.percent}%)")
"""

from __future__ import annotations

import argparse
import hashlib
import html as html_mod
import json
import random
import sys
import textwrap
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ── Models ───────────────────────────────────────────────────────────

class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class QuizConfig:
    """Configuration for quiz generation."""
    count: int = 10
    difficulty: str = "medium"
    categories: Optional[List[str]] = None
    seed: Optional[int] = None
    timed: Optional[int] = None  # seconds per question


@dataclass
class Choice:
    """A single answer choice."""
    letter: str
    text: str
    is_correct: bool = False


@dataclass
class Question:
    """A quiz question with choices and explanation."""
    id: str
    text: str
    choices: List[Choice] = field(default_factory=list)
    correct_letter: str = ""
    explanation: str = ""
    difficulty: str = "medium"
    category: str = ""
    source_id: str = ""  # KB entry ID

    def render(self, show_answer: bool = False) -> str:
        lines = [f"  {self.id}. {self.text}"]
        lines.append(f"     [Category: {self.category} | Difficulty: {self.difficulty}]")
        for c in self.choices:
            marker = " *" if (show_answer and c.is_correct) else ""
            lines.append(f"     {c.letter}) {c.text}{marker}")
        if show_answer:
            lines.append(f"     Answer: {self.correct_letter}")
            lines.append(f"     Explanation: {self.explanation}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "choices": [{"letter": c.letter, "text": c.text, "is_correct": c.is_correct} for c in self.choices],
            "correct_letter": self.correct_letter,
            "explanation": self.explanation,
            "difficulty": self.difficulty,
            "category": self.category,
            "source_id": self.source_id,
        }


@dataclass
class QuizResult:
    """Scored quiz result."""
    score: int = 0
    total: int = 0
    percent: float = 0.0
    answers: Dict[str, str] = field(default_factory=dict)
    correct: List[str] = field(default_factory=list)
    incorrect: List[str] = field(default_factory=list)
    time_taken: Optional[float] = None

    def render(self) -> str:
        lines = [
            "=" * 50,
            "  QUIZ RESULTS",
            "=" * 50,
            f"  Score: {self.score}/{self.total} ({self.percent:.0f}%)",
        ]
        if self.time_taken is not None:
            lines.append(f"  Time: {self.time_taken:.1f}s")
        grade = _grade(self.percent)
        lines.append(f"  Grade: {grade}")
        lines.append("")
        if self.incorrect:
            lines.append(f"  Missed: {', '.join(self.incorrect)}")
        lines.append("=" * 50)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score, "total": self.total, "percent": self.percent,
            "grade": _grade(self.percent),
            "answers": self.answers, "correct": self.correct,
            "incorrect": self.incorrect, "time_taken": self.time_taken,
        }


@dataclass
class Quiz:
    """A generated quiz with questions."""
    questions: List[Question] = field(default_factory=list)
    config: Optional[QuizConfig] = None

    def score(self, answers: Dict[str, str]) -> QuizResult:
        correct_list, incorrect_list = [], []
        for q in self.questions:
            ans = answers.get(q.id, "").upper()
            if ans == q.correct_letter:
                correct_list.append(q.id)
            else:
                incorrect_list.append(q.id)
        total = len(self.questions)
        sc = len(correct_list)
        return QuizResult(
            score=sc, total=total,
            percent=(sc / total * 100) if total else 0,
            answers=answers, correct=correct_list, incorrect=incorrect_list,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "questions": [q.to_dict() for q in self.questions],
            "count": len(self.questions),
        }


def _grade(pct: float) -> str:
    if pct >= 90: return "A — Excellent"
    if pct >= 80: return "B — Good"
    if pct >= 70: return "C — Satisfactory"
    if pct >= 60: return "D — Needs Improvement"
    return "F — Requires Retraining"


# ── Knowledge Base Question Templates ────────────────────────────────

# Templates indexed by question type. Each generates a question from a KB entry.

def _q_what_is(entry: Any, all_entries: List[Any], rng: random.Random) -> Optional[Question]:
    """What does this pattern/anti-pattern address?"""
    if not hasattr(entry, "description") or len(entry.description) < 20:
        return None
    correct = entry.title
    distractors = _pick_distractors(entry, all_entries, rng, key="title", n=3)
    if len(distractors) < 3:
        return None
    choices, correct_letter = _build_choices(correct, distractors, rng)
    return Question(
        id="", text=f"Which safety concept is described as: \"{_truncate(entry.description, 120)}\"?",
        choices=choices, correct_letter=correct_letter,
        explanation=f"{entry.title}: {_truncate(entry.description, 200)}",
        category=getattr(entry, "category", "general"),
        source_id=getattr(entry, "id", ""),
    )


def _q_category(entry: Any, all_entries: List[Any], rng: random.Random) -> Optional[Question]:
    """What category does this entry belong to?"""
    correct = getattr(entry, "category", None)
    if not correct:
        return None
    cats = list({getattr(e, "category", "") for e in all_entries if getattr(e, "category", "") and getattr(e, "category", "") != correct})
    if len(cats) < 3:
        return None
    distractors = rng.sample(cats, 3)
    choices, correct_letter = _build_choices(correct, distractors, rng)
    return Question(
        id="", text=f"\"{entry.title}\" falls under which safety category?",
        choices=choices, correct_letter=correct_letter,
        explanation=f"{entry.title} is categorized under '{correct}'.",
        category=correct,
        source_id=getattr(entry, "id", ""),
    )


def _q_severity(entry: Any, all_entries: List[Any], rng: random.Random) -> Optional[Question]:
    """What severity level is this?"""
    correct = getattr(entry, "severity", None)
    if not correct:
        return None
    options = ["critical", "high", "medium", "low"]
    distractors = [s for s in options if s != correct][:3]
    choices, correct_letter = _build_choices(correct, distractors, rng)
    return Question(
        id="", text=f"What is the severity level of \"{entry.title}\"?",
        choices=choices, correct_letter=correct_letter,
        explanation=f"{entry.title} has severity: {correct}.",
        category=getattr(entry, "category", "general"),
        source_id=getattr(entry, "id", ""),
    )


def _q_guidance(entry: Any, all_entries: List[Any], rng: random.Random) -> Optional[Question]:
    """Which is a recommended action for this entry?"""
    guidance = getattr(entry, "guidance", [])
    if not guidance or len(guidance) < 1:
        return None
    correct = rng.choice(guidance)
    if len(correct) < 10:
        return None
    # Pick distractors from other entries' guidance
    other_guidance = []
    for e in all_entries:
        if getattr(e, "id", "") != getattr(entry, "id", ""):
            for g in getattr(e, "guidance", []):
                if len(g) >= 10:
                    other_guidance.append(g)
    if len(other_guidance) < 3:
        return None
    distractors = rng.sample(other_guidance, min(3, len(other_guidance)))
    choices, correct_letter = _build_choices(
        _truncate(correct, 80), [_truncate(d, 80) for d in distractors], rng
    )
    return Question(
        id="", text=f"Which action is recommended for \"{entry.title}\"?",
        choices=choices, correct_letter=correct_letter,
        explanation=f"Recommended: {correct}",
        category=getattr(entry, "category", "general"),
        source_id=getattr(entry, "id", ""),
    )


def _q_kind(entry: Any, all_entries: List[Any], rng: random.Random) -> Optional[Question]:
    """Is this a pattern or anti-pattern?"""
    kind = getattr(entry, "kind", None)
    if kind not in ("pattern", "anti-pattern"):
        return None
    correct = kind
    distractors = ["anti-pattern" if kind == "pattern" else "pattern", "mitigation", "framework"]
    choices, correct_letter = _build_choices(correct, distractors, rng)
    return Question(
        id="", text=f"\"{entry.title}\" is classified as a:",
        choices=choices, correct_letter=correct_letter,
        explanation=f"{entry.title} is a {kind}.",
        category=getattr(entry, "category", "general"),
        source_id=getattr(entry, "id", ""),
    )


_GENERATORS = [_q_what_is, _q_category, _q_severity, _q_guidance, _q_kind]

_DIFFICULTY_WEIGHTS = {
    "easy": [0.4, 0.3, 0.2, 0.05, 0.05],
    "medium": [0.2, 0.15, 0.15, 0.3, 0.2],
    "hard": [0.1, 0.1, 0.1, 0.35, 0.35],
}


# ── Helpers ──────────────────────────────────────────────────────────

def _truncate(text: str, maxlen: int) -> str:
    text = text.strip().replace("\n", " ")
    return text[:maxlen - 3] + "..." if len(text) > maxlen else text


def _pick_distractors(entry: Any, all_entries: List[Any], rng: random.Random,
                      key: str, n: int = 3) -> List[str]:
    correct = getattr(entry, key, "")
    pool = [getattr(e, key, "") for e in all_entries
            if getattr(e, key, "") and getattr(e, key, "") != correct]
    pool = list(set(pool))
    return rng.sample(pool, min(n, len(pool)))


def _build_choices(correct: str, distractors: List[str],
                   rng: random.Random) -> Tuple[List[Choice], str]:
    items = [(correct, True)] + [(d, False) for d in distractors]
    rng.shuffle(items)
    letters = "ABCD"
    choices = [Choice(letter=letters[i], text=t, is_correct=c) for i, (t, c) in enumerate(items)]
    correct_letter = next(c.letter for c in choices if c.is_correct)
    return choices, correct_letter


# ── Generator ────────────────────────────────────────────────────────

class QuizGenerator:
    """Generate quizzes from the Safety Knowledge Base."""

    def __init__(self) -> None:
        self._kb: Any = None

    def _load_kb(self) -> Any:
        if self._kb is None:
            from .knowledge_base import SafetyKnowledgeBase
            self._kb = SafetyKnowledgeBase()
        return self._kb

    def generate(self, config: Optional[QuizConfig] = None) -> Quiz:
        config = config or QuizConfig()
        rng = random.Random(config.seed)
        kb = self._load_kb()
        entries = list(kb._entries.values()) if hasattr(kb, "_entries") else []
        if not entries:
            return Quiz(questions=[], config=config)

        # Filter by category
        if config.categories:
            cats_lower = {c.lower() for c in config.categories}
            entries = [e for e in entries if getattr(e, "category", "").lower() in cats_lower]

        if not entries:
            return Quiz(questions=[], config=config)

        difficulty = config.difficulty.lower()
        weights = _DIFFICULTY_WEIGHTS.get(difficulty, _DIFFICULTY_WEIGHTS["medium"])

        questions: List[Question] = []
        attempts = 0
        max_attempts = config.count * 10

        while len(questions) < config.count and attempts < max_attempts:
            attempts += 1
            entry = rng.choice(entries)
            gen_fn = rng.choices(_GENERATORS, weights=weights, k=1)[0]
            q = gen_fn(entry, entries, rng)
            if q is not None:
                q.id = f"Q{len(questions) + 1}"
                q.difficulty = difficulty
                questions.append(q)

        return Quiz(questions=questions, config=config)


# ── HTML Export ──────────────────────────────────────────────────────

def _generate_html(quiz: Quiz, timed: Optional[int] = None) -> str:
    esc = html_mod.escape
    q_json = json.dumps([q.to_dict() for q in quiz.questions])
    timer_js = f"const TIME_LIMIT = {timed};" if timed else "const TIME_LIMIT = null;"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Safety Training Quiz</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:system-ui,-apple-system,sans-serif;background:#0f172a;color:#e2e8f0;min-height:100vh;padding:2rem}}
.container{{max-width:800px;margin:0 auto}}
h1{{text-align:center;font-size:1.8rem;margin-bottom:.5rem;color:#38bdf8}}
.subtitle{{text-align:center;color:#94a3b8;margin-bottom:2rem}}
.progress{{background:#1e293b;border-radius:8px;height:8px;margin-bottom:2rem;overflow:hidden}}
.progress-bar{{height:100%;background:linear-gradient(90deg,#38bdf8,#818cf8);transition:width .3s}}
.question-card{{background:#1e293b;border-radius:12px;padding:1.5rem;margin-bottom:1.5rem;border:1px solid #334155}}
.question-card.correct{{border-color:#22c55e}}.question-card.incorrect{{border-color:#ef4444}}
.q-header{{display:flex;justify-content:space-between;margin-bottom:.8rem}}
.q-num{{color:#38bdf8;font-weight:700}}.q-meta{{color:#64748b;font-size:.85rem}}
.q-text{{font-size:1.05rem;line-height:1.5;margin-bottom:1rem}}
.choices{{display:flex;flex-direction:column;gap:.5rem}}
.choice{{padding:.7rem 1rem;background:#0f172a;border:2px solid #334155;border-radius:8px;cursor:pointer;transition:all .2s}}
.choice:hover{{border-color:#38bdf8}}.choice.selected{{border-color:#818cf8;background:#1e1b4b}}
.choice.correct-answer{{border-color:#22c55e;background:#052e16}}.choice.wrong-answer{{border-color:#ef4444;background:#450a0a}}
.explanation{{margin-top:.8rem;padding:.8rem;background:#0f172a;border-radius:8px;border-left:3px solid #38bdf8;display:none}}
.explanation.visible{{display:block}}
.timer{{text-align:center;font-size:1.2rem;color:#f59e0b;margin-bottom:1rem;display:none}}
.timer.active{{display:block}}
.results{{background:#1e293b;border-radius:12px;padding:2rem;text-align:center;display:none}}
.results.visible{{display:block}}.score-big{{font-size:3rem;font-weight:700;color:#38bdf8}}
.grade{{font-size:1.3rem;margin:.5rem 0}}.stats{{display:flex;justify-content:center;gap:2rem;margin:1rem 0}}
.stat{{text-align:center}}.stat-val{{font-size:1.5rem;font-weight:700}}.stat-label{{font-size:.85rem;color:#94a3b8}}
.btn{{display:inline-block;padding:.7rem 1.5rem;border-radius:8px;border:none;font-size:1rem;cursor:pointer;font-weight:600;margin:.3rem}}
.btn-primary{{background:#38bdf8;color:#0f172a}}.btn-secondary{{background:#334155;color:#e2e8f0}}
.btn:hover{{opacity:.9}}
.actions{{text-align:center;margin-top:1.5rem}}
</style>
</head>
<body>
<div class="container">
<h1>🛡️ AI Safety Training Quiz</h1>
<p class="subtitle" id="subtitle">Answer each question to test your safety knowledge</p>
<div class="timer" id="timer"></div>
<div class="progress"><div class="progress-bar" id="progressBar" style="width:0%"></div></div>
<div id="quizArea"></div>
<div class="results" id="results">
  <div class="score-big" id="scoreBig"></div>
  <div class="grade" id="grade"></div>
  <div class="stats">
    <div class="stat"><div class="stat-val" id="correctCount" style="color:#22c55e">0</div><div class="stat-label">Correct</div></div>
    <div class="stat"><div class="stat-val" id="incorrectCount" style="color:#ef4444">0</div><div class="stat-label">Incorrect</div></div>
    <div class="stat"><div class="stat-val" id="timeStat" style="color:#f59e0b">-</div><div class="stat-label">Time</div></div>
  </div>
  <div class="actions"><button class="btn btn-primary" onclick="reviewQuiz()">Review Answers</button> <button class="btn btn-secondary" onclick="location.reload()">Retry</button></div>
</div>
</div>
<script>
{timer_js}
const QUESTIONS={q_json};
let answers={{}},startTime=Date.now(),timerInterval=null;
function init(){{
  const area=document.getElementById('quizArea');
  QUESTIONS.forEach((q,i)=>{{
    let html=`<div class="question-card" id="qcard_${{q.id}}">
      <div class="q-header"><span class="q-num">${{q.id}}</span><span class="q-meta">${{q.category}} · ${{q.difficulty}}</span></div>
      <div class="q-text">${{q.text}}</div><div class="choices">`;
    q.choices.forEach(c=>{{html+=`<div class="choice" data-qid="${{q.id}}" data-letter="${{c.letter}}" onclick="selectChoice(this)">${{c.letter}}) ${{c.text}}</div>`}});
    html+=`</div><div class="explanation" id="exp_${{q.id}}">${{q.explanation}}</div></div>`;
    area.innerHTML+=html;
  }});
  if(TIME_LIMIT){{document.getElementById('timer').classList.add('active');startTimer()}}
}}
function selectChoice(el){{
  const qid=el.dataset.qid;
  document.querySelectorAll(`.choice[data-qid="${{qid}}"]`).forEach(c=>c.classList.remove('selected'));
  el.classList.add('selected');
  answers[qid]=el.dataset.letter;
  updateProgress();
  if(Object.keys(answers).length===QUESTIONS.length)setTimeout(submitQuiz,500);
}}
function updateProgress(){{
  const pct=Object.keys(answers).length/QUESTIONS.length*100;
  document.getElementById('progressBar').style.width=pct+'%';
}}
function startTimer(){{
  let remaining=TIME_LIMIT*QUESTIONS.length;
  const el=document.getElementById('timer');
  timerInterval=setInterval(()=>{{
    remaining--;
    const m=Math.floor(remaining/60),s=remaining%60;
    el.textContent=`⏱ ${{m}}:${{s.toString().padStart(2,'0')}}`;
    if(remaining<=0){{clearInterval(timerInterval);submitQuiz()}}
  }},1000);
}}
function submitQuiz(){{
  if(timerInterval)clearInterval(timerInterval);
  const elapsed=(Date.now()-startTime)/1000;
  let correct=0;
  QUESTIONS.forEach(q=>{{
    const card=document.getElementById('qcard_'+q.id);
    const ans=answers[q.id]||'';
    const isCorrect=ans===q.correct_letter;
    if(isCorrect){{correct++;card.classList.add('correct')}}else{{card.classList.add('incorrect')}}
    document.querySelectorAll(`.choice[data-qid="${{q.id}}"]`).forEach(c=>{{
      if(c.dataset.letter===q.correct_letter)c.classList.add('correct-answer');
      else if(c.dataset.letter===ans&&!isCorrect)c.classList.add('wrong-answer');
      c.style.pointerEvents='none';
    }});
    document.getElementById('exp_'+q.id).classList.add('visible');
  }});
  const pct=correct/QUESTIONS.length*100;
  document.getElementById('scoreBig').textContent=Math.round(pct)+'%';
  document.getElementById('grade').textContent=pct>=90?'A — Excellent':pct>=80?'B — Good':pct>=70?'C — Satisfactory':pct>=60?'D — Needs Improvement':'F — Requires Retraining';
  document.getElementById('correctCount').textContent=correct;
  document.getElementById('incorrectCount').textContent=QUESTIONS.length-correct;
  document.getElementById('timeStat').textContent=elapsed<60?Math.round(elapsed)+'s':Math.floor(elapsed/60)+'m '+Math.round(elapsed%60)+'s';
  document.getElementById('results').classList.add('visible');
  document.getElementById('progressBar').style.width='100%';
}}
function reviewQuiz(){{document.getElementById('quizArea').scrollIntoView({{behavior:'smooth'}})}}
init();
</script>
</body>
</html>"""


# ── Interactive CLI Quiz ─────────────────────────────────────────────

def _run_interactive(quiz: Quiz, timed: Optional[int] = None) -> QuizResult:
    """Run quiz interactively in terminal."""
    answers: Dict[str, str] = {}
    total = len(quiz.questions)
    start = time.time()

    print()
    print("=" * 55)
    print("  🛡️  AI SAFETY TRAINING QUIZ")
    print("=" * 55)
    print(f"  Questions: {total} | Difficulty: {quiz.config.difficulty if quiz.config else 'mixed'}")
    if timed:
        print(f"  Time limit: {timed}s per question")
    print("=" * 55)
    print()

    for i, q in enumerate(quiz.questions, 1):
        print(f"  [{i}/{total}] {q.text}")
        print(f"  Category: {q.category}")
        print()
        for c in q.choices:
            print(f"    {c.letter}) {c.text}")
        print()

        while True:
            try:
                ans = input("  Your answer (A/B/C/D): ").strip().upper()
            except (EOFError, KeyboardInterrupt):
                print("\n  Quiz cancelled.")
                elapsed = time.time() - start
                return quiz.score(answers)
            if ans in ("A", "B", "C", "D"):
                break
            print("  Please enter A, B, C, or D.")

        answers[q.id] = ans
        is_correct = ans == q.correct_letter
        if is_correct:
            print("  ✅ Correct!")
        else:
            print(f"  ❌ Wrong — correct answer: {q.correct_letter}")
        print(f"  💡 {q.explanation}")
        print()

    elapsed = time.time() - start
    result = quiz.score(answers)
    result.time_taken = elapsed
    print(result.render())
    return result


# ── CLI ──────────────────────────────────────────────────────────────

def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m replication quiz",
        description="Generate AI safety training quizzes from the knowledge base",
    )
    parser.add_argument("--count", "-n", type=int, default=10, help="Number of questions (default: 10)")
    parser.add_argument("--difficulty", "-d", choices=["easy", "medium", "hard"], default="medium")
    parser.add_argument("--category", "-c", action="append", help="Filter by KB category (repeatable)")
    parser.add_argument("--timed", "-t", type=int, metavar="SEC", help="Seconds per question")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--json", action="store_true", help="JSON output (non-interactive)")
    parser.add_argument("--html", action="store_true", help="Generate self-contained HTML quiz")
    parser.add_argument("--review", action="store_true", help="Review mode — show all answers")
    parser.add_argument("-o", "--output", metavar="FILE", help="Output file")

    args = parser.parse_args(argv)
    config = QuizConfig(
        count=args.count, difficulty=args.difficulty,
        categories=args.category, seed=args.seed, timed=args.timed,
    )

    gen = QuizGenerator()
    quiz = gen.generate(config)

    if not quiz.questions:
        print("No questions could be generated. Check that the knowledge base has entries.")
        sys.exit(1)

    if args.json:
        out = json.dumps(quiz.to_dict(), indent=2)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(out)
            print(f"Wrote {len(quiz.questions)} questions to {args.output}")
        else:
            print(out)
    elif args.html:
        html = _generate_html(quiz, timed=args.timed)
        dest = args.output or "safety_quiz.html"
        with open(dest, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"Wrote interactive quiz ({len(quiz.questions)} questions) to {dest}")
    elif args.review:
        print()
        for q in quiz.questions:
            print(q.render(show_answer=True))
            print()
    else:
        _run_interactive(quiz, timed=args.timed)


if __name__ == "__main__":
    main()
