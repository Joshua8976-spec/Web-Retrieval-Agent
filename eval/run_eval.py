import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

# Add repository root to pythonpath
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agent.loop import AgentOrchestrator
from src.agent.state import AgentResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval")


class EvalQuestion(BaseModel):
    id: str
    url: str
    question: str
    difficulty: str  # easy, medium, impossible
    expected_found: bool
    expected_answer: Optional[str] = None
    expected_keywords: List[str] = []


class QuestionResult(BaseModel):
    id: str
    question: str
    url: str
    difficulty: str
    expected_found: bool
    actual_found: bool
    is_correct: bool
    answer: Optional[str] = None
    evidence_quote: Optional[str] = None
    source_url: Optional[str] = None
    explanation: Optional[str] = None
    steps: int
    elapsed_seconds: float
    error: Optional[str] = None


def score_question(q: EvalQuestion, res: AgentResult) -> bool:
    """Evaluate if agent output matches ground truth expectations."""
    # Impossible questions: must return found == False
    if not q.expected_found:
        return res.found is False

    # Positive questions (easy/medium): must return found == True
    if not res.found:
        return False

    # Must provide evidence quote
    if not res.evidence_quote or not res.evidence_quote.strip():
        return False

    # Check keyword matches if specified
    if q.expected_keywords:
        combined_text = f"{res.answer or ''} {res.evidence_quote or ''}".lower()
        matched = any(kw.lower() in combined_text for kw in q.expected_keywords)
        return matched

    return True


async def evaluate_single(
    orchestrator: AgentOrchestrator,
    q: EvalQuestion,
) -> QuestionResult:
    """Run evaluation for a single question."""
    start_t = time.perf_counter()
    try:
        res = await orchestrator.run(start_url=q.url, question=q.question)
        elapsed = time.perf_counter() - start_t
        is_correct = score_question(q, res)

        return QuestionResult(
            id=q.id,
            question=q.question,
            url=q.url,
            difficulty=q.difficulty,
            expected_found=q.expected_found,
            actual_found=res.found,
            is_correct=is_correct,
            answer=res.answer,
            evidence_quote=res.evidence_quote,
            source_url=res.source_url,
            explanation=res.explanation,
            steps=res.steps,
            elapsed_seconds=round(elapsed, 2),
        )
    except Exception as exc:
        elapsed = time.perf_counter() - start_t
        logger.error("Error evaluating question %s: %s", q.id, exc)
        return QuestionResult(
            id=q.id,
            question=q.question,
            url=q.url,
            difficulty=q.difficulty,
            expected_found=q.expected_found,
            actual_found=False,
            is_correct=False,
            steps=0,
            elapsed_seconds=round(elapsed, 2),
            error=str(exc),
        )


def generate_report(results: List[QuestionResult]) -> Dict[str, Any]:
    """Compute aggregate accuracy, latency, and step metrics."""
    total = len(results)
    if total == 0:
        return {"total": 0}

    correct_count = sum(1 for r in results if r.is_correct)
    overall_accuracy = round((correct_count / total) * 100, 1)

    diff_groups: Dict[str, List[QuestionResult]] = {}
    for r in results:
        diff_groups.setdefault(r.difficulty, []).append(r)

    difficulty_breakdown = {}
    for diff, items in diff_groups.items():
        c = sum(1 for r in items if r.is_correct)
        difficulty_breakdown[diff] = {
            "total": len(items),
            "correct": c,
            "accuracy_pct": round((c / len(items)) * 100, 1),
        }

    # Hallucination rate on impossible questions
    impossible_items = diff_groups.get("impossible", [])
    hallucinations = sum(1 for r in impossible_items if r.actual_found is True)
    hallucination_rate = (
        round((hallucinations / len(impossible_items)) * 100, 1) if impossible_items else 0.0
    )

    avg_steps = round(sum(r.steps for r in results) / total, 2)
    avg_latency = round(sum(r.elapsed_seconds for r in results) / total, 2)

    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_evaluated": total,
        "correct": correct_count,
        "overall_accuracy_pct": overall_accuracy,
        "hallucination_rate_pct": hallucination_rate,
        "average_steps": avg_steps,
        "average_latency_sec": avg_latency,
        "difficulty_breakdown": difficulty_breakdown,
        "results": [r.model_dump() for r in results],
    }


def print_summary_table(report: Dict[str, Any]):
    """Print a clean CLI summary table."""
    print("\n" + "=" * 65)
    print("      WEB RETRIEVAL AGENT — BENCHMARK EVALUATION REPORT      ")
    print("=" * 65)
    print(f"Total Questions Evaluated:  {report['total_evaluated']}")
    print(f"Overall Accuracy:           {report['overall_accuracy_pct']}% ({report['correct']}/{report['total_evaluated']})")
    print(f"Hallucination Rate:         {report['hallucination_rate_pct']}% (on impossible queries)")
    print(f"Average Steps Taken:        {report['average_steps']}")
    print(f"Average Time Per Query:     {report['average_latency_sec']}s")
    print("-" * 65)
    print(f"{'Difficulty':<15} | {'Questions':<10} | {'Correct':<10} | {'Accuracy':<10}")
    print("-" * 65)
    for diff, data in report.get("difficulty_breakdown", {}).items():
        print(f"{diff.capitalize():<15} | {data['total']:<10} | {data['correct']:<10} | {data['accuracy_pct']:>7}%")
    print("=" * 65 + "\n")


async def main():
    parser = argparse.ArgumentParser(description="Run Web Retrieval Agent evaluation benchmark.")
    parser.add_argument("--questions", default="eval/questions.jsonl", help="Path to questions.jsonl")
    parser.add_argument("--output", default="eval/report.json", help="Path to output report.json")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions to run")
    parser.add_argument("--difficulty", choices=["easy", "medium", "impossible"], default=None, help="Filter by difficulty")
    args = parser.parse_args()

    questions_path = Path(args.questions)
    if not questions_path.exists():
        logger.error("Questions file not found: %s", questions_path)
        return

    questions: List[EvalQuestion] = []
    with open(questions_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                q = EvalQuestion.model_validate_json(line)
                if args.difficulty and q.difficulty != args.difficulty:
                    continue
                questions.append(q)

    if args.limit:
        questions = questions[:args.limit]

    logger.info("Loaded %d questions for evaluation.", len(questions))
    orchestrator = AgentOrchestrator()

    results: List[QuestionResult] = []
    for i, q in enumerate(questions, 1):
        logger.info("[%d/%d] Running %s (%s): '%s'", i, len(questions), q.id, q.difficulty, q.question)
        res = await evaluate_single(orchestrator, q)
        results.append(res)
        logger.info(
            "--> %s | Found: %s | Correct: %s | Steps: %d | Time: %.1fs",
            q.id, res.actual_found, res.is_correct, res.steps, res.elapsed_seconds,
        )

    report = generate_report(results)
    print_summary_table(report)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved full evaluation report to %s", output_path)


if __name__ == "__main__":
    asyncio.run(main())
