import json
from pathlib import Path
import pytest
from eval.run_eval import EvalQuestion, QuestionResult, score_question, generate_report
from src.agent.state import AgentResult

EVAL_DIR = Path(__file__).parent.parent / "eval"


def test_questions_jsonl_validity_and_distribution():
    questions_file = EVAL_DIR / "questions.jsonl"
    assert questions_file.exists()

    questions = []
    with open(questions_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                q = EvalQuestion.model_validate_json(line)
                questions.append(q)

    # Must contain at least 25 questions as specified in brief
    assert len(questions) >= 25

    difficulties = {q.difficulty for q in questions}
    assert "easy" in difficulties
    assert "medium" in difficulties
    assert "impossible" in difficulties

    # Check non-empty URLs and questions
    for q in questions:
        assert q.url.startswith("http")
        assert len(q.question) > 5


def test_score_question_impossible():
    q = EvalQuestion(
        id="q_test",
        url="https://example.com",
        question="Secret query",
        difficulty="impossible",
        expected_found=False,
    )

    # Correct: agent says found: False
    res_correct = AgentResult(found=False, explanation="Not on site")
    assert score_question(q, res_correct) is True

    # Hallucination: agent says found: True
    res_hallucinated = AgentResult(
        found=True,
        answer="Made up",
        evidence_quote="Made up quote",
        source_url="https://example.com",
    )
    assert score_question(q, res_hallucinated) is False


def test_score_question_positive():
    q = EvalQuestion(
        id="q_pos",
        url="https://example.com",
        question="What is X?",
        difficulty="easy",
        expected_found=True,
        expected_keywords=["fastapi", "web"],
    )

    # Missing evidence quote
    res_no_quote = AgentResult(found=True, answer="FastAPI web framework", evidence_quote="")
    assert score_question(q, res_no_quote) is False

    # Matching keywords and valid quote
    res_good = AgentResult(
        found=True,
        answer="FastAPI web framework",
        evidence_quote="FastAPI is a modern web framework.",
        source_url="https://example.com",
    )
    assert score_question(q, res_good) is True


def test_generate_report_metrics():
    mock_results = [
        QuestionResult(
            id="q1",
            question="Q1",
            url="https://example.com",
            difficulty="easy",
            expected_found=True,
            actual_found=True,
            is_correct=True,
            steps=2,
            elapsed_seconds=3.0,
        ),
        QuestionResult(
            id="q2",
            question="Q2",
            url="https://example.com",
            difficulty="impossible",
            expected_found=False,
            actual_found=False,
            is_correct=True,
            steps=3,
            elapsed_seconds=4.0,
        ),
        QuestionResult(
            id="q3",
            question="Q3",
            url="https://example.com",
            difficulty="medium",
            expected_found=True,
            actual_found=False,
            is_correct=False,
            steps=5,
            elapsed_seconds=6.0,
        ),
    ]

    report = generate_report(mock_results)
    assert report["total_evaluated"] == 3
    assert report["correct"] == 2
    assert report["overall_accuracy_pct"] == 66.7
    assert report["hallucination_rate_pct"] == 0.0
    assert report["average_steps"] == 3.33
