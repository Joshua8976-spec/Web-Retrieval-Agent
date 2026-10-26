from unittest.mock import AsyncMock, MagicMock
import pytest
from src.agent.verifier import (
    verify_evidence_in_page,
    verify_answer_with_llm,
    normalize_whitespace,
)


def test_normalize_whitespace():
    raw = "Hello   world \n\n this is \t a test  "
    assert normalize_whitespace(raw) == "Hello world this is a test"


def test_verify_evidence_in_page_exact_match():
    page = "AMD ROCm open ecosystem supports Instinct MI300X accelerators with 192GB HBM3."
    quote = "supports Instinct MI300X accelerators"
    valid, msg = verify_evidence_in_page(quote, page)
    assert valid is True
    assert "Exact raw character match" in msg


def test_verify_evidence_in_page_whitespace_normalized():
    page = "Line 1\nLine 2\n\tLine 3 contains the target fact."
    quote = "Line 1 Line 2 Line 3 contains the target fact."
    valid, msg = verify_evidence_in_page(quote, page)
    assert valid is True
    assert "Whitespace-normalized" in msg


def test_verify_evidence_in_page_tampered_or_missing():
    page = "AMD ROCm is an open software platform."
    # Tampered quote: 'completely open' instead of 'open'
    quote = "AMD ROCm is a completely open software platform."
    valid, msg = verify_evidence_in_page(quote, page)
    assert valid is False
    assert "Quote was not found" in msg


def test_verify_evidence_empty():
    valid, msg = verify_evidence_in_page("", "Some text")
    assert valid is False
    assert "empty" in msg

    valid, msg = verify_evidence_in_page("Quote", "")
    assert valid is False
    assert "empty" in msg


@pytest.mark.asyncio
async def test_verify_answer_with_llm_yes():
    mock_llm = MagicMock()
    mock_llm.async_chat_completion_json = AsyncMock(
        return_value={"is_valid": True, "reason": "The quote directly answers the memory capacity question."}
    )

    valid, reason = await verify_answer_with_llm(
        llm=mock_llm,
        question="How much memory does MI300X have?",
        evidence_quote="The AMD Instinct MI300X features 192GB of HBM3 memory.",
        proposed_answer="192GB of HBM3",
    )
    assert valid is True
    assert "directly answers" in reason


@pytest.mark.asyncio
async def test_verify_answer_with_llm_no():
    mock_llm = MagicMock()
    mock_llm.async_chat_completion_json = AsyncMock(
        return_value={"is_valid": False, "reason": "The quote discusses software, not memory capacity."}
    )

    valid, reason = await verify_answer_with_llm(
        llm=mock_llm,
        question="How much memory does MI300X have?",
        evidence_quote="AMD ROCm is an open software platform.",
        proposed_answer="192GB",
    )
    assert valid is False
    assert "software, not memory" in reason
