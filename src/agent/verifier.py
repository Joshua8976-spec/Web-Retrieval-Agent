import logging
import re
from typing import Optional, Tuple
from pydantic import BaseModel, Field

from src.llm.client import LLMClient

logger = logging.getLogger(__name__)


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace by collapsing multiple spaces, tabs, and newlines into single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def strip_markdown(text: str) -> str:
    """Strip markdown formatting (links, bold, italics, code) to extract clean visible text."""
    no_links = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    no_emphasis = re.sub(r"[*_]{1,3}([^*_]+)[*_]{1,3}", r"\1", no_links)
    return no_emphasis


def verify_evidence_in_page(evidence_quote: str, page_text: str) -> Tuple[bool, str]:
    """Verify in deterministic code that evidence_quote appears verbatim in the page text.
    
    Checks direct substring, whitespace-normalized substring, and markdown-stripped
    substring to handle HTML layout spacing or markdown link markup while enforcing
    exact word-for-word accuracy.
    """
    if not evidence_quote or not evidence_quote.strip():
        return False, "Evidence quote is empty."

    if not page_text or not page_text.strip():
        return False, "Page text is empty."

    quote_raw = evidence_quote.strip()

    # 1. Exact raw character-for-character check
    if quote_raw in page_text:
        return True, "Exact raw character match verified."

    # 2. Whitespace-normalized check (handles newlines/spaces introduced by HTML parsing)
    norm_quote = normalize_whitespace(quote_raw).lower()
    norm_page = normalize_whitespace(page_text).lower()

    if norm_quote in norm_page:
        return True, "Whitespace-normalized verbatim match verified."

    # 3. Markdown-stripped check (handles markdown link syntax from conversion)
    stripped_page = normalize_whitespace(strip_markdown(page_text)).lower()
    stripped_quote = normalize_whitespace(strip_markdown(quote_raw)).lower()
    if stripped_quote in stripped_page:
        return True, "Markdown-normalized verbatim match verified."

    return False, f"Quote was not found in the stored page text: '{quote_raw[:80]}...'"


class VerificationVerdict(BaseModel):
    """Schema for secondary LLM verification response."""
    is_valid: bool = Field(description="True if the quote directly answers the question, False otherwise.")
    reason: str = Field(description="Brief explanation of why the quote answers or does not answer the question.")


VERIFICATION_PROMPT = """You are a rigorous factual auditor.
Your job is to independently verify whether a specific evidence quote directly answers the user's question.

USER QUESTION:
{question}

PROPOSED ANSWER:
{proposed_answer}

EVIDENCE QUOTE (EXTRACTED VERBATIM FROM THE SOURCE PAGE):
\"{evidence_quote}\"

TASK:
Does this evidence quote directly, factually, and sufficiently support or answer the question?
Reply with strictly valid JSON:
{{
    "is_valid": true,
    "reason": "The quote explicitly confirms the required fact."
}}
or:
{{
    "is_valid": false,
    "reason": "The quote talks about a different topic and does not state the answer."
}}

Rules:
- Be strict: If the quote only hints at or is vaguely related to the answer, reply false.
- Only reply true if the quote directly supports the answer.
"""


async def verify_answer_with_llm(
    llm: LLMClient,
    question: str,
    evidence_quote: str,
    proposed_answer: str,
) -> Tuple[bool, str]:
    """Secondary verification pass: ask the LLM if the quote answers the question."""
    prompt = VERIFICATION_PROMPT.format(
        question=question,
        proposed_answer=proposed_answer,
        evidence_quote=evidence_quote,
    )
    messages = [
        {"role": "system", "content": "You are a factual audit verifier. Reply only with valid JSON."},
        {"role": "user", "content": prompt},
    ]

    try:
        verdict_dict = await llm.async_chat_completion_json(messages, max_tokens=250)
        is_valid = bool(verdict_dict.get("is_valid", False))
        reason = str(verdict_dict.get("reason", "No explanation provided."))
        return is_valid, reason
    except Exception as exc:
        logger.warning("LLM verification pass failed with error: %s. Defaulting to code check.", exc)
        # If the verifier model fails to respond, we fail safely to avoid false positives
        return False, f"Verifier model error: {exc}"
