from unittest.mock import AsyncMock, MagicMock
import pytest
from src.agent.loop import AgentOrchestrator
from src.tools.fetch import FetchResult


@pytest.mark.asyncio
async def test_orchestrator_multi_step_navigation():
    # Setup mock fetcher
    mock_fetcher = MagicMock()
    page1_html = """
    <html><body>
        <h1>Welcome to AMD</h1>
        <a href="/specifications">Product Specifications</a>
    </body></html>
    """
    page2_html = """
    <html><body>
        <h1>Product Specifications</h1>
        <p>The AMD Instinct MI300X features 192GB of HBM3 memory.</p>
    </body></html>
    """

    async def mock_fetch(url):
        if "specifications" in url:
            return FetchResult(
                url=url, final_url="https://amd.com/specifications", status_code=200, html=page2_html, is_success=True
            )
        return FetchResult(
            url=url, final_url="https://amd.com", status_code=200, html=page1_html, is_success=True
        )

    mock_fetcher.fetch = AsyncMock(side_effect=mock_fetch)

    # Setup mock LLM:
    # 1. step 1 action -> open_link 1
    # 2. step 2 action -> answer
    # 3. step 2 secondary verification -> is_valid: True
    mock_llm = MagicMock()
    mock_llm.async_chat_completion_json = AsyncMock(
        side_effect=[
            {"action": "open_link", "link_number": 1, "reasoning": "Navigate to specifications."},
            {
                "action": "answer",
                "answer": "192GB of HBM3 memory",
                "evidence_quote": "The AMD Instinct MI300X features 192GB of HBM3 memory.",
                "source_url": "https://amd.com/specifications",
            },
            {"is_valid": True, "reason": "Quote explicitly confirms the memory specification."},
        ]
    )

    events = []
    def progress_tracker(ev):
        events.append(ev["event"])

    orchestrator = AgentOrchestrator(llm_client=mock_llm, fetcher=mock_fetcher)
    result = await orchestrator.run(
        start_url="https://amd.com",
        question="How much memory does MI300X have?",
        progress_callback=progress_tracker,
    )

    assert result.found is True
    assert result.answer == "192GB of HBM3 memory"
    assert "192GB" in (result.evidence_quote or "")
    assert len(result.pages_visited) == 2
    assert "https://amd.com/specifications" in result.pages_visited
    assert "start" in events
    assert "verifying" in events
    assert "answer_accepted" in events


@pytest.mark.asyncio
async def test_orchestrator_rejects_hallucinated_quote_by_code_check():
    mock_fetcher = MagicMock()
    page_html = "<html><body><p>AMD ROCm software documentation.</p></body></html>"
    mock_fetcher.fetch = AsyncMock(
        return_value=FetchResult(
            url="https://amd.com",
            final_url="https://amd.com",
            status_code=200,
            html=page_html,
            is_success=True,
        )
    )

    mock_llm = MagicMock()
    # Model attempts to hallucinate a quote that is NOT on the page
    mock_llm.async_chat_completion_json = AsyncMock(
        return_value={
            "action": "answer",
            "answer": "Fabricated fact",
            "evidence_quote": "This quote does not exist anywhere on the page.",
            "source_url": "https://amd.com",
        }
    )

    events = []
    orchestrator = AgentOrchestrator(llm_client=mock_llm, fetcher=mock_fetcher)
    result = await orchestrator.run(
        start_url="https://amd.com",
        question="What is the fabricated fact?",
        progress_callback=lambda ev: events.append(ev["event"]),
    )

    # Since quote failed code check and no more links were available, answer was rejected
    assert result.found is False
    assert "answer_rejected" in events
    assert "not found verbatim" in (result.explanation or "")


@pytest.mark.asyncio
async def test_orchestrator_rejects_when_llm_verification_fails():
    mock_fetcher = MagicMock()
    page_html = "<html><body><p>AMD was founded in California.</p></body></html>"
    mock_fetcher.fetch = AsyncMock(
        return_value=FetchResult(
            url="https://amd.com",
            final_url="https://amd.com",
            status_code=200,
            html=page_html,
            is_success=True,
        )
    )

    mock_llm = MagicMock()
    # Step 1: Model outputs answer with quote that exists on page
    # Step 2: Verification pass rejects it because quote does not answer the question
    mock_llm.async_chat_completion_json = AsyncMock(
        side_effect=[
            {
                "action": "answer",
                "answer": "AMD has 50000 employees",
                "evidence_quote": "AMD was founded in California.",
                "source_url": "https://amd.com",
            },
            {
                "is_valid": False,
                "reason": "The quote talks about where AMD was founded, not how many employees it has.",
            },
        ]
    )

    events = []
    orchestrator = AgentOrchestrator(llm_client=mock_llm, fetcher=mock_fetcher)
    result = await orchestrator.run(
        start_url="https://amd.com",
        question="How many employees does AMD have?",
        progress_callback=lambda ev: events.append(ev["event"]),
    )

    assert result.found is False
    assert "answer_rejected" in events
    assert "rejected by verification pass" in (result.explanation or "")


@pytest.mark.asyncio
async def test_orchestrator_give_up():
    mock_fetcher = MagicMock()
    mock_fetcher.fetch = AsyncMock(
        return_value=FetchResult(
            url="https://amd.com",
            final_url="https://amd.com",
            status_code=200,
            html="<html><body><p>Some unrelated text.</p></body></html>",
            is_success=True,
        )
    )

    mock_llm = MagicMock()
    mock_llm.async_chat_completion_json = AsyncMock(
        return_value={"action": "give_up", "reason": "Information not present on this website."}
    )

    orchestrator = AgentOrchestrator(llm_client=mock_llm, fetcher=mock_fetcher)
    result = await orchestrator.run(
        start_url="https://amd.com",
        question="What is the CEO's favorite pet?",
    )

    assert result.found is False
    assert result.answer is None
    assert "not present" in (result.explanation or "")
