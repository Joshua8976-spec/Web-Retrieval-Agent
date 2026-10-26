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

    # Setup mock LLM: step 1 returns open_link 1, step 2 returns answer
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
    assert "action" in events


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
