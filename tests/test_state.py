import time
import pytest
from src.agent.state import AgentState, AgentResult


def test_agent_state_initialization():
    state = AgentState(question="What is ROCm?", start_url="https://rocm.docs.amd.com/en/latest/")
    assert state.target_domain == "rocm.docs.amd.com"
    assert state.current_url == "https://rocm.docs.amd.com/en/latest/"
    assert state.step_count == 0
    assert len(state.visited_urls) == 0
    assert state.is_complete is False


def test_record_visit_and_visited_check():
    state = AgentState(question="What is ROCm?", start_url="https://amd.com")
    url = "https://amd.com/page1"
    
    assert state.has_visited(url) is False
    state.record_visit(url, "Clean page content here")
    assert state.has_visited(url) is True
    assert state.step_count == 1
    assert state.page_contents[url] == "Clean page content here"
    assert state.page_history == [url]


def test_state_limits_detection(monkeypatch):
    state = AgentState(question="Query", start_url="https://amd.com")
    
    # Under limit
    reached, reason = state.is_limit_reached()
    assert reached is False

    # Simulate max pages reached
    for i in range(10):
        state.record_visit(f"https://amd.com/p{i}", f"content {i}")
    
    reached, reason = state.is_limit_reached()
    assert reached is True
    assert "Maximum page limit" in reason


def test_terminal_states():
    state = AgentState(question="Query", start_url="https://amd.com")
    state.record_visit("https://amd.com/about", "Page about AMD")
    
    # Answer terminal state
    result = state.finish_with_answer(
        answer="Answer text",
        evidence_quote="AMD was founded in 1969.",
        source_url="https://amd.com/about",
    )
    assert result.found is True
    assert result.answer == "Answer text"
    assert result.steps == 1
    assert state.is_complete is True

    # Not found terminal state
    state2 = AgentState(question="Query", start_url="https://amd.com")
    res_not_found = state2.finish_not_found(explanation="Not on site")
    assert res_not_found.found is False
    assert res_not_found.explanation == "Not on site"
