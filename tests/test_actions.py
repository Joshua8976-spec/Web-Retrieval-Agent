import pytest
from pydantic import ValidationError
from src.agent.actions import (
    ActionType,
    OpenLinkAction,
    OpenUrlAction,
    SearchSiteAction,
    AnswerAction,
    GiveUpAction,
    parse_action,
)


def test_parse_open_link_action():
    data = {"action": "open_link", "link_number": 3, "reasoning": "Looks relevant."}
    action = parse_action(data)
    assert isinstance(action, OpenLinkAction)
    assert action.link_number == 3
    assert action.reasoning == "Looks relevant."


def test_parse_open_url_action():
    data = {"action": "open_url", "url": "https://example.com/faq"}
    action = parse_action(data)
    assert isinstance(action, OpenUrlAction)
    assert action.url == "https://example.com/faq"


def test_parse_search_site_action():
    data = {"action": "search_site", "query": "pricing details"}
    action = parse_action(data)
    assert isinstance(action, SearchSiteAction)
    assert action.query == "pricing details"


def test_parse_answer_action():
    data = {
        "action": "answer",
        "answer": "The speed is 100 mph.",
        "evidence_quote": "The speed is 100 mph.",
        "source_url": "https://example.com/specs",
    }
    action = parse_action(data)
    assert isinstance(action, AnswerAction)
    assert action.answer == "The speed is 100 mph."
    assert action.evidence_quote == "The speed is 100 mph."
    assert action.source_url == "https://example.com/specs"


def test_parse_give_up_action():
    data = {"action": "give_up", "reason": "No mention of warranty on any page."}
    action = parse_action(data)
    assert isinstance(action, GiveUpAction)
    assert action.reason == "No mention of warranty on any page."


def test_parse_invalid_action():
    with pytest.raises(ValueError):
        parse_action({"action": "unknown_action_type"})

    with pytest.raises(ValueError):
        parse_action({"no_action_key": "test"})
