from enum import Enum
from typing import Annotated, Literal, Optional, Union
from pydantic import BaseModel, Field, ValidationError


class ActionType(str, Enum):
    OPEN_LINK = "open_link"
    OPEN_URL = "open_url"
    SEARCH_SITE = "search_site"
    ANSWER = "answer"
    GIVE_UP = "give_up"


class OpenLinkAction(BaseModel):
    """Choose a numbered link from the available links list."""
    action: Literal[ActionType.OPEN_LINK] = ActionType.OPEN_LINK
    link_number: int = Field(description="The 1-based index number of the link to open.")
    reasoning: Optional[str] = Field(default=None, description="Brief explanation of why this link was chosen.")


class OpenUrlAction(BaseModel):
    """Open a specific internal URL on the same site (e.g., sitemap, docs, search)."""
    action: Literal[ActionType.OPEN_URL] = ActionType.OPEN_URL
    url: str = Field(description="The internal URL on the same site to navigate to.")
    reasoning: Optional[str] = Field(default=None, description="Brief explanation of why this URL was chosen.")


class SearchSiteAction(BaseModel):
    """Execute a search query on the current site's search feature."""
    action: Literal[ActionType.SEARCH_SITE] = ActionType.SEARCH_SITE
    query: str = Field(description="The search query keywords.")
    search_url: Optional[str] = Field(default=None, description="The site search endpoint URL if known.")
    reasoning: Optional[str] = Field(default=None, description="Why searching is preferred at this step.")


class AnswerAction(BaseModel):
    """Return the final answer backed by exact verbatim supporting evidence."""
    action: Literal[ActionType.ANSWER] = ActionType.ANSWER
    answer: str = Field(description="Concise, direct answer to the user's question.")
    evidence_quote: str = Field(description="Verbatim text quote copied word-for-word from the page.")
    source_url: str = Field(description="The exact URL of the page containing the evidence quote.")
    reasoning: Optional[str] = Field(default=None, description="Explanation connecting the quote to the answer.")


class GiveUpAction(BaseModel):
    """Acknowledge that the information is not present on the site after thorough searching."""
    action: Literal[ActionType.GIVE_UP] = ActionType.GIVE_UP
    reason: str = Field(description="Explanation of why the information was not found on the site.")


# Discriminated union of all possible actions
AgentAction = Annotated[
    Union[OpenLinkAction, OpenUrlAction, SearchSiteAction, AnswerAction, GiveUpAction],
    Field(discriminator="action"),
]


def parse_action(data: dict) -> AgentAction:
    """Parse and validate raw JSON into a typed AgentAction model.
    
    Raises ValidationError if data does not conform to the schema.
    """
    action_type = data.get("action")
    if not action_type:
        raise ValueError("Missing required field 'action' in JSON.")

    if action_type == ActionType.OPEN_LINK.value:
        return OpenLinkAction.model_validate(data)
    elif action_type == ActionType.OPEN_URL.value:
        return OpenUrlAction.model_validate(data)
    elif action_type == ActionType.SEARCH_SITE.value:
        return SearchSiteAction.model_validate(data)
    elif action_type == ActionType.ANSWER.value:
        return AnswerAction.model_validate(data)
    elif action_type == ActionType.GIVE_UP.value:
        return GiveUpAction.model_validate(data)
    else:
        raise ValueError(f"Unknown action type: {action_type}. Allowed: {[a.value for a in ActionType]}")
