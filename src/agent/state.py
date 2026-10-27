import time
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse
from pydantic import BaseModel, Field

from src.config import settings


class AgentResult(BaseModel):
    """Final output schema for the retrieval agent."""

    found: bool
    status: str = Field(default="not_found_on_site", description="Outcome status: found, not_found_on_site, or access_error.")
    answer: Optional[str] = None
    evidence_quote: Optional[str] = None
    source_url: Optional[str] = None
    explanation: Optional[str] = None
    pages_visited: List[str] = Field(default_factory=list)
    steps: int = 0
    elapsed_seconds: float = 0.0


class AgentState(BaseModel):
    """State tracking for a single information retrieval run."""

    question: str
    start_url: str
    target_domain: str = ""
    current_url: str = ""
    visited_urls: Set[str] = Field(default_factory=set)
    attempted_urls: Set[str] = Field(default_factory=set)
    discovered_urls: Set[str] = Field(default_factory=set)
    page_history: List[str] = Field(default_factory=list)
    page_contents: Dict[str, str] = Field(default_factory=dict)
    step_count: int = 0
    start_time: float = Field(default_factory=time.perf_counter)
    fetch_retries: Dict[str, int] = Field(default_factory=dict)
    is_complete: bool = False
    final_output: Optional[AgentResult] = None

    def model_post_init(self, __context):
        if not self.target_domain and self.start_url:
            self.target_domain = urlparse(self.start_url).netloc.lower()
        if not self.current_url:
            self.current_url = self.start_url
        if self.start_url:
            self.discovered_urls.add(self.start_url)

    def elapsed_seconds(self) -> float:
        """Seconds elapsed since agent execution started."""
        return time.perf_counter() - self.start_time

    def is_limit_reached(self) -> Tuple[bool, str]:
        """Check if hard constraints (10 pages or 120s timeout) have been exceeded."""
        if len(self.visited_urls) >= settings.agent_max_pages:
            return True, f"Maximum page limit reached ({settings.agent_max_pages} pages visited)."
        
        if self.elapsed_seconds() >= settings.agent_timeout_seconds:
            return True, f"Time limit exceeded ({settings.agent_timeout_seconds} seconds)."
        
        return False, ""

    def has_visited(self, url: str) -> bool:
        """Check if a URL has already been visited or attempted (prevents repeated fetches)."""
        return url in self.visited_urls or url in self.attempted_urls

    def record_attempt(self, url: str):
        """Mark URL as attempted to guarantee no repeated fetches of the same URL."""
        self.attempted_urls.add(url)

    def record_visit(self, url: str, clean_text: str):
        """Record page visit and cache extracted text."""
        self.visited_urls.add(url)
        self.attempted_urls.add(url)
        if url not in self.page_history:
            self.page_history.append(url)
        self.page_contents[url] = clean_text
        self.step_count += 1

    def can_retry_fetch(self, url: str, status_code: Optional[int] = None) -> bool:
        """Check if a URL can be retried (never retry 404s or client errors)."""
        if status_code in (404, 400, 401, 403, 410):
            return False
        retries = self.fetch_retries.get(url, 0)
        return retries < settings.agent_max_retries

    def record_fetch_retry(self, url: str) -> int:
        """Increment and return retry count for a URL."""
        count = self.fetch_retries.get(url, 0) + 1
        self.fetch_retries[url] = count
        return count

    def finish_with_answer(
        self,
        answer: str,
        evidence_quote: str,
        source_url: str,
    ) -> AgentResult:
        """Mark run complete with a verified answer."""
        from src.agent.verifier import format_evidence_quote

        cleaned_quote = format_evidence_quote(evidence_quote) or evidence_quote
        self.is_complete = True
        self.final_output = AgentResult(
            found=True,
            status="found",
            answer=answer,
            evidence_quote=cleaned_quote,
            source_url=source_url,
            pages_visited=list(self.page_history),
            steps=self.step_count,
            elapsed_seconds=round(self.elapsed_seconds(), 2),
        )
        return self.final_output

    def finish_not_found(
        self,
        explanation: str,
        status: str = "not_found_on_site",
    ) -> AgentResult:
        """Mark run complete when the information is not on the site, limits are reached, or access error occurred."""
        self.is_complete = True
        self.final_output = AgentResult(
            found=False,
            status=status,
            answer=None,
            evidence_quote=None,
            source_url=None,
            explanation=explanation,
            pages_visited=list(self.page_history),
            steps=self.step_count,
            elapsed_seconds=round(self.elapsed_seconds(), 2),
        )
        return self.final_output
