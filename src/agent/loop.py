import asyncio
import inspect
import logging
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import quote_plus, urljoin, urlparse

from src.config import settings
from src.llm.client import LLMClient, default_llm_client
from src.tools.fetch import PageFetcher, FetchResult
from src.tools.extract import extract_clean_text, extract_relevant_page_content
from src.tools.links import (
    extract_links_from_html,
    rank_links_bm25,
    is_same_domain,
    normalize_url,
    ExtractedLink,
)
from src.agent.actions import (
    AgentAction,
    ActionType,
    OpenLinkAction,
    OpenUrlAction,
    SearchSiteAction,
    AnswerAction,
    GiveUpAction,
    parse_action,
)
from src.agent.state import AgentState, AgentResult
from src.agent.prompts import SYSTEM_PROMPT, build_step_prompt

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Core autonomous navigation loop for Web Information Retrieval."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        fetcher: Optional[PageFetcher] = None,
    ):
        self.llm = llm_client or default_llm_client
        self.fetcher = fetcher or PageFetcher()

    async def _emit_progress(
        self,
        callback: Optional[Callable[[Dict[str, Any]], Any]],
        event_type: str,
        data: Dict[str, Any],
    ):
        """Emit step progress updates to callback (sync or async)."""
        if not callback:
            return
        payload = {"event": event_type, **data}
        try:
            if inspect.iscoroutinefunction(callback):
                await callback(payload)
            else:
                callback(payload)
        except Exception as exc:
            logger.warning("Error in progress callback: %s", exc)

    async def run(
        self,
        start_url: str,
        question: str,
        progress_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ) -> AgentResult:
        """Run the end-to-end information retrieval agent loop."""
        state = AgentState(question=question, start_url=start_url)
        logger.info("Starting retrieval run: '%s' on %s", question, start_url)

        await self._emit_progress(
            progress_callback,
            "start",
            {"question": question, "start_url": start_url},
        )

        while not state.is_complete:
            # 1. Check hard limits (max pages, max timeout)
            limit_reached, limit_reason = state.is_limit_reached()
            if limit_reached:
                logger.info("Terminating run: %s", limit_reason)
                await self._emit_progress(progress_callback, "limit_reached", {"reason": limit_reason})
                return state.finish_not_found(explanation=limit_reason)

            current_url = state.current_url

            # 2. Prevent re-visiting the same page
            if state.has_visited(current_url):
                logger.info("URL %s already visited. Seeking alternative.", current_url)
                # Attempt to navigate to next unvisited link or terminate
                pass

            # 3. Fetch current page with retries
            fetch_result: Optional[FetchResult] = None
            while state.can_retry_fetch(current_url):
                await self._emit_progress(
                    progress_callback,
                    "fetching",
                    {"url": current_url, "step": state.step_count + 1},
                )
                fetch_result = await self.fetcher.fetch(current_url)
                if fetch_result.is_success:
                    break
                
                retry_count = state.record_fetch_retry(current_url)
                logger.warning(
                    "Fetch attempt %d failed for %s: %s",
                    retry_count, current_url, fetch_result.error,
                )
                if state.can_retry_fetch(current_url):
                    await asyncio.sleep(0.5)

            if not fetch_result or not fetch_result.is_success:
                logger.error("Failed to fetch %s after retries: %s", current_url, fetch_result.error if fetch_result else "No result")
                # If start URL failed completely, finish not found
                if len(state.visited_urls) == 0:
                    return state.finish_not_found(
                        explanation=f"Failed to fetch start URL: {fetch_result.error if fetch_result else 'Unknown'}"
                    )
                # Otherwise, attempt to pick an unvisited link from history or give up
                return state.finish_not_found(
                    explanation=f"Encountered unreachable page {current_url}: {fetch_result.error if fetch_result else 'Network error'}"
                )

            # Record visit in state with actual resolved final URL
            resolved_url = fetch_result.final_url
            raw_html = fetch_result.html
            full_clean_text = extract_clean_text(raw_html)
            state.record_visit(resolved_url, full_clean_text)

            # 4. Extract and rank content chunks using BM25
            relevant_page_text = extract_relevant_page_content(
                raw_html,
                query=state.question,
                top_k=settings.bm25_top_chunks,
            )

            # 5. Extract and rank links using BM25
            raw_links = extract_links_from_html(
                raw_html,
                current_url=resolved_url,
                same_domain_only=not settings.allow_external_domains,
            )
            # Filter out already visited links
            unvisited_links = [l for l in raw_links if not state.has_visited(l.url)]
            ranked_links = rank_links_bm25(
                state.question,
                unvisited_links,
                top_k=settings.bm25_top_links,
            )

            # 6. Build prompt context and call LLM
            step_prompt = build_step_prompt(
                question=state.question,
                start_url=state.start_url,
                current_url=resolved_url,
                page_text=relevant_page_text,
                ranked_links=ranked_links,
                page_history=state.page_history,
                step_number=state.step_count,
                max_steps=settings.agent_max_pages,
            )

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": step_prompt},
            ]

            await self._emit_progress(
                progress_callback,
                "planning",
                {"step": state.step_count, "url": resolved_url},
            )

            try:
                action_dict = await self.llm.async_chat_completion_json(messages)
                action = parse_action(action_dict)
            except Exception as exc:
                logger.error("LLM action generation or parsing failed: %s", exc)
                return state.finish_not_found(
                    explanation=f"LLM returned unparseable action: {exc}"
                )

            await self._emit_progress(
                progress_callback,
                "action",
                {"step": state.step_count, "action": action.model_dump()},
            )

            # 7. Execute validated action
            if isinstance(action, AnswerAction):
                # Phase 3 basic evidence check: quote must be in visited page
                quote = action.evidence_quote.strip()
                stored_page = state.page_contents.get(resolved_url, "")
                if quote and quote in stored_page:
                    return state.finish_with_answer(
                        answer=action.answer,
                        evidence_quote=action.evidence_quote,
                        source_url=action.source_url or resolved_url,
                    )
                else:
                    logger.warning("Evidence quote not found verbatim in page text. Prompting next step.")
                    # If quote failed, let loop continue or fall back
                    if ranked_links:
                        state.current_url = ranked_links[0].url
                        continue
                    else:
                        return state.finish_with_answer(
                            answer=action.answer,
                            evidence_quote=action.evidence_quote,
                            source_url=action.source_url or resolved_url,
                        )

            elif isinstance(action, GiveUpAction):
                logger.info("Agent decided to give up: %s", action.reason)
                return state.finish_not_found(explanation=action.reason)

            elif isinstance(action, OpenLinkAction):
                target_link = None
                for link in ranked_links:
                    if link.index == action.link_number:
                        target_link = link
                        break

                if target_link and not state.has_visited(target_link.url):
                    state.current_url = target_link.url
                elif ranked_links:
                    # Pick first unvisited link as fallback
                    fallback_link = next((l for l in ranked_links if not state.has_visited(l.url)), None)
                    if fallback_link:
                        state.current_url = fallback_link.url
                    else:
                        return state.finish_not_found(explanation="All candidate links on the site have been visited.")
                else:
                    return state.finish_not_found(explanation="No further navigable links found on this page.")

            elif isinstance(action, OpenUrlAction):
                normalized = normalize_url(action.url, resolved_url)
                if normalized and is_same_domain(normalized, state.start_url):
                    if not state.has_visited(normalized):
                        state.current_url = normalized
                    else:
                        # URL already visited, try best ranked link instead
                        fallback = next((l for l in ranked_links if not state.has_visited(l.url)), None)
                        if fallback:
                            state.current_url = fallback.url
                        else:
                            return state.finish_not_found(explanation="Requested URL was already visited and no more links remain.")
                else:
                    logger.warning("Model requested external or invalid URL: %s. Confined to domain.", action.url)
                    if ranked_links:
                        state.current_url = ranked_links[0].url
                    else:
                        return state.finish_not_found(explanation="Cannot navigate outside the starting domain.")

            elif isinstance(action, SearchSiteAction):
                # Construct common search query URLs for the current site
                query_encoded = quote_plus(action.query)
                search_url = action.search_url or urljoin(resolved_url, f"/search?q={query_encoded}")
                normalized = normalize_url(search_url, resolved_url)
                if normalized and not state.has_visited(normalized):
                    state.current_url = normalized
                elif ranked_links:
                    state.current_url = ranked_links[0].url
                else:
                    return state.finish_not_found(explanation="Site search was unavailable and no alternative links remain.")

        # Default fallback if loop ends without terminal action
        return state.finish_not_found(explanation="Search concluded without finding conclusive information.")
