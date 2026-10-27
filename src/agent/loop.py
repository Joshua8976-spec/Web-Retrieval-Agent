import asyncio
import inspect
import logging
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urljoin, urlparse

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
from src.tools.search_forms import extract_search_forms, DiscoveredSearchForm
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
from src.agent.verifier import verify_evidence_in_page, verify_answer_with_llm

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
        normalized_start = normalize_url(start_url, start_url) or start_url
        state = AgentState(question=question, start_url=normalized_start)
        logger.info("Starting retrieval run: '%s' on %s", question, state.start_url)
        unvisited_queue: List[str] = []
        last_error: Optional[str] = None
        current_search_forms: List[DiscoveredSearchForm] = []

        await self._emit_progress(
            progress_callback,
            "start",
            {"question": question, "start_url": state.start_url},
        )

        while not state.is_complete:
            # 1. Check hard limits (max pages, max timeout)
            limit_reached, limit_reason = state.is_limit_reached()
            if limit_reached:
                logger.info("Terminating run: %s", limit_reason)
                await self._emit_progress(progress_callback, "limit_reached", {"reason": limit_reason})
                return state.finish_not_found(explanation=limit_reason, status="not_found_on_site")

            current_url = state.current_url

            # 2. Prevent re-visiting already processed URLs
            if state.has_visited(current_url) and current_url in state.page_contents:
                # If we're already on this page and have content, check if we need to advance
                if not unvisited_queue and not state.page_history:
                    return state.finish_not_found(explanation="All navigable links on this site have been visited.", status="not_found_on_site")
            elif state.has_visited(current_url) and current_url not in state.page_contents:
                # URL failed previously; pick next candidate from queue
                next_url = None
                while unvisited_queue:
                    cand = unvisited_queue.pop(0)
                    if not state.has_visited(cand):
                        next_url = cand
                        break
                if next_url:
                    current_url = next_url
                    state.current_url = next_url
                else:
                    return state.finish_not_found(explanation="All navigable links on this site have been exhausted.", status="not_found_on_site")

            # 3. Fetch page if not already in memory
            raw_html = ""
            resolved_url = current_url
            if current_url in state.page_contents:
                resolved_url = current_url
                full_clean_text = state.page_contents[current_url]
            else:
                state.record_attempt(current_url)
                fetch_result: Optional[FetchResult] = None
                while True:
                    await self._emit_progress(
                        progress_callback,
                        "fetching",
                        {"url": current_url, "step": state.step_count},
                    )
                    fetch_result = await self.fetcher.fetch(current_url)
                    if fetch_result.is_success:
                        break

                    # Never retry 404s or client errors
                    if not state.can_retry_fetch(current_url, fetch_result.status_code):
                        break

                    retry_count = state.record_fetch_retry(current_url)
                    logger.warning("Fetch attempt %d failed for %s: %s", retry_count, current_url, fetch_result.error)
                    if state.can_retry_fetch(current_url, fetch_result.status_code):
                        await asyncio.sleep(0.5)

                if not fetch_result or not fetch_result.is_success:
                    state.step_count += 1
                    error_msg = fetch_result.error if fetch_result else "Network error"
                    logger.warning("Failed to fetch %s (HTTP %s): %s", current_url, fetch_result.status_code if fetch_result else 0, error_msg)

                    # If start URL failed completely on first step, finish with access_error
                    if len(state.visited_urls) == 0:
                        return state.finish_not_found(
                            explanation=f"Failed to access start URL {current_url}: {error_msg}",
                            status="access_error",
                        )

                    # Return error observation to LLM on next turn and continue
                    last_error = f"Failed to open '{current_url}': HTTP {fetch_result.status_code if fetch_result else 0} ({error_msg})."
                    await self._emit_progress(
                        progress_callback,
                        "fetch_failed",
                        {"url": current_url, "error": last_error},
                    )

                    next_url = None
                    while unvisited_queue:
                        cand = unvisited_queue.pop(0)
                        if not state.has_visited(cand):
                            next_url = cand
                            break
                    if next_url:
                        state.current_url = next_url
                        continue
                    elif state.page_history:
                        state.current_url = state.page_history[-1]
                        continue
                    else:
                        return state.finish_not_found(
                            explanation=f"Encountered unreachable page {current_url}: {error_msg}",
                            status="not_found_on_site",
                        )

                # Record visit in state with normalized URL
                resolved_url = normalize_url(fetch_result.final_url, fetch_result.final_url) or fetch_result.final_url
                raw_html = fetch_result.html
                full_clean_text = extract_clean_text(raw_html)
                state.record_visit(resolved_url, full_clean_text)

            # 4. Extract and rank content chunks using BM25
            relevant_page_text = extract_relevant_page_content(
                raw_html,
                query=state.question,
                top_k=settings.bm25_top_chunks,
            ) if raw_html else full_clean_text[:3000]

            # 5. Discover search forms and links
            if raw_html:
                current_search_forms = extract_search_forms(raw_html, resolved_url)
                raw_links = extract_links_from_html(
                    raw_html,
                    current_url=resolved_url,
                    same_domain_only=not settings.allow_external_domains,
                )
                for l in raw_links:
                    state.discovered_urls.add(l.url)
                    if not state.has_visited(l.url) and l.url not in unvisited_queue:
                        unvisited_queue.append(l.url)
                for f in current_search_forms:
                    state.discovered_urls.add(f.action)
            else:
                raw_links = []

            # Filter out already visited links for this step
            unvisited_links = [l for l in raw_links if not state.has_visited(l.url)]
            ranked_links = rank_links_bm25(
                state.question,
                unvisited_links,
                top_k=settings.bm25_top_links,
            )

            # Format search forms for prompt
            search_forms_text = None
            if current_search_forms:
                search_forms_text = "\n".join(
                    f"- Form {idx}: Action URL: {f.action}, Query Field: '{f.query_param}', Method: {f.method}"
                    + (f", Hidden: {f.hidden_fields}" if f.hidden_fields else "")
                    for idx, f in enumerate(current_search_forms, 1)
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
                search_forms_text=search_forms_text,
                last_error=last_error,
            )

            # Clear last_error after presenting observation to LLM
            last_error = None

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
                    explanation=f"LLM returned unparseable action: {exc}",
                    status="not_found_on_site",
                )

            await self._emit_progress(
                progress_callback,
                "action",
                {"step": state.step_count, "action": action.model_dump()},
            )

            # 7. Execute validated action
            if isinstance(action, AnswerAction):
                await self._emit_progress(
                    progress_callback,
                    "verifying",
                    {"step": state.step_count, "quote": action.evidence_quote},
                )

                # Check 1: Deterministic Code Check (verbatim quote in visited page)
                target_url = action.source_url or resolved_url
                page_text = state.page_contents.get(target_url, "")
                if not page_text and resolved_url in state.page_contents:
                    page_text = state.page_contents[resolved_url]
                    target_url = resolved_url

                quote_valid, code_check_msg = verify_evidence_in_page(action.evidence_quote, page_text)
                if not quote_valid:
                    # Also check if quote appears on ANY other previously visited page
                    for past_url, past_text in state.page_contents.items():
                        past_valid, _ = verify_evidence_in_page(action.evidence_quote, past_text)
                        if past_valid:
                            quote_valid = True
                            target_url = past_url
                            code_check_msg = "Match found on previously visited page."
                            break

                if not quote_valid:
                    logger.warning("Code check failed for answer quote: %s", code_check_msg)
                    last_error = f"Answer rejected: evidence quote was not found verbatim in page text ({code_check_msg})."
                    await self._emit_progress(
                        progress_callback,
                        "answer_rejected",
                        {"reason": last_error},
                    )
                    # Reject and navigate to next unvisited link or candidate
                    fallback_link = next((l for l in ranked_links if not state.has_visited(l.url)), None)
                    if not fallback_link and unvisited_queue:
                        fallback_url = next((u for u in unvisited_queue if not state.has_visited(u)), None)
                    else:
                        fallback_url = fallback_link.url if fallback_link else None

                    if fallback_url:
                        state.current_url = fallback_url
                        continue
                    else:
                        return state.finish_not_found(
                            explanation=f"Proposed answer quote was not found verbatim in page content ({code_check_msg}).",
                            status="not_found_on_site",
                        )

                # Check 2: Secondary LLM Verification Pass
                llm_valid, llm_reason = await verify_answer_with_llm(
                    llm=self.llm,
                    question=state.question,
                    evidence_quote=action.evidence_quote,
                    proposed_answer=action.answer,
                )

                if not llm_valid:
                    logger.warning("LLM verification pass rejected answer: %s", llm_reason)
                    last_error = f"Answer rejected: verification pass determined quote does not answer the question ({llm_reason})."
                    await self._emit_progress(
                        progress_callback,
                        "answer_rejected",
                        {"reason": last_error},
                    )
                    fallback_link = next((l for l in ranked_links if not state.has_visited(l.url)), None)
                    if not fallback_link and unvisited_queue:
                        fallback_url = next((u for u in unvisited_queue if not state.has_visited(u)), None)
                    else:
                        fallback_url = fallback_link.url if fallback_link else None

                    if fallback_url:
                        state.current_url = fallback_url
                        continue
                    else:
                        return state.finish_not_found(
                            explanation=f"Proposed answer was rejected by verification pass: {llm_reason}",
                            status="not_found_on_site",
                        )

                # Both checks passed! Return verified answer
                await self._emit_progress(
                    progress_callback,
                    "answer_accepted",
                    {"answer": action.answer, "source_url": target_url},
                )
                return state.finish_with_answer(
                    answer=action.answer,
                    evidence_quote=action.evidence_quote,
                    source_url=target_url,
                )

            elif isinstance(action, GiveUpAction):
                logger.info("Agent decided to give up: %s", action.reason)
                return state.finish_not_found(explanation=action.reason, status="not_found_on_site")

            elif isinstance(action, OpenLinkAction):
                target_link = None
                for link in ranked_links:
                    if link.index == action.link_number:
                        target_link = link
                        break

                if target_link and not state.has_visited(target_link.url):
                    state.current_url = target_link.url
                else:
                    # Pick next unvisited link from ranked_links or queue
                    fallback_url = next((l.url for l in ranked_links if not state.has_visited(l.url)), None)
                    if not fallback_url:
                        while unvisited_queue:
                            cand = unvisited_queue.pop(0)
                            if not state.has_visited(cand):
                                fallback_url = cand
                                break
                    if fallback_url:
                        state.current_url = fallback_url
                        last_error = f"Link #{action.link_number} was already visited or invalid. Navigated to fallback candidate."
                    else:
                        return state.finish_not_found(
                            explanation="All candidate links on the site have been visited.",
                            status="not_found_on_site",
                        )

            elif isinstance(action, OpenUrlAction):
                normalized = normalize_url(action.url, resolved_url)
                if not normalized or not is_same_domain(normalized, state.start_url):
                    logger.warning("Model requested external or invalid URL: %s. Confined to domain.", action.url)
                    last_error = f"URL '{action.url}' is outside the allowed site domain."
                    fallback = next((l.url for l in ranked_links if not state.has_visited(l.url)), None)
                    if not fallback and unvisited_queue:
                        fallback = next((u for u in unvisited_queue if not state.has_visited(u)), None)
                    if fallback:
                        state.current_url = fallback
                    else:
                        return state.finish_not_found(
                            explanation="Cannot navigate outside the starting domain.",
                            status="not_found_on_site",
                        )
                elif normalized not in state.discovered_urls and action.url not in state.discovered_urls:
                    # Model invented an unobserved speculative URL (e.g. /search?q=...)
                    logger.warning("Model invented undiscovered URL: %s. Rejecting speculative guess.", action.url)
                    last_error = (
                        f"URL '{action.url}' was not discovered on the page. "
                        f"Do not guess hypothetical URLs. To search the site, use 'search_site'. "
                        f"To navigate, choose an existing link from <available_links>."
                    )
                    fallback = next((l.url for l in ranked_links if not state.has_visited(l.url)), None)
                    if not fallback and unvisited_queue:
                        fallback = next((u for u in unvisited_queue if not state.has_visited(u)), None)
                    if fallback:
                        state.current_url = fallback
                    else:
                        return state.finish_not_found(
                            explanation=f"Invented URL '{action.url}' does not exist on the page and no alternative links remain.",
                            status="not_found_on_site",
                        )
                elif not state.has_visited(normalized):
                    state.current_url = normalized
                else:
                    fallback = next((l.url for l in ranked_links if not state.has_visited(l.url)), None)
                    if not fallback and unvisited_queue:
                        fallback = next((u for u in unvisited_queue if not state.has_visited(u)), None)
                    if fallback:
                        state.current_url = fallback
                        last_error = f"Requested URL '{normalized}' was already visited."
                    else:
                        return state.finish_not_found(
                            explanation="Requested URL was already visited and no more links remain.",
                            status="not_found_on_site",
                        )

            elif isinstance(action, SearchSiteAction):
                # Search using real discovered form actions or explicit search_url (no blind /search?q= guessing)
                search_target_url = None
                if action.search_url:
                    normalized_search = normalize_url(action.search_url, resolved_url)
                    if normalized_search and is_same_domain(normalized_search, state.start_url):
                        search_target_url = normalized_search
                    else:
                        last_error = f"Search URL '{action.search_url}' is outside the allowed site domain."
                elif current_search_forms:
                    best_form = next((f for f in current_search_forms if f.method.upper() == "GET"), current_search_forms[0])
                    search_target_url = best_form.build_query_url(action.query)

                if search_target_url and not state.has_visited(search_target_url):
                    state.current_url = search_target_url
                elif search_target_url and state.has_visited(search_target_url):
                    last_error = f"Search URL '{search_target_url}' was already visited."
                    fallback = next((l.url for l in ranked_links if not state.has_visited(l.url)), None)
                    if not fallback and unvisited_queue:
                        fallback = next((u for u in unvisited_queue if not state.has_visited(u)), None)
                    if fallback:
                        state.current_url = fallback
                    else:
                        return state.finish_not_found(
                            explanation="Search query was already visited and no more links remain.",
                            status="not_found_on_site",
                        )
                else:
                    last_error = "No search form was discovered on this page. Please use 'open_link' to navigate or 'open_url' with a valid link."
                    logger.info("Search requested but no search form discovered on %s", resolved_url)
                    fallback = next((l.url for l in ranked_links if not state.has_visited(l.url)), None)
                    if not fallback and unvisited_queue:
                        fallback = next((u for u in unvisited_queue if not state.has_visited(u)), None)
                    if fallback:
                        state.current_url = fallback
                    else:
                        return state.finish_not_found(
                            explanation="Site search was unavailable and no alternative links remain.",
                            status="not_found_on_site",
                        )

        # Default fallback if loop ends without terminal action
        return state.finish_not_found(
            explanation="Search concluded without finding conclusive information.",
            status="not_found_on_site",
        )
