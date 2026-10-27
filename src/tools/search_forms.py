import logging
from typing import Dict, List, Optional
from urllib.parse import urlencode, urljoin, urlparse
from pydantic import BaseModel, Field
from selectolax.lexbor import LexborHTMLParser as HTMLParser

logger = logging.getLogger(__name__)


class DiscoveredSearchForm(BaseModel):
    """Structured representation of a search form discovered on an HTML page."""

    action: str = Field(description="Resolved target URL for form submission.")
    method: str = Field(default="GET", description="HTTP method (GET or POST).")
    query_param: str = Field(default="q", description="Name of the input field for the search query.")
    hidden_fields: Dict[str, str] = Field(default_factory=dict, description="Hidden input fields required by the form.")
    description: Optional[str] = Field(default=None, description="Form label, placeholder, or identifier.")

    def build_query_url(self, query: str) -> str:
        """Construct a GET URL with the query and hidden parameters encoded."""
        params = dict(self.hidden_fields)
        params[self.query_param] = query
        query_string = urlencode(params)
        delimiter = "&" if "?" in self.action else "?"
        return f"{self.action}{delimiter}{query_string}"


class OpenSearchLink(BaseModel):
    """OpenSearch description link found in page <head>."""

    href: str
    title: Optional[str] = None
    type: str = "application/opensearchdescription+xml"


def extract_search_forms(html: str, current_url: str) -> List[DiscoveredSearchForm]:
    """Discover all navigable search forms on the page without site-specific hardcoding."""
    if not html or not html.strip():
        return []

    discovered: List[DiscoveredSearchForm] = []
    try:
        tree = HTMLParser(html)
        forms = tree.css("form")
        for form in forms:
            action_attr = form.attributes.get("action", "")
            action_url = urljoin(current_url, action_attr) if action_attr else current_url
            method = form.attributes.get("method", "get").upper()

            # Inspect inputs
            inputs = form.css("input, textarea")
            query_param: Optional[str] = None
            hidden_fields: Dict[str, str] = {}
            placeholder: Optional[str] = None

            text_candidates: List[str] = []
            for inp in inputs:
                inp_type = inp.attributes.get("type", "text").lower()
                inp_name = inp.attributes.get("name")
                if not inp_name:
                    continue

                if inp_type == "hidden":
                    hidden_fields[inp_name] = inp.attributes.get("value") or ""
                elif inp_type in ("text", "search"):
                    text_candidates.append(inp_name)
                    if not placeholder:
                        placeholder = inp.attributes.get("placeholder") or inp.attributes.get("aria-label")

                    # High-confidence query parameter names
                    if inp_type == "search" or inp_name.lower() in ("q", "query", "search", "s", "k", "search_term", "keyword", "wd"):
                        query_param = inp_name

            # Inspect selects for default / selected options (e.g. language selection)
            selects = form.css("select")
            for sel in selects:
                sel_name = sel.attributes.get("name")
                if not sel_name:
                    continue
                selected_opt = sel.css_first("option[selected]") or sel.css_first("option")
                if selected_opt:
                    val = selected_opt.attributes.get("value")
                    if val is None:
                        val = selected_opt.text(strip=True)
                    hidden_fields[sel_name] = val or ""

            # If no high-confidence name found, but text candidate exists
            if not query_param and text_candidates:
                query_param = text_candidates[0]

            if query_param:
                form_id = form.attributes.get("id") or form.attributes.get("class") or ""
                desc = placeholder or (f"Form #{form_id}" if form_id else f"Search form targeting {urlparse(action_url).path}")
                discovered.append(
                    DiscoveredSearchForm(
                        action=action_url,
                        method=method,
                        query_param=query_param,
                        hidden_fields=hidden_fields,
                        description=desc,
                    )
                )

    except Exception as exc:
        logger.warning("Error discovering search forms: %s", exc)

    return discovered


def extract_opensearch_links(html: str, current_url: str) -> List[OpenSearchLink]:
    """Extract OpenSearch XML declaration links from <head>."""
    if not html or not html.strip():
        return []

    links: List[OpenSearchLink] = []
    try:
        tree = HTMLParser(html)
        for link in tree.css('link[rel="search"], link[rel~="search"]'):
            href = link.attributes.get("href")
            if href:
                full_href = urljoin(current_url, href)
                mime = link.attributes.get("type", "application/opensearchdescription+xml")
                title = link.attributes.get("title")
                links.append(OpenSearchLink(href=full_href, title=title, type=mime))
    except Exception as exc:
        logger.warning("Error discovering OpenSearch links: %s", exc)

    return links
