import logging
import re
from typing import List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse, urlunparse, parse_qs, urlencode
from pydantic import BaseModel
from selectolax.lexbor import LexborHTMLParser as HTMLParser
from rank_bm25 import BM25Okapi

from src.config import settings

logger = logging.getLogger(__name__)

# Common non-navigable file extensions and link types
IGNORED_EXTENSIONS = (
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico",
    ".pdf", ".zip", ".tar", ".gz", ".rar", ".7z",
    ".mp3", ".mp4", ".wav", ".avi", ".mov",
    ".css", ".js", ".json", ".xml", ".woff", ".woff2", ".ttf",
    ".exe", ".dmg", ".pkg", ".deb", ".rpm",
)

IGNORED_SCHEMES = ("mailto:", "tel:", "javascript:", "data:", "irc:", "ftp:")


class ExtractedLink(BaseModel):
    """Structured representation of a parsed link."""

    index: int = 0
    url: str
    anchor_text: str
    is_same_domain: bool = True
    score: float = 0.0


def normalize_url(url: str, base_url: str) -> Optional[str]:
    """Resolve relative URLs, strip fragments and tracking query params, and enforce http/https."""
    if not url or not url.strip():
        return None

    cleaned = url.strip()
    if any(cleaned.lower().startswith(scheme) for scheme in IGNORED_SCHEMES):
        return None

    # Resolve relative URL against base_url
    try:
        resolved = urljoin(base_url, cleaned)
        parsed = urlparse(resolved)
    except Exception:
        return None

    if parsed.scheme not in ("http", "https"):
        return None

    # Check ignored extensions
    path_lower = parsed.path.lower()
    if any(path_lower.endswith(ext) for ext in IGNORED_EXTENSIONS):
        return None

    # Clean query parameters: strip common tracking params (utm_*, ref, gclid, etc.)
    query_params = parse_qs(parsed.query)
    clean_params = {
        k: v for k, v in query_params.items()
        if not k.lower().startswith("utm_") and k.lower() not in ("ref", "source", "fbclid", "gclid")
    }
    new_query = urlencode(clean_params, doseq=True)

    # Reconstruct URL without fragment
    normalized = urlunparse((
        parsed.scheme,
        parsed.netloc.lower(),
        parsed.path,
        parsed.params,
        new_query,
        "",  # No fragment
    ))

    # Strip trailing slash if not root
    if normalized.endswith("/") and parsed.path != "/" and parsed.path != "":
        normalized = normalized[:-1]

    return normalized


def is_same_domain(url: str, base_url: str) -> bool:
    """Check if target URL belongs to the same domain (including subdomains or exact domain)."""
    try:
        target_netloc = urlparse(url).netloc.lower().split(":")[0]
        base_netloc = urlparse(base_url).netloc.lower().split(":")[0]

        if target_netloc == base_netloc:
            return True
        # Check subdomains (e.g. docs.example.com vs example.com)
        if target_netloc.endswith("." + base_netloc):
            return True
        return False
    except Exception:
        return False


def extract_links_from_html(
    html: str,
    current_url: str,
    same_domain_only: bool = True,
) -> List[ExtractedLink]:
    """Extract all valid hyperlinks from HTML with anchor text."""
    if not html:
        return []

    tree = HTMLParser(html)
    extracted: List[ExtractedLink] = []
    seen_urls: Set[str] = set()

    for a_tag in tree.css("a[href]"):
        href = a_tag.attributes.get("href")
        if not href:
            continue

        resolved_url = normalize_url(href, current_url)
        if not resolved_url or resolved_url in seen_urls:
            continue

        same_domain = is_same_domain(resolved_url, current_url)
        if same_domain_only and not same_domain:
            continue

        # Extract anchor text or title or aria-label
        anchor_text = a_tag.text(strip=True)
        if not anchor_text:
            anchor_text = a_tag.attributes.get("title") or a_tag.attributes.get("aria-label") or ""
        
        # Clean up whitespace
        anchor_text = re.sub(r"\s+", " ", anchor_text).strip()
        if not anchor_text:
            # Fall back to URL path if anchor text is completely empty
            parsed = urlparse(resolved_url)
            anchor_text = parsed.path.strip("/").split("/")[-1] or "Home"

        seen_urls.add(resolved_url)
        extracted.append(
            ExtractedLink(
                url=resolved_url,
                anchor_text=anchor_text,
                is_same_domain=same_domain,
            )
        )

    return extracted


def rank_links_bm25(
    query: str,
    links: List[ExtractedLink],
    top_k: Optional[int] = None,
) -> List[ExtractedLink]:
    """Rank links by BM25 relevance against query based on anchor text and URL path."""
    if not links:
        return []

    k = top_k or settings.bm25_top_links
    if len(links) <= k:
        for i, link in enumerate(links, 1):
            link.index = i
        return links

    query_tokens = [w.lower() for w in re.findall(r"\w+", query)]
    if not query_tokens:
        top_slice = links[:k]
        for i, link in enumerate(top_slice, 1):
            link.index = i
        return top_slice

    # Tokenize link representation (anchor text + URL path components)
    tokenized_corpus = []
    for link in links:
        path = urlparse(link.url).path.replace("-", " ").replace("_", " ")
        doc_text = f"{link.anchor_text} {path}"
        tokenized_corpus.append([w.lower() for w in re.findall(r"\w+", doc_text)])

    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query_tokens)

    for i, score in enumerate(scores):
        links[i].score = float(score)

    # Sort descending by score
    ranked = sorted(links, key=lambda l: l.score, reverse=True)[:k]
    for i, link in enumerate(ranked, 1):
        link.index = i

    return ranked


def format_links_for_prompt(links: List[ExtractedLink]) -> str:
    """Format numbered link list for LLM context."""
    if not links:
        return "No links found on this page."

    lines = []
    for link in links:
        lines.append(f"[{link.index}] {link.url} — \"{link.anchor_text}\"")
    return "\n".join(lines)
