import logging
import re
from typing import List, Optional
import trafilatura
from selectolax.lexbor import LexborHTMLParser as HTMLParser
import markdownify
from rank_bm25 import BM25Okapi

from src.config import settings

logger = logging.getLogger(__name__)


def extract_full_visible_text(html: str) -> str:
    """Extract all human-visible text from HTML using selectolax and markdownify.
    
    Preserves text in navigation blocks, cards, and portal grids that article extractors strip.
    """
    try:
        tree = HTMLParser(html)
        for tag in tree.css("script, style, noscript, svg"):
            tag.decompose()
        body = tree.body
        if body:
            raw_html = body.html or ""
            md_text = markdownify.markdownify(raw_html, heading_style="ATX", strip=["img"])
            cleaned = re.sub(r"\n{3,}", "\n\n", md_text).strip()
            return cleaned
    except Exception as exc:
        logger.warning("Full visible text extraction failed: %s", exc)
    return ""


def extract_clean_text(html: str) -> str:
    """Extract clean readable text from HTML.
    
    Uses trafilatura as primary article extractor.
    Falls back to full visible text if trafilatura returns nothing, or if
    trafilatura output is sparse relative to the full visible body text
    (common on portals, directories, and landing pages).
    """
    if not html or not html.strip():
        return ""

    visible_fallback = extract_full_visible_text(html)

    # Primary extractor: trafilatura
    extracted_traf = ""
    try:
        extracted = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
            output_format="txt",
        )
        if extracted:
            extracted_traf = extracted.strip()
    except Exception as exc:
        logger.debug("Trafilatura extraction failed: %s", exc)

    # If trafilatura produced a substantial article (>= 5000 chars), use it directly
    if extracted_traf and len(extracted_traf) >= 5000:
        return extracted_traf

    # If trafilatura produced sparse text (< 5000 chars) while visible text is substantially larger (>= 3x),
    # this indicates a portal/hub/landing page where trafilatura stripped grids/cards.
    if extracted_traf and len(visible_fallback) >= 3 * len(extracted_traf) and len(visible_fallback) > 1000:
        logger.info(
            "Trafilatura extraction appears sparse (%d chars vs %d visible chars). Using full visible text.",
            len(extracted_traf), len(visible_fallback)
        )
        return visible_fallback

    if extracted_traf:
        return extracted_traf

    return visible_fallback


def chunk_text(text: str, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None) -> List[str]:
    """Split text into overlapping chunks of approximately chunk_size characters.
    
    Preserves paragraph or sentence boundaries where possible.
    """
    if not text:
        return []

    c_size = chunk_size or settings.chunk_size
    c_overlap = chunk_overlap or settings.chunk_overlap

    if len(text) <= c_size:
        return [text]

    # Split into paragraphs first
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    chunks: List[str] = []
    current_chunk: List[str] = []
    current_length = 0

    for para in paragraphs:
        para_len = len(para)
        # If single paragraph is larger than chunk_size, split by sentences or hard slice
        if para_len > c_size:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_length = 0
            
            # Slice large paragraph into windowed chunks
            start = 0
            while start < len(para):
                end = start + c_size
                chunks.append(para[start:end])
                start += c_size - c_overlap
            continue

        if current_length + para_len + 2 > c_size:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                # Keep overlap if possible
                overlap_text = current_chunk[-1] if len(current_chunk[-1]) <= c_overlap else ""
                current_chunk = [overlap_text, para] if overlap_text else [para]
                current_length = sum(len(p) for p in current_chunk) + 2
            else:
                current_chunk = [para]
                current_length = para_len
        else:
            current_chunk.append(para)
            current_length += para_len + 2

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


def rank_chunks_bm25(query: str, chunks: List[str], top_k: Optional[int] = None) -> List[str]:
    """Rank text chunks using BM25 against the query and return the top_k most relevant chunks."""
    if not chunks:
        return []
    
    k = top_k or settings.bm25_top_chunks
    if len(chunks) <= k:
        return chunks

    query_tokens = [w.lower() for w in re.findall(r"\w+", query)]
    if not query_tokens:
        return chunks[:k]

    tokenized_corpus = [[w.lower() for w in re.findall(r"\w+", doc)] for doc in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query_tokens)

    # Sort chunks by score descending
    ranked_indices = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)

    if k <= 1:
        return [chunks[ranked_indices[0]]]

    # Always ensure the introductory lead section (chunk 0) is included for context when k > 1
    selected_indices = [0]
    for idx in ranked_indices:
        if idx not in selected_indices:
            selected_indices.append(idx)
        if len(selected_indices) >= k:
            break

    return [chunks[i] for i in selected_indices]


def extract_relevant_page_content(html: str, query: str, top_k: Optional[int] = None) -> str:
    """End-to-end extraction: clean HTML, chunk text, and return BM25-ranked relevant sections."""
    text = extract_clean_text(html)
    if not text:
        return ""
    
    chunks = chunk_text(text)
    if len(chunks) <= (top_k or settings.bm25_top_chunks):
        return text

    top_chunks = rank_chunks_bm25(query, chunks, top_k=top_k)
    return "\n\n---\n\n".join(top_chunks)
