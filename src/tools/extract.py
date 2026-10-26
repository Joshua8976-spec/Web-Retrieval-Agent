import logging
import re
from typing import List, Optional
import trafilatura
from selectolax.lexbor import LexborHTMLParser as HTMLParser
import markdownify
from rank_bm25 import BM25Okapi

from src.config import settings

logger = logging.getLogger(__name__)


def extract_clean_text(html: str) -> str:
    """Extract clean readable text from HTML.
    
    Uses trafilatura as primary extractor (best at stripping navbars, footers, ads).
    Falls back to selectolax + markdownify if trafilatura returns insufficient text.
    """
    if not html or not html.strip():
        return ""

    # Primary extractor: trafilatura
    try:
        extracted = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
            output_format="txt",
        )
        if extracted and len(extracted.strip()) > 50:
            return extracted.strip()
    except Exception as exc:
        logger.debug("Trafilatura extraction failed, falling back: %s", exc)

    # Secondary fallback extractor: selectolax + markdownify
    try:
        tree = HTMLParser(html)
        # Strip script, style, noscript, svg, nav, footer tags
        for tag in tree.css("script, style, noscript, svg, nav, footer, header"):
            tag.decompose()
        
        body = tree.body
        if body:
            raw_html = body.html or ""
            md_text = markdownify.markdownify(raw_html, heading_style="ATX", strip=["img"])
            # Clean up excessive newlines
            cleaned = re.sub(r"\n{3,}", "\n\n", md_text).strip()
            if cleaned:
                return cleaned
    except Exception as exc:
        logger.warning("Fallback HTML parser failed: %s", exc)

    return ""


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
    return [chunks[i] for i in ranked_indices[:k]]


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
