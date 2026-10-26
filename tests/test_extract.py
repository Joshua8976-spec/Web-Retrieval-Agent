from pathlib import Path
import pytest
from src.tools.extract import (
    extract_clean_text,
    chunk_text,
    rank_chunks_bm25,
    extract_relevant_page_content,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def article_html() -> str:
    with open(FIXTURES_DIR / "article.html", "r", encoding="utf-8") as f:
        return f.read()


def test_extract_clean_text(article_html):
    text = extract_clean_text(article_html)
    assert "AMD ROCm Ecosystem" in text
    assert "192GB of HBM3 memory" in text
    assert "5.3 TB/s" in text
    # Should not include raw tags or comments
    assert "<h1>" not in text
    assert "<article>" not in text


def test_chunk_text():
    sample_text = (
        "Paragraph 1 contains some initial introductory text.\n\n"
        "Paragraph 2 discusses GPU acceleration and AMD ROCm kernels.\n\n"
        "Paragraph 3 is about memory bandwidth and latency optimization."
    )
    chunks = chunk_text(sample_text, chunk_size=80, chunk_overlap=15)
    assert len(chunks) >= 2
    # Check that all chunks have content
    for c in chunks:
        assert len(c.strip()) > 0


def test_rank_chunks_bm25():
    chunks = [
        "Cats and dogs are common domestic animals found in households.",
        "AMD Instinct MI300X features 192GB HBM3 memory with high bandwidth.",
        "Baking sourdough bread requires flour, water, salt, and wild yeast culture.",
    ]
    query = "What is the memory size of AMD MI300X?"
    ranked = rank_chunks_bm25(query, chunks, top_k=1)
    assert len(ranked) == 1
    assert "192GB HBM3" in ranked[0]


def test_extract_relevant_page_content(article_html):
    query = "memory bandwidth of MI300X"
    content = extract_relevant_page_content(article_html, query=query, top_k=2)
    assert "5.3 TB/s" in content
