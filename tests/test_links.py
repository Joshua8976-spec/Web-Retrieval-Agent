from pathlib import Path
import pytest
from src.tools.links import (
    normalize_url,
    is_same_domain,
    extract_links_from_html,
    rank_links_bm25,
    format_links_for_prompt,
    ExtractedLink,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def links_html() -> str:
    with open(FIXTURES_DIR / "links_page.html", "r", encoding="utf-8") as f:
        return f.read()


def test_normalize_url():
    base = "https://amd.com/en/docs"

    # Relative path resolution
    assert normalize_url("/guide", base) == "https://amd.com/guide"

    # Fragment stripping
    assert normalize_url("/arch#summary", base) == "https://amd.com/arch"

    # Tracking param stripping
    clean_url = normalize_url("/benchmarks?utm_source=twitter&utm_medium=cpc&run=1", base)
    assert clean_url == "https://amd.com/benchmarks?run=1"

    # Rejection of mailto, javascript, pdf, png
    assert normalize_url("mailto:test@example.com", base) is None
    assert normalize_url("javascript:void(0)", base) is None
    assert normalize_url("/files/doc.pdf", base) is None
    assert normalize_url("/images/logo.png", base) is None


def test_is_same_domain():
    base = "https://amd.com/en/docs"
    assert is_same_domain("https://amd.com/products", base) is True
    assert is_same_domain("https://sub.amd.com/portal", base) is True
    assert is_same_domain("https://google.com/search", base) is False
    assert is_same_domain("https://notamd.com", base) is False


def test_extract_links_from_html(links_html):
    current_url = "https://amd.com/resources"
    links = extract_links_from_html(links_html, current_url, same_domain_only=True)

    urls = [link.url for link in links]
    # Verify expected same-domain links are extracted
    assert "https://amd.com/docs/getting-started" in urls
    assert "https://amd.com/research/benchmarks" in urls
    assert "https://amd.com/research/vllm-rocm" in urls

    # Verify external, binary, and javascript links are filtered out
    assert not any("otherdomain.com" in u for u in urls)
    assert not any("manual.pdf" in u for u in urls)
    assert not any("diagram.png" in u for u in urls)
    assert not any("javascript" in u for u in urls)

    # Verify deduplication (Duplicate Link was present in fixture)
    assert urls.count("https://amd.com/research/benchmarks") == 1


def test_rank_links_bm25():
    links = [
        ExtractedLink(url="https://amd.com/docs/install", anchor_text="Installation Guide"),
        ExtractedLink(url="https://amd.com/products/mi300", anchor_text="MI300 GPU Hardware Specs"),
        ExtractedLink(url="https://amd.com/company/careers", anchor_text="Jobs and Careers"),
    ]
    query = "Where can I find hardware specs for the MI300 GPU?"
    ranked = rank_links_bm25(query, links, top_k=2)

    assert len(ranked) == 2
    assert ranked[0].anchor_text == "MI300 GPU Hardware Specs"
    assert ranked[0].index == 1
    assert ranked[1].index == 2


def test_format_links_for_prompt():
    links = [
        ExtractedLink(index=1, url="https://amd.com/spec", anchor_text="Specifications"),
        ExtractedLink(index=2, url="https://amd.com/faq", anchor_text="FAQ"),
    ]
    formatted = format_links_for_prompt(links)
    assert '[1] https://amd.com/spec — "Specifications"' in formatted
    assert '[2] https://amd.com/faq — "FAQ"' in formatted
