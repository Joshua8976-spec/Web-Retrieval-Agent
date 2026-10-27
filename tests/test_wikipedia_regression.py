from pathlib import Path
import pytest

from src.tools.links import (
    extract_links_from_html,
    is_same_domain,
    normalize_url,
)
from src.tools.extract import extract_clean_text
from src.tools.search_forms import extract_search_forms

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def wikipedia_portal_html() -> str:
    path = FIXTURES_DIR / "wikipedia_portal.html"
    return path.read_text(encoding="utf-8")


def test_wikipedia_cross_subdomain_registered_domain():
    portal_url = "https://www.wikipedia.org/"
    en_wiki_url = "https://en.wikipedia.org/wiki/Main_Page"
    ja_wiki_url = "https://ja.wikipedia.org/"
    external_url = "https://www.google.com/"

    assert is_same_domain(en_wiki_url, portal_url) is True
    assert is_same_domain(ja_wiki_url, portal_url) is True
    assert is_same_domain(portal_url, en_wiki_url) is True
    assert is_same_domain(external_url, portal_url) is False


def test_wikipedia_portal_link_extraction(wikipedia_portal_html):
    current_url = "https://www.wikipedia.org/"
    links = extract_links_from_html(wikipedia_portal_html, current_url, same_domain_only=True)

    urls = [link.url for link in links]
    # Verify subdomains are included rather than filtered out
    assert any("en.wikipedia.org" in u for u in urls)
    assert any("ja.wikipedia.org" in u for u in urls)
    # Total valid Wikipedia links should be well above 1 (previously was only 1)
    assert len(links) > 20


def test_trailing_slash_preservation():
    base = "https://en.wikipedia.org/wiki/"
    # Trailing slash on path must not be stripped
    normalized = normalize_url("https://en.wikipedia.org/wiki/", base)
    assert normalized == "https://en.wikipedia.org/wiki/"

    normalized_sub = normalize_url("/docs/guide/", "https://example.com")
    assert normalized_sub == "https://example.com/docs/guide/"


def test_wikipedia_portal_text_fallback(wikipedia_portal_html):
    text = extract_clean_text(wikipedia_portal_html)

    # Must preserve language statistics and portal tiles that trafilatura would discard as sparse
    assert "7,254,000+" in text or "articles" in text
    assert "English" in text
    assert len(text) > 5000


def test_wikipedia_portal_search_form(wikipedia_portal_html):
    forms = extract_search_forms(wikipedia_portal_html, "https://www.wikipedia.org/")

    assert len(forms) >= 1
    form = forms[0]
    assert "search-redirect.php" in form.action
    assert form.query_param == "search"
    assert form.hidden_fields.get("family") == "wikipedia"

    assert form.hidden_fields.get("language") == "en"

    query_url = form.build_query_url("Mahatma Gandhi")
    assert "search=Mahatma+Gandhi" in query_url or "search=Mahatma%20Gandhi" in query_url
    assert "family=wikipedia" in query_url
    assert "language=en" in query_url


def test_html_meta_refresh_redirect_extraction():
    from src.tools.fetch import extract_html_redirect

    html_meta = """
    <html>
      <head>
        <meta http-equiv="refresh" content="0; url=https://en.wikipedia.org/wiki/Mahatma_Gandhi">
      </head>
      <body>Redirecting...</body>
    </html>
    """
    dest = extract_html_redirect(html_meta, "https://www.wikipedia.org/")
    assert dest == "https://en.wikipedia.org/wiki/Mahatma_Gandhi"

    # Relative path in meta refresh
    html_rel = '<meta http-equiv="refresh" content="1;url=/wiki/Special:Search?search=test">'
    dest_rel = extract_html_redirect(html_rel, "https://en.wikipedia.org/portal")
    assert dest_rel == "https://en.wikipedia.org/wiki/Special:Search?search=test"


def test_format_evidence_quote_strips_citations_and_trims_sentence():
    from src.agent.verifier import format_evidence_quote

    raw_quote = (
        "Mohandas Karamchand Gandhi[c] (2 October 1869 \u2013 30 January 1948) was an Indian lawyer,[2] "
        "anti-colonial nationalist,[3] and political ethicist[4] who employed nonviolent resistance. Additional trailing unneeded fragment"
    )

    cleaned = format_evidence_quote(raw_quote)
    # Citations must be stripped
    assert "[c]" not in cleaned
    assert "[2]" not in cleaned
    assert "[3]" not in cleaned
    assert "[4]" not in cleaned
    # Must end at sentence boundary
    assert cleaned.endswith("who employed nonviolent resistance.")
