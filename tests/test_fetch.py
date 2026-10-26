import pytest
import respx
import httpx
from src.tools.fetch import PageFetcher, FetchResult


@pytest.mark.asyncio
async def test_fetcher_success():
    fetcher = PageFetcher()
    test_url = "https://example.com/page"
    html_content = "<html><body><h1>Hello World</h1></body></html>"

    with respx.mock:
        respx.get(test_url).mock(
            return_value=httpx.Response(200, text=html_content, headers={"content-type": "text/html; charset=utf-8"})
        )
        result = await fetcher.fetch(test_url)
        assert result.is_success is True
        assert result.status_code == 200
        assert "Hello World" in result.html
        assert result.final_url == test_url


@pytest.mark.asyncio
async def test_fetcher_ignores_binary_content():
    fetcher = PageFetcher()
    test_url = "https://example.com/archive.zip"

    with respx.mock:
        respx.get(test_url).mock(
            return_value=httpx.Response(200, content=b"PK\x03\x04...", headers={"content-type": "application/zip"})
        )
        result = await fetcher.fetch(test_url)
        assert result.is_success is False
        assert result.html == ""
        assert "Unsupported content type" in (result.error or "")


@pytest.mark.asyncio
async def test_fetcher_caps_size_limit():
    fetcher = PageFetcher(max_size_bytes=100)
    test_url = "https://example.com/huge"
    oversized_content = "A" * 500

    with respx.mock:
        respx.get(test_url).mock(
            return_value=httpx.Response(200, text=oversized_content, headers={"content-type": "text/html"})
        )
        result = await fetcher.fetch(test_url)
        assert result.is_success is False
        assert "exceeded limit" in (result.error or "")


@pytest.mark.asyncio
async def test_fetcher_handles_redirect():
    fetcher = PageFetcher()
    start_url = "https://example.com/old"
    target_url = "https://example.com/new"

    with respx.mock:
        respx.get(start_url).mock(
            return_value=httpx.Response(301, headers={"location": target_url})
        )
        respx.get(target_url).mock(
            return_value=httpx.Response(200, text="<html><body>Final Page</body></html>", headers={"content-type": "text/html"})
        )
        result = await fetcher.fetch(start_url)
        assert result.is_success is True
        assert result.final_url == target_url
        assert "Final Page" in result.html
