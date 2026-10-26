import logging
import time
from typing import Optional
from urllib.parse import urlparse
import httpx
from pydantic import BaseModel, Field

from src.config import settings
from src.safety.url_guard import is_safe_url
from src.safety.robots import default_robots_cache, default_rate_limiter

logger = logging.getLogger(__name__)

# Allowed MIME types for text extraction
ALLOWED_CONTENT_TYPES = (
    "text/html",
    "text/plain",
    "application/xhtml+xml",
    "text/xml",
    "application/xml",
)


class FetchResult(BaseModel):
    """Result of an HTTP fetch operation."""

    url: str
    final_url: str
    status_code: int
    html: str
    content_type: str = ""
    elapsed_seconds: float = 0.0
    error: Optional[str] = None
    is_success: bool = True


class PageFetcher:
    """Async HTTP page fetcher with SSRF protection, robots.txt compliance, and size capping."""

    def __init__(
        self,
        user_agent: Optional[str] = None,
        timeout: Optional[float] = None,
        max_size_bytes: Optional[int] = None,
        check_safety: bool = True,
        check_robots: bool = True,
    ):
        self.user_agent = user_agent or settings.user_agent
        self.timeout = timeout or 15.0
        self.max_size_bytes = max_size_bytes or settings.max_page_size_bytes
        self.check_safety = check_safety
        self.check_robots = check_robots
        self.headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }

    async def fetch(self, url: str) -> FetchResult:
        """Fetch URL asynchronously, enforcing SSRF safety, robots.txt, and content validation."""
        start_time = time.perf_counter()

        # 1. SSRF URL Guard check
        if self.check_safety:
            safe, reason = is_safe_url(url)
            if not safe:
                logger.warning("SSRF block on target URL %s: %s", url, reason)
                return FetchResult(
                    url=url,
                    final_url=url,
                    status_code=0,
                    html="",
                    elapsed_seconds=time.perf_counter() - start_time,
                    error=f"Security block (SSRF protection): {reason}",
                    is_success=False,
                )

        # 2. Robots.txt check
        if self.check_robots:
            allowed = await default_robots_cache.can_fetch(url)
            if not allowed:
                logger.info("URL disallowed by robots.txt: %s", url)
                return FetchResult(
                    url=url,
                    final_url=url,
                    status_code=403,
                    html="",
                    elapsed_seconds=time.perf_counter() - start_time,
                    error="Fetching disallowed by site robots.txt policy.",
                    is_success=False,
                )

        # 3. Polite host rate limiting
        host = urlparse(url).netloc.lower()
        await default_rate_limiter.wait_if_needed(host)

        try:
            async with httpx.AsyncClient(
                headers=self.headers,
                timeout=self.timeout,
                follow_redirects=True,
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            ) as client:
                response = await client.get(url)
                elapsed = time.perf_counter() - start_time
                final_url = str(response.url)
                content_type = response.headers.get("content-type", "").lower()

                # Re-verify final_url after redirects to guard against open redirect SSRF
                if self.check_safety and final_url != url:
                    final_safe, final_reason = is_safe_url(final_url)
                    if not final_safe:
                        logger.warning("SSRF block on redirected URL %s: %s", final_url, final_reason)
                        return FetchResult(
                            url=url,
                            final_url=final_url,
                            status_code=0,
                            html="",
                            elapsed_seconds=elapsed,
                            error=f"Security block on redirect target: {final_reason}",
                            is_success=False,
                        )

                # Validate content type: ignore binary downloads (pdf, zip, exe, images, etc.)
                if not any(allowed in content_type for allowed in ALLOWED_CONTENT_TYPES):
                    return FetchResult(
                        url=url,
                        final_url=final_url,
                        status_code=response.status_code,
                        html="",
                        content_type=content_type,
                        elapsed_seconds=elapsed,
                        error=f"Unsupported content type: {content_type}. Ignored binary/non-HTML resource.",
                        is_success=False,
                    )

                # Validate content size
                body_bytes = response.content
                if len(body_bytes) > self.max_size_bytes:
                    return FetchResult(
                        url=url,
                        final_url=final_url,
                        status_code=response.status_code,
                        html="",
                        content_type=content_type,
                        elapsed_seconds=elapsed,
                        error=f"Page size ({len(body_bytes)} bytes) exceeded limit ({self.max_size_bytes} bytes).",
                        is_success=False,
                    )

                # Decode text
                text = response.text
                return FetchResult(
                    url=url,
                    final_url=final_url,
                    status_code=response.status_code,
                    html=text,
                    content_type=content_type,
                    elapsed_seconds=elapsed,
                    is_success=response.is_success,
                    error=None if response.is_success else f"HTTP status error: {response.status_code}",
                )

        except httpx.TimeoutException as exc:
            elapsed = time.perf_counter() - start_time
            logger.warning("Fetch timeout for %s: %s", url, exc)
            return FetchResult(
                url=url,
                final_url=url,
                status_code=0,
                html="",
                elapsed_seconds=elapsed,
                error=f"Fetch timeout after {self.timeout}s: {exc}",
                is_success=False,
            )
        except httpx.RequestError as exc:
            elapsed = time.perf_counter() - start_time
            logger.warning("Fetch request error for %s: %s", url, exc)
            return FetchResult(
                url=url,
                final_url=url,
                status_code=0,
                html="",
                elapsed_seconds=elapsed,
                error=f"Request error: {exc}",
                is_success=False,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            logger.error("Unexpected fetch error for %s: %s", url, exc)
            return FetchResult(
                url=url,
                final_url=url,
                status_code=0,
                html="",
                elapsed_seconds=elapsed,
                error=f"Unexpected error: {exc}",
                is_success=False,
            )
