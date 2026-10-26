import logging
from typing import Optional

from src.config import settings

logger = logging.getLogger(__name__)


class BrowserFallback:
    """Optional Playwright browser fallback for dynamic JS-rendered single-page apps."""

    def __init__(self):
        self.enabled = settings.enable_playwright_fallback
        self._playwright = None
        self._browser = None

    async def fetch_rendered_html(self, url: str, timeout_seconds: float = 15.0) -> Optional[str]:
        """Fetch rendered HTML using headless Playwright if enabled and installed."""
        if not self.enabled:
            logger.debug("Playwright fallback is disabled by configuration.")
            return None

        try:
            from playwright.async_api import async_playwright
        except ImportError:
            logger.warning(
                "Playwright is enabled in config but playwright package is not installed. "
                "Run: pip install playwright && playwright install chromium"
            )
            return None

        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page(user_agent=settings.user_agent)
                await page.goto(url, wait_until="networkidle", timeout=int(timeout_seconds * 1000))
                content = await page.content()
                await browser.close()
                return content
        except Exception as exc:
            logger.warning("Playwright rendering failed for %s: %s", url, exc)
            return None
