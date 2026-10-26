import asyncio
import logging
import time
from typing import Dict, Optional
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser
import httpx

from src.config import settings

logger = logging.getLogger(__name__)


class HostRateLimiter:
    """Polite per-host rate limiter to prevent overwhelming target websites."""

    def __init__(self, min_delay_seconds: float = 0.5):
        self.min_delay = min_delay_seconds
        self.last_request_times: Dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def wait_if_needed(self, host: str):
        """Pause execution if necessary to enforce polite interval between requests to the same host."""
        if self.min_delay <= 0 or not host:
            return

        async with self._lock:
            now = time.monotonic()
            last_time = self.last_request_times.get(host, 0.0)
            elapsed = now - last_time
            if elapsed < self.min_delay:
                wait_time = self.min_delay - elapsed
                await asyncio.sleep(wait_time)
            self.last_request_times[host] = time.monotonic()


class RobotsCache:
    """Asynchronous, cached robots.txt compliance checker."""

    def __init__(self, user_agent: Optional[str] = None):
        self.user_agent = user_agent or settings.user_agent
        self.cache: Dict[str, Optional[RobotFileParser]] = {}
        self._lock = asyncio.Lock()

    async def can_fetch(self, url: str) -> bool:
        """Check whether fetching the given URL is permitted by the site's robots.txt."""
        try:
            parsed = urlparse(url)
            host = parsed.netloc.lower()
            if not host:
                return True

            async with self._lock:
                if host in self.cache:
                    rfp = self.cache[host]
                    return rfp.can_fetch(self.user_agent, url) if rfp else True

            # Fetch robots.txt
            robots_url = f"{parsed.scheme}://{host}/robots.txt"
            rfp = RobotFileParser()
            try:
                async with httpx.AsyncClient(timeout=5.0, follow_redirects=True) as client:
                    resp = await client.get(robots_url, headers={"User-Agent": self.user_agent})
                    if resp.status_code == 200:
                        rfp.parse(resp.text.splitlines())
                    else:
                        # 404 or non-200 implies no restrictions
                        rfp = None
            except Exception as exc:
                logger.debug("Could not fetch robots.txt for %s: %s", host, exc)
                rfp = None

            async with self._lock:
                self.cache[host] = rfp

            return rfp.can_fetch(self.user_agent, url) if rfp else True

        except Exception as exc:
            logger.warning("Error evaluating robots.txt for %s: %s", url, exc)
            return True


# Default singletons
default_rate_limiter = HostRateLimiter(min_delay_seconds=settings.rate_limit_delay_seconds)
default_robots_cache = RobotsCache()
