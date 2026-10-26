import pytest
import respx
import httpx
from src.safety.url_guard import is_safe_url
from src.safety.robots import HostRateLimiter, RobotsCache


def test_url_guard_blocks_schemes():
    safe, reason = is_safe_url("file:///etc/passwd")
    assert not safe
    assert "Prohibited URL scheme" in reason

    safe, reason = is_safe_url("ftp://example.com/file")
    assert not safe
    assert "Prohibited URL scheme" in reason

    safe, reason = is_safe_url("javascript:alert(1)")
    assert not safe


def test_url_guard_blocks_loopback_and_private_ips():
    # Loopback IP
    safe, reason = is_safe_url("http://127.0.0.1:8080/admin")
    assert not safe
    assert "private, loopback" in reason or "Target IP address" in reason

    # Localhost
    safe, reason = is_safe_url("http://localhost:3000/api")
    assert not safe
    assert "prohibited" in reason

    # AWS/GCP/Azure link-local cloud metadata
    safe, reason = is_safe_url("http://169.254.169.254/latest/meta-data/")
    assert not safe
    assert "prohibited" in reason or "private" in reason

    # RFC1918 Private IP ranges
    safe, reason = is_safe_url("http://10.0.0.5/internal")
    assert not safe
    assert "private, loopback" in reason

    safe, reason = is_safe_url("http://192.168.1.1/router")
    assert not safe
    assert "private, loopback" in reason

    safe, reason = is_safe_url("http://172.16.0.25/metrics")
    assert not safe
    assert "private, loopback" in reason


def test_url_guard_permits_valid_public_urls():
    safe, reason = is_safe_url("https://example.com/docs")
    assert safe
    assert reason == ""

    safe, reason = is_safe_url("https://python.org/downloads")
    assert safe
    assert reason == ""


def test_url_guard_empty_or_malformed():
    safe, reason = is_safe_url("")
    assert not safe
    assert "empty" in reason

    safe, reason = is_safe_url("http://")
    assert not safe
    assert "lacks a valid hostname" in reason


@pytest.mark.asyncio
async def test_host_rate_limiter():
    limiter = HostRateLimiter(min_delay_seconds=0.05)
    
    # First call should be immediate
    await limiter.wait_if_needed("example.com")
    assert "example.com" in limiter.last_request_times

    # Second call should throttle slightly
    import time
    t0 = time.monotonic()
    await limiter.wait_if_needed("example.com")
    elapsed = time.monotonic() - t0
    assert elapsed >= 0.04


@pytest.mark.asyncio
@respx.mock
async def test_robots_cache_disallow_and_allow():
    robots_content = """User-agent: *
Disallow: /private/
Disallow: /admin/
Allow: /
"""
    respx.get("https://safe-domain.org/robots.txt").respond(
        status_code=200,
        text=robots_content,
    )

    cache = RobotsCache(user_agent="TestBot")

    # Disallowed path
    can_fetch = await cache.can_fetch("https://safe-domain.org/private/secret.html")
    assert can_fetch is False

    # Allowed path
    can_fetch = await cache.can_fetch("https://safe-domain.org/public/docs.html")
    assert can_fetch is True

    # Caching check: Second call to same domain does not refetch
    assert len(respx.calls) == 1
    can_fetch_cached = await cache.can_fetch("https://safe-domain.org/other.html")
    assert can_fetch_cached is True
    assert len(respx.calls) == 1


@pytest.mark.asyncio
@respx.mock
async def test_robots_cache_404_allows_all():
    respx.get("https://unrestricted-site.com/robots.txt").respond(status_code=404)

    cache = RobotsCache(user_agent="TestBot")
    can_fetch = await cache.can_fetch("https://unrestricted-site.com/anything")
    assert can_fetch is True
