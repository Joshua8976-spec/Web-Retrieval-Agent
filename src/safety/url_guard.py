import ipaddress
import logging
import socket
from typing import Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

BLOCKED_HOSTNAMES = {
    "localhost",
    "metadata.google.internal",
    "instance-data",
    "169.254.169.254",  # AWS/GCP/Azure link-local metadata service
}


def is_safe_url(url: str) -> Tuple[bool, str]:
    """Validate URL against SSRF, private IPs, loopback, and disallowed schemes.
    
    Protects against internal network scanning and cloud metadata theft.
    """
    if not url or not url.strip():
        return False, "URL is empty."

    try:
        parsed = urlparse(url.strip())
    except Exception as exc:
        return False, f"Malformed URL: {exc}"

    # 1. Scheme check: only http and https allowed
    if parsed.scheme.lower() not in ("http", "https"):
        return False, f"Prohibited URL scheme '{parsed.scheme}'. Only http and https are permitted."

    hostname = parsed.hostname
    if not hostname:
        return False, "URL lacks a valid hostname."

    hostname_clean = hostname.lower()

    # 2. Blocked specific hostnames
    if hostname_clean in BLOCKED_HOSTNAMES or hostname_clean.endswith(".localhost"):
        return False, f"Target host '{hostname}' is prohibited (internal or metadata service)."

    # 3. Check if hostname is directly an IP literal
    try:
        ip_obj = ipaddress.ip_address(hostname_clean)
        if (
            ip_obj.is_loopback
            or ip_obj.is_private
            or ip_obj.is_link_local
            or ip_obj.is_multicast
            or ip_obj.is_reserved
        ):
            return False, f"Target IP address '{ip_obj}' is in a private, loopback, or reserved range."
    except ValueError:
        # Hostname is a domain name, proceed to DNS resolution check
        pass

    # 4. Resolve domain name to verify all associated IP addresses
    try:
        addr_info = socket.getaddrinfo(hostname_clean, None)
        for family, socktype, proto, canonname, sockaddr in addr_info:
            ip_str = sockaddr[0]
            try:
                ip_obj = ipaddress.ip_address(ip_str)
                if (
                    ip_obj.is_loopback
                    or ip_obj.is_private
                    or ip_obj.is_link_local
                    or ip_obj.is_multicast
                    or ip_obj.is_reserved
                ):
                    return False, f"Domain '{hostname}' resolves to private/loopback IP: {ip_obj}."
            except ValueError:
                continue
    except socket.gaierror as exc:
        logger.debug("DNS resolution failed for %s: %s", hostname_clean, exc)
        # We let the HTTP client handle unreachable host names during fetch
        pass
    except Exception as exc:
        logger.warning("Error inspecting DNS for %s: %s", hostname_clean, exc)

    return True, ""
