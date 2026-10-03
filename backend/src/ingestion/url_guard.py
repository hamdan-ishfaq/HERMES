"""
SSRF guard for user-supplied URLs.

Any URL that reaches the ingestion path is attacker-controlled, and the
process sits inside a private network. Without a check, a user can point
ingestion at loopback, the link-local metadata endpoint (169.254.169.254),
or RFC1918 hosts and read the response back through the extracted text.

``validate_public_url`` is deliberately strict and fails closed: a hostname
that cannot be resolved, resolves to nothing, or resolves to any address that
is not globally routable is rejected. Every address returned by
``getaddrinfo`` is checked, not just the first, so a hostname advertising a
public A record alongside a private AAAA record is still blocked.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

ALLOWED_SCHEMES = ("http", "https")


def _addresses_are_blocked(addr: ipaddress._BaseAddress) -> bool:
    """
    Return True when ``addr`` is anything other than a globally routable address.

    IPv4-mapped IPv6 addresses (``::ffff:127.0.0.1``) are unwrapped first and
    judged as IPv4. Python's ``is_private`` already covers much of this, but
    the flags are checked individually so a new address range is not silently
    allowed by a gap in one predicate.
    """
    candidates = [addr]
    mapped = getattr(addr, "ipv4_mapped", None)
    if mapped is not None:
        candidates = [mapped]

    for candidate in candidates:
        if (
            candidate.is_private
            or candidate.is_loopback
            or candidate.is_link_local
            or candidate.is_multicast
            or candidate.is_reserved
            or candidate.is_unspecified
        ):
            return True
    return False


def validate_public_url(url: str) -> str:
    """
    Validate that ``url`` is an http/https URL resolving only to public addresses.

    Returns the URL unchanged when it is safe to fetch.

    Raises:
        ValueError: if the scheme is not http/https, there is no hostname, the
            hostname does not resolve, or any resolved address is non-public.
    """
    try:
        parsed = urlparse(url)
    except ValueError as exc:
        raise ValueError(f"Could not parse URL: {exc}") from exc

    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        raise ValueError(
            f"Unsupported URL scheme {parsed.scheme!r}; only http and https are allowed."
        )

    try:
        hostname = parsed.hostname
    except ValueError as exc:
        raise ValueError(f"Could not parse URL host: {exc}") from exc

    if not hostname:
        raise ValueError("URL has no hostname.")

    try:
        infos = socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
    except (socket.gaierror, UnicodeError, OSError) as exc:
        raise ValueError(f"Could not resolve hostname {hostname!r}.") from exc

    if not infos:
        raise ValueError(f"Could not resolve hostname {hostname!r}.")

    for info in infos:
        sockaddr = info[4]
        try:
            addr = ipaddress.ip_address(sockaddr[0])
        except ValueError as exc:
            raise ValueError(f"Resolved to an unusable address: {sockaddr[0]!r}") from exc
        if _addresses_are_blocked(addr):
            raise ValueError(
                f"Hostname {hostname!r} resolves to non-public address {addr}, "
                "which is not allowed."
            )

    return url