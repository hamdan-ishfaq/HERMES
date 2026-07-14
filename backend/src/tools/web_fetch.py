"""Optional bounded web_fetch — disabled unless HERMES_WEB_ALLOWLIST is set."""

from __future__ import annotations

import os
from urllib.parse import urlparse

import httpx


def web_fetch(url: str, timeout: float = 10.0) -> str:
    """
    Fetch URL text if host is on HERMES_WEB_ALLOWLIST (comma-separated suffixes).

    Returns empty string when disabled or blocked.
    """
    allow = os.getenv("HERMES_WEB_ALLOWLIST", "").strip()
    if not allow:
        return ""
    host = urlparse(url).hostname or ""
    suffixes = [s.strip().lower() for s in allow.split(",") if s.strip()]
    if not any(host.endswith(s) for s in suffixes):
        return ""
    resp = httpx.get(url, timeout=timeout, follow_redirects=True)
    resp.raise_for_status()
    return resp.text[:50_000]
