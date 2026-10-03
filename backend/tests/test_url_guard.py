"""
Unit tests for the SSRF guard and the redirect-following URL loader.

No test in this module performs real network I/O. DNS is intercepted by
monkeypatching ``socket.getaddrinfo`` and HTTP by monkeypatching
``httpx.Client.stream``.
"""

import socket

import httpx
import pytest

from src.ingestion import url_loader
from src.ingestion.url_guard import validate_public_url


def fake_getaddrinfo(mapping: dict[str, list[str]]):
    """Build a getaddrinfo stub resolving only the hostnames in ``mapping``."""

    def _getaddrinfo(host, port, *args, **kwargs):
        if host not in mapping:
            raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")
        out = []
        for ip in mapping[host]:
            sockaddr = (ip, port or 0, 0, 0) if ":" not in ip else (ip, port or 0, 0, 0)
            out.append((socket.AF_INET6 if ":" in ip else socket.AF_INET,
                        socket.SOCK_STREAM, socket.IPPROTO_TCP, "", sockaddr))
        return out

    return _getaddrinfo


@pytest.fixture
def public_dns(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo({
        "example.com": ["93.184.216.34"],
        "public.example": ["93.184.216.34"],
        "internal.example": ["10.0.0.5"],
        "dual.example": ["93.184.216.34", "fd00::1"],
        "mapped.example": ["::ffff:127.0.0.1"],
        "metadata.example": ["169.254.169.254"],
        "v6loop.example": ["::1"],
    }))


BLOCKED_URLS = [
    "http://127.0.0.1/",
    "http://localhost/",
    "http://169.254.169.254/latest/meta-data/",
    "http://[::1]/",
    "http://[::ffff:127.0.0.1]/",
    "file:///etc/passwd",
    "ftp://example.com/",
    "http://internal.example/",
    "http://metadata.example/",
    "http://v6loop.example/",
    "http://mapped.example/",
    "http://dual.example/",
]


@pytest.mark.parametrize("url", BLOCKED_URLS)
def test_validate_public_url_blocks(url, public_dns):
    with pytest.raises(ValueError):
        validate_public_url(url)


def test_validate_public_url_blocks_unresolvable_host(public_dns):
    with pytest.raises(ValueError, match="Could not resolve"):
        validate_public_url("http://nowhere.invalid/")


def test_validate_public_url_blocks_missing_hostname(public_dns):
    with pytest.raises(ValueError, match="no hostname"):
        validate_public_url("http:///just-a-path")


@pytest.mark.parametrize("url", [
    "http://example.com/",
    "https://example.com/deep/path?q=1",
    "http://public.example:8080/x",
])
def test_validate_public_url_allows_public(url, public_dns):
    assert validate_public_url(url) == url


def test_fetch_url_follows_redirect_to_public_target(public_dns, monkeypatch):
    requested: list[str] = []

    class _Resp:
        def __init__(self, status, headers=b"", body=b""):
            self.status_code = status
            self.headers = headers
            self._body = body
            self.encoding = "utf-8"

        def raise_for_status(self):
            pass

        def iter_bytes(self):
            yield self._body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    responses = {
        "http://example.com/": _Resp(302, {"location": "http://public.example/final"}),
        "http://public.example/final": _Resp(200, body=b"<html><body>hi</body></html>"),
    }

    class _Client:
        def __init__(self, **kwargs):
            assert kwargs["follow_redirects"] is False
            assert kwargs["timeout"] == 15.0

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def stream(self, method, url):
            requested.append(url)
            return responses[url]

    monkeypatch.setattr(url_loader.httpx, "Client", _Client)
    monkeypatch.setattr(url_loader.trafilatura, "extract", lambda *a, **k: "text")
    monkeypatch.setattr(url_loader.trafilatura, "extract_metadata", lambda *a, **k: None)

    text, _ = url_loader.fetch_url("http://example.com/")
    assert text == "text"
    assert requested == ["http://example.com/", "http://public.example/final"]


def test_fetch_url_blocks_redirect_to_loopback(public_dns, monkeypatch):
    class _Resp:
        def __init__(self, status, headers=None, body=b""):
            self.status_code = status
            self.headers = headers or {}
            self.encoding = "utf-8"

        def raise_for_status(self):
            pass

        def iter_bytes(self):
            yield self._body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    responses = {
        "http://example.com/": _Resp(302, {"location": "http://127.0.0.1/admin"}),
    }

    class _Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def stream(self, method, url):
            return responses[url]

    monkeypatch.setattr(url_loader.httpx, "Client", _Client)
    monkeypatch.setattr(
        url_loader.trafilatura, "extract",
        lambda *a, **k: pytest.fail("extraction must not run for a blocked redirect"),
    )

    with pytest.raises(ValueError):
        url_loader.fetch_url("http://example.com/")


def test_download_html_rejects_oversized_body(public_dns, monkeypatch):
    oversized = b"x" * (url_loader.MAX_RESPONSE_BYTES + 1)

    class _Resp:
        status_code = 200
        headers: dict = {}
        encoding = "utf-8"

        def raise_for_status(self):
            pass

        def iter_bytes(self):
            yield oversized

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class _Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def stream(self, method, url):
            return _Resp()

    monkeypatch.setattr(url_loader.httpx, "Client", _Client)
    assert url_loader._download_html("http://example.com/") is None