"""
Unit tests for the ingestion API (`/api/ingest/*`).

Covers:
    - Successful URL ingestion (mocked scraper + retriever)
    - Auth requirement on ingest endpoints
    - SSRF rejection returning 400 before any job row is created
    - PDF MIME-type validation (reject non-PDF uploads)

Fixtures used (from conftest.py):
    - `client` — async ASGI test client.
    - `auth_token` — valid JWT for authenticated requests.
    - `mock_retriever` — prevents real Qdrant writes.
    - `mock_ingest_url` — returns a fixed `total_children` count.
"""

import ipaddress
import socket

import pytest

from src.db import AsyncSessionLocal, IngestionJob
from sqlalchemy import func, select


def _hermetic_getaddrinfo(host, port, *args, **kwargs):
    """
    Resolve without touching the resolver.

    Literal IPs resolve to themselves so the guard still sees the real target;
    any hostname resolves to a public address.
    """
    try:
        ipaddress.ip_address(host)
        resolved = [host]
    except ValueError:
        resolved = ["93.184.216.34"]
    out = []
    for ip in resolved:
        family = socket.AF_INET6 if ":" in ip else socket.AF_INET
        out.append((family, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip, port or 0, 0, 0)))
    return out


@pytest.fixture
def public_dns(monkeypatch):
    """Resolve every hostname to a public IP so no test touches real DNS."""
    monkeypatch.setattr(socket, "getaddrinfo", _hermetic_getaddrinfo)


@pytest.fixture
def no_dns(monkeypatch):
    """Resolve literals locally but never query a resolver for hostnames."""
    def _getaddrinfo(host, port, *args, **kwargs):
        try:
            ipaddress.ip_address(host)
        except ValueError:
            raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")
        return _hermetic_getaddrinfo(host, port, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", _getaddrinfo)


@pytest.mark.asyncio
async def test_ingest_url_success(client, auth_token, mock_retriever, mock_ingest_url,
                                  public_dns):
    """Authenticated URL ingest should return 200 with status ok and chunk count."""
    resp = await client.post(
        "/api/ingest/url",
        json={"url": "https://example.com/doc"},
        headers={"Authorization": f"Bearer {auth_token}"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["chunks_stored"] == 42


@pytest.mark.asyncio
async def test_ingest_url_requires_auth(client):
    """URL ingest without Authorization header must return 401."""
    resp = await client.post("/api/ingest/url", json={"url": "https://example.com"})
    assert resp.status_code == 401


@pytest.mark.asyncio
@pytest.mark.parametrize("url", [
    "http://127.0.0.1/",
    "http://169.254.169.254/latest/meta-data/",
    "file:///etc/passwd",
])
async def test_ingest_url_rejects_non_public_target(client, auth_token, no_dns, url):
    """Blocked URLs return 400 and must not create an IngestionJob row."""
    resp = await client.post(
        "/api/ingest/url",
        json={"url": url},
        headers={"Authorization": f"Bearer {auth_token}"},
    )
    assert resp.status_code == 400
    assert "Blocked URL" in resp.json()["detail"]

    async with AsyncSessionLocal() as db:
        count = (await db.execute(select(func.count()).select_from(IngestionJob))).scalar()
    assert count == 0


@pytest.mark.asyncio
async def test_ingest_pdf_rejects_non_pdf(client, auth_token):
    """Uploading a non-PDF file (text/plain) must be rejected with 400."""
    from io import BytesIO
    resp = await client.post(
        "/api/ingest/pdf",
        files={"file": ("document.txt", BytesIO(b"text content"), "text/plain")},
        headers={"Authorization": f"Bearer {auth_token}"},
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
@pytest.mark.parametrize("url", [
    "http://169.254.169.254/latest/meta-data/",
    "not-a-video-id",
    "https://www.youtube.com/watch?v=short",
])
async def test_ingest_youtube_rejects_bad_target(client, auth_token, no_dns, url):
    """Non-YouTube targets are rejected with 400 and create no job row."""
    resp = await client.post(
        "/api/ingest/youtube",
        json={"url": url},
        headers={"Authorization": f"Bearer {auth_token}"},
    )
    assert resp.status_code == 400

    async with AsyncSessionLocal() as db:
        count = (await db.execute(select(func.count()).select_from(IngestionJob))).scalar()
    assert count == 0
