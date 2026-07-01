"""
Unit tests for the ingestion API (`/api/ingest/*`).

Covers:
    - Successful URL ingestion (mocked scraper + retriever)
    - Auth requirement on ingest endpoints
    - PDF MIME-type validation (reject non-PDF uploads)

Fixtures used (from conftest.py):
    - `client` — async ASGI test client.
    - `auth_token` — valid JWT for authenticated requests.
    - `mock_retriever` — prevents real Qdrant writes.
    - `mock_ingest_url` — returns a fixed `total_children` count.
"""

import pytest


@pytest.mark.asyncio
async def test_ingest_url_success(client, auth_token, mock_retriever, mock_ingest_url):
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
async def test_ingest_pdf_rejects_non_pdf(client, auth_token):
    """Uploading a non-PDF file (text/plain) must be rejected with 400."""
    from io import BytesIO
    resp = await client.post(
        "/api/ingest/pdf",
        files={"file": ("document.txt", BytesIO(b"text content"), "text/plain")},
        headers={"Authorization": f"Bearer {auth_token}"},
    )
    assert resp.status_code == 400
