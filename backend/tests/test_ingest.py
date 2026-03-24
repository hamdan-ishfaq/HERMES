import pytest


@pytest.mark.asyncio
async def test_ingest_url_success(client, auth_token, mock_retriever, mock_ingest_url):
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
    resp = await client.post("/api/ingest/url", json={"url": "https://example.com"})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_ingest_pdf_rejects_non_pdf(client, auth_token):
    from io import BytesIO
    resp = await client.post(
        "/api/ingest/pdf",
        files={"file": ("document.txt", BytesIO(b"text content"), "text/plain")},
        headers={"Authorization": f"Bearer {auth_token}"},
    )
    assert resp.status_code == 400
