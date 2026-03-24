import pytest


@pytest.mark.asyncio
async def test_research_returns_answer(client, auth_token, mock_run_research):
    resp = await client.post(
        "/api/research",
        json={"query": "What is RAG?"},
        headers={"Authorization": f"Bearer {auth_token}"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert data["answer"] == "RAG combines retrieval with generation."
    assert data["model_used"] == "ollama/llama3.1:8b"
    assert data["cache_hit"] is False
    assert len(data["citations"]) == 1


@pytest.mark.asyncio
async def test_research_requires_auth(client):
    resp = await client.post("/api/research", json={"query": "test"})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_health_endpoint(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "service": "hermes"}
