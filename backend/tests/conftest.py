import pytest
import pytest_asyncio
from unittest.mock import MagicMock, patch
from httpx import AsyncClient, ASGITransport
from sqlalchemy import text

from src.main import app
from src.db import engine, Base

@pytest_asyncio.fixture(scope="session", autouse=True)
async def create_test_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

@pytest_asyncio.fixture(autouse=True)
async def truncate_tables():
    yield
    async with engine.begin() as conn:
        await conn.execute(text(
            "TRUNCATE users, ingestion_jobs, query_logs RESTART IDENTITY CASCADE"
        ))

@pytest_asyncio.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c

@pytest_asyncio.fixture
async def auth_token(client):
    resp = await client.post("/api/auth/register", json={
        "email": "ci@hermes.dev", "password": "cipass123"
    })
    return resp.json()["access_token"]

@pytest.fixture
def mock_retriever():
    with patch("src.routers.ingest.get_retriever") as mock:
        retriever = MagicMock()
        mock.return_value = retriever
        yield retriever

@pytest.fixture
def mock_ingest_url():
    with patch("src.routers.ingest.ingest_url") as mock:
        mock.return_value = {"total_children": 42}
        yield mock

@pytest.fixture
def mock_run_research():
    with patch("src.routers.research.run_research") as mock:
        mock.return_value = {
            "final_answer": "RAG combines retrieval with generation.",
            "citations": [{"source": "wiki", "page_num": 1, "context": "...", "score": 9.1}],
            "model_used": "ollama/llama3.1:8b",
            "cache_hit": False,
        }
        yield mock
