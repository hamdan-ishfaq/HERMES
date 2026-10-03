"""
Shared pytest fixtures for the Hermes backend test suite.

What is tested (across all test modules):
    - Authentication (register, login, JWT-protected routes)
    - Document ingestion (URL, PDF validation)
    - Research / RAG query pipeline
    - Optional integration tests against live Qdrant + Ollama (see test_integration_retriever.py)

How fixtures work:
    - `create_test_tables` (session-scoped, autouse): creates all SQLAlchemy tables
      before any test runs and drops them after the session ends.
    - `truncate_tables` (autouse): clears users, ingestion_jobs, and query_logs
      after each test so cases stay isolated without recreating schema.
    - `client`: async httpx ASGI client wired to the FastAPI `app`.
    - `auth_token`: registers a CI user and returns a valid Bearer token for
      protected-route tests.
    - `mock_retriever`, `mock_ingest_url`, `mock_run_research`: patch heavy
      RAG dependencies so unit tests stay fast and deterministic.
"""

import os

import pytest
import pytest_asyncio
from unittest.mock import MagicMock, patch
from httpx import AsyncClient, ASGITransport
from sqlalchemy import text

from src.main import app
from src.db import DATABASE_URL, engine, Base

# Database names the suite is allowed to run against. The tests TRUNCATE tables
# between cases and drop nothing, so pointing them at a real database destroys
# that data. Only throwaway names belong here.
_ALLOWED_TEST_DATABASES = {"hermes_test", "hermes_e2e", "hermes_ci", "test", "postgres"}


def _assert_safe_database_url() -> None:
    """Abort the session if DATABASE_URL does not point at a throwaway database."""
    name = DATABASE_URL.rsplit("/", 1)[-1].split("?", 1)[0]
    if name == "hermes_db":
        raise RuntimeError(
            "Refusing to run tests against 'hermes_db' — it is the real "
            "application database and this suite truncates tables between "
            "tests. Set DATABASE_URL to a throwaway database (e.g. hermes_test)."
        )
    if name not in _ALLOWED_TEST_DATABASES:
        raise RuntimeError(
            f"Refusing to run tests against database {name!r}. "
            f"Allowed: {sorted(_ALLOWED_TEST_DATABASES)}. Set DATABASE_URL "
            "explicitly to a throwaway database."
        )


_assert_safe_database_url()


@pytest_asyncio.fixture(scope="session", autouse=True)
async def create_test_tables():
    """
    Ensure ORM tables exist for the test session.

    Does NOT drop tables on teardown — the hermes Postgres is shared with local
    API/E2E runs; dropping would break live servers mid-session.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield



@pytest_asyncio.fixture(autouse=True)
async def truncate_tables():
    """Reset mutable tables between tests to prevent cross-test data leakage."""
    yield
    async with engine.begin() as conn:
        await conn.execute(text(
            "TRUNCATE users, ingestion_jobs, query_logs, conversation_turns "
            "RESTART IDENTITY CASCADE"
        ))


@pytest_asyncio.fixture
async def client():
    """Async HTTP client that talks to the FastAPI app in-process (no live server)."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest_asyncio.fixture
async def auth_token(client):
    """Register a test user and return a JWT suitable for Authorization headers."""
    resp = await client.post("/api/auth/register", json={
        "email": "ci@hermes.dev", "password": "cipass123"
    })
    return resp.json()["access_token"]


@pytest.fixture
def mock_retriever():
    """Patch `get_retriever` in the ingest router so vector DB is never touched."""
    with patch("src.routers.ingest.get_retriever") as mock:
        retriever = MagicMock()
        mock.return_value = retriever
        yield retriever


@pytest.fixture
def mock_ingest_url():
    """Stub `ingest_url` to return a fixed chunk count without scraping the web."""
    with patch("src.routers.ingest.ingest_url") as mock:
        mock.return_value = {"total_children": 42}
        yield mock


@pytest.fixture
def mock_run_research():
    """Stub the research pipeline so tests assert on the HTTP contract, not LLM output."""
    with patch("src.routers.research.run_research") as mock:
        mock.return_value = {
            "final_answer": "RAG combines retrieval with generation.",
            "citations": [{"source": "wiki", "page_num": 1, "context": "...", "score": 9.1}],
            "model_used": "ollama/llama3.1:8b",
            "cache_hit": False,
            "tool_trace": [],
            "rewritten_query": "What is RAG?",
        }
        yield mock
