"""
Integration tests for the real retrieval pipeline.

These exercise the actual Qdrant + Ollama stack (no mocks) and prove the two
core fixes:
  1. Parent-context expansion fires through the shared retriever.
  2. The semantic cache round-trips a stored answer.

They are marked `integration` and skipped in the default/CI run. Run locally
with the stack up:

    docker compose up -d qdrant redis
    # ollama serving nomic-embed-text on $OLLAMA_API_BASE
    uv run pytest -m integration -v

Fixtures: none from conftest — these tests manage their own Qdrant/Redis state
and skip automatically when the external stack is unreachable.
"""

import os
import uuid

import httpx
import pytest

pytestmark = pytest.mark.integration


def _stack_available() -> bool:
    """True only if both Qdrant and Ollama embeddings are reachable."""
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    ollama_base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
    try:
        httpx.get(f"{qdrant_url}/collections", timeout=3).raise_for_status()
        httpx.post(
            f"{ollama_base}/api/embed",
            json={"model": "nomic-embed-text", "input": "ping"},
            timeout=15,
        ).raise_for_status()
        return True
    except Exception:
        return False


requires_stack = pytest.mark.skipif(
    not _stack_available(),
    reason="Qdrant + Ollama (nomic-embed-text) must be running for integration tests",
)


SAMPLE_TEXT = (
    "Parent-child chunking is a retrieval technique where small child chunks "
    "are indexed for precise semantic matching, while the larger parent chunk "
    "that contains them is returned to the language model. This gives the model "
    "rich surrounding context for generation without sacrificing retrieval "
    "precision. The marker phrase ZEPHYR_QUOKKA_42 is embedded here so the test "
    "can target this specific passage deterministically across many chunks. "
) * 8


@requires_stack
def test_parent_expansion_returns_more_than_child():
    """Ingest sample text and verify query returns expanded parent context, not just the child chunk."""
    from src.rag.factory import get_retriever, reset_retriever

    reset_retriever()
    retriever = get_retriever()
    retriever.ingest(SAMPLE_TEXT, metadata={"source": f"itest-{uuid.uuid4()}"})

    results = retriever.query("What is parent-child chunking?", top_k=3)
    assert results, "expected at least one retrieved context"

    top = results[0]
    # Parent expansion: the context returned to the LLM must be the larger
    # parent text, not just the ~200-char child chunk it matched on.
    assert len(top["context"]) > len(top["child_text"])


@requires_stack
def test_semantic_cache_round_trip():
    """Store an answer in SemanticCache and verify a paraphrased query hits the cache."""
    import redis
    from src.rag.cache import SemanticCache

    r = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
    r.flushdb()

    cache = SemanticCache()
    payload = {
        "answer": "RAG combines retrieval with generation.",
        "citations": [{"source": "itest", "context": "..."}],
        "contexts": [],
    }
    cache.set("What is retrieval augmented generation?", payload)

    hit = cache.get("Can you explain retrieval augmented generation?")
    assert hit is not None
    assert hit.get("cache_hit") is True
    assert hit["answer"] == payload["answer"]
