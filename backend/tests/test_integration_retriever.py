"""
Integration tests for the real retrieval pipeline.

These exercise the actual Qdrant + Ollama stack (no mocks) and prove the two
core fixes:
  1. Parent-context expansion fires through the shared retriever.
  2. The semantic cache round-trips a stored answer.

They are marked `integration` and skipped in the default/CI run. Run locally
with the stack up:

    docker compose up -d qdrant redis
    # Ollama is only needed if EMBED_MODEL points at it; the default local
    # BGE-m3 embedder needs no server.
    DATABASE_URL=...hermes_test uv run pytest -m integration -v

Fixtures: none from conftest — these tests manage their own Qdrant/Redis state
and skip automatically when the external stack is unreachable.
"""

import os
import uuid

import httpx
import pytest

pytestmark = pytest.mark.integration


def _stack_available() -> bool:
    """
    True when Qdrant is reachable and the configured embedder can produce vectors.

    Ollama is only required when EMBED_MODEL actually points at it. The default
    embedder is the local BGE-m3 sentence-transformer, so requiring an Ollama
    embedding server here silently skipped the whole file on a machine that was
    otherwise ready to run it.
    """
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    try:
        httpx.get(f"{qdrant_url}/collections", timeout=3).raise_for_status()
    except Exception:
        return False

    if os.getenv("EMBED_MODEL", "bge-m3").strip().lower() in ("bge-m3", "bge_m3", "bge"):
        try:
            from src.rag.embeddings import dense_embed
            dense_embed(["ping"])
            return True
        except Exception:
            return False

    ollama_base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
    model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    try:
        httpx.post(
            f"{ollama_base}/api/embed",
            json={"model": model, "input": "ping"},
            timeout=15,
        ).raise_for_status()
        return True
    except Exception:
        return False


requires_stack = pytest.mark.skipif(
    not _stack_available(),
    reason="Qdrant must be running, plus Ollama if EMBED_MODEL points at it",
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


def _clear_hermes_keys(r) -> int:
    """
    Delete only this app's Redis keys instead of flushdb().

    flushdb() wipes every key in the database, including anything another
    process or a developer's local session is using. Scan-and-delete keeps the
    blast radius to hermes:* which is all this suite writes.
    """
    deleted = 0
    for key in r.scan_iter(match="hermes:*", count=500):
        deleted += r.delete(key)
    return deleted


@requires_stack
def test_semantic_cache_round_trip():
    """
    Store an answer in SemanticCache and read it back with a reworded query.

    The reworded query has to clear SIMILARITY_THRESHOLD (0.95) to be a hit.
    This previously used "Can you explain retrieval augmented generation?",
    which BGE-m3 scores at 0.9135 against the stored question -- a legitimate
    miss, so the test asserted behaviour the cache was never meant to have.
    """
    import redis
    from src.rag.cache import SIMILARITY_THRESHOLD, SemanticCache

    r = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
    _clear_hermes_keys(r)

    cache = SemanticCache()
    payload = {
        "answer": "RAG combines retrieval with generation.",
        "citations": [{"source": "itest", "context": "..."}],
        "contexts": [],
    }
    stored = "What is retrieval augmented generation?"
    cache.set(stored, payload)

    # A genuine rephrasing (measured 0.9744, above the threshold) must hit.
    hit = cache.get("What does retrieval augmented generation mean?")
    assert hit is not None, "reworded query above the threshold failed to hit"
    assert hit.get("cache_hit") is True
    assert hit["answer"] == payload["answer"]

    # A differently-worded question (measured 0.9135, below it) must miss,
    # so the cache cannot serve a stale answer to an unrelated question.
    miss = cache.get("Explain retrieval augmented generation")
    assert miss is None or miss.get("cache_hit") is not True


@requires_stack
def test_semantic_cache_scoped_cleanup_leaves_foreign_keys():
    """The scoped cleanup used by this suite must not delete non-hermes keys."""
    import redis
    from src.rag.cache import SemanticCache

    r = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
    r.set("e2e:canary:must-survive", "1")
    try:
        SemanticCache().set("What is retrieval augmented generation?",
                            {"answer": "x", "citations": [], "contexts": []})
        deleted = _clear_hermes_keys(r)
        assert deleted >= 1
        assert r.get("e2e:canary:must-survive") == b"1"
    finally:
        r.delete("e2e:canary:must-survive")
