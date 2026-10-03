"""Unit tests for the Redis semantic cache helpers."""

import json
import logging

import pytest

from src.rag import cache


def test_cosine_similarity_identical_vectors():
    v = [1.0, 2.0, 3.0]
    assert cache._cosine_similarity(v, list(v)) == pytest.approx(1.0)


def test_cosine_similarity_zero_vector_returns_zero():
    assert cache._cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


def test_cosine_similarity_raises_on_length_mismatch():
    with pytest.raises(ValueError, match="vector length mismatch"):
        cache._cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0])


def test_cosine_similarity_raises_on_768_vs_1024_mismatch():
    stale = [0.1] * 768
    current = [0.1] * 1024
    with pytest.raises(ValueError, match="768 != 1024"):
        cache._cosine_similarity(stale, current)


def test_embed_single_delegates_to_dense_embed(monkeypatch):
    captured: dict = {}

    def fake_dense_embed(texts):
        captured["texts"] = texts
        return [[0.5, 0.25, 0.125]]

    monkeypatch.setattr(cache, "dense_embed", fake_dense_embed)
    assert cache._embed_single("What is RAG?") == [0.5, 0.25, 0.125]
    assert captured["texts"] == ["What is RAG?"]


class _FakeRedis:
    def __init__(self, entries: list[str]):
        self._entries = entries

    def lrange(self, key, start, end):
        return self._entries

    def llen(self, key):
        return len(self._entries)


def _entry(vec: list[float], result: dict) -> str:
    return json.dumps({
        "query": "cached query",
        "embedding": vec,
        "result": result,
        "user_id": None,
    })


def _make_cache(monkeypatch, entries: list[str], query_vec: list[float]):
    c = cache.SemanticCache()
    c.redis = _FakeRedis(entries)
    monkeypatch.setattr(cache, "_embed_single", lambda text: list(query_vec))
    return c


def test_get_returns_none_when_only_entry_has_stale_dimension(monkeypatch):
    stale = [0.1] * 768
    c = _make_cache(monkeypatch, [_entry(stale, {"answer": "old"})], [0.1] * 1024)
    assert c.get("q") is None
    assert c.misses == 1
    assert c.hits == 0


def test_get_matches_valid_entry_alongside_stale_dimension_entry(monkeypatch):
    stale = [0.1] * 768
    valid = [0.5] * 1024
    entries = [
        _entry(stale, {"answer": "stale"}),
        _entry(valid, {"answer": "fresh"}),
    ]
    c = _make_cache(monkeypatch, entries, [0.5] * 1024)
    out = c.get("q")
    assert out is not None
    assert out["answer"] == "fresh"
    assert out["cache_hit"] is True
    assert c.hits == 1


def test_get_logs_one_warning_on_dimension_mismatch(monkeypatch, caplog):
    stale_a = [0.1] * 768
    stale_b = [0.2] * 512
    entries = [_entry(stale_a, {"answer": "a"}), _entry(stale_b, {"answer": "b"})]
    c = _make_cache(monkeypatch, entries, [0.1] * 1024)

    with caplog.at_level(logging.WARNING, logger="src.rag.cache"):
        assert c.get("q") is None

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert warnings[0].name == "src.rag.cache"
    assert "skipped 2 entries with mismatched embedding dimension" in warnings[0].getMessage()
