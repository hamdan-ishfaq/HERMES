"""Unit tests for the Redis semantic cache helpers."""

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