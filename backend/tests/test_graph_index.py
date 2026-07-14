"""Unit tests for GraphRAG-lite helpers."""

from src.rag.graph_index import _tokenize_entities, expand_graph, graph_enabled


def test_tokenize_entities_filters_stopwords():
    tokens = _tokenize_entities("What is hybrid search in a vector database?")
    assert "hybrid" in tokens
    assert "what" not in tokens
    assert "the" not in tokens


def test_graph_disabled_expand_empty(monkeypatch):
    monkeypatch.setenv("HERMES_GRAPH_RAG", "0")
    assert graph_enabled() is False
    assert expand_graph("RAG retrieval") == []


def test_graph_enabled(monkeypatch):
    monkeypatch.setenv("HERMES_GRAPH_RAG", "1")
    assert graph_enabled() is True
