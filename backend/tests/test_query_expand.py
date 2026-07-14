"""Unit tests for multi-query merge."""

from unittest.mock import patch

from src.rag import query_expand


def test_merge_keeps_best_score_per_parent():
    a = {"parent_id": "p1", "context": "a", "reranker_score": 0.3}
    b = {"parent_id": "p1", "context": "a long", "reranker_score": 0.9}
    c = {"parent_id": "p2", "context": "c", "reranker_score": 0.5}
    merged = query_expand.merge_contexts_by_parent([[a], [b, c]])
    assert len(merged) == 2
    by_id = {m["parent_id"]: m for m in merged}
    assert by_id["p1"]["reranker_score"] == 0.9


def test_expand_queries_disabled():
    with patch.dict("os.environ", {"HERMES_MULTI_QUERY": "0"}, clear=False):
        assert query_expand.expand_queries("What is RAG?") == ["What is RAG?"]
