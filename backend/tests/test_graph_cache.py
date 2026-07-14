"""Unit tests for cache_check node and graph entry routing."""

from unittest.mock import MagicMock, patch

from src.agents.cache_check import cache_check_node
from src.agents.graph import build_graph, route_after_cache


def _base_state(**overrides):
    state = {
        "query": "What is RAG?",
        "query_complexity": "",
        "next_agent": "",
        "retrieved_contexts": [],
        "citations": [],
        "draft_answer": "",
        "final_answer": "",
        "cache_hit": False,
        "model_used": "",
        "error": None,
    }
    state.update(overrides)
    return state


def test_cache_hit_routes_to_end():
    retriever = MagicMock()
    retriever.use_cache = True
    retriever.cache.get.return_value = {
        "answer": "Cached answer",
        "citations": [{"source": "wiki", "context": "...", "score": 1.0}],
        "contexts": [{"context": "..."}],
    }
    with patch("src.agents.cache_check.get_retriever", return_value=retriever):
        out = cache_check_node(_base_state(user_id="42"))
    assert out["cache_hit"] is True
    assert out["next_agent"] == "END"
    assert out["final_answer"] == "Cached answer"
    retriever.cache.get.assert_called_once_with("What is RAG?", user_id="42")
    assert route_after_cache(out) == "END"


def test_cache_miss_routes_to_supervisor():
    retriever = MagicMock()
    retriever.use_cache = True
    retriever.cache.get.return_value = None
    with patch("src.agents.cache_check.get_retriever", return_value=retriever):
        out = cache_check_node(_base_state())
    assert out["cache_hit"] is False
    assert out["next_agent"] == "supervisor"
    assert route_after_cache(out) == "supervisor"


def test_build_graph_has_cache_check_node():
    graph = build_graph()
    # Compiled graph exposes nodes via get_graph / internal structure
    nodes = set(graph.get_graph().nodes)
    assert "cache_check" in nodes
    assert "supervisor" in nodes
    assert "query_rewrite" in nodes
    assert "research_agent" in nodes
