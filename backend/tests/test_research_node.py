"""Unit tests for research_node filtering, routing, and cache write."""

from unittest.mock import MagicMock, patch

import src.agents.research as research_mod
from src.agents.research import research_node


def _state(complexity="simple", **overrides):
    state = {
        "query": "What is RAG?",
        "query_complexity": complexity,
        "next_agent": "",
        "retrieved_contexts": [],
        "citations": [],
        "draft_answer": "",
        "final_answer": "",
        "cache_hit": False,
        "model_used": "",
        "error": None,
        "tool_trace": [],
    }
    state.update(overrides)
    return state


def _ctx(score, text="good context"):
    return {
        "context": text,
        "child_text": text[:20],
        "score": score,
        "reranker_score": score,
        "parent_id": "p1",
        "metadata": {"source": "doc", "url": None, "title": None, "page_num": 1},
    }


def test_filters_below_min_rerank(monkeypatch):
    monkeypatch.setenv("MIN_RERANK_SCORE", "0.35")
    monkeypatch.setenv("HERMES_CRAG_LITE", "0")
    monkeypatch.setattr(research_mod, "expand_queries", lambda q: [q])
    retriever = MagicMock()
    retriever.use_cache = False
    with (
        patch("src.agents.research.get_retriever", return_value=retriever),
        patch(
            "src.agents.research.hybrid_search",
            return_value=[_ctx(0.1, "weak"), _ctx(5.0, "strong")],
        ),
        patch("src.agents.research.get_completion", return_value="Draft from strong") as mock_llm,
    ):
        out = research_node(_state())
    assert len(out["citations"]) == 1
    assert out["citations"][0]["context"] == "strong"
    mock_llm.assert_called_once()


def test_no_context_early_exit():
    retriever = MagicMock()
    retriever.use_cache = False
    with (
        patch("src.agents.research.get_retriever", return_value=retriever),
        patch("src.agents.research.expand_queries", return_value=["What is RAG?"]),
        patch("src.agents.research.hybrid_search", return_value=[]),
        patch("src.agents.research.get_completion") as mock_llm,
    ):
        out = research_node(_state())
    assert out["error"] == "no_context"
    assert out["next_agent"] == "END"
    assert out["citations"] == []
    mock_llm.assert_not_called()


def test_simple_routes_end_and_caches():
    retriever = MagicMock()
    retriever.use_cache = True
    with (
        patch("src.agents.research.get_retriever", return_value=retriever),
        patch("src.agents.research.expand_queries", return_value=["What is RAG?"]),
        patch("src.agents.research.hybrid_search", return_value=[_ctx(5.0)]),
        patch("src.agents.research.get_completion", return_value="Simple draft"),
    ):
        out = research_node(_state("simple"))
    assert out["next_agent"] == "END"
    retriever.cache.set.assert_called_once()


def test_multi_hop_routes_synthesis_without_cache():
    retriever = MagicMock()
    retriever.use_cache = True
    with (
        patch("src.agents.research.get_retriever", return_value=retriever),
        patch("src.agents.research.expand_queries", return_value=["What is RAG?"]),
        patch("src.agents.research.hybrid_search", return_value=[_ctx(5.0)]),
        patch("src.agents.research.get_completion", return_value="Multi draft"),
    ):
        out = research_node(_state("multi_hop"))
    assert out["next_agent"] == "synthesis_agent"
    retriever.cache.set.assert_not_called()
