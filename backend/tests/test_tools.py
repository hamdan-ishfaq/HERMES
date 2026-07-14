"""HE-7: forced hybrid_search tool trace and empty-KB honesty."""

from unittest.mock import MagicMock, patch

from src.agents.research import research_node


def _state(**overrides):
    state = {
        "query": "What is RAG?",
        "query_complexity": "simple",
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


def _ctx(score=5.0, source="doc-a.pdf"):
    return {
        "context": "Retrieval-augmented generation grounds answers.",
        "child_text": "RAG grounds answers",
        "score": score,
        "reranker_score": score,
        "parent_id": "p1",
        "metadata": {
            "source": source,
            "url": "https://example.com/doc",
            "title": "Doc A",
            "page_num": 1,
        },
    }


def test_hybrid_search_always_in_trace():
    retriever = MagicMock()
    retriever.use_cache = False
    with (
        patch("src.agents.research.get_retriever", return_value=retriever),
        patch("src.agents.research.expand_queries", return_value=["What is RAG?"]),
        patch("src.agents.research.hybrid_search", return_value=[_ctx()]) as mock_search,
        patch("src.agents.research.get_completion", return_value="Grounded answer"),
        patch.dict("os.environ", {"HERMES_CRAG_LITE": "0"}, clear=False),
    ):
        out = research_node(_state())
    mock_search.assert_called_once()
    names = [t["name"] for t in out["tool_trace"]]
    assert "hybrid_search" in names


def test_empty_tools_no_invented_source():
    retriever = MagicMock()
    retriever.use_cache = False
    with (
        patch("src.agents.research.get_retriever", return_value=retriever),
        patch("src.agents.research.expand_queries", return_value=["What is RAG?"]),
        patch("src.agents.research.hybrid_search", return_value=[]),
        patch("src.agents.research.get_completion") as mock_llm,
        patch.dict("os.environ", {"HERMES_CRAG_LITE": "0"}, clear=False),
    ):
        out = research_node(_state())
    assert out["error"] == "no_context"
    assert out["citations"] == []
    assert "http" not in (out["final_answer"] or "").lower()
    assert any(t["name"] == "hybrid_search" for t in out["tool_trace"])
    mock_llm.assert_not_called()


def test_citations_subset_of_tool_sources():
    retriever = MagicMock()
    retriever.use_cache = False
    contexts = [_ctx(source="alpha.pdf"), _ctx(source="beta.pdf")]
    with (
        patch("src.agents.research.get_retriever", return_value=retriever),
        patch("src.agents.research.expand_queries", return_value=["What is RAG?"]),
        patch("src.agents.research.hybrid_search", return_value=contexts),
        patch("src.agents.research.get_completion", return_value="Answer"),
        patch.dict("os.environ", {"HERMES_CRAG_LITE": "0"}, clear=False),
    ):
        out = research_node(_state())
    tool_sources = {c["metadata"]["source"] for c in contexts}
    for cit in out["citations"]:
        assert cit["source"] in tool_sources
