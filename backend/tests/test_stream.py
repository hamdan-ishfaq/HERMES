"""Unit tests for streaming research helper (HE-5)."""

from unittest.mock import MagicMock, patch

from src.agents.stream_research import iter_research_events


def test_stream_cache_hit_emits_token_then_done():
    cached_state = {
        "query": "q",
        "cache_hit": True,
        "final_answer": "Cached answer",
        "citations": [{"source": "a"}],
        "tool_trace": [],
        "model_used": "",
        "next_agent": "END",
    }
    with patch(
        "src.agents.stream_research.cache_check_node",
        return_value=cached_state,
    ):
        events = list(iter_research_events("q", session_id="s1"))
    names = [e[0] for e in events]
    assert names == ["status", "token", "done"]
    assert events[0][1]["stage"] == "cache"
    assert events[1][1]["text"] == "Cached answer"
    assert events[2][1]["cache_hit"] is True
    assert events[2][1]["answer"] == "Cached answer"


def test_stream_generate_concatenates_tokens():
    miss_state = {
        "query": "What is RAG?",
        "cache_hit": False,
        "next_agent": "supervisor",
        "messages": [],
        "tool_trace": [],
        "citations": [],
        "final_answer": "",
        "model_used": "",
    }
    rewrite_state = {**miss_state, "rewritten_query": "What is RAG?", "query_complexity": "simple"}
    ctx = {
        "context": "RAG retrieves then generates.",
        "child_text": "RAG",
        "score": 5.0,
        "reranker_score": 5.0,
        "parent_id": "p1",
        "metadata": {"source": "doc", "url": None, "title": None, "page_num": 1},
    }
    retriever = MagicMock()
    retriever.use_cache = False

    with (
        patch("src.agents.stream_research.cache_check_node", return_value=miss_state),
        patch("src.agents.stream_research.classify_complexity", return_value="simple"),
        patch("src.agents.stream_research.query_rewrite_node", return_value=rewrite_state),
        patch("src.agents.stream_research._retrieve_contexts", return_value=[ctx]),
        patch("src.agents.stream_research._crag_lite_enabled", return_value=False),
        patch(
            "src.agents.stream_research.get_completion_stream",
            return_value=iter(["Hel", "lo", "!"]),
        ),
        patch("src.agents.stream_research.get_retriever", return_value=retriever),
    ):
        events = list(iter_research_events("What is RAG?", session_id="s2"))

    tokens = "".join(e[1]["text"] for e in events if e[0] == "token")
    assert tokens == "Hello!"
    done = next(e[1] for e in events if e[0] == "done")
    assert done["answer"] == "Hello!"
    assert done["cache_hit"] is False
    assert len(done["citations"]) == 1
