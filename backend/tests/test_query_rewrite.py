"""Unit tests for query_rewrite_node (HE-2)."""

from unittest.mock import patch

from src.agents.query_rewrite import query_rewrite_node


def _state(query="follow-up?", messages=None):
    return {
        "query": query,
        "query_complexity": "simple",
        "next_agent": "",
        "retrieved_contexts": [],
        "citations": [],
        "draft_answer": "",
        "final_answer": "",
        "messages": messages or [],
        "rewritten_query": "",
        "tool_trace": [],
        "user_id": None,
        "cache_hit": False,
        "model_used": "",
        "error": None,
    }


def test_passthrough_without_history():
    out = query_rewrite_node(_state("What is RAG?"))
    assert out["rewritten_query"] == "What is RAG?"
    assert out["next_agent"] == "research_agent"


def test_rewrite_with_history(monkeypatch):
    monkeypatch.setenv("HERMES_MULTI_TURN", "1")
    # Reload module flag — set on the module directly
    import src.agents.query_rewrite as qr
    monkeypatch.setattr(qr, "HERMES_MULTI_TURN", True)

    messages = [
        {"role": "user", "content": "Tell me about ZEPHYR_ENTITY_99"},
        {"role": "assistant", "content": "ZEPHYR_ENTITY_99 is a marker phrase."},
    ]
    with patch(
        "src.agents.query_rewrite.get_completion",
        return_value="What else about ZEPHYR_ENTITY_99?",
    ):
        out = query_rewrite_node(_state("What else about it?", messages))
    assert "ZEPHYR_ENTITY_99" in out["rewritten_query"]


def test_multi_turn_disabled(monkeypatch):
    import src.agents.query_rewrite as qr
    monkeypatch.setattr(qr, "HERMES_MULTI_TURN", False)
    messages = [{"role": "user", "content": "prior"}]
    out = query_rewrite_node(_state("follow-up", messages))
    assert out["rewritten_query"] == "follow-up"
