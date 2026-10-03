"""HE-6: MCP tool schemas and hermes_search wiring."""

import json
from unittest.mock import patch

from src.mcp.server import hermes_research, hermes_search, mcp


def test_mcp_lists_hermes_search():
    tools = getattr(mcp, "_tool_manager", None)
    assert tools is not None
    names = list(tools._tools.keys())
    assert "hermes_search" in names
    assert "hermes_research" in names


def test_hermes_search_schema_requires_query():
    tool = mcp._tool_manager._tools["hermes_search"]
    schema = tool.parameters
    assert "query" in schema.get("required", [])
    assert "query" in schema.get("properties", {})


def test_hermes_search_returns_json_with_mocked_retriever(monkeypatch):
    fake = [
        {
            "context": "hello",
            "score": 1.0,
            "metadata": {"source": "a.pdf"},
        }
    ]
    monkeypatch.setenv("HERMES_MCP_USER_ID", "42")
    with patch("src.mcp.server.hybrid_search", return_value=fake) as mock_search:
        out = hermes_search("test query", top_k=3)
    assert "hello" in out
    assert "a.pdf" in out
    assert mock_search.call_args.kwargs["user_id"] == "42"


def test_hermes_search_errors_without_user_id_env(monkeypatch):
    monkeypatch.delenv("HERMES_MCP_USER_ID", raising=False)
    with patch("src.mcp.server.hybrid_search") as mock_search:
        out = hermes_search("test query")
    mock_search.assert_not_called()
    assert json.loads(out)["error"] == "HERMES_MCP_USER_ID is not set."


def test_hermes_search_errors_on_blank_user_id_env(monkeypatch):
    monkeypatch.setenv("HERMES_MCP_USER_ID", "   ")
    with patch("src.mcp.server.hybrid_search") as mock_search:
        out = hermes_search("test query")
    mock_search.assert_not_called()
    assert "error" in json.loads(out)


def test_hermes_research_errors_without_user_id_env(monkeypatch):
    monkeypatch.delenv("HERMES_MCP_USER_ID", raising=False)
    with patch("src.agents.graph.run_research") as mock_research:
        out = hermes_research("test query")
    mock_research.assert_not_called()
    assert json.loads(out)["error"] == "HERMES_MCP_USER_ID is not set."


def test_hermes_research_passes_user_id(monkeypatch):
    monkeypatch.setenv("HERMES_MCP_USER_ID", "42")
    with patch("src.agents.graph.run_research") as mock_research:
        mock_research.return_value = {"final_answer": "42"}
        out = hermes_research("test query")
    assert mock_research.call_args.kwargs["user_id"] == "42"
    assert json.loads(out)["answer"] == "42"
