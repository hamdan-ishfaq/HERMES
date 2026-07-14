"""HE-6: MCP tool schemas and hermes_search wiring."""

from unittest.mock import patch

from src.mcp.server import hermes_search, mcp


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


def test_hermes_search_returns_json_with_mocked_retriever():
    fake = [
        {
            "context": "hello",
            "score": 1.0,
            "metadata": {"source": "a.pdf"},
        }
    ]
    with patch("src.mcp.server.hybrid_search", return_value=fake):
        out = hermes_search("test query", top_k=3)
    assert "hello" in out
    assert "a.pdf" in out
