"""
MCP stdio server — expose Hermes retrieval tools to MCP Inspector / clients.

Tools:
    hermes_search — hybrid KB search (shares HE-7 hybrid_search)
    hermes_research — full research pipeline (local/dev; no JWT)

Run:
    uv run python -m src.mcp.server
"""

from __future__ import annotations

import json

from mcp.server.fastmcp import FastMCP

from src.tools.hybrid_search import hybrid_search

mcp = FastMCP("hermes")


@mcp.tool()
def hermes_search(query: str, top_k: int = 5) -> str:
    """
    Hybrid dense+BM25 search over the Hermes knowledge base.

    Returns JSON list of {context, score, metadata} (and related fields).
    """
    results = hybrid_search(query, top_k=top_k, user_id=None)
    return json.dumps(results, default=str)


@mcp.tool()
def hermes_research(query: str) -> str:
    """
    Run the Hermes research pipeline for a query (local/dev tool).

    No JWT — do not expose without auth in production. Returns JSON with
    answer, citations, and tool_trace.
    """
    from src.agents.graph import run_research

    state = run_research(query, session_id=None, messages=[], user_id=None)
    payload = {
        "answer": state.get("final_answer") or state.get("draft_answer") or "",
        "citations": state.get("citations") or [],
        "tool_trace": state.get("tool_trace") or [],
        "model_used": state.get("model_used", ""),
        "cache_hit": state.get("cache_hit", False),
    }
    return json.dumps(payload, default=str)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
