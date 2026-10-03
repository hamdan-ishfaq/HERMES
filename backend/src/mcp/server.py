"""
MCP stdio server — expose Hermes retrieval tools to MCP Inspector / clients.

Tools:
    hermes_search — hybrid KB search (shares HE-7 hybrid_search)
    hermes_research — full research pipeline (local/dev; no JWT)

Both tools run under the ACL of the operator declared in HERMES_MCP_USER_ID.
There is no request-level identity on stdio MCP, so that env var is the only
identity available; if it is unset the tools refuse to run rather than
falling back to an unscoped (user_id=None) search.

Run:
    HERMES_MCP_USER_ID=<user id> uv run python -m src.mcp.server
"""

from __future__ import annotations

import json
import os

from mcp.server.fastmcp import FastMCP

from src.tools.hybrid_search import hybrid_search

mcp = FastMCP("hermes")

MCP_USER_ID_ENV = "HERMES_MCP_USER_ID"


def _resolve_user_id() -> tuple[str | None, str | None]:
    """
    Resolve the acting user from the environment.

    Returns (user_id, None) when HERMES_MCP_USER_ID is set to a non-blank
    value, otherwise (None, error_json) so the caller can return the failure
    to the MCP client instead of running an unscoped query.
    """
    raw = os.getenv(MCP_USER_ID_ENV, "").strip()
    if not raw:
        return None, json.dumps({
            "error": f"{MCP_USER_ID_ENV} is not set.",
            "detail": (
                "This server has no request-level identity, so it cannot "
                "choose an ACL scope. Export HERMES_MCP_USER_ID with the "
                "numeric id of the user to act as and restart the server."
            ),
        })
    return raw, None


@mcp.tool()
def hermes_search(query: str, top_k: int = 5) -> str:
    """
    Hybrid dense+BM25 search over the Hermes knowledge base.

    Returns JSON list of {context, score, metadata} (and related fields), or a
    JSON {error, detail} object when HERMES_MCP_USER_ID is not configured.
    """
    user_id, error = _resolve_user_id()
    if error is not None:
        return error
    results = hybrid_search(query, top_k=top_k, user_id=user_id)
    return json.dumps(results, default=str)


@mcp.tool()
def hermes_research(query: str) -> str:
    """
    Run the Hermes research pipeline for a query (local/dev tool).

    No JWT — do not expose without auth in production. Returns JSON with
    answer, citations, and tool_trace, or a JSON {error, detail} object when
    HERMES_MCP_USER_ID is not configured.
    """
    user_id, error = _resolve_user_id()
    if error is not None:
        return error

    from src.agents.graph import run_research

    state = run_research(query, session_id=None, messages=[], user_id=user_id)
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
