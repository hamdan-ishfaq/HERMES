"""Named retrieval tools used by research agent and MCP (HE-7 / HE-6)."""

from src.tools.hybrid_search import hybrid_search
from src.tools.fetch_parent import fetch_parent
from src.tools.web_fetch import web_fetch

TOOL_SCHEMAS = [
    {
        "name": "hybrid_search",
        "description": "Hybrid dense+BM25 search over the Hermes knowledge base.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_parent",
        "description": "Fetch full parent chunk text by parent_id.",
        "parameters": {
            "type": "object",
            "properties": {"parent_id": {"type": "string"}},
            "required": ["parent_id"],
        },
    },
]

__all__ = ["hybrid_search", "fetch_parent", "web_fetch", "TOOL_SCHEMAS"]
