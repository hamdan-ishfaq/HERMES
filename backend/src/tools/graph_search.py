"""graph_search tool — 1-hop entity-linked chunk expansion."""

from __future__ import annotations

from src.rag.factory import get_retriever
from src.rag.graph_index import fetch_graph_contexts, graph_enabled


def graph_search(
    query: str,
    user_id: str | None = None,
    limit: int = 5,
) -> list[dict]:
    if not graph_enabled():
        return []
    retriever = get_retriever()
    return fetch_graph_contexts(query, retriever, user_id=user_id)[:limit]
