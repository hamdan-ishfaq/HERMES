"""hybrid_search tool — wraps HermesRetriever.query."""

from src.rag.factory import get_retriever


def hybrid_search(
    query: str,
    top_k: int = 5,
    user_id: str | None = None,
) -> list[dict]:
    """
    Run hybrid retrieval for ``query``.

    Returns list of context dicts with metadata for citations.
    """
    retriever = get_retriever()
    return retriever.query(query, top_k=top_k, user_id=user_id)
