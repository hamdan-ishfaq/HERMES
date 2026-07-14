"""fetch_parent tool — expand parent context by id."""

from src.rag.factory import get_retriever


def fetch_parent(parent_id: str) -> str:
    """Return parent_text for ``parent_id`` or empty string if missing."""
    text = get_retriever().fetch_parent_text(parent_id)
    return text or ""
