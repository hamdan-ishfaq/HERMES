"""HE-4: workspace-scoped Qdrant user_id filter."""

from unittest.mock import MagicMock, patch

from qdrant_client.models import FieldCondition, Filter, MatchValue, SparseVector

from src.rag.retriever import HermesRetriever
from src.tools.hybrid_search import hybrid_search


def _mock_qdrant_client():
    client = MagicMock()
    coll = MagicMock()
    coll.name = "hermes_docs"
    client.get_collections.return_value = MagicMock(collections=[coll])
    client.query_points.return_value = MagicMock(points=[])
    return client


def test_query_passes_user_id_filter():
    client = _mock_qdrant_client()

    with (
        patch("src.rag.retriever.QdrantClient", return_value=client),
        patch("src.rag.retriever.vector_dimension", return_value=768),
        patch("src.rag.retriever.dense_embed", return_value=[[0.1] * 768]),
        patch(
            "src.rag.retriever.sparse_embed",
            return_value=[SparseVector(indices=[1], values=[1.0])],
        ),
    ):
        retriever = HermesRetriever(use_reranker=False, use_cache=False)
        retriever.query("secret marker", top_k=3, user_id="user-a")

    kwargs = client.query_points.call_args.kwargs
    qf = kwargs.get("query_filter")
    assert isinstance(qf, Filter)
    assert any(
        isinstance(c, FieldCondition)
        and c.key == "user_id"
        and isinstance(c.match, MatchValue)
        and c.match.value == "user-a"
        for c in (qf.must or [])
    )
    for prefetch in kwargs.get("prefetch") or []:
        assert prefetch.filter is not None


def test_query_without_user_id_has_no_filter():
    client = _mock_qdrant_client()

    with (
        patch("src.rag.retriever.QdrantClient", return_value=client),
        patch("src.rag.retriever.vector_dimension", return_value=768),
        patch("src.rag.retriever.dense_embed", return_value=[[0.1] * 768]),
        patch(
            "src.rag.retriever.sparse_embed",
            return_value=[SparseVector(indices=[1], values=[1.0])],
        ),
    ):
        retriever = HermesRetriever(use_reranker=False, use_cache=False)
        retriever.query("open search", top_k=3, user_id=None)

    kwargs = client.query_points.call_args.kwargs
    assert kwargs.get("query_filter") is None


def test_hybrid_search_forwards_user_id():
    retriever = MagicMock()
    retriever.query.return_value = []
    with patch("src.tools.hybrid_search.get_retriever", return_value=retriever):
        hybrid_search("q", top_k=2, user_id="42")
    retriever.query.assert_called_once_with("q", top_k=2, user_id="42")
