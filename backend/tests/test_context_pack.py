"""Unit tests for context_pack dedup and cap."""

from src.rag.context_pack import pack_contexts


def _ctx(text: str, score: float, page: int = 0) -> dict:
    return {
        "context": text,
        "reranker_score": score,
        "metadata": {"page_num": page},
    }


def test_dedup_by_normalized_text():
    contexts = [
        _ctx("Same text here.", 0.9),
        _ctx("Same   text here.", 0.5),
        _ctx("Different passage.", 0.8),
    ]
    packed = pack_contexts(contexts, top_k=5)
    assert len(packed) == 2
    assert packed[0]["context"] == "Same text here."


def test_caps_to_top_k():
    contexts = [_ctx(f"chunk {i}", float(i)) for i in range(10)]
    packed = pack_contexts(contexts, top_k=3)
    assert len(packed) == 3
    assert packed[0]["reranker_score"] == 9.0
