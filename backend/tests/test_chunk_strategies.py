"""Unit tests for chunk strategy factory."""

from src.rag.chunk_strategies import SemanticChunker, build_chunker, get_chunk_config


def test_fixed_large_config():
    cfg = get_chunk_config()
    assert cfg.child_chunk_size > 0


def test_semantic_chunker_produces_parents():
    chunker = SemanticChunker(get_chunk_config())
    text = (
        "Retrieval-Augmented Generation combines search with generation.\n\n"
        "Hybrid search uses dense and sparse vectors together.\n\n"
        "Cross-encoders rerank candidate passages for better precision."
    ) * 3
    parents = chunker.chunk(text, {"source": "test"})
    assert len(parents) >= 1
    assert all(len(p.children) >= 1 for p in parents)


def test_build_chunker_returns_hierarchical_by_default():
    chunker = build_chunker()
    assert hasattr(chunker, "chunk")
