"""
Dense embedding providers — Ollama nomic-embed-text (default) or local BGE-m3.
"""

from __future__ import annotations

import os

from qdrant_client.models import SparseVector

_embed_model = None
_sparse_model = None

OLLAMA_MODEL = "nomic-embed-text"
BGE_MODEL = "BAAI/bge-m3"

DIMENSIONS = {
    "ollama": 768,
    "nomic": 768,
    "bge-m3": 1024,
    "bge_m3": 1024,
}


def embed_model_name() -> str:
    return os.getenv("EMBED_MODEL", "ollama").strip().lower()


def vector_dimension() -> int:
    name = embed_model_name()
    if name in DIMENSIONS:
        return DIMENSIONS[name]
    return int(os.getenv("VECTOR_DIM", "768"))


def _use_bge() -> bool:
    return embed_model_name() in ("bge-m3", "bge_m3", "bge")


def _get_bge_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer

        device = os.getenv("EMBED_DEVICE", "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu")
        try:
            _embed_model = SentenceTransformer(BGE_MODEL, device=device)
        except Exception:
            _embed_model = SentenceTransformer(BGE_MODEL, device="cpu")
        print(f"BGE-m3 loaded on {_embed_model.device}")
    return _embed_model


def dense_embed(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    if _use_bge():
        model = _get_bge_model()
        vectors = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=int(os.getenv("EMBED_BATCH_SIZE", "32")),
        )
        return [v.tolist() for v in vectors]

    import httpx

    base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
    model = os.getenv("OLLAMA_EMBED_MODEL", OLLAMA_MODEL)
    vectors = []
    for text in texts:
        resp = httpx.post(
            f"{base}/api/embed",
            json={"model": model, "input": text},
            timeout=float(os.getenv("OLLAMA_EMBED_TIMEOUT", "120")),
        )
        resp.raise_for_status()
        vectors.append(resp.json()["embeddings"][0])
    return vectors


def sparse_embed(texts: list[str]) -> list[SparseVector]:
    global _sparse_model
    if not texts:
        return []
    if _sparse_model is None:
        from fastembed import SparseTextEmbedding

        _sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
    results = []
    for embedding in _sparse_model.embed(texts):
        results.append(
            SparseVector(
                indices=embedding.indices.tolist(),
                values=embedding.values.tolist(),
            )
        )
    return results
