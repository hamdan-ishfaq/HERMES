"""
Cross-encoder reranker — ms-marco-MiniLM (default) or BGE-reranker-v2-m3.
"""

from __future__ import annotations

import os

from sentence_transformers import CrossEncoder

_model = None
_model_name = None


def _resolve_model() -> str:
    return os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")


def get_reranker() -> CrossEncoder:
    global _model, _model_name
    name = _resolve_model()
    if _model is None or _model_name != name:
        device = os.getenv("RERANK_DEVICE", "").strip()
        if not device:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        print(f"Loading reranker {name} on {device}...")
        try:
            _model = CrossEncoder(name, max_length=512, device=device)
        except Exception:
            _model = CrossEncoder(name, max_length=512, device="cpu")
        _model_name = name
        print("Reranker loaded")
    return _model


def rerank(query: str, candidates: list[dict], top_k: int = 3) -> list[dict]:
    if not candidates:
        return []

    reranker = get_reranker()
    pairs = [(query, c["context"]) for c in candidates]
    scores = reranker.predict(pairs)

    for i, candidate in enumerate(candidates):
        candidate["reranker_score"] = round(float(scores[i]), 4)
        candidate["vector_score"] = candidate.get("score", 0)

    reranked = sorted(candidates, key=lambda x: x["reranker_score"], reverse=True)
    return reranked[:top_k]
