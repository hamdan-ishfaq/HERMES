"""
Post-retrieval context packing — dedup and cap before LLM prompt.

Lightweight EAg-RAG-style post-processor: normalize-text dedup, sort by
reranker score, optionally order by document position, cap to top-N.
"""

from __future__ import annotations

import os
import re

CONTEXT_PACK_TOP_K = int(os.getenv("CONTEXT_PACK_TOP_K", "5"))


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def pack_contexts(contexts: list[dict], top_k: int | None = None) -> list[dict]:
    """
    Deduplicate by normalized context text, sort by score, cap to ``top_k``.

    When ``page_num`` is present in metadata, secondary sort preserves
    reading order within the same source.
    """
    if not contexts:
        return []

    limit = top_k if top_k is not None else CONTEXT_PACK_TOP_K
    seen: set[str] = set()
    deduped: list[dict] = []

    ranked = sorted(
        contexts,
        key=lambda c: (
            -c.get("reranker_score", c.get("score", 0)),
            c.get("metadata", {}).get("page_num") or 0,
        ),
    )

    for ctx in ranked:
        key = _normalize(ctx.get("context", ""))
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(ctx)

    return deduped[:limit]
