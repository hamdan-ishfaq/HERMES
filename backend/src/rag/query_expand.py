"""
Multi-query expansion — generate paraphrases for standalone retrieval.

Gated by ``HERMES_MULTI_QUERY=1``. Each variant runs hybrid search; results
are merged by parent_id keeping the best reranker score per parent.
"""

from __future__ import annotations

import os

from src.llm.providers import get_completion


def expand_queries(query: str) -> list[str]:
    """
    Return the original query plus up to ``HERMES_MULTI_QUERY_COUNT`` paraphrases.
    Falls back to ``[query]`` on LLM failure or when multi-query is disabled.
    """
    if os.getenv("HERMES_MULTI_QUERY", "1") == "0":
        return [query]

    count = int(os.getenv("HERMES_MULTI_QUERY_COUNT", "2"))
    prompt = [
        {
            "role": "system",
            "content": (
                "Generate search-query paraphrases for retrieval. "
                "Output one paraphrase per line, no numbering or bullets. "
                "Do not repeat the original wording."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Original question: {query}\n\n"
                f"Write exactly {count} alternative search queries:"
            ),
        },
    ]

    try:
        raw = get_completion(prompt, complexity="classify")
        variants = [
            line.strip().strip('"').strip("'")
            for line in raw.splitlines()
            if line.strip() and len(line.strip()) >= 5
        ]
        seen = {query.lower()}
        out = [query]
        for v in variants:
            key = v.lower()
            if key not in seen:
                seen.add(key)
                out.append(v)
            if len(out) >= count + 1:
                break
        return out
    except Exception as e:
        print(f"[MultiQuery] expansion failed ({e}) — using original query")
        return [query]


def merge_contexts_by_parent(context_lists: list[list[dict]]) -> list[dict]:
    """Merge retrieval results; keep highest reranker score per parent_id."""
    best: dict[str, dict] = {}
    for contexts in context_lists:
        for ctx in contexts:
            pid = ctx.get("parent_id") or ctx.get("context", "")[:80]
            score = ctx.get("reranker_score", ctx.get("score", 0))
            prev = best.get(pid)
            prev_score = (
                prev.get("reranker_score", prev.get("score", 0)) if prev else -1
            )
            if prev is None or score > prev_score:
                best[pid] = ctx
    return sorted(
        best.values(),
        key=lambda c: c.get("reranker_score", c.get("score", 0)),
        reverse=True,
    )
