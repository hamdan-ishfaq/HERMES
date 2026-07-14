"""
Query-rewrite node — expand follow-ups using prior turns before retrieval.

What this file does:
    If multi-turn memory is enabled and prior messages exist, asks a small LLM
    to rewrite the latest user question into a standalone search query.
    Otherwise passes the original query through unchanged.

Where it sits:
    After supervisor, before research_agent in the LangGraph pipeline.
"""

from __future__ import annotations

import os

from src.agents.state import ResearchState
from src.llm.providers import get_completion

HERMES_MULTI_TURN = os.getenv("HERMES_MULTI_TURN", "1") != "0"


def query_rewrite_node(state: ResearchState) -> ResearchState:
    """
    Set rewritten_query for retrieval; leave original query for cache keying.
    """
    query = state["query"]
    messages = state.get("messages") or []

    if not HERMES_MULTI_TURN or not messages:
        print("[Rewrite] passthrough (no history or multi-turn disabled)")
        return {**state, "rewritten_query": query, "next_agent": "research_agent"}

    history_lines = []
    for m in messages[-8:]:
        role = m.get("role", "user") if isinstance(m, dict) else "user"
        content = m.get("content", "") if isinstance(m, dict) else str(m)
        history_lines.append(f"{role}: {content}")
    history = "\n".join(history_lines)

    prompt = [
        {
            "role": "system",
            "content": (
                "Rewrite the follow-up question into a standalone search query "
                "that includes any needed entities from the prior conversation. "
                "Output ONLY the rewritten query text. No quotes or explanation."
            ),
        },
        {
            "role": "user",
            "content": f"Prior turns:\n{history}\n\nFollow-up: {query}\n\nStandalone query:",
        },
    ]

    try:
        rewritten = get_completion(prompt, complexity="classify").strip().strip('"')
        if not rewritten or len(rewritten) < 3:
            rewritten = query
    except Exception as e:
        print(f"[Rewrite] LLM failed ({e}) — using original query")
        rewritten = query

    print(f"[Rewrite] '{query[:40]}...' → '{rewritten[:60]}...'")
    return {
        **state,
        "rewritten_query": rewritten,
        "next_agent": "research_agent",
    }
