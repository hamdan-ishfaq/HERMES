"""
ResearchState — the single shared object passed between all agents.

Every agent reads from this state and writes back to it.
LangGraph manages the transitions between agents based on `next_agent`.
"""

from typing import TypedDict, Optional


class Citation(TypedDict):
    source: str
    page_num: Optional[int]
    context: str
    score: float


class ResearchState(TypedDict):
    # Input
    query: str

    # Routing
    query_complexity: str        # "simple" | "multi_hop" | "synthesis"
    next_agent: str              # which agent runs next

    # Retrieval
    retrieved_contexts: list[dict]   # raw results from retriever
    citations: list[Citation]        # formatted citations

    # Generation
    draft_answer: str
    final_answer: str

    # Metadata
    cache_hit: bool
    model_used: str
    error: Optional[str]
