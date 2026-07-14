"""
ResearchState — the shared data contract between all LangGraph agents.

What this file does:
    Defines ``ResearchState`` and ``Citation`` TypedDicts that describe every
    field passed through the agent pipeline. LangGraph merges partial updates
    from each node into this single state object.

Where it sits in the HERMES pipeline:
    Imported by every agent node and by ``graph.py`` when building/running the
    graph. Think of it as the "clipboard" all agents read and write.

What calls this:
    - ``src/agents/graph.py`` — initial state and type hints
    - ``src/agents/cache_check.py``, ``supervisor.py``, ``research.py``,
      ``synthesis.py`` — each node returns a partial ``ResearchState`` update

What this calls:
    Nothing — pure type definitions only.

Every agent reads from this state and writes back to it.
LangGraph manages the transitions between agents based on ``next_agent``.
"""

from typing import TypedDict, Optional


class Citation(TypedDict, total=False):
    """
    One source reference attached to an answer for the UI.

    Fields vary by source type: PDFs use ``page_num``; web/YouTube use
    ``url`` and ``title``; YouTube may also set ``timestamp``.
    """
    source: str
    title: Optional[str]          # web page / video title (None for PDFs)
    url: Optional[str]            # link for web + YouTube sources
    page_num: Optional[int]      # PDF page
    timestamp: Optional[str]     # YouTube timestamp (e.g. "2:05")
    context: str
    score: float


class ResearchState(TypedDict):
    """
    Complete state bag for one research run through the LangGraph pipeline.

    Nodes return dict spreads like ``{**state, "final_answer": "..."}`` so
    unchanged keys are preserved. Routing uses ``next_agent``; metadata
    fields support logging and the frontend response.
    """
    # Input
    query: str

    # Routing — set by supervisor; consumed by graph conditional edges
    query_complexity: str        # "simple" | "multi_hop" | "synthesis"
    next_agent: str              # which agent runs next (or "END")

    # Retrieval — populated by research_node (or cache_check on hit)
    retrieved_contexts: list[dict]   # raw retriever results with scores/metadata
    citations: list[Citation]        # UI-friendly citation list

    # Generation — draft from research; final may be refined by synthesis
    draft_answer: str
    final_answer: str

    # Multi-turn memory (HE-2) — prior turns + rewritten retrieval query
    messages: list[dict]
    rewritten_query: str

    # Tool calling (HE-7) — list of {name, args, ok, summary}
    tool_trace: list[dict]

    # Workspace ACL (HE-4) — string form of user id for Qdrant filter
    user_id: Optional[str]

    # Metadata — surfaced in API response and QueryLog
    cache_hit: bool
    model_used: str
    error: Optional[str]
