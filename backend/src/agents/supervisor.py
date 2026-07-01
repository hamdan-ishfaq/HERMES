"""
Supervisor agent — classifies query complexity and sets routing.

What this file does:
    Uses a small local LLM (Ollama 3b) to label each question as simple,
    multi_hop, or synthesis, then writes that label and the next graph node
    into ``ResearchState``. It does not retrieve documents or generate answers.

Where it sits in the HERMES pipeline:
    Runs only on cache miss, between ``cache_check`` and ``research_agent``.
    The complexity label controls whether synthesis runs after research.

What calls this:
    - ``src/agents/graph.py`` — ``supervisor`` node

What this calls:
    - ``src.llm.providers.get_completion`` with ``complexity="classify"``

Routing logic:
  simple    → research_agent only (retrieve + answer, then END)
  multi_hop → research_agent → synthesis_agent
  synthesis → research_agent → synthesis_agent (cross-doc style prompt)
"""

from src.agents.state import ResearchState
from src.llm.providers import get_completion


# All complexity classes still enter research first; synthesis is decided later.
ROUTE_MAP = {
    "simple":    "research_agent",
    "multi_hop": "research_agent",
    "synthesis": "research_agent",
}


def classify_complexity(query: str) -> str:
    """
    Use a fast local LLM to bucket the query into one of three categories.

    The model is instructed to reply with a single word. We scan the response
    for known category names and default to ``"simple"`` if parsing fails.

    Parameters:
        query: Raw user question.

    Returns:
        One of ``"simple"``, ``"multi_hop"``, or ``"synthesis"``.
    """
    prompt = [
        {
            "role": "system",
            "content": (
                "Classify this research query into exactly one category:\n"
                "- simple: single fact, definition, straightforward lookup\n"
                "- multi_hop: requires connecting multiple pieces of information\n"
                "- synthesis: requires comparing or synthesising across multiple documents\n"
                "Reply with ONLY the category word."
            )
        },
        {"role": "user", "content": query}
    ]
    result = get_completion(prompt, complexity="classify").strip().lower()
    # Substring match tolerates extra whitespace or brief model chatter.
    for category in ["simple", "multi_hop", "synthesis"]:
        if category in result:
            return category
    return "simple"


def supervisor_node(state: ResearchState) -> ResearchState:
    """
    LangGraph node — classify complexity and set ``next_agent``.

    Clears any stale error and ensures ``cache_hit`` is False (only the
    cache node sets True).

    Parameters:
        state: Must contain ``query``.

    Returns:
        State update with ``query_complexity``, ``next_agent``, and metadata.
    """
    print(f"\n[Supervisor] Query: {state['query'][:60]}...")

    complexity = classify_complexity(state["query"])
    next_agent = ROUTE_MAP[complexity]

    print(f"[Supervisor] Complexity: {complexity} → routing to: {next_agent}")

    return {
        **state,
        "query_complexity": complexity,
        "next_agent": next_agent,
        "cache_hit": False,
        "error": None,
    }
