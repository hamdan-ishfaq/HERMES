"""
Supervisor agent — classifies query and routes to correct sub-agent.

Routing logic:
  simple    → research_agent only (retrieve + answer)
  multi_hop → research_agent → synthesis_agent
  synthesis → research_agent → synthesis_agent (with cross-doc prompt)
"""

from src.agents.state import ResearchState
from src.llm.providers import get_completion


ROUTE_MAP = {
    "simple":    "research_agent",
    "multi_hop": "research_agent",
    "synthesis": "research_agent",
}


def classify_complexity(query: str) -> str:
    """Use local Ollama to classify query complexity."""
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
    for category in ["simple", "multi_hop", "synthesis"]:
        if category in result:
            return category
    return "simple"


def supervisor_node(state: ResearchState) -> ResearchState:
    """
    Entry point for every query.
    Classifies complexity and sets routing.
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
