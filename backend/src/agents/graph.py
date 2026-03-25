"""
LangGraph state machine — wires all agents together.

Graph structure:
  START → supervisor → research_agent → [END or synthesis_agent] → END

Conditional routing based on state["next_agent"].
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from src.agents.state import ResearchState
from src.agents.supervisor import supervisor_node
from src.agents.research import research_node
from src.agents.synthesis import synthesis_node


def route_after_supervisor(state: ResearchState) -> str:
    return state["next_agent"]


def route_after_research(state: ResearchState) -> str:
    return state["next_agent"]


def build_graph():
    """Build and compile the LangGraph research pipeline."""
    graph = StateGraph(ResearchState)

    # Add nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("research_agent", research_node)
    graph.add_node("synthesis_agent", synthesis_node)

    # Entry point
    graph.add_edge(START, "supervisor")

    # Supervisor routes to research agent
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "research_agent": "research_agent",
        }
    )

    # Research routes to synthesis or END
    graph.add_conditional_edges(
        "research_agent",
        route_after_research,
        {
            "synthesis_agent": "synthesis_agent",
            "END": END,
        }
    )

    # Synthesis always ends
    graph.add_edge("synthesis_agent", END)

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


# Singleton graph instance
_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_research(query: str, session_id: str | None = None) -> dict:
    """
    Main entry point for running a research query through the full pipeline.
    Returns final state with answer and citations.
    """
    import uuid
    if not session_id:
        session_id = str(uuid.uuid4())

    graph = get_graph()

    initial_state: ResearchState = {
        "query": query,
        "query_complexity": "",
        "next_agent": "",
        "retrieved_contexts": [],
        "citations": [],
        "draft_answer": "",
        "final_answer": "",
        "cache_hit": False,
        "model_used": "",
        "error": None,
    }

    config = {"configurable": {"thread_id": session_id}}
    final_state = graph.invoke(initial_state, config=config)
    return final_state


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    test_queries = [
        ("simple",    "What is retrieval augmented generation?"),
        ("multi_hop", "How does hybrid search improve over dense-only retrieval and why does reranking help?"),
    ]

    for expected_type, query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query ({expected_type}): {query}")
        print('='*60)

        result = run_research(query)

        print(f"\nComplexity detected: {result['query_complexity']}")
        print(f"Model used: {result['model_used']}")
        print(f"Cache hit: {result['cache_hit']}")
        print(f"Citations: {len(result['citations'])}")
        print(f"\nFinal Answer:\n{result['final_answer']}")
        print(f"\nCitations:")
        for c in result['citations']:
            print(f"  - {c['source']} p.{c['page_num']} (score: {c['score']:.3f})")
