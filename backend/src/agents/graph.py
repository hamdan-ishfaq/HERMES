"""
LangGraph state machine — wires all research agents into one pipeline.

What this file does:
    Builds a compiled LangGraph ``StateGraph`` that orchestrates cache lookup,
    query classification, retrieval + draft generation, and optional
    synthesis. Exposes ``run_research`` as the single function other modules
    call to answer a question.

Where it sits in the HERMES pipeline:
    Core "brain" between the HTTP layer and the RAG stack. When a user hits
    ``POST /api/research``, the router calls ``run_research`` defined here.

What calls this:
    - ``src/routers/research.py`` — production API path
    - ``src/evaluation/ragas_eval.py`` — scores answers against golden Q&A
    - ``__main__`` block — local smoke tests

What this calls:
    - ``src.agents.cache_check.cache_check_node`` — semantic cache gate
    - ``src.agents.supervisor.supervisor_node`` — complexity classification
    - ``src.agents.research.research_node`` — retrieve + draft answer
    - ``src.agents.synthesis.synthesis_node`` — refine multi-hop answers

Graph structure:
  START → cache_check → [END on hit | supervisor]
          supervisor → research_agent → [END or synthesis_agent] → END

Conditional routing reads ``state["next_agent"]`` set by each node.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from src.agents.state import ResearchState
from src.agents.cache_check import cache_check_node
from src.agents.supervisor import supervisor_node
from src.agents.research import research_node
from src.agents.synthesis import synthesis_node


def route_after_cache(state: ResearchState) -> str:
    """
    LangGraph routing function after the cache-check node.

    Parameters:
        state: Current graph state; ``next_agent`` is either ``"supervisor"``
               or ``"END"`` (cache hit).

    Returns:
        The name of the next node or ``"END"`` to terminate the graph.
    """
    return state["next_agent"]


def route_after_supervisor(state: ResearchState) -> str:
    """
    LangGraph routing function after the supervisor classifies the query.

    Parameters:
        state: ``next_agent`` is always ``"research_agent"`` today.

    Returns:
        Next node name from ``state["next_agent"]``.
    """
    return state["next_agent"]


def route_after_research(state: ResearchState) -> str:
    """
    LangGraph routing function after retrieval and draft generation.

    Parameters:
        state: ``next_agent`` is ``"synthesis_agent"`` for complex queries or
               ``"END"`` for simple ones.

    Returns:
        Next node name from ``state["next_agent"]``.
    """
    return state["next_agent"]


def build_graph():
    """
    Construct and compile the LangGraph research pipeline.

    Registers four agent nodes and wires conditional edges based on each
    node's ``next_agent`` field. Attaches an in-memory checkpointer so
    LangGraph can resume thread state by ``session_id``.

    Returns:
        Compiled graph ready for ``.invoke(initial_state, config=...)``.
    """
    graph = StateGraph(ResearchState)

    # Register each pipeline stage as a named node.
    graph.add_node("cache_check", cache_check_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("research_agent", research_node)
    graph.add_node("synthesis_agent", synthesis_node)

    # Every query enters through the semantic cache gate first.
    graph.add_edge(START, "cache_check")

    # Cache hit ends immediately; miss continues to the supervisor.
    graph.add_conditional_edges(
        "cache_check",
        route_after_cache,
        {
            "supervisor": "supervisor",
            "END": END,
        }
    )

    # Supervisor always routes to the research agent (complexity is stored in state).
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "research_agent": "research_agent",
        }
    )

    # Simple queries end here; multi-hop/synthesis go to synthesis_agent.
    graph.add_conditional_edges(
        "research_agent",
        route_after_research,
        {
            "synthesis_agent": "synthesis_agent",
            "END": END,
        }
    )

    # Synthesis is always the final step for routes that reach it.
    graph.add_edge("synthesis_agent", END)

    # MemorySaver enables per-session checkpointing via thread_id in config.
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


# Process-wide singleton — building the graph loads models and is expensive.
_graph = None


def get_graph():
    """
    Return the shared compiled graph, creating it on first use.

    Returns:
        Compiled LangGraph instance from ``build_graph()``.
    """
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_research(query: str, session_id: str | None = None) -> dict:
    """
    Main entry point for running a research query through the full pipeline.

    Initializes ``ResearchState`` with empty retrieval/generation fields,
    invokes the graph once, and returns the final state dict (answer,
    citations, cache_hit, model_used, etc.).

    Parameters:
        query: Natural-language question from the user.
        session_id: Optional LangGraph thread id for conversational checkpointing;
                    a random UUID is generated if omitted.

    Returns:
        Final ``ResearchState`` dict after all nodes have run.
    """
    import uuid
    if not session_id:
        session_id = str(uuid.uuid4())

    graph = get_graph()

    # Fresh state for each invoke — nodes merge updates into this dict.
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

    # thread_id ties this run to a LangGraph checkpoint thread.
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
