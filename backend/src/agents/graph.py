"""
LangGraph state machine — wires all research agents into one pipeline.

Graph structure:
  START → cache_check → [END on hit | supervisor]
          supervisor → query_rewrite → research_agent → [END or synthesis_agent] → END
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from src.agents.state import ResearchState
from src.agents.cache_check import cache_check_node
from src.agents.supervisor import supervisor_node
from src.agents.query_rewrite import query_rewrite_node
from src.agents.research import research_node
from src.agents.synthesis import synthesis_node


def route_after_cache(state: ResearchState) -> str:
    return state["next_agent"]


def route_after_supervisor(state: ResearchState) -> str:
    return "query_rewrite"


def route_after_rewrite(state: ResearchState) -> str:
    return state.get("next_agent") or "research_agent"


def route_after_research(state: ResearchState) -> str:
    return state["next_agent"]


def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("cache_check", cache_check_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("query_rewrite", query_rewrite_node)
    graph.add_node("research_agent", research_node)
    graph.add_node("synthesis_agent", synthesis_node)

    graph.add_edge(START, "cache_check")

    graph.add_conditional_edges(
        "cache_check",
        route_after_cache,
        {"supervisor": "supervisor", "END": END},
    )
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {"query_rewrite": "query_rewrite"},
    )
    graph.add_conditional_edges(
        "query_rewrite",
        route_after_rewrite,
        {"research_agent": "research_agent"},
    )
    graph.add_conditional_edges(
        "research_agent",
        route_after_research,
        {"synthesis_agent": "synthesis_agent", "END": END},
    )
    graph.add_edge("synthesis_agent", END)

    return graph.compile(checkpointer=MemorySaver())


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_research(
    query: str,
    session_id: str | None = None,
    messages: list[dict] | None = None,
    user_id: str | None = None,
) -> dict:
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
        "messages": messages or [],
        "rewritten_query": "",
        "tool_trace": [],
        "user_id": user_id,
        "cache_hit": False,
        "model_used": "",
        "error": None,
    }
    config = {"configurable": {"thread_id": session_id}}
    return graph.invoke(initial_state, config=config)
