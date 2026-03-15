"""
orchestration/graph.py

Your DAG structure is unchanged — same nodes, same edges, same supervisor routing.

The one architectural fix: VectorMemory is now instantiated inside build_graph()
and passed via closure. This means each call to graph.invoke() gets the same
fresh memory instance for the duration of that query — no cross-query contamination.

If you ever move to a multi-worker FastAPI setup, instantiate VectorMemory
inside the route handler and pass it into the graph invocation instead.
"""

from langgraph.graph import StateGraph, END

from orchestration.state import ResearchState
from memory.vector_memory import VectorMemory
from agents.supervisor import supervisor_agent
from agents.researcher import research_agent
from agents.memory_agent import memory_agent
from agents.analyst import analyst_agent
from agents.context_builder import context_builder_agent
from agents.summarizer import summarizer_agent


def build_graph():
    # Fresh memory per graph build — no disk persistence, no cross-contamination
    vector_mem = VectorMemory()

    graph = StateGraph(ResearchState)

    # ── Nodes ──────────────────────────────────────────────────
    graph.add_node("supervisor", supervisor_agent)
    graph.add_node("research",   research_agent)
    graph.add_node("memory",     lambda state: memory_agent(state, vector_mem))
    graph.add_node("analysis",   lambda state: analyst_agent(state, vector_mem))
    graph.add_node("context",    context_builder_agent)
    graph.add_node("summarize",  summarizer_agent)

    # ── Entry ──────────────────────────────────────────────────
    graph.set_entry_point("supervisor")

    # ── Supervisor is the only router ─────────────────────────
    graph.add_conditional_edges(
        "supervisor",
        lambda state: state["next_step"],
        {
            "research":  "research",
            "context":   "context",
            "summarize": "summarize",
            "end":       END,
        },
    )

    # ── Execution chain (unchanged) ────────────────────────────
    graph.add_edge("research",  "memory")
    graph.add_edge("memory",    "analysis")
    graph.add_edge("analysis",  "supervisor")   # analyst reports back to supervisor
    graph.add_edge("context",   "supervisor")   # context flows back to supervisor
    graph.add_edge("summarize", END)

    return graph.compile()