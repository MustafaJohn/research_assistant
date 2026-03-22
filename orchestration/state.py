from typing import TypedDict, List, Dict, Any


class ResearchState(TypedDict):
    # ── Input ──────────────────────────────────────────────────
    query:   str
    sort_by: str   # "relevance" | "recent" | "cited"

    # ── Researcher output ──────────────────────────────────────
    fetched_docs: List[Dict[str, Any]]

    # ── Memory / retrieval ─────────────────────────────────────
    vector_results: List[Dict[str, Any]]
    graph_results:  List[Dict[str, Any]]

    # ── Assembled context string passed to LLM ─────────────────
    final_context: str

    # ── Routing ────────────────────────────────────────────────
    next_step:         str
    analysis_decision: str

    # ── Metadata ───────────────────────────────────────────────
    sources:     List[Dict[str, Any]]
    max_results: int
    logs:        List[str]
