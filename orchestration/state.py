from typing import TypedDict, List, Dict, Any


class ResearchState(TypedDict):
    # ── Input ──────────────────────────────────────────────────
    query: str

    # ── Researcher output ──────────────────────────────────────
    # Each dict: {source, title, authors, year, abstract,
    #              citations, url, is_open_access, doi, arxiv_id}
    fetched_docs: List[Dict[str, Any]]

    # ── Memory / retrieval ─────────────────────────────────────
    vector_results: List[Dict[str, Any]]   # [{score, url, chunk}]
    graph_results:  List[Dict[str, Any]]   # reserved for graph memory

    # ── Assembled context string passed to LLM ─────────────────
    final_context: str

    # ── Routing ────────────────────────────────────────────────
    next_step:         str   # supervisor sets this
    analysis_decision: str   # "ready" | "need_more_info"

    # ── Metadata ───────────────────────────────────────────────
    # Structured paper metadata forwarded to the API response
    # so the frontend can render real source cards
    sources:     List[Dict[str, Any]]

    # How many papers to fetch — set by API request, defaults to 10
    max_results: int

    logs: List[str]
