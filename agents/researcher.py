"""
agents/researcher.py

Fetches real academic papers via OpenAlex + arXiv.
Semantic Scholar was dropped — IP-banned on Render's shared IPs.
"""

import logging
from orchestration.state import ResearchState
from tools.fetch_web import fetch_papers

logger = logging.getLogger(__name__)


def research_agent(state: ResearchState) -> ResearchState:
    query       = state["query"]
    sub_areas   = [a.get("title", "") for a in state.get("sources", []) if a.get("title")]
    max_results = state.get("max_results") or 10
    sort_by     = state.get("sort_by") or "relevance"

    logger.info("[researcher] Fetching %d papers for: %s (sort: %s, sub_areas: %s)",
                max_results, query, sort_by, sub_areas or "none")

    result     = fetch_papers(
        topic       = query,
        sub_areas   = sub_areas or None,
        max_results = max_results,
        sort_by     = sort_by,
    )
    all_papers = result["papers"]

    if not result["api_worked"]:
        logger.warning("[researcher] Both APIs returned nothing for query: %s", query)

    valid_docs = [p for p in all_papers if _is_valid_doc(p)]
    skipped    = len(all_papers) - len(valid_docs)
    if skipped:
        logger.info("[researcher] Skipped %d papers with insufficient abstracts", skipped)

    logger.info("[researcher] %d valid papers from %s", len(valid_docs), result["sources_used"])

    state["fetched_docs"] = valid_docs
    state["sources"] = [
        {
            "title":          p["title"],
            "authors":        p["authors"],
            "year":           p["year"],
            "url":            p["url"],
            "citations":      p["citations"],
            "is_open_access": p["is_open_access"],
            "source":         p["source"],
        }
        for p in valid_docs
    ]
    return state


def _is_valid_doc(paper: dict) -> bool:
    text = paper.get("abstract") or paper.get("text") or ""
    return len(text.strip()) >= 100 and "\x00" not in text
