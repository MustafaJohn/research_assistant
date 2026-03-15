"""
agents/researcher.py

Fetches real academic papers via Semantic Scholar + arXiv.
Replaces the old DuckDuckGo + BeautifulSoup approach.

Changes from original:
  - FetchWebTool() → fetch_papers()
  - Documents now carry structured metadata (title, authors, year, citations, url)
  - Sources stored in state["sources"] for the API response / frontend
  - is_valid_text() kept for defensive filtering of empty abstracts
"""

import logging
from orchestration.state import ResearchState
from tools.fetch_web import fetch_papers

logger = logging.getLogger(__name__)


def research_agent(state: ResearchState) -> ResearchState:
    query     = state["query"]
    sub_areas = [a.get("title", "") for a in state.get("sources", []) if a.get("title")]

    logger.info("[researcher] Fetching papers for: %s", query)

    result      = fetch_papers(topic=query, sub_areas=sub_areas or None, max_results=12)
    all_papers  = result["papers"]

    if not result["api_worked"]:
        logger.warning("[researcher] Both APIs returned nothing for query: %s", query)

    # Filter out anything with a suspiciously short or empty abstract
    valid_docs = [p for p in all_papers if _is_valid_doc(p)]
    skipped    = len(all_papers) - len(valid_docs)

    if skipped:
        logger.info("[researcher] Skipped %d papers with insufficient abstracts", skipped)

    logger.info(
        "[researcher] %d valid papers from %s",
        len(valid_docs),
        result["sources_used"],
    )

    state["fetched_docs"] = valid_docs
    # Store structured metadata for the API response
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
    if len(text.strip()) < 100:
        return False
    if "\x00" in text:
        return False
    return True