"""
tools/fetch_web.py

Replaces DuckDuckGo + BeautifulSoup with two purpose-built academic APIs:

  • Semantic Scholar  — 200 M+ papers, abstracts, citation counts, open-access PDFs
  • arXiv             — open-access preprints, always freely accessible

Both are free, require no API key, and return structured JSON/Atom data
instead of HTML pages that may be paywalled or return 404.
"""

import time
import logging
import re
from typing import Optional

import requests
import feedparser

logger = logging.getLogger(__name__)

SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
ARXIV_URL            = "https://export.arxiv.org/api/query"

_HEADERS = {
    "Accept":     "application/json",
    "User-Agent": "mini-research-agent/2.0 (academic research tool)",
}


# ─────────────────────────────────────────────────────────────
# Semantic Scholar
# ─────────────────────────────────────────────────────────────

def _fetch_semantic_scholar(query: str, limit: int = 6) -> list[dict]:
    fields = "title,abstract,authors,year,citationCount,openAccessPdf,externalIds,url"
    try:
        resp = requests.get(
            SEMANTIC_SCHOLAR_URL,
            params={"query": query, "fields": fields, "limit": limit},
            headers=_HEADERS,
            timeout=12,
        )
        resp.raise_for_status()
        papers = resp.json().get("data", [])
    except requests.exceptions.Timeout:
        logger.warning("Semantic Scholar timed out for: %s", query)
        return []
    except requests.exceptions.RequestException as exc:
        logger.warning("Semantic Scholar error: %s", exc)
        return []

    results = []
    for p in papers:
        if not p.get("title") or not p.get("abstract"):
            continue
        open_pdf = (p.get("openAccessPdf") or {})
        url      = open_pdf.get("url") or f"https://www.semanticscholar.org/paper/{p.get('paperId','')}"
        results.append({
            "source":         "semantic_scholar",
            "title":          p["title"],
            "authors":        ", ".join(a["name"] for a in (p.get("authors") or [])[:3]),
            "year":           p.get("year"),
            "abstract":       p["abstract"],
            "citations":      p.get("citationCount", 0),
            "url":            url,
            "is_open_access": bool(open_pdf.get("url")),
            "doi":            (p.get("externalIds") or {}).get("DOI"),
            "arxiv_id":       (p.get("externalIds") or {}).get("ArXiv"),
            # text field for backwards-compat with memory_agent chunking
            "text":           p["abstract"],
        })
    return results


# ─────────────────────────────────────────────────────────────
# arXiv
# ─────────────────────────────────────────────────────────────

def _fetch_arxiv(query: str, limit: int = 5) -> list[dict]:
    try:
        resp = requests.get(
            ARXIV_URL,
            params={
                "search_query": f"all:{query}",
                "start":        0,
                "max_results":  limit,
                "sortBy":       "relevance",
                "sortOrder":    "descending",
            },
            timeout=12,
        )
        resp.raise_for_status()
        feed = feedparser.parse(resp.text)
    except Exception as exc:
        logger.warning("arXiv error: %s", exc)
        return []

    results = []
    for entry in feed.entries:
        arxiv_id = getattr(entry, "id", "").split("/abs/")[-1].strip()
        title    = getattr(entry, "title",   "").replace("\n", " ").strip()
        abstract = getattr(entry, "summary", "").replace("\n", " ").strip()
        if not title or not abstract or not arxiv_id:
            continue

        authors   = ", ".join(getattr(a, "name", "") for a in getattr(entry, "authors", [])[:3])
        published = getattr(entry, "published", "")
        year      = int(published[:4]) if published else None

        results.append({
            "source":         "arxiv",
            "title":          title,
            "authors":        authors,
            "year":           year,
            "abstract":       abstract,
            "citations":      None,
            "url":            f"https://arxiv.org/abs/{arxiv_id}",
            "is_open_access": True,
            "doi":            None,
            "arxiv_id":       arxiv_id,
            "text":           abstract,
        })
    return results


# ─────────────────────────────────────────────────────────────
# Merge + deduplicate
# ─────────────────────────────────────────────────────────────

def _normalise(title: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", title.lower()).strip()[:80]


def _merge_and_rank(ss: list, arxiv: list) -> list[dict]:
    seen, merged = set(), []
    for paper in ss + arxiv:
        key = _normalise(paper["title"])
        if key in seen or len(paper["title"]) < 10:
            continue
        seen.add(key)
        merged.append(paper)

    merged.sort(key=lambda p: (
        not p["is_open_access"],
        -(p["citations"] or -1),
    ))
    return merged


# ─────────────────────────────────────────────────────────────
# Public entrypoint  — called by researcher.py
# ─────────────────────────────────────────────────────────────

def fetch_papers(
    topic:       str,
    sub_areas:   Optional[list[str]] = None,
    max_results: int = 10,
) -> dict:
    """
    Fetch real academic papers for a research topic.

    Args:
        topic:       Broad research topic string
        sub_areas:   Optional sub-area keywords (from Stage 1 exploration)
        max_results: Cap on returned papers after merging both sources

    Returns:
        {
            "papers":      list[dict],   # merged, ranked paper dicts
            "api_worked":  bool,
            "sources_used": list[str],
        }
    """
    queries = list(dict.fromkeys(
        [topic] + [f"{topic} {a}" for a in (sub_areas or [])[:3]]
    ))[:4]

    all_ss, all_arxiv, sources_used = [], [], []

    for i, query in enumerate(queries):
        if i > 0:
            time.sleep(0.3)   # polite rate-limiting

        ss = _fetch_semantic_scholar(query, limit=5)
        if ss:
            all_ss.extend(ss)
            if "semantic_scholar" not in sources_used:
                sources_used.append("semantic_scholar")

        if i < 2:
            ax = _fetch_arxiv(query, limit=4)
            if ax:
                all_arxiv.extend(ax)
                if "arxiv" not in sources_used:
                    sources_used.append("arxiv")

    papers = _merge_and_rank(all_ss, all_arxiv)[:max_results]

    return {
        "papers":      papers,
        "api_worked":  len(papers) > 0,
        "sources_used": sources_used,
    }


# ─────────────────────────────────────────────────────────────
# LLM context formatter  — called by summarizer / analyst
# ─────────────────────────────────────────────────────────────

def papers_to_llm_context(papers: list[dict], max_abstract_chars: int = 400) -> str:
    """
    Format paper list into a clean numbered string for LLM prompts.
    Keeps each abstract short to stay within context limits.
    """
    if not papers:
        return "No papers could be fetched from academic APIs."

    lines = []
    for i, p in enumerate(papers, 1):
        author_year   = f"{p['authors']}, {p['year']}" if p.get("year") else p.get("authors", "")
        snippet       = (p.get("abstract") or "")[:max_abstract_chars].rstrip()
        if len(p.get("abstract") or "") > max_abstract_chars:
            snippet += "…"
        citation_note = f" [{p['citations']:,} citations]" if p.get("citations") else ""
        access_note   = " [OPEN ACCESS]" if p["is_open_access"] else ""

        lines.append(
            f"{i}. \"{p['title']}\" ({author_year}){citation_note}{access_note}\n"
            f"   URL: {p['url']}\n"
            f"   Abstract: {snippet}"
        )
    return "\n\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Quick standalone test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    result = fetch_papers("Financial Technology", sub_areas=["blockchain", "machine learning"])
    print(f"API worked: {result['api_worked']}")
    print(f"Sources:    {result['sources_used']}")
    print(f"Papers:     {len(result['papers'])}\n")
    print(papers_to_llm_context(result["papers"]))