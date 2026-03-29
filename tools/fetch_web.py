"""
tools/fetch_web.py

Academic paper fetcher using OpenAlex + arXiv.

Semantic Scholar was dropped — its shared IPs are banned on Render's free tier.
OpenAlex (250M+ scholarly works, free, no auth) replaces it as the primary source.
arXiv covers preprints and CS/ML/physics where OpenAlex may lag.

Both sources run in parallel via ThreadPoolExecutor.
"""

import re
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import requests
import feedparser

logger = logging.getLogger(__name__)

OPENALEX_URL = "https://api.openalex.org/works"
ARXIV_URL    = "https://export.arxiv.org/api/query"

_HEADERS = {
    "Accept":     "application/json",
    "User-Agent": "mini-research-agent/3.0 (academic research tool; mailto:research@example.com)",
}

_OA_SORT = {
    "relevance": None,                    # omit sort — OA default relevance with filter
    "recent":    "publication_year:desc",
    "cited":     "cited_by_count:desc",
}

_ARXIV_SORT = {
    "relevance": ("relevance",     "descending"),
    "recent":    ("submittedDate", "descending"),
    "cited":     ("relevance",     "descending"),
}


def _fetch_openalex(query: str, limit: int = 8, sort_by: str = "recent") -> list[dict]:
    sort_param = _OA_SORT.get(sort_by)

    # OpenAlex filter syntax requires commas between conditions.
    # requests encodes commas as %2C which breaks the filter.
    # Build the URL manually to preserve the raw comma.
    select = ("id,title,abstract_inverted_index,authorships,publication_year,"
              "cited_by_count,open_access,doi,primary_location")

    # Encode only the query value, not the filter structure
    from urllib.parse import quote
    encoded_query = quote(query, safe="")
    filter_str    = f"title_and_abstract.search:{encoded_query},has_abstract:true"

    url = (
        f"{OPENALEX_URL}"
        f"?filter={filter_str}"
        f"&per-page={limit}"
        f"&select={select}"
    )
    if sort_param:
        url += f"&sort={sort_param}"

    try:
        resp = requests.get(url, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        works = resp.json().get("results", [])
    except requests.exceptions.Timeout:
        logger.warning("OpenAlex timed out for: %s", query)
        return []
    except requests.exceptions.RequestException as exc:
        logger.warning("OpenAlex error: %s", exc)
        return []

    results = []
    for w in works:
        title    = w.get("title") or ""
        abstract = _reconstruct_abstract(w.get("abstract_inverted_index"))
        if not title or not abstract:
            continue

        authors = ", ".join(
            a.get("author", {}).get("display_name", "")
            for a in (w.get("authorships") or [])[:3]
            if a.get("author", {}).get("display_name")
        )

        doi = w.get("doi", "")
        oa  = w.get("open_access", {})
        url = oa.get("oa_url") or (f"https://doi.org/{doi}" if doi else "") or w.get("id", "")
        is_open = bool(oa.get("is_oa"))

        results.append({
            "source":         "openalex",
            "title":          title,
            "authors":        authors,
            "year":           w.get("publication_year"),
            "abstract":       abstract,
            "citations":      w.get("cited_by_count", 0),
            "url":            url,
            "is_open_access": is_open,
            "doi":            doi or None,
            "arxiv_id":       None,
            "text":           abstract,
        })
    return results


def _reconstruct_abstract(inverted_index: dict | None) -> str:
    if not inverted_index:
        return ""
    try:
        positions = {}
        for word, pos_list in inverted_index.items():
            for pos in pos_list:
                positions[pos] = word
        return " ".join(positions[i] for i in sorted(positions))
    except Exception:
        return ""


def _fetch_arxiv(query: str, limit: int = 5, sort_by: str = "recent") -> list[dict]:
    sort_by_param, sort_order = _ARXIV_SORT.get(sort_by, ("relevance", "descending"))
    try:
        resp = requests.get(
            ARXIV_URL,
            params={
                "search_query": f"all:{query}",
                "start":        0,
                "max_results":  limit,
                "sortBy":       sort_by_param,
                "sortOrder":    sort_order,
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

        authors   = ", ".join(
            getattr(a, "name", "") for a in getattr(entry, "authors", [])[:3]
        )
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


def _normalise(title: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", title.lower()).strip()[:80]


def _merge_and_rank(oa: list, arxiv: list, sort_by: str = "relevance") -> list[dict]:
    seen, merged = set(), []
    for paper in oa + arxiv:
        key = _normalise(paper["title"])
        if key in seen or len(paper["title"]) < 10:
            continue
        seen.add(key)
        merged.append(paper)

    if sort_by == "cited":
        merged.sort(key=lambda p: -(p["citations"] or -1))
    elif sort_by == "recent":
        merged.sort(key=lambda p: -(p["year"] or 0))
    else:
        merged.sort(key=lambda p: (not p["is_open_access"], -(p["citations"] or -1)))

    return merged


def fetch_papers(
    topic:       str,
    sub_areas:   Optional[list[str]] = None,
    max_results: int = 10,
    sort_by:     str = "relevance",
) -> dict:
    """
    Fetch real academic papers for a research topic.

    Query strategy:
      - If sub_areas provided, use them as primary queries (they are more specific)
      - Always include the broad topic as one query for coverage
      - Deduplicate and cap at 4 queries total
    """
    if sub_areas:
        # Sub-areas are more specific — use them as primary queries
        # Include the broad topic once for coverage
        queries = list(dict.fromkeys(
            sub_areas[:3] + [topic]
        ))[:4]
    else:
        queries = [topic]

    per_query_limit = max(5, max_results // max(len(queries), 1) + 2)

    all_oa, all_arxiv = [], []

    def fetch_oa_query(q):
        return _fetch_openalex(q, limit=per_query_limit, sort_by=sort_by)

    def fetch_arxiv_query(q):
        return _fetch_arxiv(q, limit=per_query_limit, sort_by=sort_by)

    with ThreadPoolExecutor(max_workers=6) as pool:
        oa_futures    = {pool.submit(fetch_oa_query, q): q for q in queries}
        arxiv_futures = {pool.submit(fetch_arxiv_query, q): q for q in queries[:2]}

        for future in as_completed(oa_futures):
            try:
                all_oa.extend(future.result())
            except Exception as exc:
                logger.warning("OpenAlex future failed: %s", exc)

        for future in as_completed(arxiv_futures):
            try:
                all_arxiv.extend(future.result())
            except Exception as exc:
                logger.warning("arXiv future failed: %s", exc)

    sources_used = []
    if all_oa:    sources_used.append("openalex")
    if all_arxiv: sources_used.append("arxiv")

    papers = _merge_and_rank(all_oa, all_arxiv, sort_by)[:max_results]

    logger.info("fetch_papers: queries=%s → %d papers from %s", queries, len(papers), sources_used)

    return {
        "papers":       papers,
        "api_worked":   len(papers) > 0,
        "sources_used": sources_used,
    }


def papers_to_llm_context(papers: list[dict], max_abstract_chars: int = 400) -> str:
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = fetch_papers("Financial Technology", sub_areas=["blockchain"], sort_by="cited")
    print(f"API worked: {result['api_worked']}")
    print(f"Sources:    {result['sources_used']}")
    print(f"Papers:     {len(result['papers'])}\n")
    print(papers_to_llm_context(result["papers"]))
