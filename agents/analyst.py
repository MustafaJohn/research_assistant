"""
agents/analyst.py

Retrieves relevant chunks from vector memory using semantic search
and assembles context for the summarizer.

RAG is preserved but lightweight — vector search over a small corpus
(10-20 papers) is near-instant after batch embedding is done.
"""

import logging
from orchestration.state import ResearchState
from memory.vector_memory import VectorMemory

logger = logging.getLogger(__name__)

MIN_HITS      = 3
MIN_AVG_SCORE = 0.25


def analyst_agent(state: ResearchState, vector_mem: VectorMemory) -> ResearchState:
    query = state["query"]

    vector_hits = vector_mem.search(query, k=12)
    logger.info("[analyst] %d vector hits, avg score: %.4f",
                len(vector_hits),
                sum(v["score"] for v in vector_hits) / max(len(vector_hits), 1))

    state["vector_results"] = vector_hits

    if len(vector_hits) < MIN_HITS:
        logger.warning("[analyst] Insufficient hits — using all fetched docs as fallback")
        # Fallback: use all paper abstracts directly
        context_blocks = [
            f"[SOURCE: {p['url']}]\n{p.get('abstract') or p.get('text', '')}"
            for p in state.get("fetched_docs", [])
        ]
        state["final_context"]     = "\n\n".join(context_blocks)
        state["analysis_decision"] = "ready"
        return state

    avg_score = sum(v["score"] for v in vector_hits) / len(vector_hits)
    if avg_score < MIN_AVG_SCORE:
        logger.warning("[analyst] Low avg score %.4f — proceeding anyway", avg_score)

    context_blocks = [f"[SOURCE: {v['url']}]\n{v['chunk']}" for v in vector_hits]
    state["final_context"]     = "\n\n".join(context_blocks)
    state["analysis_decision"] = "ready"
    return state
