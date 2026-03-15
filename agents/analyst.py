"""
agents/analyst.py

Retrieves relevant chunks from vector memory and decides whether
the context is sufficient to proceed to summarization.

Changes from original:
  - Thresholds tuned for real paper abstracts (shorter, denser than scraped HTML)
  - Cleaner decision logic with explicit logging
  - context_builder step is bypassed — analyst assembles context directly
    (context_builder still exists in the graph for future use)
"""

import logging
from orchestration.state import ResearchState
from memory.vector_memory import VectorMemory

logger = logging.getLogger(__name__)

# With real abstracts these thresholds are intentionally lower —
# a single high-quality abstract chunk is worth 10 scraped nav-menu chunks.
MIN_VECTOR_HITS = 3
MIN_AVG_SCORE   = 0.30


def analyst_agent(state: ResearchState, vector_mem: VectorMemory) -> ResearchState:
    query = state["query"]

    vector_hits = vector_mem.search(query, k=10)
    logger.info("[analyst] Retrieved %d vector hits", len(vector_hits))
    state["vector_results"] = vector_hits

    if not vector_hits:
        logger.warning("[analyst] No vector hits — flagging need_more_info")
        state["analysis_decision"] = "need_more_info"
        return state

    avg_score = sum(v["score"] for v in vector_hits) / len(vector_hits)
    logger.info("[analyst] Avg similarity score: %.4f", avg_score)

    if len(vector_hits) < MIN_VECTOR_HITS or avg_score < MIN_AVG_SCORE:
        logger.warning(
            "[analyst] Insufficient hits (%d) or low avg score (%.4f) — need_more_info",
            len(vector_hits), avg_score,
        )
        state["analysis_decision"] = "need_more_info"
        return state

    state["analysis_decision"] = "ready"

    # Assemble context: top chunks with their source URLs
    context_blocks = []
    for v in vector_hits:
        context_blocks.append(f"[SOURCE: {v['url']}]\n{v['chunk']}")

    state["final_context"] = "\n\n".join(context_blocks)
    logger.info("[analyst] Context assembled — ready for summarization")
    return state