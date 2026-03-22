"""
agents/memory_agent.py

Chunks all paper abstracts and stores them in vector memory using
batch embedding — one model inference for all chunks combined.
"""

import logging
from memory.chunker import chunk_text
from memory.vector_memory import VectorMemory
from orchestration.state import ResearchState

logger = logging.getLogger(__name__)


def memory_agent(state: ResearchState, vector_mem: VectorMemory) -> ResearchState:
    docs = state["fetched_docs"]
    logger.info("[memory] Chunking %d papers…", len(docs))

    # Collect all (url, chunk_id, text) tuples across all papers
    all_entries = []
    for doc in docs:
        text = doc.get("abstract") or doc.get("text") or ""
        url  = doc.get("url", "unknown")
        if not text.strip():
            continue
        for chunk_id, chunk_text_ in chunk_text(text):
            all_entries.append((url, chunk_id, chunk_text_))

    # Single batch embedding call for all chunks
    stored = vector_mem.add_chunks_batch(all_entries)
    logger.info("[memory] %d chunks stored (batch embedded), index size: %d",
                stored, vector_mem.size())
    return state
