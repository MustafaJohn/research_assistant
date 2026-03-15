"""
agents/memory_agent.py

Chunks fetched paper abstracts and stores them in vector memory.

Your original logic is preserved exactly. The only change is that
fetched_docs now contains structured paper dicts (with an "abstract" /
"text" key) rather than raw scraped HTML, so the chunks are cleaner.
"""

import logging
from memory.chunker import chunk_text
from memory.vector_memory import VectorMemory
from orchestration.state import ResearchState

logger = logging.getLogger(__name__)


def memory_agent(state: ResearchState, vector_mem: VectorMemory) -> ResearchState:
    logger.info("[memory] Storing fetched documents into memory...")
    all_chunks = []

    for doc in state["fetched_docs"]:
        # Use abstract as the text to chunk — clean, focused, no HTML noise
        text = doc.get("abstract") or doc.get("text") or ""
        url  = doc.get("url", "unknown")

        if not text.strip():
            continue

        chunks = chunk_text(text)
        for chunk_id, chunk_text_ in chunks:
            all_chunks.append((url, chunk_id, chunk_text_))

    logger.info("[memory] Chunked into %d segments, storing...", len(all_chunks))

    for url, chunk_id, text in all_chunks:
        vector_mem.add_chunks(url, [(chunk_id, text)])

    logger.info("[memory] Vector store size: %d", vector_mem.size())
    return state