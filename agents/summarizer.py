"""
agents/summarizer.py

Uses RAG context (from analyst) plus real paper metadata.
Uses Pro model — this is the heavy academic task.
"""

import logging
from time import sleep

from tools.call_llm import call_llm
from tools.fetch_web import papers_to_llm_context
from orchestration.state import ResearchState

logger    = logging.getLogger(__name__)
MAX_RETRY = 3


def summarizer_agent(state: ResearchState) -> ResearchState:
    query         = state["query"]
    rag_context   = state.get("final_context", "")
    papers        = state.get("fetched_docs", [])
    paper_context = papers_to_llm_context(papers, max_abstract_chars=300)

    prompt = f"""You are a research advisor identifying potential research areas.

Research Topic: {query}

Real academic papers retrieved:
{paper_context}

Relevant context retrieved via semantic search:
{rag_context}

Based on the above, identify 5-7 distinct research areas or directions.
For each area:
1. Clear title
2. What it involves and why it matters (2-3 sentences)
3. The gap or open question it addresses
4. A realistic methodology
5. Reference at least one real paper from the list above

Format with numbered sections. Only reference papers listed above.
"""

    for attempt in range(1, MAX_RETRY + 1):
        try:
            logger.info("[summarizer] Calling LLM (attempt %d/%d)", attempt, MAX_RETRY)
            result = call_llm(prompt)
            state["final_context"] = result
            logger.info("[summarizer] Done.")
            return state
        except RuntimeError as exc:
            logger.warning("[summarizer] Attempt %d failed: %s", attempt, exc)
            if attempt < MAX_RETRY:
                sleep(2 ** attempt)
            else:
                state["final_context"] = (
                    "Error: Could not generate summary. Check your GEMINI_API_KEY."
                )
    return state
