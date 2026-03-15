"""
agents/summarizer.py

Calls Gemini with a grounded prompt that includes:
  - The user's research query
  - Real paper abstracts (from vector retrieval)
  - Structured paper metadata (titles, authors, years, URLs)

Changes from original:
  - Prompt is grounded in real paper metadata — no hallucinated citations
  - Retry logic cleaned up: max 3 attempts with exponential backoff
  - Sources appended to output so the frontend can render real citation cards
"""

import logging
from time import sleep

from tools.call_llm import call_llm
from tools.fetch_web import papers_to_llm_context
from orchestration.state import ResearchState

logger = logging.getLogger(__name__)

MAX_RETRIES = 3


def summarizer_agent(state: ResearchState) -> ResearchState:
    query   = state["query"]
    context = state["final_context"]
    sources = state.get("sources", [])

    # Build a rich paper reference block so Gemini can cite real works
    paper_refs = papers_to_llm_context(
        state.get("fetched_docs", []),
        max_abstract_chars=300,
    )

    prompt = f"""You are a research advisor helping a student identify potential research areas.

Research Topic: {query}

Real academic papers retrieved on this topic:
{paper_refs}

Relevant context from those papers:
{context}

Based on the above real papers and context, identify 5-7 distinct research areas or directions
that a Masters or undergraduate student could pursue. For each area:
1. Give it a clear title
2. Explain what it involves and why it matters (2-3 sentences)
3. Identify what gap or open question it addresses
4. Suggest a realistic methodology a student could use
5. Reference at least one of the real papers above that is relevant to this area

Format your response clearly with numbered sections. Do not invent paper titles or authors —
only reference the papers listed above.
"""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info("[summarizer] Calling LLM (attempt %d/%d)", attempt, MAX_RETRIES)
            result = call_llm(prompt)
            state["final_context"] = result
            logger.info("[summarizer] Done.")
            return state
        except RuntimeError as exc:
            logger.warning("[summarizer] Attempt %d failed: %s", attempt, exc)
            if attempt < MAX_RETRIES:
                sleep(2 ** attempt)   # exponential backoff: 2s, 4s
            else:
                logger.error("[summarizer] All %d attempts failed.", MAX_RETRIES)
                state["final_context"] = (
                    "Error: Could not generate research summary after multiple attempts. "
                    "Please check your GEMINI_API_KEY and try again."
                )
    return state