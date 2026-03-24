"""
api.py — FastAPI wrapper.

Endpoints:
  POST /api/fetch      — fetch + RAG pipeline (~10-15s), returns papers
  POST /api/summarize  — summarise papers (~20-30s), called in background
  POST /api/llm        — Gemini proxy for frontend
  GET  /api/health
  GET  /               — serves frontend

Split design:
  /api/fetch runs: researcher → memory (batch embed) → analyst (vector search)
  Returns papers immediately. Frontend renders them.

  /api/summarize runs: summarizer only on provided papers + RAG context
  Called in background while user browses papers.
"""

import os
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Startup warmup
# ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== Research Agent starting up ===")
    t0 = time.time()

    # Preload embedding model once — all requests share this singleton
    try:
        from memory.vector_memory import get_model
        model = get_model()
        list(model.embed(["warmup"]))   # triggers ONNX compilation
        logger.info("✓ Embedding model loaded (%.1fs)", time.time() - t0)
    except Exception as e:
        logger.warning("Embedding warmup failed: %s", e)

    # Warm up Gemini
    try:
        from tools.call_llm import call_llm
        call_llm("Reply with only the word: ready", model="gemini-2.5-flash")
        logger.info("✓ Gemini warmed up (%.1fs)", time.time() - t0)
    except Exception as e:
        logger.warning("Gemini warmup failed: %s", e)

    logger.info("=== Ready in %.1fs ===", time.time() - t0)
    yield
    logger.info("=== Shutting down ===")


app = FastAPI(title="Mini Research Agent API", version="4.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

frontend_path = Path(__file__).parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")


# ─────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────

class FetchRequest(BaseModel):
    query:       str       = Field(..., min_length=3, max_length=500)
    sub_areas:   list[str] = Field(default_factory=list)
    max_results: int       = Field(default=10, ge=5, le=25)
    sort_by:     str       = Field(default="relevance", pattern="^(relevance|recent|cited)$")

class SourceItem(BaseModel):
    title:          str
    authors:        str | None
    year:           int | None
    url:            str
    citations:      int | None
    is_open_access: bool
    source:         str

class FetchResponse(BaseModel):
    query:           str
    sources:         list[SourceItem]
    sources_used:    list[str]        # which APIs returned data
    rag_context:     str
    elapsed_seconds: float

class SummarizeRequest(BaseModel):
    query:       str
    papers:      list[dict]
    rag_context: str = ""

class SummarizeResponse(BaseModel):
    summary:         str
    elapsed_seconds: float

class LLMProxyRequest(BaseModel):
    prompt:     str
    max_tokens: int = 1000
    model:      str = "pro"


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "version": "4.0.0"}


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    index = frontend_path / "index.html"
    if not index.exists():
        return HTMLResponse("<h2>Frontend not found.</h2>", status_code=404)
    return HTMLResponse(index.read_text(encoding="utf-8"))


@app.post("/api/fetch", response_model=FetchResponse)
def fetch_endpoint(req: FetchRequest):
    """
    Runs: researcher → memory (batch embed) → analyst (vector search)
    Returns papers + RAG context immediately (~10-15s).
    Frontend renders papers. RAG context is passed to /api/summarize.
    """
    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured.")

    logger.info("Fetch: %s (sort=%s, max=%d)", req.query, req.sort_by, req.max_results)
    t0 = time.time()

    try:
        from tools.fetch_web import fetch_papers
        from memory.vector_memory import VectorMemory
        from memory.chunker import chunk_text
        from agents.analyst import analyst_agent
        from orchestration.state import ResearchState

        # 1. Fetch papers
        result = fetch_papers(
            topic       = req.query,
            sub_areas   = req.sub_areas or None,
            max_results = req.max_results,
            sort_by     = req.sort_by,
        )
        papers = [p for p in result["papers"] if _is_valid(p)]
        logger.info("Fetched %d valid papers from %s", len(papers), result["sources_used"])

        # 2. Batch embed all chunks
        vector_mem  = VectorMemory()
        all_entries = []
        for p in papers:
            text = p.get("abstract") or p.get("text") or ""
            url  = p.get("url", "")
            if text.strip():
                for cid, ctxt in chunk_text(text):
                    all_entries.append((url, cid, ctxt))

        stored = vector_mem.add_chunks_batch(all_entries)
        logger.info("Batch embedded %d chunks → %d stored (%.1fs)",
                    len(all_entries), stored, time.time() - t0)

        # 3. Vector search — near-instant after batch embed
        hits = vector_mem.search(req.query, k=12)
        rag_context = "\n\n".join(
            f"[SOURCE: {h['url']}]\n{h['chunk']}" for h in hits
        ) if hits else ""

        sources = [
            {
                "title":          p["title"],
                "authors":        p["authors"],
                "year":           p["year"],
                "url":            p["url"],
                "citations":      p["citations"],
                "is_open_access": p["is_open_access"],
                "source":         p["source"],
            }
            for p in papers
        ]

    except Exception as exc:
        logger.exception("Fetch failed: %s", req.query)
        raise HTTPException(status_code=500, detail=str(exc))

    elapsed = round(time.time() - t0, 2)
    logger.info("Fetch complete in %.2fs — %d papers, %d RAG chunks", elapsed, len(papers), len(hits))

    return FetchResponse(
        query           = req.query,
        sources         = sources,
        sources_used    = result["sources_used"],
        rag_context     = rag_context,
        elapsed_seconds = elapsed,
    )


@app.post("/api/summarize", response_model=SummarizeResponse)
def summarize_endpoint(req: SummarizeRequest):
    """
    Runs Gemini Pro on paper metadata + RAG context.
    Called in background after /api/fetch returns.
    """
    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured.")

    logger.info("Summarize: %s (%d papers)", req.query, len(req.papers))
    t0 = time.time()

    try:
        from tools.fetch_web import papers_to_llm_context
        from tools.call_llm import call_llm

        paper_context = papers_to_llm_context(req.papers, max_abstract_chars=300)

        prompt = f"""You are a research advisor identifying potential research areas.

Research Topic: {req.query}

Real academic papers retrieved:
{paper_context}

Relevant context retrieved via semantic search:
{req.rag_context or '(not available)'}

Based on the above, identify 5-7 distinct research areas or directions.
For each area:
1. Clear title
2. What it involves and why it matters (2-3 sentences)
3. The gap or open question it addresses
4. A realistic methodology
5. Reference at least one real paper from the list above

Format with numbered sections. Only reference papers listed above.
"""
        summary = call_llm(prompt)

    except Exception as exc:
        logger.exception("Summarize failed: %s", req.query)
        raise HTTPException(status_code=500, detail=str(exc))

    elapsed = round(time.time() - t0, 2)
    logger.info("Summarize complete in %.2fs", elapsed)

    return SummarizeResponse(summary=summary, elapsed_seconds=elapsed)


@app.post("/api/llm")
def llm_proxy(req: LLMProxyRequest):
    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured.")
    try:
        from tools.call_llm import call_llm
        model_map  = {"flash": "gemini-2.5-flash", "pro": "gemini-3.1-pro-preview"}
        model_name = model_map.get(req.model, "gemini-3.1-pro-preview")
        text       = call_llm(req.prompt, model=model_name)
        return {"content": [{"type": "text", "text": text}]}
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except RuntimeError as e:
        code = getattr(e, "status_code", None)
        raise HTTPException(status_code=503 if code in (503, 429) else 500, detail=str(e))
    except Exception as e:
        logger.error("LLM proxy error: %s", e)
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")


def _is_valid(paper: dict) -> bool:
    text = paper.get("abstract") or paper.get("text") or ""
    return len(text.strip()) >= 100 and "\x00" not in text
