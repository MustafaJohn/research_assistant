"""
api.py — FastAPI wrapper around the LangGraph research pipeline.

Endpoints:
  POST /api/research  — run the full multi-agent pipeline
  POST /api/llm       — LLM proxy (Gemini) for frontend stages
  GET  /api/health    — health check
  GET  /              — serves the frontend HTML

Run locally:
  uvicorn api:app --reload --port 8000

Deploy:
  Set GEMINI_API_KEY, start command: uvicorn api:app --host 0.0.0.0 --port $PORT
"""

import os
import logging
import time
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

from orchestration.graph import build_graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Mini Research Agent API",
    description="Multi-agent academic research assistant",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_path = Path(__file__).parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")


# ─────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────

class ResearchRequest(BaseModel):
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

class ResearchResponse(BaseModel):
    query:           str
    summary:         str
    sources:         list[SourceItem]
    elapsed_seconds: float
    warning:         str = ""

class LLMProxyRequest(BaseModel):
    prompt:     str
    max_tokens: int = 1000
    model:      str = "pro"   # "pro" | "flash"


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "version": "3.0.0"}


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    index = frontend_path / "index.html"
    if not index.exists():
        return HTMLResponse("<h2>Frontend not found.</h2>", status_code=404)
    return HTMLResponse(index.read_text(encoding="utf-8"))


@app.post("/api/llm")
def llm_proxy(req: LLMProxyRequest):
    """
    Proxy all frontend LLM calls through Gemini.
    model="flash" → gemini-2.0-flash  (Stage 1 topic exploration)
    model="pro"   → gemini-2.5-pro    (Stages 2–5, academic tasks)
    Falls back to the other model if the primary 503s.
    """
    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured.")

    try:
        from tools.call_llm import call_llm
        # Map "flash"/"pro" shorthand to actual model names
        model_map = {
            "flash": "gemini-2.5-flash",
            "pro":   "gemini-3.1-pro",
        }
        model_name = model_map.get(req.model, "gemini-2.5-pro")
        text = call_llm(req.prompt, model=model_name)
        return {"content": [{"type": "text", "text": text}]}

    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except RuntimeError as e:
        code = getattr(e, "status_code", None)
        http_status = 503 if code in (503, 429) else 500
        raise HTTPException(status_code=http_status, detail=str(e))
    except Exception as e:
        logger.error("LLM proxy unexpected error: %s", e)
        raise HTTPException(status_code=500, detail="An unexpected error occurred. Please try again.")


@app.post("/api/research", response_model=ResearchResponse)
def run_research(req: ResearchRequest):
    """
    Run the full multi-agent pipeline (fetch → memory → analyse → summarise).
    Uses Pro model for summarisation — this is the heavy academic task.
    """
    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured.")

    logger.info("Research request: %s (sort: %s, max: %d)", req.query, req.sort_by, req.max_results)
    t0 = time.time()

    try:
        graph = build_graph()
        initial_state = {
            "query":             req.query,
            "sort_by":           req.sort_by,
            "fetched_docs":      [],
            "vector_results":    [],
            "graph_results":     [],
            "final_context":     "",
            "next_step":         "",
            "analysis_decision": "",
            "sources":           [{"title": a} for a in req.sub_areas] if req.sub_areas else [],
            "max_results":       req.max_results,
            "logs":              [],
        }
        result = graph.invoke(initial_state)
    except Exception as exc:
        logger.exception("Pipeline failed for query: %s", req.query)
        raise HTTPException(status_code=500, detail=str(exc))

    elapsed = round(time.time() - t0, 2)
    logger.info("Completed in %.2fs — %d sources", elapsed, len(result.get("sources", [])))

    warning = next((l for l in result.get("logs", []) if l.startswith("⚠")), "")

    return ResearchResponse(
        query           = req.query,
        summary         = result.get("final_context", ""),
        sources         = result.get("sources", []),
        elapsed_seconds = elapsed,
        warning         = warning,
    )
