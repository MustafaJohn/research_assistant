"""
api.py — FastAPI wrapper around the LangGraph research pipeline.

Endpoints:
  POST /api/research        — run the full multi-agent pipeline
  GET  /api/health          — health check
  GET  /                    — serves the frontend HTML

Run locally:
  uvicorn api:app --reload --port 8000

Deploy (Render / Railway / Fly.io):
  Set env var GEMINI_API_KEY, then point start command to:
  uvicorn api:app --host 0.0.0.0 --port $PORT
"""

import os
import logging
import time
import json
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
    version="2.0.0",
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
# Request / Response models
# ─────────────────────────────────────────────────────────────

class ResearchRequest(BaseModel):
    query:       str       = Field(..., min_length=3, max_length=500)
    sub_areas:   list[str] = Field(default_factory=list)
    max_results: int       = Field(default=10, ge=5, le=25)

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

class LLMProxyRequest(BaseModel):
    prompt:     str
    max_tokens: int = 1000


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "version": "2.0.0"}


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    index = frontend_path / "index.html"
    if not index.exists():
        return HTMLResponse("<h2>Frontend not found. Place index.html in /frontend.</h2>", status_code=404)
    return HTMLResponse(index.read_text(encoding="utf-8"))


@app.post("/api/llm")
def llm_proxy(req: LLMProxyRequest):
    """
    Proxy all frontend LLM calls through Gemini on the server.
    Uses the fallback chain: gemini-2.5-pro → gemini-1.5-flash.
    The browser never calls any external API directly — no CORS issues.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not configured on the server.",
        )

    try:
        from tools.call_llm import call_llm
        text = call_llm(req.prompt)
        # Return in the shape the frontend expects
        return {"content": [{"type": "text", "text": text}]}

    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except RuntimeError as e:
        # Clean user-facing message from call_llm
        code = getattr(e, "status_code", None)
        http_status = 503 if code in (503, 429) else 500
        raise HTTPException(status_code=http_status, detail=str(e))
    except Exception as e:
        logger.error("LLM proxy unexpected error: %s", e)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred. Please try again."
        )


@app.post("/api/research", response_model=ResearchResponse)
def run_research(req: ResearchRequest):
    """
    Run the full multi-agent pipeline for a research query.
    Fresh VectorMemory per request — no cross-query contamination.
    """
    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not configured on the server.",
        )

    logger.info("Research request: %s", req.query)
    t0 = time.time()

    try:
        graph = build_graph()
        initial_state = {
            "query":             req.query,
            "fetched_docs":      [],
            "vector_results":    [],
            "graph_results":     [],
            "final_context":     "",
            "next_step":         "",
            "analysis_decision": "",
            "sources":           [{"title": a} for a in req.sub_areas] if req.sub_areas else [],
            "logs":              [],
            "max_results":       req.max_results,
        }
        result = graph.invoke(initial_state)
    except Exception as exc:
        logger.exception("Pipeline failed for query: %s", req.query)
        raise HTTPException(status_code=500, detail=str(exc))

    elapsed = round(time.time() - t0, 2)
    logger.info("Completed in %.2fs — %d sources", elapsed, len(result.get("sources", [])))

    return ResearchResponse(
        query           = req.query,
        summary         = result.get("final_context", ""),
        sources         = result.get("sources", []),
        elapsed_seconds = elapsed,
    )
