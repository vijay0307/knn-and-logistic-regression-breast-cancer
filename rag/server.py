"""FastAPI service for the RAG chatbot.

  uvicorn rag.server:app --port 8100

Env vars
  ANTHROPIC_API_KEY   Claude API key (required to answer questions)
  RAG_API_KEYS        comma-separated client keys; requests need `Authorization: Bearer <key>`
  RAG_ALLOW_NO_AUTH   set to 1 for local development only
  RAG_RATE_LIMIT      requests per minute per client (default 20)
  RAG_ENV             "production" disables /docs and /openapi.json
  RAG_FEEDBACK_LOG    where 👍/👎 feedback is appended (default data/rag_feedback.jsonl)
  RAG_EVENTS_DB       SQLite log of every chat request for the dashboard (default data/rag_events.db)
  RAG_REPORT_DIR      where evaluate.py writes its reports (default reports)
  CLASSIFIER_URL      e.g. http://api:8000 - adds the classifier's metrics to /dashboard
"""
from __future__ import annotations

import hashlib
import sqlite3
import hmac
import json
import logging
import os
import threading
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Literal

import anthropic
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from rag import dashboard
from rag.chatbot import RAGChatbot
from rag.guardrails import MAX_HISTORY_TURNS, MAX_QUESTION_CHARS, redact_pii

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("rag.server")

API_KEYS = [k.strip() for k in os.getenv("RAG_API_KEYS", "").split(",") if k.strip()]
ALLOW_NO_AUTH = os.getenv("RAG_ALLOW_NO_AUTH") == "1"
RATE_LIMIT = int(os.getenv("RAG_RATE_LIMIT", "20"))
PROD = os.getenv("RAG_ENV", "").lower() == "production"
FEEDBACK_LOG = Path(os.getenv("RAG_FEEDBACK_LOG", "data/rag_feedback.jsonl"))
EVENTS_DB = Path(os.getenv("RAG_EVENTS_DB", "data/rag_events.db"))
STATIC = Path(__file__).parent / "static"

REG = CollectorRegistry()
REQS = Counter("rag_requests_total", "Chat requests by outcome", ["outcome"], registry=REG)
LAT = Histogram("rag_latency_seconds", "End-to-end chat latency", registry=REG)
TOKENS = Counter("rag_tokens_total", "Claude tokens used", ["kind"], registry=REG)
FEEDBACK = Counter("rag_feedback_total", "User feedback", ["rating"], registry=REG)

app = FastAPI(title="Breast-cancer classifier docs chatbot",
              docs_url=None if PROD else "/docs", redoc_url=None,
              openapi_url=None if PROD else "/openapi.json")


# ---- security middleware ---------------------------------------------------------------------
@app.middleware("http")
async def security_headers(request: Request, call_next):
    if int(request.headers.get("content-length") or 0) > 64_000:
        return JSONResponse({"detail": "payload too large"}, status_code=413)
    resp = await call_next(request)
    resp.headers.update({
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Referrer-Policy": "no-referrer",
        "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                                   "style-src 'self' 'unsafe-inline'; frame-ancestors 'none'",
        "Cache-Control": "no-store",
    })
    return resp


_hits: dict[str, deque] = defaultdict(deque)
_lock = threading.Lock()


def authenticate(request: Request) -> str:
    """Returns a client id (hash of the key, or the IP in no-auth dev mode) and enforces rate limits."""
    if API_KEYS:
        auth = request.headers.get("authorization", "")
        token = auth[7:] if auth.lower().startswith("bearer ") else request.headers.get("x-api-key", "")
        if not any(hmac.compare_digest(token.encode(), k.encode()) for k in API_KEYS):
            raise HTTPException(401, "invalid or missing API key", headers={"WWW-Authenticate": "Bearer"})
        client_id = hashlib.sha256(token.encode()).hexdigest()[:12]
    elif ALLOW_NO_AUTH:
        client_id = request.client.host if request.client else "anon"
    else:  # fail closed: never expose an unauthenticated, paid endpoint by accident
        raise HTTPException(503, "server has no RAG_API_KEYS configured")

    now = time.monotonic()
    with _lock:
        q = _hits[client_id]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= RATE_LIMIT:
            raise HTTPException(429, "rate limit exceeded", headers={"Retry-After": "60"})
        q.append(now)
    return client_id


# ---- chatbot ---------------------------------------------------------------------------------
_bot: RAGChatbot | None = None


def get_bot() -> RAGChatbot:
    global _bot
    if _bot is None:
        _bot = RAGChatbot()
    return _bot


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    history: list[Turn] = Field(default_factory=list, max_length=2 * MAX_HISTORY_TURNS)


class FeedbackRequest(BaseModel):
    response_id: str = Field(max_length=64)
    rating: Literal["up", "down"]
    question: str = Field(max_length=MAX_QUESTION_CHARS)
    answer: str = Field(max_length=8000)
    comment: str = Field("", max_length=1000)


def record_event(outcome: str, latency_ms: int, usage: dict | None = None, n_citations: int = 0):
    """Durable, PII-free request log (no question text) that feeds /dashboard."""
    usage = usage or {}
    try:
        EVENTS_DB.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(EVENTS_DB, timeout=5) as con:
            con.execute(dashboard.EVENTS_SCHEMA)
            con.execute("INSERT INTO events VALUES (?,?,?,?,?,?)",
                        (time.time(), outcome, latency_ms, usage.get("input_tokens", 0),
                         usage.get("output_tokens", 0), n_citations))
    except sqlite3.Error:
        log.exception("could not record event")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def ui():
    return FileResponse(STATIC / "index.html")


@app.get("/metrics")
def metrics():
    return Response(generate_latest(REG), media_type=CONTENT_TYPE_LATEST)


@app.post("/chat")
def chat(req: ChatRequest, client_id: str = Depends(authenticate), bot: RAGChatbot = Depends(get_bot)):
    rid = uuid.uuid4().hex
    t0 = time.perf_counter()
    elapsed = lambda: int((time.perf_counter() - t0) * 1000)  # noqa: E731
    with LAT.time():
        try:
            r = bot.ask(req.question, [t.model_dump() for t in req.history])
        except anthropic.RateLimitError:
            REQS.labels("upstream_rate_limited").inc()
            record_event("upstream_rate_limited", elapsed())
            raise HTTPException(503, "model is busy, retry shortly", headers={"Retry-After": "30"})
        except (anthropic.APIConnectionError, anthropic.APIStatusError) as e:
            REQS.labels("upstream_error").inc()
            record_event("upstream_error", elapsed())
            log.error("anthropic error rid=%s type=%s", rid, type(e).__name__)
            raise HTTPException(502, "upstream model error")
        except Exception as e:  # e.g. missing ANTHROPIC_API_KEY; never leak internals to clients
            REQS.labels("internal_error").inc()
            record_event("internal_error", elapsed())
            log.exception("chat failed rid=%s type=%s", rid, type(e).__name__)
            raise HTTPException(503, "chat backend unavailable")
    outcome = r.blocked or ("answered" if r.answerable else "no_answer")
    REQS.labels(outcome).inc()
    record_event(outcome, elapsed(), r.usage, len(r.citations))
    for k, v in r.usage.items():
        TOKENS.labels(k).inc(v)
    # never log raw user text: redact PII and truncate
    log.info("chat rid=%s client=%s outcome=%s latency_ms=%d q=%r", rid, client_id, outcome,
             r.latency_ms, redact_pii(req.question)[:200])
    return {"response_id": rid, "answer": r.answer, "answerable": r.answerable,
            "citations": r.citations, "blocked": r.blocked is not None,
            "disclaimer": "Educational demo. Not medical advice."}


@app.post("/chat/feedback")
def feedback(fb: FeedbackRequest, client_id: str = Depends(authenticate)):
    FEEDBACK.labels(fb.rating).inc()
    FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
    rec = {"ts": time.time(), "response_id": fb.response_id, "rating": fb.rating,
           "question": redact_pii(fb.question), "answer": redact_pii(fb.answer),
           "comment": redact_pii(fb.comment)}
    with _lock, FEEDBACK_LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    return {"ok": True}


@app.get("/dashboard", include_in_schema=False)
def dashboard_page():
    return FileResponse(STATIC / "dashboard.html")


@app.get("/dashboard/data")
def dashboard_data(days: int = 7, client_id: str = Depends(authenticate)):
    days = max(1, min(days, 90))
    return dashboard.collect(EVENTS_DB, FEEDBACK_LOG, Path(os.getenv("RAG_REPORT_DIR", "reports")),
                             days, os.getenv("CLASSIFIER_URL"))
