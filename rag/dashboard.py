"""Aggregates every metric the dashboard shows into one JSON payload.

Sources
  data/rag_events.db              one row per /chat request (outcome, latency, tokens) - no user text
  data/rag_feedback.jsonl         👍/👎 from the UI (already PII-redacted)
  reports/rag_eval*.json          latest evaluation runs; rag_eval_history.jsonl for trends
  $CLASSIFIER_URL                 the classifier API's /model-info and /monitoring
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from rag.chatbot import MODEL
from rag.evaluate import MAXIMUMS, THRESHOLDS

EVENTS_SCHEMA = ("CREATE TABLE IF NOT EXISTS events (ts REAL, outcome TEXT, latency_ms INTEGER, "
                 "input_tokens INTEGER, output_tokens INTEGER, n_citations INTEGER)")

# USD per million tokens; defaults are claude-opus-5-5 list prices
PRICE_IN = float(os.getenv("RAG_PRICE_IN", "4"))
PRICE_OUT = float(os.getenv("RAG_PRICE_OUT", "20"))

GROUPS = {  # outcome -> dashboard category
    "answered": "answered", "no_answer": "no_answer",
    "prompt_injection": "blocked", "too_long": "blocked", "empty": "blocked",
    "model_refusal": "blocked", "prompt_leak": "blocked",
}


def group(outcome: str) -> str:
    return GROUPS.get(outcome, "error")


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    v = sorted(values)
    return float(v[min(len(v) - 1, max(0, math.ceil(q * len(v)) - 1))])


def _events(db: Path, since: float) -> list[tuple]:
    if not db.exists():
        return []
    with sqlite3.connect(db, timeout=5) as con:
        con.execute(EVENTS_SCHEMA)
        return con.execute("SELECT ts, outcome, latency_ms, input_tokens, output_tokens, n_citations "
                           "FROM events WHERE ts >= ? ORDER BY ts", (since,)).fetchall()


def service_metrics(db: Path, days: int) -> dict:
    now = time.time()
    rows = _events(db, now - days * 86400)
    step = 3600 if days <= 2 else 86400
    start = (now - days * 86400) // step * step
    buckets = [start + i * step for i in range(int((now - start) // step) + 1)]
    per = {b: {"counts": Counter(), "lat": [], "tin": 0, "tout": 0} for b in buckets}
    outcomes = Counter()
    for ts, outcome, lat, tin, tout, _ in rows:
        b = per.get(ts // step * step)
        if b is None:
            continue
        g = group(outcome)
        b["counts"][g] += 1
        outcomes[outcome] += 1
        if g in ("answered", "no_answer"):
            b["lat"].append(lat)
        b["tin"] += tin or 0
        b["tout"] += tout or 0

    cost = lambda tin, tout: tin / 1e6 * PRICE_IN + tout / 1e6 * PRICE_OUT  # noqa: E731
    series = [{"t": b, **{k: per[b]["counts"][k] for k in ("answered", "no_answer", "blocked", "error")},
               "p50": pct(per[b]["lat"], .5), "p95": pct(per[b]["lat"], .95),
               "input_tokens": per[b]["tin"], "output_tokens": per[b]["tout"],
               "cost": round(cost(per[b]["tin"], per[b]["tout"]), 4)} for b in buckets]

    total = len(rows)
    groups = Counter(group(o) for _, o, *_ in rows)
    lat = [r[2] for r in rows if group(r[1]) in ("answered", "no_answer")]
    tin, tout = sum(r[3] or 0 for r in rows), sum(r[4] or 0 for r in rows)
    answered_rows = [r for r in rows if r[1] == "answered"]
    model_calls = sum(1 for r in rows if (r[3] or 0) > 0)
    return {
        "bucket": "hour" if step == 3600 else "day",
        "totals": {
            "requests": total,
            "answered": groups["answered"], "no_answer": groups["no_answer"],
            "blocked": groups["blocked"], "error": groups["error"],
            "answer_rate": groups["answered"] / total if total else None,
            "block_rate": groups["blocked"] / total if total else None,
            "error_rate": groups["error"] / total if total else None,
            "p50_ms": pct(lat, .5), "p95_ms": pct(lat, .95),
            "input_tokens": tin, "output_tokens": tout,
            "cost_usd": round(cost(tin, tout), 4),
            "cost_per_answer_usd": round(cost(tin, tout) / model_calls, 5) if model_calls else None,
            "avg_citations": (sum(r[5] or 0 for r in answered_rows) / len(answered_rows)) if answered_rows else None,
        },
        "outcomes": dict(outcomes.most_common()),
        "series": series,
    }


def feedback_metrics(path: Path, days: int) -> dict:
    since = time.time() - days * 86400
    recs = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("ts", 0) >= since:
                recs.append(r)
    up = sum(r["rating"] == "up" for r in recs)
    down = [r for r in recs if r["rating"] == "down"]
    return {"up": up, "down": len(down),
            "satisfaction": up / len(recs) if recs else None,
            "recent_negative": [{"ts": r["ts"], "question": r["question"][:300], "answer": r["answer"][:400],
                                 "comment": r.get("comment", "")[:300]} for r in sorted(down, key=lambda r: r["ts"], reverse=True)[:10]]}


def eval_metrics(report_dir: Path) -> dict:
    """Merge the latest full and offline reports; full-run values win where both exist."""
    latest, sources = {}, {}
    for name in ("rag_eval_offline.json", "rag_eval.json"):
        p = report_dir / name
        if p.exists():
            try:
                d = json.loads(p.read_text())
            except json.JSONDecodeError:
                continue
            for k, v in d.get("metrics", {}).items():
                latest[k] = v
                sources[k] = {"file": name, "mtime": p.stat().st_mtime}
    rows = []
    for k, t in THRESHOLDS.items():
        v = latest.get(k)
        ok = None if v is None else (v <= t if k in MAXIMUMS else v >= t)
        rows.append({"metric": k, "value": v, "threshold": t, "direction": "max" if k in MAXIMUMS else "min",
                     "pass": ok, "measured_at": sources.get(k, {}).get("mtime")})
    history = []
    hp = report_dir / "rag_eval_history.jsonl"
    if hp.exists():
        for line in hp.read_text().splitlines()[-50:]:
            try:
                history.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    details = {}
    full = report_dir / "rag_eval.json"
    if full.exists():
        try:
            d = json.loads(full.read_text()).get("details", {})
            details = {"redteam": [{k: x[k] for k in ("id", "category", "pass", "why")} for x in d.get("redteam", [])],
                       "e2e_failures": [{k: x.get(k) for k in ("id", "correctness", "faithfulness", "reason")}
                                        for x in d.get("e2e", []) if x.get("correctness", 5) < 4 or x.get("faithfulness", 5) < 4],
                       "retrieval_misses": d.get("retrieval_misses", [])}
        except json.JSONDecodeError:
            pass
    return {"metrics": rows, "history": history, "details": details}


def _get(url: str, timeout: float = 4.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 - operator-configured URL
        return json.loads(r.read().decode())


def classifier_metrics(base_url: str | None, days: int) -> dict:
    if not base_url:
        return {"configured": False}
    base = base_url.rstrip("/")
    out = {"configured": True}
    try:
        out["model"] = _get(f"{base}/model-info")
        out["monitoring"] = _get(f"{base}/monitoring?days={days}", timeout=15)
    except Exception as e:  # noqa: BLE001 - surface any failure as a status, never crash the dashboard
        out["error"] = f"could not reach classifier API ({type(e).__name__})"
    return out


def collect(events_db: Path, feedback_log: Path, report_dir: Path, days: int, classifier_url: str | None) -> dict:
    return {"generated_at": time.time(), "window_days": days, "model": MODEL,
            "pricing": {"input_per_mtok": PRICE_IN, "output_per_mtok": PRICE_OUT},
            "service": service_metrics(events_db, days),
            "feedback": feedback_metrics(feedback_log, days),
            "evaluation": eval_metrics(report_dir),
            "classifier": classifier_metrics(classifier_url, days)}
