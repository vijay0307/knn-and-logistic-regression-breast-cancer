"""
Model-serving API (step 2 of deployment).

  GET  /health      liveness/readiness probe for Docker / Kubernetes / load balancers
  GET  /model-info  which model version is live + its offline test metrics
  POST /predict     {"records": [{feature: value, ...}, ...]} -> predictions
  POST /feedback    {"prediction_id": ..., "actual": ...} -> ground truth arrives later
  GET  /metrics     Prometheus metrics (latency, traffic, errors, class mix)
  GET  /            web UI: predict form + feedback + monitoring dashboard
  GET  /sample      a random training row (to prefill the UI form)
  GET  /monitoring  drift + live-performance report (same as monitoring/monitor.py)

Every prediction is logged (inputs + output + id) to SQLite. The monitoring
job (monitoring/monitor.py) reads that log to detect data drift and, once
feedback labels arrive, to compute LIVE accuracy / recall / F1.
"""
import json
import os
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import FileResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field


# --- Prometheus metrics ------------------------------------------------------
REQUESTS = Counter("predict_requests_total", "Prediction requests", ["status"])
ROWS = Counter("predict_rows_total", "Rows scored")
LATENCY = Histogram("predict_latency_seconds", "Latency of /predict",
                    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5))
PRED_CLASS = Counter("prediction_class_total", "Predicted class counts", ["label"])
CONFIDENCE = Histogram("prediction_confidence", "Max class probability",
                       buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0))
FEEDBACK = Counter("feedback_total", "Ground-truth labels received", ["correct"])
MODEL_INFO = Gauge("model_info", "Loaded model (value always 1)", ["name", "version"])

STATIC_DIR = Path(__file__).parent / "static"
state: dict[str, Any] = {}


def init_db(db: Path):
    db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS predictions (
            id TEXT PRIMARY KEY, ts TEXT, model_version TEXT,
            features TEXT, prediction TEXT, confidence REAL,
            actual TEXT, feedback_ts TEXT)""")


@asynccontextmanager
async def lifespan(_: FastAPI):
    artifact_dir = Path(os.getenv("ARTIFACT_DIR", "artifacts"))
    state["artifact_dir"] = artifact_dir
    state["db"] = Path(os.getenv("PREDICTION_DB", "data/predictions.db"))
    state["model"] = joblib.load(artifact_dir / "model.joblib")
    state["meta"] = json.loads((artifact_dir / "metadata.json").read_text())
    state["reference"] = pd.read_csv(artifact_dir / "reference.csv")
    MODEL_INFO.labels(state["meta"]["model_name"], state["meta"]["model_version"]).set(1)
    init_db(state["db"])
    yield
    state.clear()


app = FastAPI(title="Classifier API", lifespan=lifespan)


class PredictRequest(BaseModel):
    records: list[dict[str, Any]] = Field(..., min_length=1, max_length=1000)


class Feedback(BaseModel):
    prediction_id: str
    actual: str


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.get("/model-info")
def model_info():
    return state["meta"]


@app.post("/predict")
def predict(req: PredictRequest):
    meta, model = state["meta"], state["model"]
    start = time.perf_counter()
    expected = list(meta["features"])
    missing = sorted({c for r in req.records for c in expected if c not in r})
    if missing:
        REQUESTS.labels("invalid").inc()
        raise HTTPException(422, f"Missing features: {missing[:10]}")
    try:
        X = pd.DataFrame(req.records)[expected]
        proba = model.predict_proba(X)
        preds = proba.argmax(axis=1)
    except Exception as e:  # bad dtypes etc.
        REQUESTS.labels("error").inc()
        raise HTTPException(400, f"Could not score input: {e}") from e

    classes, now = meta["classes"], datetime.now(timezone.utc).isoformat()
    results, rows = [], []
    for rec, p, pr in zip(req.records, preds, proba):
        pid, label, conf = str(uuid.uuid4()), classes[int(p)], float(pr.max())
        results.append({"prediction_id": pid, "prediction": label, "confidence": round(conf, 4),
                        "probabilities": {c: round(float(v), 4) for c, v in zip(classes, pr)}})
        rows.append((pid, now, meta["model_version"], json.dumps(rec), label, conf))
        PRED_CLASS.labels(label).inc()
        CONFIDENCE.observe(conf)
    with sqlite3.connect(state["db"]) as con:
        con.executemany("INSERT INTO predictions (id, ts, model_version, features, prediction, confidence) "
                        "VALUES (?,?,?,?,?,?)", rows)

    REQUESTS.labels("ok").inc()
    ROWS.inc(len(rows))
    LATENCY.observe(time.perf_counter() - start)
    return {"model_version": meta["model_version"], "results": results}


@app.post("/feedback")
def feedback(fb: Feedback):
    with sqlite3.connect(state["db"]) as con:
        row = con.execute("SELECT prediction FROM predictions WHERE id=?", (fb.prediction_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "Unknown prediction_id")
        con.execute("UPDATE predictions SET actual=?, feedback_ts=? WHERE id=?",
                    (fb.actual, datetime.now(timezone.utc).isoformat(), fb.prediction_id))
    FEEDBACK.labels(str(row[0] == fb.actual).lower()).inc()
    return {"status": "recorded", "was_correct": row[0] == fb.actual}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/", include_in_schema=False)
def ui():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/sample")
def sample():
    """Random training row; `actual` is its true label so the UI can demo /feedback."""
    row = state["reference"].sample(1).iloc[0]
    features = {c: (row[c].item() if hasattr(row[c], "item") else row[c]) for c in state["meta"]["features"]}
    return {"features": features, "actual": state["meta"]["classes"][int(row["__target__"])]}


@app.get("/monitoring")
def monitoring(days: int = Query(7, ge=1, le=365)):
    from monitoring.monitor import run
    return run(state["artifact_dir"], state["db"], days)
