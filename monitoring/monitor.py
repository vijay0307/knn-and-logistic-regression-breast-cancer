"""
Model monitoring job (step 3 of deployment). Run on a schedule (cron,
Airflow, GitHub Actions, Kubernetes CronJob), e.g. daily.

A deployed model degrades silently: the code doesn't change, the WORLD does.
We check three things:

1. DATA DRIFT -- are incoming features distributed like the training data?
   PSI (Population Stability Index) per feature:
       < 0.1 stable | 0.1-0.25 moderate shift | > 0.25 significant shift
   plus a KS test for numeric features. Drift needs no labels, so it is the
   EARLY warning.
2. PREDICTION DRIFT -- has the share of each predicted class moved away from
   the training class balance? (e.g. suddenly 3x more "malignant")
3. LIVE PERFORMANCE -- once ground truth comes back via /feedback, compute
   accuracy / precision / recall / F1 and compare to the offline test metrics
   stored in metadata.json. This is the definitive signal, but labels arrive
   late (biopsy results, loan default after months...).

Exit code 1 if any alert fires, so the scheduler can page someone or trigger
retraining.

Usage:
  python monitoring/monitor.py --days 7 --report reports/monitoring.json
"""
import argparse
import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

PSI_ALERT = 0.25
PERF_DROP_ALERT = 0.05      # absolute drop vs offline test metric
MIN_ROWS_FOR_DRIFT = 50
MIN_LABELS_FOR_PERF = 30


def psi(ref, cur, bins=10):
    """Population Stability Index. Bins come from reference quantiles."""
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-6, None), np.clip(c, 1e-6, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_categorical(ref, cur):
    cats = set(ref.astype(str)) | set(cur.astype(str))
    r = ref.astype(str).value_counts(normalize=True).reindex(list(cats), fill_value=0).clip(lower=1e-6)
    c = cur.astype(str).value_counts(normalize=True).reindex(list(cats), fill_value=0).clip(lower=1e-6)
    return float(np.sum((c - r) * np.log(c / r)))


def load_live(db, since):
    with sqlite3.connect(db) as con:
        df = pd.read_sql("SELECT * FROM predictions WHERE ts >= ?", con, params=(since.isoformat(),))
    if df.empty:
        return df, pd.DataFrame()
    return df, pd.DataFrame([json.loads(f) for f in df["features"]])


def run(artifact_dir, db, days):
    artifact_dir = Path(artifact_dir)
    meta = json.loads((artifact_dir / "metadata.json").read_text())
    ref = pd.read_csv(artifact_dir / "reference.csv")
    since = datetime.now(timezone.utc) - timedelta(days=days)
    log, live_X = load_live(db, since)

    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "window_days": days,
              "model_version": meta["model_version"], "n_predictions": len(log),
              "alerts": [], "feature_drift": {}, "prediction_drift": {}, "performance": None}
    if len(log) < MIN_ROWS_FOR_DRIFT:
        report["note"] = f"Only {len(log)} predictions in window (< {MIN_ROWS_FOR_DRIFT}); drift not computed."
        return report

    # 1. Feature drift
    for col, kind in meta["features"].items():
        if col not in live_X:
            continue
        if kind == "numeric":
            cur = pd.to_numeric(live_X[col], errors="coerce").dropna()
            score = psi(ref[col].dropna(), cur)
            ks_p = float(ks_2samp(ref[col].dropna(), cur).pvalue)
            report["feature_drift"][col] = {"psi": round(score, 4), "ks_pvalue": round(ks_p, 5)}
        else:
            score = psi_categorical(ref[col].dropna(), live_X[col].dropna())
            report["feature_drift"][col] = {"psi": round(score, 4)}
        if score > PSI_ALERT:
            report["alerts"].append(f"Feature drift: {col} PSI={score:.3f}")
    drifted = sum(v["psi"] > PSI_ALERT for v in report["feature_drift"].values())
    report["drifted_feature_share"] = round(drifted / max(len(report["feature_drift"]), 1), 3)

    # 2. Prediction drift (live predicted-class mix vs training label mix)
    classes = meta["classes"]
    ref_mix = ref["__target__"].map(lambda i: classes[int(i)]).value_counts(normalize=True)
    live_mix = log["prediction"].value_counts(normalize=True)
    for c in classes:
        r, l = float(ref_mix.get(c, 0)), float(live_mix.get(c, 0))
        report["prediction_drift"][c] = {"train_share": round(r, 3), "live_share": round(l, 3)}
        if abs(l - r) > 0.15:
            report["alerts"].append(f"Prediction drift: class '{c}' {r:.0%} -> {l:.0%}")
    report["mean_confidence"] = round(float(log["confidence"].mean()), 4)

    # 3. Live performance (only rows that received ground truth)
    labelled = log.dropna(subset=["actual"])
    if len(labelled) >= MIN_LABELS_FOR_PERF:
        y, p = labelled["actual"], labelled["prediction"]
        binary = len(classes) == 2
        kw = {"average": "binary", "pos_label": classes[1]} if binary else {"average": "weighted"}
        live = {"accuracy": accuracy_score(y, p),
                "precision": precision_score(y, p, zero_division=0, **kw),
                "recall": recall_score(y, p, zero_division=0, **kw),
                "f1": f1_score(y, p, zero_division=0, **kw)}
        live = {k: round(float(v), 4) for k, v in live.items()}
        report["performance"] = {"n_labelled": len(labelled), "live": live, "offline_test": meta["test_metrics"]}
        for k, v in live.items():
            base = meta["test_metrics"].get(k)
            if base is not None and base - v > PERF_DROP_ALERT:
                report["alerts"].append(f"Performance drop: {k} {base:.3f} -> {v:.3f}")
    else:
        report["performance"] = {"n_labelled": len(labelled),
                                 "note": f"Need >= {MIN_LABELS_FOR_PERF} labelled rows for live metrics."}
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--db", default="data/predictions.db")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--report", default="reports/monitoring.json")
    args = ap.parse_args()

    report = run(args.artifacts, args.db, args.days)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("n_predictions", "alerts", "performance")}, indent=2))
    sys.exit(1 if report["alerts"] else 0)


if __name__ == "__main__":
    main()
