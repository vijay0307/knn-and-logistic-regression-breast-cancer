# Classifier: from notebook to production

`notebooks/ML.ipynb` compares about 20 classical models on Breast Cancer Wisconsin, or on any CSV such as `bank.csv`, and saves the best pipeline with `joblib`. This repo covers what comes after: **deploy** the model as an API and **monitor** it once it's live.

```
 notebook (explore)  ──►  train.py  ──►  artifacts/  ──►  app/ (FastAPI, Docker)  ──►  users
                                          model.joblib            │  every prediction logged
                                          metadata.json           ▼
                                          reference.csv     data/predictions.db ◄── /feedback (ground truth)
                                                │                 │
                                                └──► monitoring/monitor.py (drift + live accuracy, scheduled)
                                                     Prometheus + Grafana (latency, errors, confidence, real-time)
```

## 1. Train (reproducible)

```bash
pip install -r requirements-dev.txt
python train.py                                        # breast cancer, logistic regression
python train.py --model knn                            # or knn / rf
python train.py --csv bank.csv --sep ";" --target y    # your own data
```

This writes `artifacts/model.joblib` (preprocessing and model as one pipeline, so there is no train/serve skew), `metadata.json` (version, feature schema, offline test metrics) and `reference.csv` (training data used as the drift baseline).

> If you'd rather ship the notebook's own winner, copy its `best_model_pipeline.joblib` to `artifacts/model.joblib`. You still need to run `train.py` once on the same data to produce `metadata.json` and `reference.csv`. Pin the same scikit-learn version as the notebook, because a joblib file only loads reliably with the version that created it.

## 2. Deploy

**Locally**

```bash
uvicorn app.main:app --port 8000        # interactive docs: http://localhost:8000/docs
```

**Docker, with Prometheus and Grafana**

```bash
docker compose up --build
# API :8000   Prometheus :9090   Grafana :3000 (admin/admin)
```

**Call it**

```bash
curl -X POST localhost:8000/predict -H 'content-type: application/json' \
  -d '{"records":[{"mean radius":14.1,"mean texture":19.3, ... all 30 features ...}]}'
# -> {"results":[{"prediction_id":"…","prediction":"benign","confidence":0.97,"probabilities":{…}}]}

# Later, once the true outcome is known (e.g. biopsy result):
curl -X POST localhost:8000/feedback -H 'content-type: application/json' \
  -d '{"prediction_id":"…","actual":"malignant"}'
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness probe for Docker, K8s or a load balancer |
| `GET /model-info` | Live model version and its offline metrics |
| `POST /predict` | Batch scoring (up to 1000 rows). Validates the schema and logs every prediction |
| `POST /feedback` | Attaches ground truth to a past prediction |
| `GET /metrics` | Prometheus metrics |

**Where to host it.** The Docker image runs anywhere containers do: AWS ECS/App Runner, GCP Cloud Run, Azure Container Apps, Kubernetes, Render or Fly.io. For Cloud Run, for example:
`gcloud run deploy classifier --source . --port 8000`.
In production, swap SQLite for a managed database (Postgres or BigQuery) so that several replicas can share one prediction log.

## 3. Monitor performance

A deployed model rarely breaks in a loud way. More often the incoming data drifts and accuracy slowly drops. You can watch for this at two levels.

**A. Real-time service health (Prometheus, Grafana)**: `monitoring/alerts.yml`
- p95 latency above 500 ms, error rate above 5%, API down
- **Median prediction confidence below 0.7.** This is an early sign that the model is seeing unfamiliar inputs.
- The predicted-class mix (`prediction_class_total`) is tracked too.

**B. Model quality (scheduled job)**: `monitoring/monitor.py`

```bash
python monitoring/monitor.py --days 7     # exits 1 if any alert fires, so it can page someone
```

| Check | Needs labels? | Alert when |
|---|---|---|
| **Feature drift**: PSI per feature (plus a KS test for numeric features) against `reference.csv` | No (early warning) | PSI > 0.25 |
| **Prediction drift**: share of each predicted class vs. the training label mix | No | Shift > 15 points |
| **Live performance**: accuracy, precision, recall and F1 on rows that received `/feedback` | Yes (definitive, but delayed) | Drop > 0.05 vs. offline test metrics |

Run it daily with cron, a Kubernetes CronJob, Airflow or `.github/workflows/monitor.yml`. The full report is written to `reports/monitoring.json`.

**Try it end to end**

```bash
uvicorn app.main:app --port 8000 &
python monitoring/simulate_traffic.py --n 300              # normal traffic
python monitoring/monitor.py --days 1                      # no alerts, live accuracy ≈ 0.97
python monitoring/simulate_traffic.py --n 300 --drift 1.5  # inputs scaled up 50%
python monitoring/monitor.py --days 1                      # PSI alerts, malignant share 37% → 68%,
                                                           # accuracy 0.96 → 0.70, recall 0.94 → 0.51
```

**When alerts fire:** first check the data, because a broken upstream feed or a changed unit is the most common cause. If the drift is real, retrain on recent labelled data with `train.py`, compare the new model against the current one on the same holdout, and deploy the new image. Roll it out gradually (shadow or canary) and keep the old image tag so you can roll back.

> **Medical note:** in this dataset, missing a *malignant* tumour (a false negative) costs far more than a false alarm. When choosing or tuning a model, put recall on the malignant class first and consider lowering its decision threshold. Don't rely on accuracy alone.

## Tests

```bash
pytest -q    # train → serve → predict → feedback → monitor, including a drift-detection check
```
