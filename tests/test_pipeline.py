"""End-to-end check: train -> serve -> predict -> feedback -> monitor."""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_breast_cancer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("run")
    art = tmp / "artifacts"
    subprocess.run([sys.executable, str(ROOT / "train.py"), "--out", str(art)], check=True, capture_output=True)
    return art, tmp / "predictions.db"


@pytest.fixture(scope="module")
def client(env):
    art, db = env
    import os
    os.environ["ARTIFACT_DIR"], os.environ["PREDICTION_DB"] = str(art), str(db)
    from app.main import app
    with TestClient(app) as c:
        yield c


def _records(n, scale=1.0):
    d = load_breast_cancer()
    X = pd.DataFrame(d.data, columns=d.feature_names).sample(n, replace=True, random_state=1)
    labels = [d.target_names[i] for i in d.target[X.index]]
    return (X * scale).to_dict("records"), labels


def test_trained_model_is_good(env):
    meta = json.loads((env[0] / "metadata.json").read_text())
    assert meta["test_metrics"]["accuracy"] > 0.9


def test_health_and_validation(client):
    assert client.get("/health").json()["model_loaded"]
    assert client.post("/predict", json={"records": [{"foo": 1}]}).status_code == 422


def test_predict_feedback_monitor(client, env):
    from monitoring.monitor import run
    recs, labels = _records(120)
    res = client.post("/predict", json={"records": recs}).json()["results"]
    assert {r["prediction"] for r in res} <= {"malignant", "benign"}
    for r, y in zip(res, labels):
        assert client.post("/feedback", json={"prediction_id": r["prediction_id"], "actual": y}).status_code == 200
    assert b"predict_latency_seconds" in client.get("/metrics").content

    report = run(env[0], env[1], days=1)
    assert report["performance"]["live"]["accuracy"] > 0.9
    assert not any(a.startswith("Feature drift") for a in report["alerts"])

    drifted, _ = _records(200, scale=1.6)
    client.post("/predict", json={"records": drifted})
    report = run(env[0], env[1], days=1)
    assert any(a.startswith("Feature drift") for a in report["alerts"])
