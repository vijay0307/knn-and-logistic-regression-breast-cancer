"""
Demo: send production-like traffic to a running API, then send ground truth
back via /feedback, so you can see monitoring react.

  python monitoring/simulate_traffic.py --n 300            # normal traffic
  python monitoring/simulate_traffic.py --n 300 --drift 1.5  # shifted inputs

--drift multiplies numeric features (simulates e.g. a new measuring device),
which should trigger PSI alerts and usually a performance drop.
"""
import argparse
import random

import httpx
import pandas as pd
from sklearn.datasets import load_breast_cancer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--drift", type=float, default=1.0)
    ap.add_argument("--label-rate", type=float, default=0.8, help="share of predictions that get feedback")
    args = ap.parse_args()

    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names).sample(args.n, replace=True, random_state=0)
    y = [data.target_names[i] for i in data.target[X.index]]
    X = X * args.drift

    with httpx.Client(base_url=args.url, timeout=30) as c:
        res = c.post("/predict", json={"records": X.to_dict("records")}).raise_for_status().json()["results"]
        sent = 0
        for r, actual in zip(res, y):
            if random.random() < args.label_rate:
                c.post("/feedback", json={"prediction_id": r["prediction_id"], "actual": actual}).raise_for_status()
                sent += 1
    print(f"Scored {len(res)} rows, sent {sent} feedback labels.")


if __name__ == "__main__":
    main()
