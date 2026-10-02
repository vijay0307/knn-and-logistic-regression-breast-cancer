"""
Production training script (step 1 of deployment).

The notebook (notebooks/ML.ipynb) is for EXPLORATION: it compares ~20 models
and ends with `joblib.dump(best_pipe, "best_model_pipeline.joblib")`.
This script is the REPRODUCIBLE version of that last step. It retrains the
chosen pipeline from scratch and writes three artifacts that the API and the
monitoring job need:

  artifacts/model.joblib       fitted Pipeline (preprocessing + model)
  artifacts/metadata.json      model version, feature schema, test metrics,
                               decision threshold
  artifacts/reference.csv      training-feature sample = "what normal data
                               looks like", used later for drift detection

Usage:
  python train.py                                     # breast-cancer demo
  python train.py --csv bank.csv --sep ";" --target y # your own CSV
  python train.py --model knn
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

RANDOM_STATE = 42

MODELS = {
    "logreg": lambda: LogisticRegression(max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE),
    "knn": lambda: KNeighborsClassifier(n_neighbors=7, weights="distance"),
    "rf": lambda: RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE),
}


def load_data(csv, sep, target):
    if csv:
        df = pd.read_csv(csv, sep=sep)
        X, y = df.drop(columns=[target]), df[target]
        classes = sorted(y.unique().tolist())
    else:
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name="target")
        classes = data.target_names.tolist()  # ['malignant', 'benign']
    if y.dtype == object:
        le = LabelEncoder().fit(y)
        y = pd.Series(le.transform(y), name="target")
        classes = le.classes_.tolist()
    return X, y, [str(c) for c in classes]


def build_pipeline(X, model):
    num = X.select_dtypes(include=np.number).columns.tolist()
    cat = X.select_dtypes(exclude=np.number).columns.tolist()
    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]), num),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])
    return Pipeline([("preprocess", pre), ("model", model)])


def evaluate(pipe, X, y):
    pred = pipe.predict(X)
    avg = "binary" if y.nunique() == 2 else "weighted"
    out = {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, average=avg, zero_division=0),
        "recall": recall_score(y, pred, average=avg, zero_division=0),
        "f1": f1_score(y, pred, average=avg, zero_division=0),
    }
    if y.nunique() == 2:
        out["roc_auc"] = roc_auc_score(y, pipe.predict_proba(X)[:, 1])
    return {k: round(float(v), 4) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--sep", default=",")
    ap.add_argument("--target", default="y")
    ap.add_argument("--model", choices=MODELS, default="logreg")
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args()

    X, y, classes = load_data(args.csv, args.sep, args.target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    pipe = build_pipeline(X, MODELS[args.model]())
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_f1 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_weighted")
    pipe.fit(X_train, y_train)
    test_metrics = evaluate(pipe, X_test, y_test)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    version = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    joblib.dump(pipe, out / "model.joblib")
    X_train.assign(**{"__target__": y_train.values}).to_csv(out / "reference.csv", index=False)
    meta = {
        "model_name": args.model,
        "model_version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "classes": classes,
        "features": {c: ("numeric" if pd.api.types.is_numeric_dtype(X[c]) else "categorical") for c in X.columns},
        "cv_f1_mean": round(float(cv_f1.mean()), 4),
        "cv_f1_std": round(float(cv_f1.std()), 4),
        "test_metrics": test_metrics,
        "n_train": len(X_train),
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
