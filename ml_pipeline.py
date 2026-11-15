#!/usr/bin/env python3
"""
ml_pipeline.py — ML pipeline with subject-level cross-validation.

Compares two model configs on trial-level features, with the target
being the simulated gait condition (0 = healthy, 1 = impaired).
Splits are GroupKFold by subject, so no subject ever appears in both
train and test of the same fold — the honest way to evaluate a
subject-level gait classifier.

Models compared:
  A. Logistic regression (standardized features)
  B. Random forest (500 trees)

Metrics per fold (and mean ± SD across folds):
  ROC-AUC, accuracy, Brier score (calibration), sensitivity, specificity.

Outputs:
  outputs/ml_results.json    full per-fold + aggregate metrics
  outputs/ml_metrics.csv     one row per model per fold

Run:  python ml_pipeline.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent
OUT_DIR = BASE / "outputs"
RANDOM_STATE = 42

# Features used by the models (all computed from the signals in features.py).
FEATURES = [
    "cadence_spm", "mean_stride_time_s", "stride_time_cv",
    "step_time_asymmetry", "speed_median_m_s", "amp_asymmetry_az",
    "sma_acc",
    "ax_rms", "ay_rms", "az_rms", "gx_rms", "gy_rms", "gz_rms",
    "az_jerk_rms", "gy_jerk_rms", "az_dom_freq_hz",
]


def fold_metrics(y_true, y_prob, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "n_test": int(len(y_true)),
    }


def evaluate(model, X, y, groups):
    gkf = GroupKFold(n_splits=4)
    folds = []
    for fold_i, (tr, te) in enumerate(gkf.split(X, y, groups)):
        model_clone = model
        from sklearn.base import clone
        m = clone(model_clone)
        m.fit(X.iloc[tr], y.iloc[tr])
        prob = m.predict_proba(X.iloc[te])[:, 1]
        pred = (prob >= 0.5).astype(int)
        fm = fold_metrics(y.iloc[te], prob, pred)
        fm["fold"] = fold_i
        fm["test_subjects"] = sorted(set(groups.iloc[te]))
        folds.append(fm)
    agg = {k: {"mean": round(float(np.mean([f[k] for f in folds])), 4),
               "sd": round(float(np.std([f[k] for f in folds], ddof=1)), 4)}
           for k in ["roc_auc", "accuracy", "brier", "sensitivity", "specificity"]}
    return folds, agg


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(OUT_DIR / "features_trial.csv")
    X = df[FEATURES].copy()
    y = df["condition"]
    groups = df["subject_id"]

    models = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        "random_forest": RandomForestClassifier(
            n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1),
    }

    results = {"n_trials": int(len(df)), "n_subjects": int(groups.nunique()),
               "n_features": len(FEATURES), "cv": "GroupKFold(n_splits=4) by subject",
               "features": FEATURES, "models": {}}
    rows = []
    for name, model in models.items():
        folds, agg = evaluate(model, X, y, groups)
        results["models"][name] = {"per_fold": folds, "aggregate": agg}
        for f in folds:
            rows.append({"model": name, **{k: f[k] for k in
                         ["fold", "roc_auc", "accuracy", "brier",
                          "sensitivity", "specificity", "n_test"]}})
        print(f"{name}: ROC-AUC {agg['roc_auc']['mean']:.3f}±{agg['roc_auc']['sd']:.3f}, "
              f"acc {agg['accuracy']['mean']:.3f}±{agg['accuracy']['sd']:.3f}, "
              f"Brier {agg['brier']['mean']:.3f}±{agg['brier']['sd']:.3f}, "
              f"sens {agg['sensitivity']['mean']:.3f}, spec {agg['specificity']['mean']:.3f}")

    best = max(results["models"], key=lambda m: results["models"][m]["aggregate"]["roc_auc"]["mean"])
    results["best_model"] = best
    with open(OUT_DIR / "ml_results.json", "w") as f:
        json.dump(results, f, indent=2)
    pd.DataFrame(rows).to_csv(OUT_DIR / "ml_metrics.csv", index=False)
    print(f"best model by mean ROC-AUC: {best}")
    print(f"note: {len(df)} trials from only {groups.nunique()} subjects — "
          "metrics are a demo on synthetic data, not a clinical claim.")
    print("outputs: outputs/ml_results.json, outputs/ml_metrics.csv")


if __name__ == "__main__":
    main()
