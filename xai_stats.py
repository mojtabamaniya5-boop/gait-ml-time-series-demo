#!/usr/bin/env python3
"""
xai_stats.py — SHAP explanations, mixed-effects statistics, bootstrap.

Three analyses, all computed from the real outputs of features.py /
ml_pipeline.py (no hand-written numbers):

  1. SHAP — for both models (logistic regression via LinearExplainer,
     random forest via TreeExplainer), fit on all trials. Stability
     check: refit per GroupKFold fold (by subject) and compare the
     mean-|SHAP| feature rankings across folds (mean pairwise Spearman
     rho + top-3 overlap).
  2. Mixed-effects model — statsmodels MixedLM, trial-level:
     stride_time_cv ~ condition + age, random intercept per subject.
     (Also cadence ~ condition for reference.)
  3. Participant-level bootstrap — cluster bootstrap over subjects
     (resample subjects with replacement, 2000 draws) for the mean
     difference in stride_time_cv between conditions; 95% percentile CI.

Outputs:
  outputs/shap_importance.csv   mean |SHAP| per feature, both models
  outputs/shap_stability.json   fold-stability metrics
  outputs/mixedlm_summary.txt   full MixedLM summaries
  outputs/stats_results.json    MixedLM coefficients + bootstrap CI

Run:  python xai_stats.py
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.formula.api import mixedlm

BASE = Path(__file__).resolve().parent
OUT_DIR = BASE / "outputs"
RANDOM_STATE = 42
N_BOOT = 2000

FEATURES = [
    "cadence_spm", "mean_stride_time_s", "stride_time_cv",
    "step_time_asymmetry", "speed_median_m_s", "amp_asymmetry_az",
    "sma_acc",
    "ax_rms", "ay_rms", "az_rms", "gx_rms", "gy_rms", "gz_rms",
    "az_jerk_rms", "gy_jerk_rms", "az_dom_freq_hz",
]


def shap_for_model(model, X, model_name):
    """Fit model on all data, return mean |SHAP| per feature (DataFrame)."""
    m = clone(model)
    m.fit(X, y_global)
    if model_name == "logistic_regression":
        # Pipeline: scaler + LR. Explain the linear step on scaled data.
        scaler = m.named_steps["standardscaler"]
        lr = m.named_steps["logisticregression"]
        expl = shap.LinearExplainer(lr, scaler.transform(X))
        sv = expl.shap_values(scaler.transform(X))
    else:
        expl = shap.TreeExplainer(m)
        sv = expl.shap_values(X)
        if isinstance(sv, list):  # older shap returns per-class list
            sv = sv[1]
        elif isinstance(sv, np.ndarray) and sv.ndim == 3:  # shap>=0.45: (n, feats, classes)
            sv = sv[:, :, 1]
    return pd.DataFrame({"feature": X.columns,
                         f"mean_abs_shap_{model_name}": np.abs(sv).mean(axis=0)})


def shap_stability(model, X, y, groups, model_name):
    """Mean-|SHAP| ranking per GroupKFold fold -> stability metrics."""
    gkf = GroupKFold(n_splits=4)
    fold_ranks = []
    for tr, _ in gkf.split(X, y, groups):
        m = clone(model)
        m.fit(X.iloc[tr], y.iloc[tr])
        if model_name == "logistic_regression":
            scaler = m.named_steps["standardscaler"]
            lr = m.named_steps["logisticregression"]
            expl = shap.LinearExplainer(lr, scaler.transform(X.iloc[tr]))
            sv = expl.shap_values(scaler.transform(X.iloc[tr]))
        else:
            expl = shap.TreeExplainer(m)
            sv = expl.shap_values(X.iloc[tr])
            if isinstance(sv, list):
                sv = sv[1]
            elif isinstance(sv, np.ndarray) and sv.ndim == 3:
                sv = sv[:, :, 1]
        imp = np.abs(sv).mean(axis=0)
        fold_ranks.append(pd.Series(imp, index=X.columns).rank(ascending=False))
    ranks = pd.DataFrame(fold_ranks)
    rhos = []
    for i in range(len(ranks)):
        for j in range(i + 1, len(ranks)):
            rhos.append(spearmanr(ranks.iloc[i], ranks.iloc[j]).correlation)
    top3 = [set(ranks.iloc[i].nsmallest(3).index) for i in range(len(ranks))]
    overlap = np.mean([len(top3[0] & t) / 3 for t in top3[1:]])
    return {"mean_pairwise_spearman_rho": round(float(np.mean(rhos)), 3),
            "top3_overlap_vs_fold0": round(float(overlap), 3),
            "fold_rankings": ranks.round(1).to_dict(orient="index")}


def mixedlm_analysis(df, formula):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = mixedlm(formula, df, groups=df["subject_id"]).fit(reml=False, maxiter=200)
    return res


def cluster_bootstrap(df_subj, value_col, n_boot=N_BOOT, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    s = df_subj[["subject_id", "condition", value_col]].dropna()
    diffs = []
    ids = s["subject_id"].unique()
    for _ in range(n_boot):
        sampled = rng.choice(ids, size=len(ids), replace=True)
        d = s.set_index("subject_id").loc[sampled].reset_index()
        a = d.loc[d["condition"] == 0, value_col].mean()
        b = d.loc[d["condition"] == 1, value_col].mean()
        diffs.append(b - a)
    diffs = np.array(diffs)
    return {"effect": "impaired - healthy mean " + value_col,
            "observed_diff": round(float(
                s.loc[s["condition"] == 1, value_col].mean()
                - s.loc[s["condition"] == 0, value_col].mean()), 4),
            "bootstrap_mean": round(float(diffs.mean()), 4),
            "ci_95": [round(float(np.percentile(diffs, 2.5)), 4),
                      round(float(np.percentile(diffs, 97.5)), 4)],
            "n_boot": n_boot, "n_subjects": int(len(ids))}


def main():
    global y_global
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(OUT_DIR / "features_trial.csv")
    X = df[FEATURES].copy()
    y_global = df["condition"]
    groups = df["subject_id"]

    models = {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        "random_forest": RandomForestClassifier(
            n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1),
    }

    # --- 1. SHAP ---
    imp = None
    stability = {}
    for name, model in models.items():
        d = shap_for_model(model, X, name)
        imp = d if imp is None else imp.merge(d, on="feature")
        stability[name] = shap_stability(model, X, y_global, groups, name)
        top = d.sort_values(f"mean_abs_shap_{name}", ascending=False).head(3)
        print(f"SHAP {name} top-3: {', '.join(top['feature'])}")
        print(f"  stability: mean Spearman rho {stability[name]['mean_pairwise_spearman_rho']}, "
              f"top-3 overlap {stability[name]['top3_overlap_vs_fold0']}")
    imp = imp.sort_values("mean_abs_shap_random_forest", ascending=False)
    imp.to_csv(OUT_DIR / "shap_importance.csv", index=False)
    with open(OUT_DIR / "shap_stability.json", "w") as f:
        json.dump(stability, f, indent=2)

    # --- 2. Mixed-effects ---
    # stride_time_cv is fit in percentage points (x100): the raw 0-0.1 scale
    # makes the MixedLM optimizer stall (converged=False); scaling is a
    # pure unit change and the coefficient is reported in %-points.
    df["stride_time_cv_pct"] = df["stride_time_cv"] * 100
    summaries = {}
    stats = {}
    for outcome, unit in [("stride_time_cv_pct", "percentage points"),
                          ("cadence_spm", "steps/min")]:
        res = mixedlm_analysis(df, f"{outcome} ~ condition + age")
        summaries[outcome] = str(res.summary())
        p = res.params
        key = "stride_time_cv" if outcome == "stride_time_cv_pct" else outcome
        stats[key] = {
            "condition_coef": round(float(p["condition"]), 4),
            "condition_coef_unit": unit,
            "condition_pvalue": round(float(res.pvalues["condition"]), 4),
            "age_coef": round(float(p["age"]), 4),
            "converged": bool(res.converged),
            "n_obs": int(res.nobs),
        }
        print(f"MixedLM {key} ~ condition + age + (1|subject): "
              f"condition coef {stats[key]['condition_coef']} {unit} "
              f"(p={stats[key]['condition_pvalue']}), converged={stats[key]['converged']}")
    with open(OUT_DIR / "mixedlm_summary.txt", "w") as f:
        for k, v in summaries.items():
            f.write(f"===== {k} ~ condition + age + (1|subject) =====\n{v}\n\n")

    # --- 3. Participant-level bootstrap ---
    subj = pd.read_csv(OUT_DIR / "features_subject.csv")
    boot = cluster_bootstrap(subj, "stride_time_cv")
    stats["bootstrap_stride_time_cv_diff"] = boot
    print(f"bootstrap (subjects, n={boot['n_boot']}): impaired-healthy stride_time_cv "
          f"diff {boot['observed_diff']}, 95% CI [{boot['ci_95'][0]}, {boot['ci_95'][1]}]")

    with open(OUT_DIR / "stats_results.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("outputs: outputs/shap_importance.csv, outputs/shap_stability.json, "
          "outputs/mixedlm_summary.txt, outputs/stats_results.json")


if __name__ == "__main__":
    main()
