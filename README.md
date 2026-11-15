# Gait ML — Time-Series Demo (Data Audit + ML Pipeline + SHAP + Mixed Effects)

A small, fully working demonstration of the exact workflow used for
IMU/joint-kinematics gait analysis:

1. **Data Audit** — structure, time alignment, gait events, speed
   estimation, signal quality, demographics.
2. **Feature extraction** — window/statistical/stride features.
3. **ML pipeline** — subject-level GroupKFold cross-validation,
   two model configs compared, discrimination + calibration.
4. **XAI & statistics** — SHAP explanations with a fold-stability
   check, mixed-effects models, participant-level bootstrap.

## ⚠️ Honest framing — read this first

- **The data is synthetic.** `generate_data.py` synthesizes IMU-like
  gait signals (accelerometer + gyroscope, 100 Hz, heel-strike events)
  from a harmonic gait model. There are **no real patients, no real
  Xsens/IMU hardware, and no real clinical data** in this repository.
- **The group difference is built in by construction.** The two
  simulated conditions differ in cadence and stride variability on
  purpose, so the classifiers separate them almost perfectly
  (ROC-AUC 1.00). These numbers demonstrate that the *pipeline*
  runs correctly — they are **not** a clinical performance claim.
- **Small sample.** 12 subjects × 4 trials = 48 trials. Confidence
  intervals are wide and any real study would need far more subjects.
- **Speed is estimated**, not measured: the inverted-pendulum method
  (Zijlstra & Hof) derives step length from vertical displacement.
  It is reported with a plausibility flag per trial.

## Layout

```
generate_data.py   synthesize 12 subjects x 4 trials -> data/
audit.py           6 Data Audit checks -> outputs/audit_report.json, audit_summary.csv
features.py        per-trial + per-subject features -> outputs/features_*.csv
ml_pipeline.py     GroupKFold CV, LR vs RF -> outputs/ml_results.json, ml_metrics.csv
xai_stats.py       SHAP + MixedLM + bootstrap -> outputs/shap_*.csv/json,
                   outputs/mixedlm_summary.txt, outputs/stats_results.json
data/              generated CSVs (subjects.csv, trials.csv, *_trial_*.csv)
outputs/           all computed results (JSON/CSV/TXT — no figures)
```

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python generate_data.py
python audit.py
python features.py
python ml_pipeline.py
python xai_stats.py
```

Total runtime is under a minute on a laptop. Each script is
self-contained and reads only the files produced by the previous step.

## What the numbers say (on this synthetic data)

Data Audit (48 trials, 12 subjects):

- Sampling rate 100 Hz in all trials; trial length 21–30 s;
  39–40 heel strikes per trial (L/R balanced).
- 8 timestamp gaps (max 0.31 s) in 8 trials — detected, not hidden.
- Stride-time CV 0.021–0.171; 44/48 trials rated "reliable".
- Speed (inverted pendulum): median 1.10 m/s, plausible in 48/48 trials.
- az drift up to 0.026 m/s² per second; 172 ax outliers (|z| > 5) —
  the injected spikes, recovered by the audit.
- Demographics: age 65.2 ± 6.2 y; weight missing for 2/12 subjects.

ML (GroupKFold by subject, 4 folds, 16 features):

| model | ROC-AUC | accuracy | Brier |
|---|---|---|---|
| logistic regression | 1.000 ± 0.000 | 0.917 ± 0.118 | 0.046 ± 0.047 |
| random forest | 1.000 ± 0.000 | 0.958 ± 0.083 | 0.041 ± 0.064 |

SHAP (mean |SHAP|, both models agree on the top-3):
`mean_stride_time_s`, `cadence_spm`, `amp_asymmetry_az`.
Stability across folds: mean Spearman ρ 0.85 (LR) / 0.90 (RF).

Mixed-effects (trial level, random intercept per subject):

- `stride_time_cv ~ condition + age`: condition +3.61 percentage
  points (p = 0.0006), converged.
- `cadence_spm ~ condition + age`: condition −19.9 steps/min (p < 0.001).

Participant-level bootstrap (2000 resamples over subjects):
impaired − healthy stride_time_cv = 0.033, 95% CI [0.013, 0.053].

## Notes for a real project

- Replace `generate_data.py` with the real IMU export; the audit,
  features, and modeling scripts only assume the CSV schema
  (`timestamp_s, ax, ay, az, gx, gy, gz, heel_strike, foot`).
- The audit is written to *find* problems (gaps, drift, missing
  demographics, unreliable strides) — on real data, expect the
  "check" flags to fire and the Stage-1 report to list them.
- With real, noisier data, add: sensor re-calibration, stride
  segmentation from the signal itself (not just event flags),
  and nested CV for hyperparameter tuning.
