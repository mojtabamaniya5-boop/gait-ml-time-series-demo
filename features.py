#!/usr/bin/env python3
"""
features.py — feature extraction from IMU-like gait signals.

Per trial, computes window/statistical/stride features from the raw
channels and the heel-strike events, then aggregates to one row per
subject (mean across trials) for the subject-level model.

Features (per trial):
  - cadence_spm, mean_stride_time_s, stride_time_cv (from heel strikes)
  - step_time_asymmetry: |mean L step - mean R step| / mean step
  - speed_median_m_s (inverted-pendulum estimate, see audit.py)
  - per channel (ax, ay, az, gx, gy, gz): RMS, std, range, jerk RMS
    (jerk = first difference * fs), dominant frequency (Welch peak 0.3-5 Hz)
  - signal magnitude area (SMA) of acceleration
  - amplitude asymmetry: |RMS of L strides - RMS of R strides| / mean RMS (az)

Outputs:
  outputs/features_trial.csv     one row per trial
  outputs/features_subject.csv   one row per subject (mean of trials) + condition

Run:  python features.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
OUT_DIR = BASE / "outputs"
FS = 100.0
CHANNELS = ["ax", "ay", "az", "gx", "gy", "gz"]


def stride_features(df):
    hs = df.index[df["heel_strike"] == 1].tolist()
    times = df.loc[hs, "timestamp_s"].values
    feet = df.loc[hs, "foot"].values
    step_intervals = np.diff(times)
    out = {}
    if len(step_intervals):
        out["mean_step_time_s"] = float(step_intervals.mean())
        l_steps = step_intervals[feet[:-1] == "L"]
        r_steps = step_intervals[feet[:-1] == "R"]
        if len(l_steps) and len(r_steps):
            out["step_time_asymmetry"] = float(
                abs(l_steps.mean() - r_steps.mean()) / step_intervals.mean())
        else:
            out["step_time_asymmetry"] = np.nan
    else:
        out["mean_step_time_s"] = np.nan
        out["step_time_asymmetry"] = np.nan
    # Same-foot stride times.
    strides = []
    for ft in ("L", "R"):
        tt = times[feet == ft]
        strides.extend(np.diff(tt).tolist())
    strides = np.array(strides)
    if len(strides):
        out["mean_stride_time_s"] = float(strides.mean())
        out["stride_time_cv"] = float(strides.std(ddof=1) / strides.mean())
        out["cadence_spm"] = float(120.0 / strides.mean())
    else:
        out["mean_stride_time_s"] = out["stride_time_cv"] = out["cadence_spm"] = np.nan
    return out


def channel_features(df):
    out = {}
    for c in CHANNELS:
        x = df[c].values.astype(float)
        out[f"{c}_rms"] = float(np.sqrt(np.mean(x ** 2)))
        out[f"{c}_std"] = float(x.std())
        out[f"{c}_range"] = float(x.max() - x.min())
        jerk = np.diff(x) * FS
        out[f"{c}_jerk_rms"] = float(np.sqrt(np.mean(jerk ** 2)))
        f, pxx = signal.welch(x, fs=FS, nperseg=min(256, len(x)))
        band = (f >= 0.3) & (f <= 5.0)
        out[f"{c}_dom_freq_hz"] = float(f[band][np.argmax(pxx[band])]) if band.any() else np.nan
    # Signal magnitude area of acceleration (gravity removed from az).
    acc = np.abs(df[["ax", "ay"]].values).sum(axis=1) + np.abs(df["az"].values - 9.81)
    out["sma_acc"] = float(acc.mean())
    return out


def amplitude_asymmetry(df):
    """|RMS(az) on L strides - RMS(az) on R strides| / mean RMS."""
    hs = df.index[df["heel_strike"] == 1].tolist()
    times = df.loc[hs, "timestamp_s"].values
    feet = df.loc[hs, "foot"].values
    t = df["timestamp_s"].values
    az = (df["az"].values - 9.81)
    rms = {}
    for ft in ("L", "R"):
        tt = times[feet == ft]
        vals = []
        for a, b in zip(tt[:-1], tt[1:]):
            m = (t >= a) & (t < b)
            if m.sum() > 10:
                vals.append(np.sqrt(np.mean(az[m] ** 2)))
        rms[ft] = float(np.mean(vals)) if vals else np.nan
    if np.isnan(rms["L"]) or np.isnan(rms["R"]) or (rms["L"] + rms["R"]) == 0:
        return np.nan
    return float(abs(rms["L"] - rms["R"]) / ((rms["L"] + rms["R"]) / 2))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    subjects = pd.read_csv(DATA_DIR / "subjects.csv")
    trials = pd.read_csv(DATA_DIR / "trials.csv")
    audit = pd.read_csv(OUT_DIR / "audit_summary.csv")
    audit = audit.set_index("file")

    rows = []
    for _, trow in trials.iterrows():
        fname = f"{trow['subject_id']}_trial_{int(trow['trial'])}.csv"
        df = pd.read_csv(DATA_DIR / fname)
        subj = subjects.loc[subjects["subject_id"] == trow["subject_id"]].iloc[0]
        r = {"subject_id": trow["subject_id"], "trial": int(trow["trial"]),
             "condition": int(subj["condition"]), "age": int(subj["age"]),
             "sex": subj["sex"], "height_m": float(subj["height_m"])}
        r.update(stride_features(df))
        r.update(channel_features(df))
        r["amp_asymmetry_az"] = amplitude_asymmetry(df)
        # Speed from the audit (same method, no recomputation here).
        r["speed_median_m_s"] = float(audit.loc[fname, "speed_median_m_s"])
        rows.append(r)

    trial_df = pd.DataFrame(rows)
    trial_df.to_csv(OUT_DIR / "features_trial.csv", index=False)

    # Subject-level aggregation (mean across trials).
    drop = ["trial"]
    subj_df = (trial_df.drop(columns=drop)
               .groupby(["subject_id", "condition", "age", "sex", "height_m"], as_index=False)
               .mean(numeric_only=True))
    subj_df.to_csv(OUT_DIR / "features_subject.csv", index=False)

    print(f"trial features: {len(trial_df)} rows x {len(trial_df.columns)} cols")
    print(f"subject features: {len(subj_df)} rows x {len(subj_df.columns)} cols")
    print("condition means (subject level):")
    print(subj_df.groupby("condition")[["cadence_spm", "stride_time_cv",
          "step_time_asymmetry", "speed_median_m_s", "amp_asymmetry_az",
          "sma_acc"]].mean().round(3).to_string())
    print("outputs: outputs/features_trial.csv, outputs/features_subject.csv")


if __name__ == "__main__":
    main()
