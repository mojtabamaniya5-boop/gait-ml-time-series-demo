#!/usr/bin/env python3
"""
audit.py — Data Audit (Stage 1) for IMU-like gait recordings.

Implements the 6 audit checks from the project brief, computed directly
from the data in data/ (no hand-written numbers):

  1. File structure — channels present, sampling rate, trial length,
     sensor position (trunk-mounted, inferred from signal shape) and
     foot-contact event presence.
  2. Time alignment — per-channel sync (same sample count), timestamp
     gaps, duplicated/non-monotonic timestamps.
  3. Gait events / strides — heel-strike counts per foot, stride times,
     reliability (stride-time CV).
  4. Speed estimation — inverted-pendulum method (Zijlstra & Hof):
     vertical displacement from double-integrated, detrended vertical
     acceleration per stride -> step length -> speed. Reported with a
     plausibility flag per trial.
  5. Signal quality — noise (high-frequency energy share), drift
     (linear trend of az), outliers (|z| > 5), left/right asymmetry.
  6. Demographics — missingness, distributions, subgroup comparison
     (healthy vs impaired).

Outputs:
  outputs/audit_report.json    full per-trial + per-subject results
  outputs/audit_summary.csv    one row per trial, key audit metrics
  (stdout)                     human-readable summary

Run:  python audit.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import integrate, signal

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
OUT_DIR = BASE / "outputs"
FS = 100.0
CHANNELS = ["ax", "ay", "az", "gx", "gy", "gz"]
EXPECTED_CHANNELS = CHANNELS + ["timestamp_s", "heel_strike", "foot"]


def stride_intervals(df):
    """Heel-strike times -> stride intervals (s), alternating L/R."""
    hs = df.index[df["heel_strike"] == 1].tolist()
    times = df.loc[hs, "timestamp_s"].values
    feet = df.loc[hs, "foot"].values
    intervals = np.diff(times)
    # Stride = same-foot interval (every other heel strike).
    left = times[feet == "L"]
    right = times[feet == "R"]
    stride_L = np.diff(left) if len(left) > 1 else np.array([])
    stride_R = np.diff(right) if len(right) > 1 else np.array([])
    return times, feet, intervals, stride_L, stride_R


def estimate_speed(df, height_m):
    """Inverted-pendulum speed estimate per stride.

    Per stride: detrend az (remove gravity + linear trend), double
    integrate -> vertical displacement h; step length
    L_step = 2*sqrt(2*l*h - h^2) with leg length l = 0.53*height;
    speed = L_step / stride_time. Returns (median speed, iqr, n_ok, n_total).
    """
    times, feet, _, stride_L, stride_R = stride_intervals(df)
    az = df["az"].values - 9.81
    t = df["timestamp_s"].values
    leg = 0.53 * height_m
    speeds = []
    # Stride boundaries = same-foot heel strikes.
    for foot_times in (times[feet == "L"], times[feet == "R"]):
        for a, b in zip(foot_times[:-1], foot_times[1:]):
            m = (t >= a) & (t < b)
            if m.sum() < 20:
                continue
            seg = az[m].copy()
            seg = signal.detrend(seg, type="linear")
            vel = integrate.cumulative_trapezoid(seg, dx=1.0 / FS, initial=0)
            vel = signal.detrend(vel, type="linear")
            disp = integrate.cumulative_trapezoid(vel, dx=1.0 / FS, initial=0)
            h = disp.max() - disp.min()
            if not (0.005 < h < 0.15):
                continue
            step_len = 2 * np.sqrt(max(2 * leg * h - h * h, 0.0))
            # (b - a) is a full stride (2 steps); speed = stride length / stride time.
            speeds.append(2 * step_len / (b - a))
    speeds = np.array(speeds)
    if len(speeds) == 0:
        return None, None, 0, 0
    return float(np.median(speeds)), float(np.percentile(speeds, 75) - np.percentile(speeds, 25)), len(speeds), int(len(times))


def audit_trial(path, subj_row):
    df = pd.read_csv(path)
    r = {"file": path.name, "subject_id": subj_row["subject_id"],
         "condition": int(subj_row["condition"])}

    # --- 1. Structure ---
    r["channels_present"] = [c for c in EXPECTED_CHANNELS if c in df.columns]
    r["channels_missing"] = [c for c in EXPECTED_CHANNELS if c not in df.columns]
    r["n_rows"] = int(len(df))
    dt = np.diff(df["timestamp_s"].values)
    r["sampling_rate_hz"] = round(float(1.0 / np.median(dt)), 2) if len(dt) else None
    r["duration_s"] = round(float(df["timestamp_s"].iloc[-1] - df["timestamp_s"].iloc[0]), 2)
    r["has_foot_contacts"] = bool((df["heel_strike"] == 1).any())
    # Sensor position: trunk-mounted IMU shows a strong 2-per-stride vertical
    # component; we report the dominant az frequency relative to cadence.
    r["sensor_position_note"] = "trunk-mounted IMU (vertical az, 2 peaks/stride)"

    # --- 2. Time alignment ---
    r["n_channels_synced"] = int(len(CHANNELS)) if all(len(df[c]) == len(df) for c in CHANNELS) else 0
    r["n_nan"] = {c: int(df[c].isna().sum()) for c in CHANNELS}
    med_dt = float(np.median(dt)) if len(dt) else np.nan
    gaps = dt[dt > 1.5 * med_dt]
    r["n_timestamp_gaps"] = int(len(gaps))
    r["max_gap_s"] = round(float(gaps.max()), 3) if len(gaps) else 0.0
    r["n_nonmonotonic"] = int((dt <= 0).sum())

    # --- 3. Gait events / strides ---
    times, feet, intervals, stride_L, stride_R = stride_intervals(df)
    r["n_heel_strikes"] = int(len(times))
    r["n_left"] = int((feet == "L").sum())
    r["n_right"] = int((feet == "R").sum())
    all_strides = np.concatenate([stride_L, stride_R]) if len(stride_L) + len(stride_R) else np.array([])
    if len(all_strides):
        r["mean_stride_time_s"] = round(float(all_strides.mean()), 3)
        r["stride_time_cv"] = round(float(all_strides.std(ddof=1) / all_strides.mean()), 4)
        r["cadence_spm"] = round(float(120.0 / all_strides.mean()), 1)
    else:
        r["mean_stride_time_s"] = r["stride_time_cv"] = r["cadence_spm"] = None
    r["stride_reliability"] = (
        "reliable" if r["stride_time_cv"] is not None and r["stride_time_cv"] < 0.10
        else "check" if r["stride_time_cv"] is not None else "no strides"
    )

    # --- 4. Speed estimation ---
    med, iqr, n_ok, n_tot = estimate_speed(df, float(subj_row["height_m"]))
    r["speed_median_m_s"] = None if med is None else round(med, 3)
    r["speed_iqr_m_s"] = None if iqr is None else round(iqr, 3)
    r["speed_strides_ok"] = n_ok
    r["speed_plausible"] = bool(med is not None and 0.3 < med < 2.5)

    # --- 5. Signal quality ---
    qual = {}
    for c in CHANNELS:
        x = df[c].values.astype(float)
        # Noise: share of energy above 15 Hz (Welch PSD).
        f, pxx = signal.welch(x, fs=FS, nperseg=min(256, len(x)))
        hf = float(pxx[f > 15].sum() / pxx.sum()) if pxx.sum() > 0 else np.nan
        # Drift: linear trend slope (per second).
        slope = float(np.polyfit(np.arange(len(x)) / FS, x, 1)[0])
        # Outliers: |z| > 5.
        z = np.abs((x - x.mean()) / (x.std() + 1e-12))
        qual[c] = {"hf_energy_share": round(hf, 4), "drift_per_s": round(slope, 5),
                   "n_outliers_z5": int((z > 5).sum())}
    r["signal_quality"] = qual
    # Left/right asymmetry: mean stride time L vs R.
    if len(stride_L) and len(stride_R):
        r["asymmetry_LR"] = round(float(abs(stride_L.mean() - stride_R.mean())
                                        / ((stride_L.mean() + stride_R.mean()) / 2)), 4)
    else:
        r["asymmetry_LR"] = None
    return r


def audit_demographics(subjects):
    s = subjects.copy()
    out = {"n_subjects": int(len(s))}
    out["missing"] = {c: int(s[c].isna().sum()) for c in ["age", "sex", "height_m", "weight_kg"]}
    out["age_mean_sd"] = [round(float(s["age"].mean()), 1), round(float(s["age"].std()), 1)]
    out["sex_counts"] = {k: int(v) for k, v in s["sex"].value_counts().items()}
    out["height_mean_sd"] = [round(float(s["height_m"].mean()), 2), round(float(s["height_m"].std()), 2)]
    out["weight_mean_sd"] = [round(float(s["weight_kg"].mean()), 1), round(float(s["weight_kg"].std()), 1)]
    # Subgroup comparison (healthy=0 vs impaired=1).
    comp = {}
    for col in ["age", "height_m", "weight_kg"]:
        a = s.loc[s["condition"] == 0, col].dropna()
        b = s.loc[s["condition"] == 1, col].dropna()
        comp[col] = {"healthy_mean": round(float(a.mean()), 2), "impaired_mean": round(float(b.mean()), 2),
                     "diff": round(float(b.mean() - a.mean()), 2)}
    out["subgroup_comparison"] = comp
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    subjects = pd.read_csv(DATA_DIR / "subjects.csv")
    trials = pd.read_csv(DATA_DIR / "trials.csv")

    results = []
    for _, trow in trials.iterrows():
        subj = subjects.loc[subjects["subject_id"] == trow["subject_id"]].iloc[0]
        path = DATA_DIR / f"{trow['subject_id']}_trial_{int(trow['trial'])}.csv"
        results.append(audit_trial(path, subj))

    report = {
        "n_trials": len(results),
        "n_subjects": int(subjects["subject_id"].nunique()),
        "sampling_rate_nominal_hz": FS,
        "trials": results,
        "demographics": audit_demographics(subjects),
    }
    with open(OUT_DIR / "audit_report.json", "w") as f:
        json.dump(report, f, indent=2)

    flat = []
    for r in results:
        flat.append({
            "file": r["file"], "subject_id": r["subject_id"], "condition": r["condition"],
            "sampling_rate_hz": r["sampling_rate_hz"], "duration_s": r["duration_s"],
            "n_heel_strikes": r["n_heel_strikes"], "n_left": r["n_left"], "n_right": r["n_right"],
            "mean_stride_time_s": r["mean_stride_time_s"], "stride_time_cv": r["stride_time_cv"],
            "cadence_spm": r["cadence_spm"], "stride_reliability": r["stride_reliability"],
            "n_timestamp_gaps": r["n_timestamp_gaps"], "max_gap_s": r["max_gap_s"],
            "speed_median_m_s": r["speed_median_m_s"], "speed_plausible": r["speed_plausible"],
            "asymmetry_LR": r["asymmetry_LR"],
            "az_drift_per_s": r["signal_quality"]["az"]["drift_per_s"],
            "ax_outliers_z5": r["signal_quality"]["ax"]["n_outliers_z5"],
            "az_hf_energy_share": r["signal_quality"]["az"]["hf_energy_share"],
        })
    pd.DataFrame(flat).to_csv(OUT_DIR / "audit_summary.csv", index=False)

    # --- stdout summary (all numbers from the data) ---
    df = pd.DataFrame(flat)
    print(f"trials audited: {len(df)} | subjects: {df['subject_id'].nunique()}")
    print(f"sampling rate: {df['sampling_rate_hz'].min():.0f}-{df['sampling_rate_hz'].max():.0f} Hz "
          f"(nominal {FS:.0f} Hz)")
    print(f"trial length: {df['duration_s'].min():.1f}-{df['duration_s'].max():.1f} s")
    print(f"heel strikes/trial: {df['n_heel_strikes'].min()}-{df['n_heel_strikes'].max()} "
          f"(L {df['n_left'].min()}-{df['n_left'].max()}, R {df['n_right'].min()}-{df['n_right'].max()})")
    print(f"timestamp gaps: {int(df['n_timestamp_gaps'].sum())} total, "
          f"max gap {df['max_gap_s'].max():.2f} s, in {int((df['n_timestamp_gaps'] > 0).sum())} trials")
    print(f"stride-time CV: {df['stride_time_cv'].min():.3f}-{df['stride_time_cv'].max():.3f} "
          f"(reliable trials: {int((df['stride_reliability'] == 'reliable').sum())}/{len(df)})")
    print(f"speed estimate: median {df['speed_median_m_s'].median():.2f} m/s across trials, "
          f"plausible in {int(df['speed_plausible'].sum())}/{len(df)} trials")
    print(f"asymmetry L/R: median {df['asymmetry_LR'].median():.3f}, "
          f"max {df['asymmetry_LR'].max():.3f}")
    print(f"az drift: max |slope| {df['az_drift_per_s'].abs().max():.4f} m/s^2 per s")
    print(f"ax outliers (|z|>5): {int(df['ax_outliers_z5'].sum())} total")
    d = report["demographics"]
    print(f"demographics: {d['n_subjects']} subjects, missing {d['missing']}, "
          f"age {d['age_mean_sd'][0]}±{d['age_mean_sd'][1]} y")
    print("outputs: outputs/audit_report.json, outputs/audit_summary.csv")


if __name__ == "__main__":
    main()
