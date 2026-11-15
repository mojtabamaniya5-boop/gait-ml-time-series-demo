#!/usr/bin/env python3
"""
generate_data.py — synthesize IMU-like gait data for demonstration.

Generates multi-subject, multi-trial accelerometer/gyroscope time series
that mimic wearable IMU gait recordings (100 Hz), with realistic
imperfections: sensor noise, slow drift, occasional spikes, timestamp
gaps, and some missing demographics.

IMPORTANT: this is SYNTHETIC data, generated for demonstration only.
No real patients, no real Xsens/IMU hardware, no real clinical data.

Output:
  data/subject_XX_trial_Y.csv   per-trial signals + foot-contact events
  data/subjects.csv             subject demographics + condition label
  data/trials.csv               per-trial metadata (duration, gaps, drift)

Run:  python generate_data.py
"""

import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
FS = 100.0          # sampling rate (Hz)
N_SUBJECTS = 12
TRIALS_PER_SUBJECT = 4
N_STRIDES_PER_TRIAL = 20
SEED = 42


def gait_waveform(phase, harmonics):
    """Sum-of-harmonics periodic waveform over stride phase in [0, 1)."""
    out = np.zeros_like(phase)
    for amp, k, phi in harmonics:
        out += amp * np.sin(2.0 * np.pi * k * phase + phi)
    return out


# Harmonic templates (amplitude, harmonic number, phase) per channel.
# Vertical acceleration has the classic double-peak-per-stride shape.
CHANNELS = {
    "ax": [(0.35, 1, 0.4), (0.12, 2, 1.9)],                    # mediolateral
    "ay": [(0.55, 1, 2.1), (0.20, 2, 0.6)],                    # anteroposterior
    "az": [(2.00, 2, 0.0), (0.30, 1, 1.2), (0.40, 4, 2.4)],    # vertical (w/o gravity)
    "gx": [(0.55, 1, 0.9), (0.18, 2, 2.8)],                    # gyro roll
    "gy": [(0.90, 2, 0.3), (0.30, 1, 1.7)],                    # gyro pitch
    "gz": [(0.40, 1, 1.5), (0.15, 2, 0.5)],                    # gyro yaw
}
NOISE_SD = {"ax": 0.06, "ay": 0.06, "az": 0.08, "gx": 0.05, "gy": 0.05, "gz": 0.05}


def make_subjects(rng):
    subjects = []
    for i in range(N_SUBJECTS):
        sid = f"S{i + 1:02d}"
        condition = 1 if i >= N_SUBJECTS // 2 else 0  # 6 healthy, 6 impaired
        age = int(rng.normal(62, 8)) if condition == 0 else int(rng.normal(68, 9))
        age = int(np.clip(age, 45, 85))
        sex = "F" if rng.random() < 0.5 else "M"
        height = float(np.clip(rng.normal(1.68 if sex == "M" else 1.58, 0.07), 1.45, 1.90))
        weight = float(np.clip(rng.normal(78 if sex == "M" else 64, 10), 45, 110))
        # Cadence: healthy ~108 steps/min, impaired ~88 with more variability.
        cadence = float(rng.normal(108, 5)) if condition == 0 else float(rng.normal(88, 7))
        stride_cv = float(rng.uniform(0.02, 0.04)) if condition == 0 else float(rng.uniform(0.06, 0.10))
        # Left/right asymmetry factor (impaired: larger).
        asym = float(rng.uniform(0.01, 0.04)) if condition == 0 else float(rng.uniform(0.05, 0.12))
        subjects.append(
            dict(subject_id=sid, condition=condition, age=age, sex=sex,
                 height_m=round(height, 2), weight_kg=round(weight, 1),
                 cadence_spm=round(cadence, 1), stride_cv=stride_cv, asym=round(asym, 3))
        )
    # Realistic missing demographics: weight missing for 2 subjects.
    subjects[3]["weight_kg"] = None
    subjects[9]["weight_kg"] = None
    return subjects


def make_trial(rng, subj, trial_idx):
    """Build one trial. Returns (rows dict, trial meta)."""
    mean_stride = 120.0 / subj["cadence_spm"]  # seconds per stride (2 steps)
    stride_times = rng.normal(mean_stride, mean_stride * subj["stride_cv"],
                              N_STRIDES_PER_TRIAL)
    stride_times = np.clip(stride_times, 0.6, 2.0)
    starts = np.concatenate([[0.0], np.cumsum(stride_times)])
    duration = starts[-1]
    t = np.arange(0.0, duration, 1.0 / FS)

    # Stride index + fractional phase for each sample.
    idx = np.searchsorted(starts, t, side="right") - 1
    idx = np.clip(idx, 0, N_STRIDES_PER_TRIAL - 1)
    phase = (t - starts[idx]) / stride_times[idx]

    # Left/right alternation: even strides = left, odd = right.
    is_right = (idx % 2 == 1)
    asym_amp = np.where(is_right, 1.0 + subj["asym"], 1.0 - subj["asym"])

    data = {"timestamp_s": t}
    for ch, harmonics in CHANNELS.items():
        sig = gait_waveform(phase, harmonics) * asym_amp
        sig = sig + rng.normal(0, NOISE_SD[ch], size=t.shape)
        data[ch] = sig
    data["az"] = data["az"] + 9.81  # add gravity to vertical channel

    # Slow drift on az in ~1/3 of trials (sensor drift imperfection).
    drift = 0.0
    if rng.random() < 0.35:
        drift = float(rng.uniform(0.01, 0.03))  # m/s^2 per second
        data["az"] = data["az"] + drift * t

    # Occasional spikes (outliers) on ax.
    n_spikes = int(rng.integers(2, 6))
    spike_idx = rng.choice(len(t), size=n_spikes, replace=False)
    data["ax"][spike_idx] += rng.choice([-1, 1], size=n_spikes) * rng.uniform(2.0, 4.0, n_spikes)

    # Foot-contact events: two heel strikes per gait cycle (one per step),
    # at phase 0 and phase 0.5, alternating feet.
    hs = np.zeros(len(t), dtype=int)
    foot = np.array([""] * len(t), dtype=object)
    for k, s in enumerate(starts[:-1]):
        for j, ph in enumerate((0.0, 0.5)):
            tt = s + ph * stride_times[k]
            jj = int(np.argmin(np.abs(t - tt)))
            hs[jj] = 1
            foot[jj] = "R" if (2 * k + j) % 2 == 1 else "L"
    data["heel_strike"] = hs
    data["foot"] = foot

    # Timestamp gap imperfection: drop a ~0.3 s block in ~2 trials per subject set.
    gap = False
    gap_start = None
    if rng.random() < 0.15:
        gap = True
        gap_start = float(rng.uniform(5.0, duration - 5.0))
        keep = ~((t >= gap_start) & (t < gap_start + 0.3))
        data = {k: (v[keep] if isinstance(v, np.ndarray) else v) for k, v in data.items()}

    meta = dict(
        subject_id=subj["subject_id"], trial=trial_idx + 1,
        condition=subj["condition"], duration_s=round(float(data["timestamp_s"][-1]), 2),
        n_samples=int(len(data["timestamp_s"])),
        n_strides=int(data["heel_strike"].sum() // 2),
        drift_az_per_s=round(drift, 4), timestamp_gap=gap,
        gap_start_s=(round(gap_start, 2) if gap_start is not None else None),
    )
    return data, meta


def main():
    rng = np.random.default_rng(SEED)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    subjects = make_subjects(rng)
    trials_meta = []

    for subj in subjects:
        for tr in range(TRIALS_PER_SUBJECT):
            data, meta = make_trial(rng, subj, tr)
            fname = DATA_DIR / f"{subj['subject_id']}_trial_{tr + 1}.csv"
            import csv
            with open(fname, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["timestamp_s", "ax", "ay", "az", "gx", "gy", "gz",
                            "heel_strike", "foot"])
                for i in range(len(data["timestamp_s"])):
                    w.writerow([
                        f"{data['timestamp_s'][i]:.3f}",
                        f"{data['ax'][i]:.4f}", f"{data['ay'][i]:.4f}",
                        f"{data['az'][i]:.4f}", f"{data['gx'][i]:.4f}",
                        f"{data['gy'][i]:.4f}", f"{data['gz'][i]:.4f}",
                        int(data["heel_strike"][i]), data["foot"][i],
                    ])
            trials_meta.append(meta)

    import csv
    with open(DATA_DIR / "subjects.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["subject_id", "condition", "age", "sex",
                                          "height_m", "weight_kg", "cadence_spm",
                                          "stride_cv", "asym"])
        w.writeheader()
        w.writerows(subjects)
    with open(DATA_DIR / "trials.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(trials_meta[0].keys()))
        w.writeheader()
        w.writerows(trials_meta)

    print(f"subjects: {len(subjects)} (6 healthy / 6 impaired)")
    print(f"trials:   {len(trials_meta)} files in {DATA_DIR}")
    print(f"sampling: {FS:.0f} Hz, ~{N_STRIDES_PER_TRIAL} strides/trial")
    print("imperfections: noise, az drift (35% of trials), spikes, "
          "timestamp gaps (15% of trials), weight missing for S04/S10")


if __name__ == "__main__":
    main()
