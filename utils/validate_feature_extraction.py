"""Quick validation script for feature extraction.

Generates or loads mock data using load_gazebase_data (invoked with an invalid path
so the loader produces synthetic data), preprocesses it, runs feature extraction,
and prints summary stats to stdout.

Run (from repo root):
    python -m utils.validate_feature_extraction
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from data.gazebase_loader import load_gazebase_data, preprocess_gaze_data, validate_data_quality
from pipeline.feature_extractor import extract_gaze_features


def main():
    # Step 1: get synthetic mock data (loader will raise if we pass empty string now; so we trigger mock manually)
    # Construct minimal synthetic sample matching loader schema
    rng = np.random.default_rng(7)
    users = 3
    samples = 90 * 20  # 20 seconds at 90 Hz
    rows = []
    for u in range(1, users + 1):
        t_sec = np.arange(samples) / 90.0
        x = np.clip(0.5 + np.cumsum(rng.normal(0, 0.005, size=samples)), 0, 1)
        y = np.clip(0.5 + np.cumsum(rng.normal(0, 0.005, size=samples)), 0, 1)
        for ti, xv, yv in zip(t_sec, x, y):
            rows.append({
                "user_id": u,
                "timestamp_ms": float(ti * 1000.0),
                "gaze_x": float(xv),
                "gaze_y": float(yv),
                "session": 1,
                "round": 1,
                "task": "VRG",
            })
    df_raw = pd.DataFrame(rows)

    print(f"Raw rows: {len(df_raw)}  Columns: {list(df_raw.columns)}")

    # Step 2: preprocess
    df_pre = preprocess_gaze_data(df_raw)
    quality = validate_data_quality(df_pre)
    print("Quality checks:")
    for k, v in quality.items():
        print(f"  {k}: {v}")

    # Step 3: feature extraction
    feats = extract_gaze_features(df_pre, window_size_sec=5, overlap_sec=1)
    print(f"Feature rows: {len(feats)}  Feature cols: {len(feats.columns)}")
    print("First 5 feature rows:")
    print(feats.head().to_string())

    # Step 4: basic sanity assertions
    required_feature_keys = [
        "fix_ratio", "fix_count", "sac_count", "vel_mean", "path_len", "grid_entropy"
    ]
    missing = [k for k in required_feature_keys if k not in feats.columns]
    if missing:
        print(f"WARNING: Missing expected feature columns: {missing}")
    else:
        print("All expected core feature columns present.")

    if not feats.empty:
        if feats["t_end"].max() <= feats["t_start"].min():
            print("WARNING: Window time boundaries look incorrect.")
        if (feats.filter(like='vel_').isna().all().all()):
            print("WARNING: All velocity stats are NaN.")

    print("Validation complete.")


if __name__ == "__main__":
    main()
