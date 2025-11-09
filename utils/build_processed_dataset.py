"""Build processed feature dataset from data/raw subset.

- Loads only PUR/TEX/RAN tasks from data/raw
- Preprocesses (normalize gaze, compute kinematics, clip outliers)
- Extracts windowed features
- Computes per-feature z-score stats from Session 1 (baseline) and applies to all rows (optional)
- Saves features to data/processed/feature_windows.parquet

Run:
  python -m utils.build_processed_dataset --window 5 --overlap 1 --standardize
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from data.gazebase_loader import load_gazebase_data, preprocess_gaze_data
from pipeline.feature_extractor import extract_gaze_features


def compute_and_save_baseline_stats(feats: pd.DataFrame, out_dir: Path) -> Path:
    """Compute per-user baseline stats from Session 1 and save to parquet.

    Returns path to saved parquet.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'baseline_stats.parquet'
    if 'session' not in feats.columns:
        raise ValueError('session column required to compute baseline stats')
    non_feature_cols = {'user_id','t_start','t_end','session'}
    feat_cols = [c for c in feats.columns if c not in non_feature_cols]
    s1 = feats[feats['session']=='S1']
    if s1.empty:
        raise ValueError('No Session 1 rows to compute baseline stats')
    grouped = s1.groupby('user_id')
    mu = grouped[feat_cols].mean(numeric_only=True)
    sigma = grouped[feat_cols].std(numeric_only=True).replace(0, 1.0)
    stats_df = pd.concat({'mean': mu, 'std': sigma}, axis=1)
    stats_df.columns = [f'{lvl2}_{lvl1}' for lvl1, lvl2 in stats_df.columns]
    stats_df = stats_df.reset_index()
    stats_df.to_parquet(out_path, index=False)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw_dir', default='data/raw', help='Raw data directory')
    ap.add_argument('--window', type=float, default=5.0, help='Window size (sec)')
    ap.add_argument('--overlap', type=float, default=1.0, help='Overlap (sec)')
    ap.add_argument('--standardize', action='store_true', help='Apply z-score using S1 baseline stats')
    ap.add_argument('--out', default='data/processed/feature_windows.parquet', help='Output parquet path')
    ap.add_argument('--save_baseline', action='store_true', help='Also compute and save per-user baseline stats from S1')
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_gazebase_data(str(raw_dir), tasks=['PUR','TEX','RAN'])
    df_p = preprocess_gaze_data(df, normalize=True, infer_fixations=False)

    feats = extract_gaze_features(df_p, window_size_sec=args.window, overlap_sec=args.overlap)

    # Quality filtering: remove windows with too few samples or too many NaNs
    if 'sample_count' in feats.columns:
        feats = feats[feats['sample_count'] >= 0.6 * args.window * 90]  # assume ~90Hz; require >=60% of expected samples
    # Drop rows where more than 30% of feature columns are NaN
    feature_cols = [c for c in feats.columns if c not in {'user_id','t_start','t_end','session'}]
    nan_frac = feats[feature_cols].isna().mean(axis=1)
    feats = feats[nan_frac <= 0.3]

    # Attach session info by merging back minimal keys
    key_cols = ['user_id', 'timestamp_sec', 'session']
    if 'session' in df_p.columns:
        # Map window start to nearest session per user using first timestamp per session
        # For robustness, approximate by taking session of the nearest sample at t_start
        # Build a per-user mapping from timestamp to session via merge_asof
        df_keys = df_p[['user_id','timestamp_sec','session']].drop_duplicates().sort_values(['user_id','timestamp_sec'])
        feats = feats.sort_values(['user_id','t_start'])
        merged = []
        for uid, sub in feats.groupby('user_id'):
            keys = df_keys[df_keys['user_id']==uid]
            sub = pd.merge_asof(sub.sort_values('t_start'), keys[['timestamp_sec','session']].rename(columns={'timestamp_sec':'t_start'}), on='t_start')
            merged.append(sub)
        feats = pd.concat(merged, ignore_index=True)

    # Standardize features using S1 baseline (z-score)
    non_feature_cols = {'user_id','t_start','t_end','session'}
    feat_cols = [c for c in feats.columns if c not in non_feature_cols]
    raw_feats = feats.copy()
    if args.standardize and 'session' in feats.columns:
        s1 = feats[feats['session']=='S1']
        if not s1.empty:
            mu = s1[feat_cols].mean(numeric_only=True)
            sigma = s1[feat_cols].std(numeric_only=True).replace(0, 1.0)
            feats_z = feats.copy()
            feats_z[feat_cols] = (feats_z[feat_cols] - mu) / sigma
            feats = feats_z

    feats.to_parquet(out_path, index=False)
    print(f'Saved features (after quality filter): {feats.shape} -> {out_path}')

    if args.save_baseline and 'session' in raw_feats.columns:
        # Use raw (pre-standardization) feats for baseline stats
        try:
            stats_path = compute_and_save_baseline_stats(raw_feats, out_path.parent)
        except ValueError as e:
            print(f'Baseline stats not saved: {e}')
            stats_path = None
        print(f'Saved baseline stats -> {stats_path}')


if __name__ == '__main__':
    main()
