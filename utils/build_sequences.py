"""Build temporal sequences from windowed features.

Given the windowed feature parquet, generate fixed-length sequences per user,
optionally with stride, and save as a new parquet (one row per sequence) or
NumPy npz for deep models.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd
import numpy as np


def build_sequences(df: pd.DataFrame, seq_len: int = 5, stride: int = 1) -> tuple[pd.DataFrame, list[str]]:
    non_feature_cols = {'user_id','t_start','t_end','session'}
    feat_cols = [c for c in df.columns if c not in non_feature_cols]
    seq_rows = []
    for uid, g in df.sort_values(['user_id','t_start']).groupby('user_id'):
        g = g.reset_index(drop=True)
        for i in range(0, len(g) - seq_len + 1, stride):
            window = g.iloc[i:i+seq_len]
            row = {
                'user_id': int(uid),
                't_start': float(window['t_start'].iloc[0]),
                't_end': float(window['t_end'].iloc[-1]),
            }
            if 'session' in g.columns:
                row['session'] = window['session'].mode().iat[0] if not window['session'].empty else None
            # flatten features as <feat>_t{k}
            for k, (_, wrow) in enumerate(window.iterrows()):
                for c in feat_cols:
                    row[f'{c}_t{k}'] = wrow[c]
            seq_rows.append(row)
    seq_df = pd.DataFrame(seq_rows)
    return seq_df, feat_cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in_parquet', default='data/processed/feature_windows.parquet')
    ap.add_argument('--out_parquet', default='data/processed/sequences.parquet')
    ap.add_argument('--seq_len', type=int, default=5)
    ap.add_argument('--stride', type=int, default=1)
    args = ap.parse_args()

    in_path = Path(args.in_parquet)
    out_path = Path(args.out_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(in_path)
    seq_df, feat_cols = build_sequences(df, seq_len=args.seq_len, stride=args.stride)
    seq_df.to_parquet(out_path, index=False)
    print(f'Saved sequences: {seq_df.shape} -> {out_path}')


if __name__ == '__main__':
    main()
