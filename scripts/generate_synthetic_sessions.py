"""
Generate additional synthetic sessions (S3+) per user calibrated to real drift.

- Uses create_longitudinal_dataset with tuned drift (default: exponential, 0.06)
- Writes per-user, per-task CSVs in data/raw_synthetic/ following GazebaseVR naming:
  S_{SubjectID}_S{Session}_{Round}_{Task}.csv

Note: This approximates rounds/tasks by inheriting distributions from S1 files.
"""
from pathlib import Path
import sys
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.gazebase_loader import load_gazebase_data, preprocess_gaze_data, parse_filename  # type: ignore
from data.simulated_drift import create_longitudinal_dataset  # type: ignore


def subjects_with_session(raw_dir: Path, session: int, tasks: List[str]) -> List[int]:
    files = list(raw_dir.glob('S_*.csv'))
    subs = set()
    for f in files:
        try:
            meta = parse_filename(f.name)
        except Exception:
            continue
        if meta['session'] == session and (not tasks or meta['task'] in tasks):
            subs.add(meta['subject_id'])
    return sorted(list(subs))


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic sessions S3+ using calibrated drift")
    ap.add_argument('--from-session', type=int, default=1, help='Base session to simulate from (use 1)')
    ap.add_argument('--start-session', type=int, default=3, help='First synthetic session index (e.g., 3)')
    ap.add_argument('--end-session', type=int, default=5, help='Last synthetic session index (inclusive)')
    ap.add_argument('--drift-type', type=str, default='exponential', choices=['linear','exponential','periodic','none'])
    ap.add_argument('--drift-mag', type=float, default=0.06)
    ap.add_argument('--max-subjects', type=int, default=10)
    args = ap.parse_args()

    raw_dir = ROOT / 'data' / 'raw'
    out_dir = ROOT / 'data' / 'raw_synthetic'
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = ['PUR','TEX','RAN']

    subjects = subjects_with_session(raw_dir, args.from_session, tasks)[: args.max_subjects]
    if not subjects:
        print('No subjects found in base session')
        sys.exit(1)

    # Load base session data
    df_base = preprocess_gaze_data(load_gazebase_data(str(raw_dir), subjects=subjects, sessions=[args.from_session], tasks=tasks))

    # For each target session, generate drifted data and write CSVs per file/task/round
    for target_session in range(args.start_session, args.end_session + 1):
        print(f"\n=== Generating synthetic session S{target_session} ===")
        df_long = create_longitudinal_dataset(df_base, num_periods=2, drift_type=args.drift_type, drift_magnitude=args.drift_mag)
        df_syn = df_long[df_long['time_period'] == df_long['time_period'].max()].copy()
        df_syn['session'] = target_session

        # Group by original base file structure (approximate rounds/tasks)
        # We use the base file_path to preserve file splits; create new filenames with session swapped
        for (user_id, task), grp in df_syn.groupby(['user_id','task']):
            # Reconstruct approximate rounds by splitting evenly into existing count from S1
            base_files = [p for p in (raw_dir.glob(f"S_{user_id}_S{args.from_session}_*_*.csv")) if parse_filename(p.name)['task'] == task]
            if not base_files:
                # Write one file as fallback
                filename = out_dir / f"S_{user_id}_S{target_session}_1_{task}.csv"
                grp.drop(columns=['file_path'], errors='ignore').to_csv(filename, index=False)
                continue
            base_files_sorted = sorted(base_files)
            n_files = len(base_files_sorted)
            # Split by time into n_files chunks
            grp_sorted = grp.sort_values('timestamp_ms') if 'timestamp_ms' in grp.columns else grp
            splits = np.array_split(grp_sorted, n_files)
            for i, split_df in enumerate(splits, start=1):
                filename = out_dir / f"S_{user_id}_S{target_session}_{i}_{task}.csv"
                split_df.drop(columns=['file_path'], errors='ignore').to_csv(filename, index=False)
        print(f"S{target_session} saved to {out_dir}")


if __name__ == '__main__':
    main()
