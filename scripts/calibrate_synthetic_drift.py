"""
Compute calibration parameters from real S1 vs S2 and save JSON.
Then optionally preview a calibrated synthetic S2' and compare to real.
"""
from pathlib import Path
import sys
import json
import argparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.gazebase_loader import load_gazebase_data, preprocess_gaze_data, parse_filename  # type: ignore
from data.simulated_drift import estimate_drift_from_sessions, create_longitudinal_dataset_calibrated  # type: ignore


def main():
    ap = argparse.ArgumentParser(description='Calibrate synthetic drift from real S1→S2')
    ap.add_argument('--groupby', type=str, default='task', choices=['task','user_task'])
    ap.add_argument('--max-subjects', type=int, default=10)
    ap.add_argument('--save', type=str, default=str(ROOT / 'results' / 'synthetic_calibration.json'))
    args = ap.parse_args()

    raw_dir = ROOT / 'data' / 'raw'
    # Choose up to N subjects with both S1 and S2
    files = list((raw_dir).glob('S_*.csv'))
    by_sub = {}
    for f in files:
        try:
            meta = parse_filename(f.name)
        except Exception:
            continue
        by_sub.setdefault(meta['subject_id'], set()).add(meta['session'])
    subjects_all = sorted([s for s, sess in by_sub.items() if 1 in sess and 2 in sess])
    subjects = subjects_all[: args.max_subjects] if subjects_all else None

    df_s1 = preprocess_gaze_data(load_gazebase_data(str(raw_dir), subjects=subjects, sessions=[1], tasks=['PUR','TEX','RAN']))
    df_s2 = preprocess_gaze_data(load_gazebase_data(str(raw_dir), subjects=subjects, sessions=[2], tasks=['PUR','TEX','RAN']))

    calib = estimate_drift_from_sessions(df_s1, df_s2, groupby=args.groupby)

    outp = Path(args.save)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, 'w') as f:
        json.dump(calib, f, indent=2)
    print(f"Saved calibration to {outp}")

    # Optional: preview calibrated longitudinal generation (2 periods)
    preview = create_longitudinal_dataset_calibrated(df_s1, calib, groupby=args.groupby, num_periods=2)
    print(f"Preview generated rows: {len(preview)}")


if __name__ == '__main__':
    main()
