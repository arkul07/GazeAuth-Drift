"""
Sweep synthetic drift settings and compare against real S1→S2.

Outputs a ranked table and JSON with delta metrics (accuracy/EER differences) for CNN and LSTM
at window and sequence levels. Optionally computes KS distances on genuine/impostor scores.
"""
from pathlib import Path
import sys
import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.gazebase_loader import load_gazebase_data, preprocess_gaze_data, parse_filename  # type: ignore
from data.simulated_drift import create_longitudinal_dataset  # type: ignore
from pipeline.feature_extractor import extract_gaze_features  # type: ignore
from models.baselines import prepare_features  # type: ignore
from utils.metrics import (
    build_user_score_vectors,
    aggregate_global_eer,
    sequence_level_identification_accuracy,
    sequence_level_scores_for_verification,
)  # type: ignore


def subjects_with_both_sessions(raw_dir: Path, tasks: List[str]) -> List[int]:
    files = list(raw_dir.glob('S_*.csv'))
    by_subj = {}
    for f in files:
        try:
            meta = parse_filename(f.name)
        except Exception:
            continue
        if tasks and meta["task"] not in tasks:
            continue
        s = meta["subject_id"]
        sess = meta["session"]
        by_subj.setdefault(s, set()).add(sess)
    return sorted([s for s, sess_set in by_subj.items() if 1 in sess_set and 2 in sess_set])


def clean_features(X: np.ndarray) -> np.ndarray:
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    # remove constant columns
    keep = []
    for c in range(X.shape[1]):
        col = X[:, c]
        if np.nanmax(col) - np.nanmin(col) >= 1e-12:
            keep.append(c)
    if len(keep) and len(keep) != X.shape[1]:
        X = X[:, keep]
    # drop all-zero columns
    nz = np.where(np.any(np.abs(X) > 0, axis=0))[0]
    if len(nz) and len(nz) != X.shape[1]:
        X = X[:, nz]
    return X


def split_adapt(X, y, seq, ratio=0.2):
    users = np.unique(y)
    parts = []
    for u in users:
        m = (y == u)
        Xu = X[m]; yu = y[m]; su = np.array(seq)[m]
        k = max(1, int(len(Xu)*ratio))
        parts.append((Xu[:k], yu[:k], su[:k], Xu[k:], yu[k:], su[k:]))
    Xa = np.concatenate([p[0] for p in parts]); ya = np.concatenate([p[1] for p in parts]); sa = np.concatenate([p[2] for p in parts])
    Xt = np.concatenate([p[3] for p in parts]); yt = np.concatenate([p[4] for p in parts]); st = np.concatenate([p[5] for p in parts])
    return Xa, ya, sa, Xt, yt, st


def seq_id(df: pd.DataFrame) -> np.ndarray:
    if all(c in df.columns for c in ['user_id', 'session', 'round', 'task']):
        return df.apply(lambda r: f"{int(r['user_id'])}_S{int(r['session'])}_R{int(r['round'])}_{str(r['task'])}", axis=1).values
    return df.apply(lambda r: f"{int(r['user_id'])}_S{int(r['session'])}", axis=1).values


def run_models(Xtr, ytr, Xa, ya, Xt, yt, st) -> Dict[str, Any]:
    import torch  # noqa: F401
    from models.temporal.gaze_cnn import GazeCNNClassifier
    from models.temporal.gaze_lstm import GazeLSTMClassifier

    def train_eval(model_cls):
        m0 = model_cls(seq_length=10, epochs=15, batch_size=32, learning_rate=1e-3)
        m0.train(Xtr, ytr)
        acc_s = float((m0.predict(Xt) == yt).mean())
        proba_s = m0.predict_proba(Xt)
        m1 = model_cls(seq_length=10, epochs=15, batch_size=32, learning_rate=1e-3)
        m1.train(Xtr, ytr)
        m1.train(Xa, ya, continue_training=True)
        acc_a = float((m1.predict(Xt) == yt).mean())
        proba_a = m1.predict_proba(Xt)
        # ID (seq) and EER (window/seq)
        seq_acc_s = sequence_level_identification_accuracy(yt, proba_s, st, list(m0.classes_))
        seq_acc_a = sequence_level_identification_accuracy(yt, proba_a, st, list(m1.classes_))
        def eer_block(proba, classes):
            proba = np.nan_to_num(proba, nan=0.0, posinf=1.0, neginf=0.0)
            sc = build_user_score_vectors(yt, proba, classes)
            geer, gthr = aggregate_global_eer(sc)
            ssc = sequence_level_scores_for_verification(yt, proba, st, list(classes))
            seer, _ = aggregate_global_eer(ssc)
            return geer, seer
        we_s, se_s = eer_block(proba_s, m0.classes_)
        we_a, se_a = eer_block(proba_a, m1.classes_)
        return {
            'acc_static': acc_s, 'acc_adapt': acc_a,
            'seq_acc_static': float(seq_acc_s), 'seq_acc_adapt': float(seq_acc_a),
            'eer_window_static': float(we_s), 'eer_window_adapt': float(we_a),
            'eer_seq_static': float(se_s), 'eer_seq_adapt': float(se_a),
        }

    return {
        'CNN': train_eval(GazeCNNClassifier),
        'LSTM': train_eval(GazeLSTMClassifier),
    }


def main():
    ap = argparse.ArgumentParser(description="Sweep synthetic drift to match real S1→S2")
    ap.add_argument("--max-subjects", type=int, default=10)
    ap.add_argument("--drift-types", type=str, default="linear,exponential,periodic")
    ap.add_argument("--drift-mags", type=str, default="0.03,0.06,0.09,0.12,0.15")
    args = ap.parse_args()

    raw_dir = ROOT / 'data' / 'raw'
    tasks = ['PUR', 'TEX', 'RAN']
    subjects = subjects_with_both_sessions(raw_dir, tasks)[: args.max_subjects]
    if not subjects:
        print("No subjects with both S1 and S2")
        sys.exit(1)

    # Load real S1/S2
    df_s1 = preprocess_gaze_data(load_gazebase_data(str(raw_dir), subjects=subjects, sessions=[1], tasks=tasks))
    df_s2 = preprocess_gaze_data(load_gazebase_data(str(raw_dir), subjects=subjects, sessions=[2], tasks=tasks))

    f_s1 = extract_gaze_features(df_s1, window_size_sec=5.0, overlap_sec=1.0)
    f_s2 = extract_gaze_features(df_s2, window_size_sec=5.0, overlap_sec=1.0)

    Xtr, ytr = prepare_features(f_s1)
    Xt_real, yt_real = prepare_features(f_s2)
    Xtr = clean_features(Xtr)
    Xt_real = clean_features(Xt_real)

    st_real = seq_id(f_s2)
    Xa_real, ya_real, sa_real, Xtest_real, ytest_real, stest_real = split_adapt(Xt_real, yt_real, st_real, ratio=0.2)

    # Baseline metrics on real drift
    real_metrics = run_models(Xtr, ytr, Xa_real, ya_real, Xtest_real, ytest_real, stest_real)

    drift_types = [s.strip() for s in args.drift_types.split(',') if s.strip()]
    drift_mags = [float(s.strip()) for s in args.drift_mags.split(',') if s.strip()]

    rows = []
    all_results = {
        'real': real_metrics,
        'sweeps': []
    }

    for dt in drift_types:
        for dm in drift_mags:
            print(f"\n=== Synthetic sweep: type={dt} mag={dm} ===")
            syn = create_longitudinal_dataset(df_s1, num_periods=2, drift_type=dt, drift_magnitude=dm)
            syn_last = syn[syn['time_period'] == syn['time_period'].max()].copy()
            syn_last['session'] = 2
            f_syn = extract_gaze_features(syn_last, window_size_sec=5.0, overlap_sec=1.0)
            Xsyn, ysyn = prepare_features(f_syn)
            Xsyn = clean_features(Xsyn)
            st_syn = seq_id(f_syn)
            Xa_syn, ya_syn, sa_syn, Xtest_syn, ytest_syn, stest_syn = split_adapt(Xsyn, ysyn, st_syn, ratio=0.2)

            syn_metrics = run_models(Xtr, ytr, Xa_syn, ya_syn, Xtest_syn, ytest_syn, stest_syn)
            all_results['sweeps'].append({'drift_type': dt, 'drift_mag': dm, 'metrics': syn_metrics})

            # Distance to real (sum of absolute deltas across key metrics)
            def dist(model):
                r = real_metrics[model]; s = syn_metrics[model]
                keys = ['acc_static','acc_adapt','eer_window_static','eer_window_adapt','eer_seq_static','eer_seq_adapt']
                return float(sum(abs(r[k]-s[k]) for k in keys))
            score_cnn = dist('CNN')
            score_lstm = dist('LSTM')
            rows.append({'drift_type': dt, 'drift_mag': dm, 'score_cnn': score_cnn, 'score_lstm': score_lstm, 'score_sum': score_cnn+score_lstm})

    df_rank = pd.DataFrame(rows).sort_values('score_sum')
    print("\nTop candidates (lower is better):")
    print(df_rank.head(10))

    out_dir = ROOT / 'results'
    out_dir.mkdir(parents=True, exist_ok=True)
    df_rank.to_csv(out_dir / 'synthetic_sweep_ranking.csv', index=False)
    with open(out_dir / 'synthetic_sweep_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved ranking to {out_dir/'synthetic_sweep_ranking.csv'} and all results to {out_dir/'synthetic_sweep_results.json'}")


if __name__ == '__main__':
    main()
