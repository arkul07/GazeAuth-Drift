"""
Multi-session temporal adaptation over real + synthetic sessions.
- Train on S1
- For each next session (S2..Sk), fine-tune on first 20% windows per user, test on remaining 80%
- Report per-session window accuracy, sequence identification accuracy, and verification EERs for CNN/LSTM
"""
from pathlib import Path
import sys
import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.gazebase_loader import load_gazebase_data, preprocess_gaze_data, parse_filename  # type: ignore
from pipeline.feature_extractor import extract_gaze_features  # type: ignore
from models.baselines import prepare_features  # type: ignore
from utils.metrics import (
    build_user_score_vectors,
    aggregate_global_eer,
    far_frr_at_threshold,
    sequence_level_identification_accuracy,
    sequence_level_scores_for_verification,
)  # type: ignore


def subjects_available(raw_roots: List[Path], tasks: List[str]) -> List[int]:
    files = []
    for root in raw_roots:
        files += list(root.glob('S_*.csv'))
    subs = set()
    by_sub_sess = {}
    for f in files:
        try:
            meta = parse_filename(f.name)
        except Exception:
            continue
        if tasks and meta['task'] not in tasks:
            continue
        subs.add(meta['subject_id'])
        by_sub_sess.setdefault(meta['subject_id'], set()).add(meta['session'])
    # Only keep subjects with S1 available
    subs = [s for s in subs if 1 in by_sub_sess.get(s, set())]
    return sorted(subs)


def split_adapt(X, y, meta_seq, ratio=0.2):
    users = np.unique(y)
    parts = []
    for u in users:
        m = (y == u)
        X_u = X[m]; y_u = y[m]; s_u = np.array(meta_seq)[m]
        k = max(1, int(len(X_u)*ratio))
        parts.append((X_u[:k], y_u[:k], s_u[:k], X_u[k:], y_u[k:], s_u[k:]))
    Xa = np.concatenate([p[0] for p in parts]); ya = np.concatenate([p[1] for p in parts]); sa = np.concatenate([p[2] for p in parts])
    Xt = np.concatenate([p[3] for p in parts]); yt = np.concatenate([p[4] for p in parts]); st = np.concatenate([p[5] for p in parts])
    return Xa, ya, sa, Xt, yt, st


def seq_ids(df_feat: pd.DataFrame) -> np.ndarray:
    if all(c in df_feat.columns for c in ['user_id','session','round','task']):
        return df_feat.apply(lambda r: f"{int(r['user_id'])}_S{int(r['session'])}_R{int(r['round'])}_{str(r['task'])}", axis=1).values
    return df_feat.apply(lambda r: f"{int(r['user_id'])}_S{int(r['session'])}", axis=1).values


def clean(X: np.ndarray) -> np.ndarray:
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    keep = []
    for c in range(X.shape[1]):
        col = X[:, c]
        if np.nanmax(col) - np.nanmin(col) >= 1e-12:
            keep.append(c)
    if len(keep) and len(keep) != X.shape[1]:
        X = X[:, keep]
    nz = np.where(np.any(np.abs(X) > 0, axis=0))[0]
    if len(nz) and len(nz) != X.shape[1]:
        X = X[:, nz]
    return X


def run_session(model_cls, model_name: str, Xtr, ytr, sessions_data: List[Dict[str, Any]]):
    # Initialize a fresh model for this family (static copy to adapt sequentially)
    m = model_cls(seq_length=10, epochs=15, batch_size=32, learning_rate=1e-3)
    m.train(Xtr, ytr)
    per_session = {}

    for sess in sessions_data:
        label = sess['label']
        Xs, ys, seqs = sess['X'], sess['y'], sess['seq']
        Xa, ya, sa, Xt, yt, st = split_adapt(Xs, ys, seqs, ratio=0.2)
        # Evaluate static
        acc_static = float((m.predict(Xt) == yt).mean())
        proba_static = m.predict_proba(Xt)
        seq_acc_static = sequence_level_identification_accuracy(yt, proba_static, st, list(m.classes_))
        scores = build_user_score_vectors(yt, proba_static, m.classes_)
        we_static, wthr = aggregate_global_eer(scores)
        seq_scores = sequence_level_scores_for_verification(yt, proba_static, st, list(m.classes_))
        se_static, _ = aggregate_global_eer(seq_scores)
        # Fine-tune
        m.train(Xa, ya, continue_training=True)
        acc_adapt = float((m.predict(Xt) == yt).mean())
        proba_adapt = m.predict_proba(Xt)
        seq_acc_adapt = sequence_level_identification_accuracy(yt, proba_adapt, st, list(m.classes_))
        scores_a = build_user_score_vectors(yt, proba_adapt, m.classes_)
        we_adapt, _ = aggregate_global_eer(scores_a)
        seq_scores_a = sequence_level_scores_for_verification(yt, proba_adapt, st, list(m.classes_))
        se_adapt, _ = aggregate_global_eer(seq_scores_a)

        per_session[label] = {
            'window_accuracy': {'static': acc_static, 'adapted': acc_adapt},
            'sequence_identification_accuracy': {'static': float(seq_acc_static), 'adapted': float(seq_acc_adapt)},
            'eer_window': {'static': float(we_static), 'adapted': float(we_adapt)},
            'eer_sequence': {'static': float(se_static), 'adapted': float(se_adapt)},
        }
    return per_session


def main():
    ap = argparse.ArgumentParser(description='Multi-session temporal adaptation over real + synthetic sessions')
    ap.add_argument('--max-subjects', type=int, default=8)
    ap.add_argument('--include-synthetic', action='store_true', help='Include data/raw_synthetic sessions (S3+)')
    args = ap.parse_args()

    raw_real = ROOT / 'data' / 'raw'
    raw_syn = ROOT / 'data' / 'raw_synthetic'
    tasks = ['PUR','TEX','RAN']

    subs = subjects_available([raw_real] + ([raw_syn] if args.include_synthetic else []), tasks)[: args.max_subjects]
    if not subs:
        print('No subjects found')
        sys.exit(1)

    # Load S1 for training
    df_s1 = preprocess_gaze_data(load_gazebase_data(str(raw_real), subjects=subs, sessions=[1], tasks=tasks))
    f_s1 = extract_gaze_features(df_s1, window_size_sec=5.0, overlap_sec=1.0)
    Xtr, ytr = prepare_features(f_s1)
    Xtr = clean(Xtr)

    # Prepare sessions S2 (real) and optional S3+ (synthetic)
    sessions: List[Dict[str, Any]] = []
    # Real S2
    df_s2 = preprocess_gaze_data(load_gazebase_data(str(raw_real), subjects=subs, sessions=[2], tasks=tasks))
    f_s2 = extract_gaze_features(df_s2, window_size_sec=5.0, overlap_sec=1.0)
    X2, y2 = prepare_features(f_s2); X2 = clean(X2); s2 = seq_ids(f_s2)
    sessions.append({'label': 'S2 (real)', 'X': X2, 'y': y2, 'seq': s2})

    # Synthetic sessions
    if args.include_synthetic and raw_syn.exists():
        # Discover available synthetic session numbers
        sess_nums = sorted({parse_filename(p.name)['session'] for p in raw_syn.glob('S_*.csv')})
        for sn in sess_nums:
            if sn <= 2:
                continue
            df_s = preprocess_gaze_data(load_gazebase_data(str(raw_syn), subjects=subs, sessions=[sn], tasks=tasks))
            f_s = extract_gaze_features(df_s, window_size_sec=5.0, overlap_sec=1.0)
            Xs, ys = prepare_features(f_s); Xs = clean(Xs); ss = seq_ids(f_s)
            sessions.append({'label': f'S{sn} (synthetic)', 'X': Xs, 'y': ys, 'seq': ss})

    import torch  # noqa: F401
    from models.temporal.gaze_cnn import GazeCNNClassifier
    from models.temporal.gaze_lstm import GazeLSTMClassifier

    results: Dict[str, Any] = {
        'subjects': subs,
        'sessions': [s['label'] for s in sessions],
    }

    results['CNN'] = run_session(GazeCNNClassifier, 'CNN', Xtr, ytr, sessions)
    results['LSTM'] = run_session(GazeLSTMClassifier, 'LSTM', Xtr, ytr, sessions)

    out = ROOT / 'results' / 'temporal_multisession_metrics.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved multi-session metrics to {out}')


if __name__ == '__main__':
    main()
