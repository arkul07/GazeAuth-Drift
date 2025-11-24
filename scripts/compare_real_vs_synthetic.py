"""
Compare Real vs Synthetic Drift for Gaze Authentication

Goal:
- Assess whether synthetic drift (generated from Session 1) produces model behavior similar to real cross-session drift (S1→S2).

Protocol:
1) Real drift: Train on S1; evaluate static and adapted on S2.
2) Synthetic drift: Generate synthetic S2' from S1 using data/simulated_drift.py; evaluate static and adapted on S2'.
3) Compute metrics for CNN and LSTM:
   - Window accuracy on test windows
   - Sequence-level identification accuracy
   - Verification scores→global EER (+ FAR/FRR)
4) Save side-by-side results to results/synthetic_comparison.json.

Notes:
- We reuse feature extraction and metrics utilities already used by run_temporal_adaptation.py.
- For synthetic S2', we preserve user labels and per-file grouping (seq_id) best-effort.
"""

from pathlib import Path
import sys
import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.gazebase_loader import load_gazebase_data, preprocess_gaze_data, parse_filename  # type: ignore
from data.simulated_drift import create_longitudinal_dataset, create_longitudinal_dataset_calibrated  # type: ignore
from pipeline.feature_extractor import extract_gaze_features  # type: ignore
from models.baselines import prepare_features  # type: ignore
from utils.metrics import (
    build_user_score_vectors,
    compute_eer_from_scores,
    aggregate_global_eer,
    far_frr_at_threshold,
    sequence_level_identification_accuracy,
    sequence_level_scores_for_verification,
)  # type: ignore


def _subjects_with_both_sessions(raw_dir: Path, tasks):
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


def _split_adaptation(X, y, adapt_ratio=0.2, per_user: bool = True, meta=None):
    if not per_user:
        n = len(X)
        k = max(1, int(n * adapt_ratio))
        if meta is None:
            return (X[:k], y[:k], X[k:], y[k:])
        else:
            return (X[:k], y[:k], X[k:], y[k:], meta[:k], meta[k:])

    adapt_X = []
    adapt_y = []
    test_X = []
    test_y = []
    users = np.unique(y)
    if meta is not None:
        adapt_meta_list = []
        test_meta_list = []
    for u in users:
        mask = (y == u)
        X_u = X[mask]
        y_u = y[mask]
        k = max(1, int(len(X_u) * adapt_ratio))
        adapt_X.append(X_u[:k])
        adapt_y.append(y_u[:k])
        test_X.append(X_u[k:])
        test_y.append(y_u[k:])
        if meta is not None:
            m_u = np.array(meta)[mask]
            adapt_meta_list.append(m_u[:k])
            test_meta_list.append(m_u[k:])
    X_adapt = np.concatenate(adapt_X, axis=0)
    y_adapt = np.concatenate(adapt_y, axis=0)
    X_test = np.concatenate(test_X, axis=0)
    y_test = np.concatenate(test_y, axis=0)
    if meta is None:
        return X_adapt, y_adapt, X_test, y_test
    else:
        meta_adapt = np.concatenate(adapt_meta_list, axis=0) if len(adapt_meta_list) else np.array([])
        meta_test = np.concatenate(test_meta_list, axis=0) if len(test_meta_list) else np.array([])
        return X_adapt, y_adapt, X_test, y_test, meta_adapt, meta_test


def _clean_features(X: np.ndarray):
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
    non_zero_cols = np.where(np.any(np.abs(X) > 0, axis=0))[0]
    if len(non_zero_cols) and len(non_zero_cols) != X.shape[1]:
        X = X[:, non_zero_cols]
    return X


def _seq_ids_from_features(df_features: pd.DataFrame) -> np.ndarray:
    if all(c in df_features.columns for c in ['user_id', 'session', 'round', 'task']):
        return df_features.apply(
            lambda r: f"{int(r['user_id'])}_S{int(r['session'])}_R{int(r['round'])}_{str(r['task'])}", axis=1
        ).values
    return df_features.apply(lambda r: f"{int(r['user_id'])}_S{int(r['session'])}", axis=1).values


def _train_eval_model(model_cls, X_train, y_train, X_adapt, y_adapt, X_test, y_test, seq_test, label_prefix: str):
    # init
    model_static = model_cls(seq_length=10, epochs=20, batch_size=32, learning_rate=1e-3)
    model_static.train(X_train, y_train)
    y_pred_static = model_static.predict(X_test)
    acc_static = float((y_pred_static == y_test).mean())
    proba_static = model_static.predict_proba(X_test)

    model_adapt = model_cls(seq_length=10, epochs=20, batch_size=32, learning_rate=1e-3)
    model_adapt.train(X_train, y_train)
    model_adapt.train(X_adapt, y_adapt, continue_training=True)
    y_pred_adapt = model_adapt.predict(X_test)
    acc_adapt = float((y_pred_adapt == y_test).mean())
    proba_adapt = model_adapt.predict_proba(X_test)

    # sequence-level ID
    seq_acc_static = sequence_level_identification_accuracy(y_test, proba_static, seq_test, list(model_static.classes_))
    seq_acc_adapt = sequence_level_identification_accuracy(y_test, proba_adapt, seq_test, list(model_adapt.classes_))

    # verification metrics (window + sequence pooled)
    def _metrics_for(proba, classes):
        proba = np.nan_to_num(proba, nan=0.0, posinf=1.0, neginf=0.0)
        scores = build_user_score_vectors(y_test, proba, classes)
        geer, gthr = aggregate_global_eer(scores)
        far, frr = far_frr_at_threshold(
            np.concatenate([scores[u]['genuine'] for u in scores]),
            np.concatenate([scores[u]['impostor'] for u in scores]),
            gthr,
        )
        # sequence pooled
        seq_scores = sequence_level_scores_for_verification(y_test, proba, seq_test, list(classes))
        s_geer, s_gthr = aggregate_global_eer(seq_scores)
        s_far, s_frr = far_frr_at_threshold(
            np.concatenate([seq_scores[u]['genuine'] for u in seq_scores]),
            np.concatenate([seq_scores[u]['impostor'] for u in seq_scores]),
            s_gthr,
        )
        return {
            'window': {'global_eer': float(geer), 'global_threshold': float(gthr), 'far': float(far), 'frr': float(frr)},
            'sequence': {'global_eer': float(s_geer), 'global_threshold': float(s_gthr), 'far': float(s_far), 'frr': float(s_frr)},
        }

    metrics_static = _metrics_for(proba_static, model_static.classes_)
    metrics_adapt = _metrics_for(proba_adapt, model_adapt.classes_)

    return {
        f'{label_prefix} Static': {
            'accuracy': acc_static,
            'sequence_identification_accuracy': float(seq_acc_static),
            'verification': metrics_static,
        },
        f'{label_prefix} Adapted': {
            'accuracy': acc_adapt,
            'sequence_identification_accuracy': float(seq_acc_adapt),
            'verification': metrics_adapt,
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Compare real vs synthetic drift for CNN/LSTM")
    parser.add_argument("--drift-type", type=str, default="linear", choices=["linear", "exponential", "periodic", "none"], help="Synthetic drift type")
    parser.add_argument("--drift-mag", type=float, default=0.12, help="Synthetic drift magnitude (0-1)")
    parser.add_argument("--num-periods", type=int, default=2, help="Number of synthetic periods (>=2 to emulate S2)")
    parser.add_argument("--calibration-json", type=str, default="", help="Path to calibration JSON (if provided, overrides drift-type/mag)")
    parser.add_argument("--calibration-groupby", type=str, default="task", choices=["task", "user_task"], help="Grouping used in calibration JSON and to generate synthetic")
    parser.add_argument("--max-subjects", type=int, default=10, help="Cap number of subjects for quick runs")
    args = parser.parse_args()

    raw_dir = ROOT / 'data' / 'raw'
    tasks = ['PUR', 'TEX', 'RAN']
    subjects = _subjects_with_both_sessions(raw_dir, tasks)
    if len(subjects) == 0:
        print("No subjects with both sessions found.")
        sys.exit(1)
    subjects = subjects[: args.max_subjects]
    print(f"Subjects: {subjects}")

    # Load S1 and S2 (real)
    df_s1 = preprocess_gaze_data(load_gazebase_data(str(raw_dir), subjects=subjects, sessions=[1], tasks=tasks))
    df_s2 = preprocess_gaze_data(load_gazebase_data(str(raw_dir), subjects=subjects, sessions=[2], tasks=tasks))

    # Synthetic S2' from S1
    if args.calibration_json:
        import json as _json
        with open(args.calibration_json, 'r') as f:
            calib = _json.load(f)
        df_s2_syn = create_longitudinal_dataset_calibrated(df_s1, calib, groupby=args.calibration_groupby, num_periods=args.num_periods)
    else:
        df_s2_syn = create_longitudinal_dataset(df_s1, num_periods=args.num_periods, drift_type=args.drift_type, drift_magnitude=args.drift_mag)
    # approximate: take last period as synthetic S2'
    df_s2_syn = df_s2_syn[df_s2_syn['time_period'] == df_s2_syn['time_period'].max()].copy()
    # Keep metadata consistent: session=2 label for synthetic
    df_s2_syn['session'] = 2

    # Feature extraction
    feats_s1 = extract_gaze_features(df_s1, window_size_sec=5.0, overlap_sec=1.0)
    feats_s2 = extract_gaze_features(df_s2, window_size_sec=5.0, overlap_sec=1.0)
    feats_syn = extract_gaze_features(df_s2_syn, window_size_sec=5.0, overlap_sec=1.0)

    # Sequence ids
    for df_feat in (feats_s2, feats_syn):
        if all(c in df_feat.columns for c in ['user_id', 'session', 'round', 'task']):
            df_feat['seq_id'] = df_feat.apply(
                lambda r: f"{int(r['user_id'])}_S{int(r['session'])}_R{int(r['round'])}_{str(r['task'])}", axis=1
            )
        else:
            df_feat['seq_id'] = df_feat.apply(lambda r: f"{int(r['user_id'])}_S{int(r['session'])}", axis=1)

    X_train, y_train = prepare_features(feats_s1)
    X_s2, y_s2 = prepare_features(feats_s2)
    X_syn, y_syn = prepare_features(feats_syn)

    # Clean and align columns independently; models do their own scaling
    X_train = _clean_features(X_train)
    X_s2 = _clean_features(X_s2)
    X_syn = _clean_features(X_syn)

    # Splits for real S2 and synthetic S2'
    seq_s2 = feats_s2['seq_id'].values
    seq_syn = feats_syn['seq_id'].values

    X_adapt_s2, y_adapt_s2, X_test_s2, y_test_s2, seq_adapt_s2, seq_test_s2 = _split_adaptation(
        X_s2, y_s2, adapt_ratio=0.2, per_user=True, meta=seq_s2
    )
    X_adapt_syn, y_adapt_syn, X_test_syn, y_test_syn, seq_adapt_syn, seq_test_syn = _split_adaptation(
        X_syn, y_syn, adapt_ratio=0.2, per_user=True, meta=seq_syn
    )

    # Import models
    import torch  # noqa: F401
    from models.temporal.gaze_cnn import GazeCNNClassifier
    from models.temporal.gaze_lstm import GazeLSTMClassifier

    results: Dict[str, Any] = {
        'subjects': subjects,
        'adapt_ratio': 0.2,
        'seq_length': 10,
        'settings': {
            'synthetic': {
                'drift_type': args.drift_type,
                'drift_magnitude': args.drift_mag,
                'num_periods': args.num_periods,
                'use_last_period_as_S2': True,
                'calibration': {
                    'json': args.calibration_json if args.calibration_json else None,
                    'groupby': args.calibration_groupby if args.calibration_json else None,
                }
            }
        }
    }

    # CNN on real vs synthetic
    results.update(_train_eval_model(GazeCNNClassifier, X_train, y_train,
                                     X_adapt_s2, y_adapt_s2, X_test_s2, y_test_s2, seq_test_s2, 'CNN Real'))
    results.update(_train_eval_model(GazeCNNClassifier, X_train, y_train,
                                     X_adapt_syn, y_adapt_syn, X_test_syn, y_test_syn, seq_test_syn, 'CNN Synthetic'))

    # LSTM on real vs synthetic
    results.update(_train_eval_model(GazeLSTMClassifier, X_train, y_train,
                                     X_adapt_s2, y_adapt_s2, X_test_s2, y_test_s2, seq_test_s2, 'LSTM Real'))
    results.update(_train_eval_model(GazeLSTMClassifier, X_train, y_train,
                                     X_adapt_syn, y_adapt_syn, X_test_syn, y_test_syn, seq_test_syn, 'LSTM Synthetic'))

    out_path = ROOT / 'results' / 'synthetic_comparison.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved comparison metrics to {out_path}")


if __name__ == '__main__':
    main()
