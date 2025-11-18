"""
Spec-compliant temporal adaptation test for CNN/LSTM.

Protocol:
- Data: data/raw, tasks=['PUR','TEX','RAN'] (present in repo)
- Subjects: only those with both Session 1 and Session 2 available
- Train: Session 1
- Adapt: first 20% of Session 2 (by window order)
- Test: remaining 80% of Session 2

Outputs: prints static vs adapted accuracies for CNN and LSTM.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import json

# Ensure project root on path when running as script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.gazebase_loader import load_gazebase_data, preprocess_gaze_data, parse_filename  # type: ignore
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


def subjects_with_both_sessions(raw_dir: Path, tasks):
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


def split_adaptation(X, y, adapt_ratio=0.2, per_user: bool = True, meta=None):
    """Split into adaptation and test sets.

    If per_user=True, take the first adapt_ratio fraction of windows *for each user* to ensure
    representation across all subjects. This avoids skew where early windows belong to only a few users.
    """
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
            if 'adapt_meta' not in locals():
                adapt_meta = []
                test_meta = []
            adapt_meta.append(m_u[:k])
            test_meta.append(m_u[k:])
    X_adapt = np.concatenate(adapt_X, axis=0)
    y_adapt = np.concatenate(adapt_y, axis=0)
    X_test = np.concatenate(test_X, axis=0)
    y_test = np.concatenate(test_y, axis=0)
    if meta is None:
        return X_adapt, y_adapt, X_test, y_test
    else:
        meta_adapt = np.concatenate(adapt_meta, axis=0) if len(adapt_meta) else np.array([])
        meta_test = np.concatenate(test_meta, axis=0) if len(test_meta) else np.array([])
        return X_adapt, y_adapt, X_test, y_test, meta_adapt, meta_test


def main():
    root = Path(__file__).resolve().parents[1]
    raw_dir = root / 'data' / 'raw'
    tasks = ['PUR', 'TEX', 'RAN']

    if not raw_dir.exists():
        print(f"✗ Raw data dir not found: {raw_dir}")
        sys.exit(1)

    subjects = subjects_with_both_sessions(raw_dir, tasks)
    if len(subjects) == 0:
        print("✗ No subjects with both Session 1 and 2.")
        sys.exit(1)

    # Keep a modest subset to keep runtime reasonable if very large
    subjects = subjects[:10]
    print(f"✓ Subjects (both sessions): {subjects}")

    # Load S1
    print("\n=== Load & extract Session 1 (train) ===")
    df_s1 = load_gazebase_data(str(raw_dir), subjects=subjects, sessions=[1], tasks=tasks)
    df_s1 = preprocess_gaze_data(df_s1)
    feats_s1 = extract_gaze_features(df_s1, window_size_sec=5.0, overlap_sec=1.0)
    X_train, y_train = prepare_features(feats_s1)
    # Basic cleaning: drop columns that are all nan or constant
    col_mask = []
    for col in range(X_train.shape[1]):
        col_values = X_train[:, col]
        if np.all(np.isnan(col_values)):
            continue
        if np.nanmax(col_values) - np.nanmin(col_values) < 1e-12:
            continue
        col_mask.append(col)
    if len(col_mask) != X_train.shape[1]:
        X_train = X_train[:, col_mask]
        print(f"Filtered features: kept {X_train.shape[1]} of original")
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    # final drop of any columns that are all zeros after cleaning
    non_zero_cols = np.where(np.any(np.abs(X_train) > 0, axis=0))[0]
    if len(non_zero_cols) and len(non_zero_cols) != X_train.shape[1]:
        X_train = X_train[:, non_zero_cols]
        col_mask = [col_mask[i] for i in non_zero_cols] if len(col_mask) else list(non_zero_cols)
        print(f"Removed zero-variance cols post-cleaning. Now {X_train.shape[1]} features")
    print(f"Train windows: {len(X_train)} | users: {len(np.unique(y_train))}")

    # Load S2
    print("\n=== Load & extract Session 2 (drift) ===")
    df_s2 = load_gazebase_data(str(raw_dir), subjects=subjects, sessions=[2], tasks=tasks)
    df_s2 = preprocess_gaze_data(df_s2)
    feats_s2 = extract_gaze_features(df_s2, window_size_sec=5.0, overlap_sec=1.0)
    # Build sequence ids per window to treat a recording/trial as one biometric instance
    # Use (user, session, round, task) to group windows from the same file/trial
    if all(c in feats_s2.columns for c in ['user_id', 'session', 'round', 'task']):
        feats_s2['seq_id'] = feats_s2.apply(
            lambda r: f"{int(r['user_id'])}_S{int(r['session'])}_R{int(r['round'])}_{str(r['task'])}", axis=1
        )
    else:
        # fallback: group by (user, session) only
        feats_s2['seq_id'] = feats_s2.apply(
            lambda r: f"{int(r['user_id'])}_S{int(r['session'])}", axis=1
        )
    X_s2, y_s2 = prepare_features(feats_s2)
    window_seq_ids = feats_s2['seq_id'].values
    if len(col_mask):
        X_s2 = X_s2[:, col_mask]
    X_s2 = np.nan_to_num(X_s2, nan=0.0, posinf=0.0, neginf=0.0)
    # align zero-variance cleanup on S2
    if 'non_zero_cols' in locals() and len(non_zero_cols):
        X_s2 = X_s2[:, list(range(len(non_zero_cols)))]
    print(f"S2 windows: {len(X_s2)}")

    # Split, keeping sequence ids aligned for test windows
    X_adapt, y_adapt, X_test, y_test, seq_adapt, seq_test = split_adaptation(
        X_s2, y_s2, adapt_ratio=0.2, per_user=True, meta=window_seq_ids
    )
    print(f"Adapt windows: {len(X_adapt)} | Test windows: {len(X_test)}")

    # Try CNN/LSTM
    try:
        import torch  # noqa: F401
        from models.temporal.gaze_cnn import GazeCNNClassifier
        from models.temporal.gaze_lstm import GazeLSTMClassifier
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    except Exception as e:
        print(f"⚠️  PyTorch not available or failed to import: {e}")
        print("Install torch/torchvision to run CNN/LSTM tests.")
        sys.exit(2)

    results = []
    seq_metrics = {}

    # CNN: static
    print("\n=== CNN (static) S1→S2 ===")
    cnn = GazeCNNClassifier(seq_length=10, epochs=20, batch_size=32, learning_rate=1e-3, device=device)
    cnn.train(X_train, y_train)
    y_pred_static = cnn.predict(X_test)
    acc_static = float((y_pred_static == y_test).mean())
    print(f"CNN static accuracy on S2: {acc_static:.3f}")
    proba_static_cnn = cnn.predict_proba(X_test)
    results.append(("CNN", "Static", acc_static))
    # Sequence-level identification accuracy (treat gaze trial as whole)
    seq_acc_cnn_static = sequence_level_identification_accuracy(y_test, proba_static_cnn, seq_test, list(cnn.classes_))
    seq_metrics['CNN Static'] = {'sequence_identification_accuracy': float(seq_acc_cnn_static)}

    # CNN: adapted
    print("\n=== CNN (adapted) fine-tune on 20% S2 ===")
    cnn_adapt = GazeCNNClassifier(seq_length=10, epochs=20, batch_size=32, learning_rate=1e-3, device=device)
    cnn_adapt.train(X_train, y_train)
    cnn_adapt.train(X_adapt, y_adapt, continue_training=True)  # fine-tune retaining weights & scaling
    y_pred_adapt = cnn_adapt.predict(X_test)
    acc_adapt = float((y_pred_adapt == y_test).mean())
    proba_adapt_cnn = cnn_adapt.predict_proba(X_test)
    print(f"CNN adapted accuracy on S2: {acc_adapt:.3f} (Δ {acc_adapt - acc_static:+.3f})")
    results.append(("CNN", "Adapted", acc_adapt))
    seq_acc_cnn_adapt = sequence_level_identification_accuracy(y_test, proba_adapt_cnn, seq_test, list(cnn_adapt.classes_))
    seq_metrics['CNN Adapted'] = {'sequence_identification_accuracy': float(seq_acc_cnn_adapt)}

    # LSTM: static
    print("\n=== LSTM (static) S1→S2 ===")
    lstm = GazeLSTMClassifier(seq_length=10, epochs=20, batch_size=32, learning_rate=1e-3, device=device)
    lstm.train(X_train, y_train)
    y_pred_static_lstm = lstm.predict(X_test)
    acc_static_l = float((y_pred_static_lstm == y_test).mean())
    proba_static_lstm = lstm.predict_proba(X_test)
    print(f"LSTM static accuracy on S2: {acc_static_l:.3f}")
    results.append(("LSTM", "Static", acc_static_l))
    seq_acc_lstm_static = sequence_level_identification_accuracy(y_test, proba_static_lstm, seq_test, list(lstm.classes_))
    seq_metrics['LSTM Static'] = {'sequence_identification_accuracy': float(seq_acc_lstm_static)}

    # LSTM: adapted
    print("\n=== LSTM (adapted) fine-tune on 20% S2 ===")
    lstm_adapt = GazeLSTMClassifier(seq_length=10, epochs=20, batch_size=32, learning_rate=1e-3, device=device)
    lstm_adapt.train(X_train, y_train)
    lstm_adapt.train(X_adapt, y_adapt, continue_training=True)
    y_pred_adapt_lstm = lstm_adapt.predict(X_test)
    acc_adapt_l = float((y_pred_adapt_lstm == y_test).mean())
    proba_adapt_lstm = lstm_adapt.predict_proba(X_test)
    print(f"LSTM adapted accuracy on S2: {acc_adapt_l:.3f} (Δ {acc_adapt_l - acc_static_l:+.3f})")
    results.append(("LSTM", "Adapted", acc_adapt_l))
    seq_acc_lstm_adapt = sequence_level_identification_accuracy(y_test, proba_adapt_lstm, seq_test, list(lstm_adapt.classes_))
    seq_metrics['LSTM Adapted'] = {'sequence_identification_accuracy': float(seq_acc_lstm_adapt)}

    # Summary
    print("\n=== Summary (accuracy on S2) ===")
    for m, t, a in results:
        print(f"{m:5s} | {t:7s} | {a:.3f}")

    # ================= Biometric Metrics =================
    print("\n=== Biometric Metrics (Identification-like to Verification) ===")
    def report(proba, label: str, classes):
        scores_dict = build_user_score_vectors(y_test, proba, classes)
        global_eer, global_thr = aggregate_global_eer(scores_dict)
        print(f"{label} Global EER: {global_eer:.3f} @ threshold {global_thr:.4f}")
        # Per-user EER (optional concise)
        sample_users = list(scores_dict.keys())[:5]
        for u in sample_users:
            g = scores_dict[u]['genuine']
            imp = scores_dict[u]['impostor']
            eer_u, thr_u = compute_eer_from_scores(g, imp)
            print(f"  User {u} EER: {eer_u:.3f} @ {thr_u:.4f} (g={len(g)}, imp={len(imp)})")
        # FAR/FRR at global threshold
        far, frr = far_frr_at_threshold(np.concatenate([scores_dict[u]['genuine'] for u in scores_dict]),
                                        np.concatenate([scores_dict[u]['impostor'] for u in scores_dict]),
                                        global_thr)
        print(f"  FAR: {far:.3f} | FRR: {frr:.3f}")

    metrics_store = {}
    def capture(label, proba, classes):
        # sanitize probabilities to avoid non-finite values downstream
        proba = np.nan_to_num(proba, nan=0.0, posinf=1.0, neginf=0.0)
        scores_dict = build_user_score_vectors(y_test, proba, classes)
        global_eer, global_thr = aggregate_global_eer(scores_dict)
        far, frr = far_frr_at_threshold(np.concatenate([scores_dict[u]['genuine'] for u in scores_dict]),
                                        np.concatenate([scores_dict[u]['impostor'] for u in scores_dict]),
                                        global_thr)
        # Sequence-level pooled scores and EER
        seq_scores = sequence_level_scores_for_verification(y_test, proba, seq_test, list(classes))
        seq_global_eer, seq_global_thr = aggregate_global_eer(seq_scores)
        # clamp non-finite thresholds for JSON serialization
        if not np.isfinite(seq_global_thr):
            seq_global_thr = 0.0
        seq_far, seq_frr = far_frr_at_threshold(
            np.concatenate([seq_scores[u]['genuine'] for u in seq_scores]),
            np.concatenate([seq_scores[u]['impostor'] for u in seq_scores]),
            seq_global_thr,
        )
        per_user = {}
        for u, d in scores_dict.items():
            eer_u, thr_u = compute_eer_from_scores(d['genuine'], d['impostor'])
            per_user[u] = {
                'eer': float(eer_u),
                'threshold': float(thr_u),
                'genuine_count': int(len(d['genuine'])),
                'impostor_count': int(len(d['impostor']))
            }
        metrics_store[label] = {
            'global_eer': float(global_eer),
            'global_threshold': float(global_thr),
            'far_at_global_threshold': float(far),
            'frr_at_global_threshold': float(frr),
            'per_user': per_user,
            'sequence_level': {
                'global_eer': float(seq_global_eer),
                'global_threshold': float(seq_global_thr),
                'far_at_global_threshold': float(seq_far),
                'frr_at_global_threshold': float(seq_frr),
                'identification_accuracy': float(seq_metrics.get(label, {}).get('sequence_identification_accuracy', 0.0))
            }
        }
        report(proba, label, classes)

    capture("CNN Static", proba_static_cnn, cnn.classes_)
    capture("CNN Adapted", proba_adapt_cnn, cnn_adapt.classes_)
    capture("LSTM Static", proba_static_lstm, lstm.classes_)
    capture("LSTM Adapted", proba_adapt_lstm, lstm_adapt.classes_)

    # Persist
    out_path = root / 'results' / 'temporal_adaptation_metrics.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'subjects': subjects,
        'adapt_ratio': 0.2,
        'seq_length': 10,
        'accuracy_summary': {f"{m}-{t}": a for m, t, a in results},
        'biometric_metrics': metrics_store,
        'sequence_level_identification': seq_metrics
    }
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved metrics to {out_path}")


if __name__ == "__main__":
    main()
