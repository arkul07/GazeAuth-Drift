import json
from copy import deepcopy
import numpy as np
import torch
import sys
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.gazebase_loader import load_gazebase_data, preprocess_gaze_data
from pipeline.feature_extractor import extract_gaze_features
from models.baselines import prepare_features
from data.calibrated_synthetic_drift import DriftAnalyzer, CalibratedSyntheticDrift
from models.baselines import BaselineClassifier
from models.temporal.gaze_cnn import GazeCNNClassifier
from models.temporal.gaze_lstm import GazeLSTMClassifier


def set_seeds(seed: int = 42):
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


# Local copies to avoid importing the full experiment script (which runs at import time)
ADAPT_RATIO = 0.40
SEQ_LENGTH = 5
EPOCHS = 15
device = "cuda" if torch.cuda.is_available() else "cpu"


def split_adaptation_by_user(X, y, adapt_ratio=ADAPT_RATIO):
    adapt_X, adapt_y, test_X, test_y = [], [], [], []
    for user in np.unique(y):
        mask = y == user
        user_X, user_y = X[mask], y[mask]
        n = len(user_X)
        if n <= 1:
            test_X.append(user_X)
            test_y.append(user_y)
            continue
        split = max(1, min(int(n * adapt_ratio), n - 1))
        adapt_X.append(user_X[:split])
        adapt_y.append(user_y[:split])
        test_X.append(user_X[split:])
        test_y.append(user_y[split:])
    return (np.concatenate(adapt_X), np.concatenate(adapt_y),
            np.concatenate(test_X), np.concatenate(test_y))


def create_mixed_replay(X_s1, y_s1, X_s2, y_s2, ratio=0.5):
    n_s1 = min(int(len(X_s2) * ratio / (1 - ratio)), len(X_s1))
    if n_s1 > 0:
        idx = np.random.choice(len(X_s1), n_s1, replace=False)
        X_mix = np.concatenate([X_s1[idx], X_s2])
        y_mix = np.concatenate([y_s1[idx], y_s2])
    else:
        X_mix, y_mix = X_s2, y_s2
    perm = np.random.permutation(len(X_mix))
    return X_mix[perm], y_mix[perm]


def run_adaptation_pipeline(X_train, y_train, X_adapt, y_adapt, X_test, y_test, model_type: str = "KNN"):
    results = {}
    if model_type == "KNN":
        model = BaselineClassifier(model_type="knn", n_neighbors=5)
        model.train(X_train, y_train)
        results['static'] = (model.predict(X_test) == y_test).mean()
        Xc = np.concatenate([X_train, X_adapt]); yc = np.concatenate([y_train, y_adapt])
        model_a = BaselineClassifier(model_type="knn", n_neighbors=5)
        model_a.train(Xc, yc)
        results['adapted'] = (model_a.predict(X_test) == y_test).mean()
    elif model_type == "SVM":
        model = BaselineClassifier(model_type="svm", kernel="rbf", C=1.0)
        model.train(X_train, y_train)
        results['static'] = (model.predict(X_test) == y_test).mean()
        Xc = np.concatenate([X_train, X_adapt]); yc = np.concatenate([y_train, y_adapt])
        model_a = BaselineClassifier(model_type="svm", kernel="rbf", C=1.0)
        model_a.train(Xc, yc)
        results['adapted'] = (model_a.predict(X_test) == y_test).mean()
    elif model_type == "CNN":
        model = GazeCNNClassifier(seq_length=SEQ_LENGTH, epochs=EPOCHS, device=device)
        model.train(X_train, y_train)
        results['static'] = (model.predict(X_test) == y_test).mean()
        X_mix, y_mix = create_mixed_replay(X_train, y_train, X_adapt, y_adapt)
        model.train(X_mix, y_mix, continue_training=True)
        results['adapted'] = (model.predict(X_test) == y_test).mean()
    elif model_type == "LSTM":
        model = GazeLSTMClassifier(seq_length=SEQ_LENGTH, epochs=EPOCHS, device=device)
        model.train(X_train, y_train)
        results['static'] = (model.predict(X_test) == y_test).mean()
        X_mix, y_mix = create_mixed_replay(X_train, y_train, X_adapt, y_adapt)
        model.train(X_mix, y_mix, continue_training=True)
        results['adapted'] = (model.predict(X_test) == y_test).mean()
    else:
        results['static'] = 0.0
        results['adapted'] = 0.0
    return results


def evaluate_combo(X_s1, y_s1, X_s2, y_s2, profile, params):
    # Build cov from S1->S2 deltas
    n = min(len(X_s1), len(X_s2))
    delta = X_s2[:n] - X_s1[:n]
    delta = delta - np.nanmean(delta, axis=0)
    delta_cov = np.cov(delta.T) if len(delta) > 1 else np.eye(X_s1.shape[1])

    gen = CalibratedSyntheticDrift(profile)
    X_syn, y_syn = gen.generate_synthetic_session2(
        X_s1, y_s1,
        strength=params.get('strength', 0.8),
        calibration_scale=params.get('calibration_scale', 0.6),
        temporal_decay=params.get('temporal_decay', 'linear'),
        period_index=params.get('period_index', 1),
        total_periods=params.get('total_periods', 3),
        correlated_noise=params.get('correlated_noise', True),
        noise_scale_factor=params.get('noise_scale_factor', 0.5),
        mean_shift_only=params.get('mean_shift_only', True),
        s1_s2_delta_cov=delta_cov,
    )

    X_adapt_real, y_adapt_real, X_test_real, y_test_real = split_adaptation_by_user(X_s2, y_s2)
    X_adapt_syn, y_adapt_syn, X_test_syn, y_test_syn = split_adaptation_by_user(X_syn, y_syn)

    models = ["KNN", "SVM", "CNN", "LSTM"]
    gaps = {}
    real_acc = {}
    syn_acc = {}
    for m in models:
        res_real = run_adaptation_pipeline(X_s1, y_s1, X_adapt_real, y_adapt_real, X_test_real, y_test_real, m)
        res_syn = run_adaptation_pipeline(X_s1, y_s1, X_adapt_syn, y_adapt_syn, X_test_syn, y_test_syn, m)
        real_adapt = res_real['adapted']
        syn_adapt = res_syn['adapted']
        gaps[m] = float(abs(real_adapt - syn_adapt))
        real_acc[m] = float(real_adapt)
        syn_acc[m] = float(syn_adapt)

    avg_gap = float(np.mean(list(gaps.values())))
    return avg_gap, gaps, real_acc, syn_acc


def main():
    set_seeds(42)
    print("Loading sessions...")
    data_root = Path(__file__).resolve().parents[1] / "data" / "raw"
    subjects = list(range(1002, 1010))
    df_s1 = load_gazebase_data(str(data_root), subjects, sessions=[1], tasks=["PUR", "TEX", "RAN"])
    df_s2 = load_gazebase_data(str(data_root), subjects, sessions=[2], tasks=["PUR", "TEX", "RAN"])

    print("Preprocess & extract features...")
    s1_clean = preprocess_gaze_data(df_s1)
    s2_clean = preprocess_gaze_data(df_s2)
    feats_s1 = extract_gaze_features(s1_clean, window_size_sec=5.0, overlap_sec=1.0)
    feats_s2 = extract_gaze_features(s2_clean, window_size_sec=5.0, overlap_sec=1.0)
    X_s1, y_s1 = prepare_features(feats_s1)
    X_s2, y_s2 = prepare_features(feats_s2)

    print("Analyzing drift...")
    analyzer = DriftAnalyzer()
    profile = analyzer.analyze(X_s1, y_s1, X_s2, y_s2)

    grid = [
        dict(name="A", strength=0.9, calibration_scale=0.7, temporal_decay="early_strong_poly", total_periods=3, noise_scale_factor=0.5, mean_shift_only=False),
        dict(name="B", strength=0.8, calibration_scale=0.6, temporal_decay="linear", total_periods=3, noise_scale_factor=0.6, mean_shift_only=True),
        dict(name="C", strength=1.0, calibration_scale=0.8, temporal_decay="linear", total_periods=3, noise_scale_factor=0.5, mean_shift_only=False),
        dict(name="D", strength=0.7, calibration_scale=0.7, temporal_decay="linear", total_periods=2, noise_scale_factor=0.6, mean_shift_only=True),
        dict(name="E", strength=0.9, calibration_scale=0.5, temporal_decay="early_strong_poly", total_periods=3, noise_scale_factor=0.3, mean_shift_only=True),
        dict(name="F", strength=1.1, calibration_scale=0.9, temporal_decay="early_strong_poly", total_periods=3, noise_scale_factor=0.6, mean_shift_only=False),
        dict(name="G", strength=0.85, calibration_scale=0.65, temporal_decay="linear", total_periods=3, noise_scale_factor=0.4, mean_shift_only=False),
        dict(name="H", strength=0.75, calibration_scale=0.55, temporal_decay="linear", total_periods=3, noise_scale_factor=0.5, mean_shift_only=False),
    ]

    results = []
    best = None
    for params in grid:
        print(f"Evaluating combo {params['name']}...", flush=True)
        avg_gap, gaps, real_acc, syn_acc = evaluate_combo(X_s1, y_s1, X_s2, y_s2, profile, params)
        row = dict(params=deepcopy(params), avg_gap=avg_gap, gaps=gaps, real=real_acc, synthetic=syn_acc)
        results.append(row)
        if best is None or avg_gap < best['avg_gap']:
            best = row
        print(f"  -> avg gap: {avg_gap:.3f} | per-model: {gaps}")

    out_path = "results/seq_friendly_tuning.json"
    with open(out_path, "w") as f:
        json.dump({"results": results, "best": best}, f, indent=2)
    print(f"Saved tuning results to {out_path}")
    print("Best params:")
    print(best)


if __name__ == "__main__":
    main()
