"""
Test Advanced Synthetic Drift Strategies

New strategies:
1. feature_specific - Different drift per feature type
2. user_specific - Each user has own drift pattern  
3. distribution_match - Transform to exactly match S2 distribution
4. distribution_70pct - 70% match to S2 distribution
5. temporal_decay - Gradual drift over time
6. adversarial - Perturbations that confuse classifiers
7. mixup_50 - 50% blend of S1 and real S2
8. mixup_80 - 80% real S2 (almost cheating, but sets upper bound)
"""

import numpy as np
import pandas as pd
from pathlib import Path

print("=" * 70)
print("ADVANCED SYNTHETIC DRIFT COMPARISON")
print("=" * 70)

from data.gazebase_loader import load_gazebase_data, preprocess_gaze_data
from pipeline.feature_extractor import extract_gaze_features
from models.baselines import BaselineClassifier, prepare_features
from data.advanced_synthetic_drift import create_advanced_drift_variants

try:
    from models.temporal.gaze_cnn import GazeCNNClassifier
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

DATA_PATH = Path(__file__).parent / "data" / "raw"
subjects = list(range(1002, 1022))

# Quick helpers
def split_adapt_test(X, y, ratio=0.4):
    adapt_X, adapt_y, test_X, test_y = [], [], [], []
    for user in np.unique(y):
        mask = y == user
        uX, uy = X[mask], y[mask]
        n = len(uX)
        if n <= 1:
            test_X.append(uX); test_y.append(uy)
            continue
        s = max(1, min(int(n * ratio), n-1))
        adapt_X.append(uX[:s]); adapt_y.append(uy[:s])
        test_X.append(uX[s:]); test_y.append(uy[s:])
    return (np.concatenate(adapt_X), np.concatenate(adapt_y),
            np.concatenate(test_X), np.concatenate(test_y))

def test_knn(X_train, y_train, X_adapt, y_adapt, X_test, y_test):
    # Adapted KNN
    X_comb = np.concatenate([X_train, X_adapt])
    y_comb = np.concatenate([y_train, y_adapt])
    model = BaselineClassifier(model_type="knn", n_neighbors=5)
    model.train(X_comb, y_comb)
    return (model.predict(X_test) == y_test).mean()

# Load data
print("\nLoading data...")
df_s1 = load_gazebase_data(str(DATA_PATH), subjects, [1], ["PUR", "TEX", "RAN"])
df_s1 = preprocess_gaze_data(df_s1)
feat_s1 = extract_gaze_features(df_s1, window_size_sec=5.0, overlap_sec=1.0)
X_s1, y_s1 = prepare_features(feat_s1)

df_s2 = load_gazebase_data(str(DATA_PATH), subjects, [2], ["PUR", "TEX", "RAN"])
df_s2 = preprocess_gaze_data(df_s2)
feat_s2 = extract_gaze_features(df_s2, window_size_sec=5.0, overlap_sec=1.0)
X_s2, y_s2 = prepare_features(feat_s2)

print(f"S1: {len(X_s1)} windows, S2: {len(X_s2)} windows")

# Real drift baseline
print("\n--- REAL DRIFT BASELINE ---")
X_adapt_real, y_adapt_real, X_test_real, y_test_real = split_adapt_test(X_s2, y_s2)
real_acc = test_knn(X_s1, y_s1, X_adapt_real, y_adapt_real, X_test_real, y_test_real)
print(f"Real drift adapted accuracy: {real_acc:.1%}")

# Generate advanced variants
print("\n--- GENERATING ADVANCED SYNTHETIC VARIANTS ---")
variants = create_advanced_drift_variants(X_s1, y_s1, X_s2, y_s2)
print(f"Created {len(variants)} variants")

# Test each
print("\n--- TESTING SYNTHETIC VARIANTS ---")
print(f"{'Strategy':<20} {'Adapted Acc':>12} {'Gap from Real':>14}")
print("-" * 48)

results = []
for name, (X_syn, y_syn) in variants.items():
    X_adapt, y_adapt, X_test, y_test = split_adapt_test(X_syn, y_syn)
    acc = test_knn(X_s1, y_s1, X_adapt, y_adapt, X_test, y_test)
    gap = abs(acc - real_acc)
    quality = "✅" if gap < 0.05 else "⚠️" if gap < 0.15 else "❌"
    print(f"{name:<20} {acc:>12.1%} {gap:>13.1%} {quality}")
    results.append({'strategy': name, 'accuracy': acc, 'gap': gap})

# Summary
print("\n" + "=" * 48)
results_df = pd.DataFrame(results).sort_values('gap')
best = results_df.iloc[0]
print(f"🎯 BEST MATCH: {best['strategy']} (gap: {best['gap']:.1%})")
print(f"   Real: {real_acc:.1%}, Synthetic: {best['accuracy']:.1%}")

if best['gap'] < 0.05:
    print("\n✅ SUCCESS! Synthetic drift closely matches real drift!")
elif best['gap'] < 0.10:
    print("\n⚠️ CLOSE! Synthetic drift is reasonably close to real drift.")
else:
    print("\n❌ Gap still significant. Real drift has unique characteristics.")

# Save results
results_df.to_csv("advanced_synthetic_results.csv", index=False)
print("\nSaved: advanced_synthetic_results.csv")

