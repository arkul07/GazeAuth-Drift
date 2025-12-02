"""
Real vs Synthetic Drift Comparison Experiment

This is the NOVEL contribution of the project:
- Train models on Session 1
- Test on REAL Session 2 drift vs SYNTHETIC drift
- Goal: Synthetic drift should produce similar adaptation results

If successful, this proves synthetic drift can be used to:
1. Augment training data
2. Test drift-aware systems without real longitudinal data
3. Simulate various drift scenarios
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple

# =============================================================================
# CONFIGURATION
# =============================================================================
ADAPT_RATIO = 0.40
SEQ_LENGTH = 5
EPOCHS = 15  # Faster for comparison
MIXED_REPLAY_RATIO = 0.5

# Reproducibility
np.random.seed(42)

print("=" * 70)
print("REAL vs SYNTHETIC DRIFT COMPARISON")
print("=" * 70)

from data.gazebase_loader import load_gazebase_data, preprocess_gaze_data
from pipeline.feature_extractor import extract_gaze_features
from models.baselines import BaselineClassifier, prepare_features
from data.calibrated_synthetic_drift import (
    DriftAnalyzer,
    CalibratedSyntheticDrift,
    create_drift_variants,
)

try:
    import torch
    from models.temporal.gaze_cnn import GazeCNNClassifier
    from models.temporal.gaze_lstm import GazeLSTMClassifier
    TORCH_AVAILABLE = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"PyTorch: {device}")
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available")

DATA_PATH = Path(__file__).parent / "data" / "raw"
subjects = list(range(1002, 1022))  # 20 subjects


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def split_adaptation_by_user(X, y, adapt_ratio=0.4):
    """Per-user temporal split."""
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
    """Mix S1 + S2 data for fine-tuning."""
    n_s1 = min(int(len(X_s2) * ratio / (1 - ratio)), len(X_s1))
    if n_s1 > 0:
        idx = np.random.choice(len(X_s1), n_s1, replace=False)
        X_mix = np.concatenate([X_s1[idx], X_s2])
        y_mix = np.concatenate([y_s1[idx], y_s2])
    else:
        X_mix, y_mix = X_s2, y_s2
    perm = np.random.permutation(len(X_mix))
    return X_mix[perm], y_mix[perm]


def evaluate_model(model, X_test, y_test):
    """Get accuracy for a model."""
    pred = model.predict(X_test)
    return (pred == y_test).mean()


def run_adaptation_pipeline(X_train, y_train, X_adapt, y_adapt, X_test, y_test,
                             model_type: str = "KNN") -> Dict[str, float]:
    """
    Run the full adaptation pipeline for a given model type.
    
    Returns:
        Dict with 'static' and 'adapted' accuracy
    """
    results = {}
    
    if model_type == "KNN":
        # Static
        model = BaselineClassifier(model_type="knn", n_neighbors=5)
        model.train(X_train, y_train)
        results['static'] = evaluate_model(model, X_test, y_test)
        
        # Adapted (incremental)
        X_combined = np.concatenate([X_train, X_adapt])
        y_combined = np.concatenate([y_train, y_adapt])
        model_adapted = BaselineClassifier(model_type="knn", n_neighbors=5)
        model_adapted.train(X_combined, y_combined)
        results['adapted'] = evaluate_model(model_adapted, X_test, y_test)
        
    elif model_type == "SVM":
        # Static
        model = BaselineClassifier(model_type="svm", kernel="rbf", C=1.0)
        model.train(X_train, y_train)
        results['static'] = evaluate_model(model, X_test, y_test)
        
        # Adapted
        X_combined = np.concatenate([X_train, X_adapt])
        y_combined = np.concatenate([y_train, y_adapt])
        model_adapted = BaselineClassifier(model_type="svm", kernel="rbf", C=1.0)
        model_adapted.train(X_combined, y_combined)
        results['adapted'] = evaluate_model(model_adapted, X_test, y_test)
        
    elif model_type == "CNN" and TORCH_AVAILABLE:
        # Static
        model = GazeCNNClassifier(seq_length=SEQ_LENGTH, epochs=EPOCHS, device=device)
        model.train(X_train, y_train)
        results['static'] = evaluate_model(model, X_test, y_test)
        
        # Adapted (fine-tune with mixed replay)
        X_mix, y_mix = create_mixed_replay(X_train, y_train, X_adapt, y_adapt)
        model.train(X_mix, y_mix, continue_training=True)
        results['adapted'] = evaluate_model(model, X_test, y_test)
        
    elif model_type == "LSTM" and TORCH_AVAILABLE:
        # Static
        model = GazeLSTMClassifier(seq_length=SEQ_LENGTH, epochs=EPOCHS, device=device)
        model.train(X_train, y_train)
        results['static'] = evaluate_model(model, X_test, y_test)
        
        # Adapted
        X_mix, y_mix = create_mixed_replay(X_train, y_train, X_adapt, y_adapt)
        model.train(X_mix, y_mix, continue_training=True)
        results['adapted'] = evaluate_model(model, X_test, y_test)
    else:
        results['static'] = 0.0
        results['adapted'] = 0.0
    
    return results


# =============================================================================
# STEP 1: Load Real Data
# =============================================================================
print("\n" + "=" * 70)
print("STEP 1: Load Real Data (Sessions 1 & 2)")
print("=" * 70)

print("\nLoading Session 1...")
df_s1 = load_gazebase_data(str(DATA_PATH), subjects, sessions=[1], tasks=["PUR", "TEX", "RAN"])
df_s1_clean = preprocess_gaze_data(df_s1)
features_s1 = extract_gaze_features(df_s1_clean, window_size_sec=5.0, overlap_sec=1.0)
X_s1, y_s1 = prepare_features(features_s1)
print(f"✅ Session 1: {len(X_s1)} windows, {len(np.unique(y_s1))} users")

print("\nLoading Session 2...")
df_s2 = load_gazebase_data(str(DATA_PATH), subjects, sessions=[2], tasks=["PUR", "TEX", "RAN"])
df_s2_clean = preprocess_gaze_data(df_s2)
features_s2 = extract_gaze_features(df_s2_clean, window_size_sec=5.0, overlap_sec=1.0)
X_s2, y_s2 = prepare_features(features_s2)
print(f"✅ Session 2: {len(X_s2)} windows, {len(np.unique(y_s2))} users")

# =============================================================================
# STEP 2: Analyze Real Drift
# =============================================================================
print("\n" + "=" * 70)
print("STEP 2: Analyze Real Drift Characteristics")
print("=" * 70)

analyzer = DriftAnalyzer()
profile = analyzer.analyze(X_s1, y_s1, X_s2, y_s2)
analyzer.print_summary()

# Build S1->S2 delta covariance for correlated noise
n_cov = min(len(X_s1), len(X_s2))
_delta = X_s2[:n_cov] - X_s1[:n_cov]
_delta = _delta - np.nanmean(_delta, axis=0)
_delta_cov = np.cov(_delta.T) if len(_delta) > 1 else np.cov((X_s1 - np.nanmean(X_s1, axis=0)).T)

# =============================================================================
# STEP 3: Generate Synthetic Drift Variants
# =============================================================================
print("\n" + "=" * 70)
print("STEP 3: Generate Synthetic Drift Variants")
print("=" * 70)

drift_variants = create_drift_variants(X_s1, y_s1, profile, s1_s2_delta_cov=_delta_cov)
print(f"Created {len(drift_variants)} synthetic drift variants:")
for name in drift_variants:
    X_var, y_var = drift_variants[name]
    print(f"  - {name}: {len(X_var)} samples")
    try:
        mad_vs_s1 = float(np.nanmean(np.abs(X_var - X_s1[:len(X_var)])))
        mad_vs_s2 = float(np.nanmean(np.abs(X_var - X_s2[:len(X_var)])))
        print(f"    · mean|X_syn-X_s1|={mad_vs_s1:.4e}, mean|X_syn-X_s2|={mad_vs_s2:.4e}")
    except Exception:
        pass

# Add tuned calibrated variant using temporal decay and correlated noise
delta = (X_s2[:min(len(X_s2), len(X_s1))] - X_s1[:min(len(X_s2), len(X_s1))])
delta = delta - np.nanmean(delta, axis=0)
delta_cov = np.cov(delta.T) if len(delta) > 1 else np.eye(X_s1.shape[1])
cal_gen = CalibratedSyntheticDrift(profile)
X_cal_tuned, y_cal_tuned = cal_gen.generate_synthetic_session2(
    X_s1, y_s1,
    strength=0.9,
    calibration_scale=0.7,
    temporal_decay="early_strong_poly",
    period_index=2,
    total_periods=3,
    correlated_noise=True,
    s1_s2_delta_cov=delta_cov,
)
drift_variants['calibrated_tuned'] = (X_cal_tuned, y_cal_tuned)

# Add a sequence-friendly calibrated variant (gentler for CNN/LSTM)
try:
    delta = (X_s2[:min(len(X_s2), len(X_s1))] - X_s1[:min(len(X_s2), len(X_s1))])
    delta = delta - np.nanmean(delta, axis=0)
    delta_cov = np.cov(delta.T) if len(delta) > 1 else np.eye(X_s1.shape[1])
    cal_gen_seq = CalibratedSyntheticDrift(profile)
    X_cal_seq, y_cal_seq = cal_gen_seq.generate_synthetic_session2(
        X_s1, y_s1,
        strength=0.8,
        calibration_scale=0.5,
        temporal_decay="linear",
        period_index=1,
        total_periods=3,
        correlated_noise=True,
        noise_scale_factor=0.5,
        mean_shift_only=True,
        s1_s2_delta_cov=delta_cov,
    )
    drift_variants['calibrated_seq_friendly'] = (X_cal_seq, y_cal_seq)
    print(f"  - calibrated_seq_friendly: {len(X_cal_seq)} samples")
    try:
        mad_vs_s1 = float(np.nanmean(np.abs(X_cal_seq - X_s1[:len(X_cal_seq)])))
        mad_vs_s2 = float(np.nanmean(np.abs(X_cal_seq - X_s2[:len(X_cal_seq)])))
        print(f"    · mean|X_syn-X_s1|={mad_vs_s1:.4e}, mean|X_syn-X_s2|={mad_vs_s2:.4e}")
    except Exception:
        pass
except Exception as e:
    print(f"Failed to create calibrated_seq_friendly: {e}")

# Add magnitude-matched calibrated variant to align average |S2-S1| drift
try:
    n = min(len(X_s1), len(X_s2))
    target_mag = float(np.nanmean(np.abs(X_s2[:n] - X_s1[:n])))
    cal_gen_mm = CalibratedSyntheticDrift(profile)
    # Initial pass
    X_tmp, _ = cal_gen_mm.generate_synthetic_session2(
        X_s1, y_s1,
        strength=0.9,
        calibration_scale=1.0,
        temporal_decay="early_strong_poly",
        period_index=2,
        total_periods=3,
        correlated_noise=True,
        noise_scale_factor=0.5,
        mean_shift_only=False,
        s1_s2_delta_cov=delta_cov,
    )
    curr_mag = float(np.nanmean(np.abs(X_tmp[:n] - X_s1[:n]))) + 1e-8
    scale = np.clip(target_mag / curr_mag, 0.5, 5.0)
    X_cal_mm, y_cal_mm = cal_gen_mm.generate_synthetic_session2(
        X_s1, y_s1,
        strength=0.9,
        calibration_scale=scale,
        temporal_decay="early_strong_poly",
        period_index=2,
        total_periods=3,
        correlated_noise=True,
        noise_scale_factor=0.5,
        mean_shift_only=False,
        s1_s2_delta_cov=delta_cov,
    )
    drift_variants['calibrated_magmatch'] = (X_cal_mm, y_cal_mm)
    print(f"  - calibrated_magmatch: {len(X_cal_mm)} samples (scale={scale:.3f})")
except Exception as e:
    print(f"Failed to create calibrated_magmatch: {e}")

# Add two extra seq-friendly variants with different strengths for CNN/LSTM calibration
try:
    delta = (X_s2[:min(len(X_s2), len(X_s1))] - X_s1[:min(len(X_s2), len(X_s1))])
    delta = delta - np.nanmean(delta, axis=0)
    delta_cov = np.cov(delta.T) if len(delta) > 1 else np.eye(X_s1.shape[1])
    cal_gen_seq2 = CalibratedSyntheticDrift(profile)
    X_cal_seq2, y_cal_seq2 = cal_gen_seq2.generate_synthetic_session2(
        X_s1, y_s1,
        strength=0.6,
        calibration_scale=0.45,
        temporal_decay="linear",
        period_index=1,
        total_periods=2,
        correlated_noise=True,
        noise_scale_factor=0.4,
        mean_shift_only=True,
        s1_s2_delta_cov=delta_cov,
    )
    drift_variants['calibrated_seq_friendly_v2'] = (X_cal_seq2, y_cal_seq2)
    print(f"  - calibrated_seq_friendly_v2: {len(X_cal_seq2)} samples")
    try:
        mad_vs_s1 = float(np.nanmean(np.abs(X_cal_seq2 - X_s1[:len(X_cal_seq2)])))
        mad_vs_s2 = float(np.nanmean(np.abs(X_cal_seq2 - X_s2[:len(X_cal_seq2)])))
        print(f"    · mean|X_syn-X_s1|={mad_vs_s1:.4e}, mean|X_syn-X_s2|={mad_vs_s2:.4e}")
    except Exception:
        pass
except Exception as e:
    print(f"Failed to create calibrated_seq_friendly_v2: {e}")

try:
    delta = (X_s2[:min(len(X_s2), len(X_s1))] - X_s1[:min(len(X_s2), len(X_s1))])
    delta = delta - np.nanmean(delta, axis=0)
    delta_cov = np.cov(delta.T) if len(delta) > 1 else np.eye(X_s1.shape[1])
    cal_gen_seq3 = CalibratedSyntheticDrift(profile)
    X_cal_seq3, y_cal_seq3 = cal_gen_seq3.generate_synthetic_session2(
        X_s1, y_s1,
        strength=0.7,
        calibration_scale=0.55,
        temporal_decay="linear",
        period_index=1,
        total_periods=3,
        correlated_noise=True,
        noise_scale_factor=0.5,
        mean_shift_only=True,
        s1_s2_delta_cov=delta_cov,
    )
    drift_variants['calibrated_seq_friendly_v3'] = (X_cal_seq3, y_cal_seq3)
    print(f"  - calibrated_seq_friendly_v3: {len(X_cal_seq3)} samples")
    try:
        mad_vs_s1 = float(np.nanmean(np.abs(X_cal_seq3 - X_s1[:len(X_cal_seq3)])))
        mad_vs_s2 = float(np.nanmean(np.abs(X_cal_seq3 - X_s2[:len(X_cal_seq3)])))
        print(f"    · mean|X_syn-X_s1|={mad_vs_s1:.4e}, mean|X_syn-X_s2|={mad_vs_s2:.4e}")
    except Exception:
        pass
except Exception as e:
    print(f"Failed to create calibrated_seq_friendly_v3: {e}")

# =============================================================================
# STEP 4: Run Comparison Experiments
# =============================================================================
print("\n" + "=" * 70)
print("STEP 4: Compare Real vs Synthetic Drift")
print("=" * 70)

# Prepare real S2 splits
X_adapt_real, y_adapt_real, X_test_real, y_test_real = split_adaptation_by_user(X_s2, y_s2)
print(f"\nReal S2 split: {len(X_adapt_real)} adapt, {len(X_test_real)} test")

# Store all results
all_results = {}

# Test on REAL drift
print("\n--- Testing on REAL drift ---")
all_results['REAL'] = {}
for model_type in ["KNN", "SVM", "CNN", "LSTM"]:
    print(f"  {model_type}...", end=" ")
    try:
        res = run_adaptation_pipeline(X_s1, y_s1, X_adapt_real, y_adapt_real,
                                       X_test_real, y_test_real, model_type)
        all_results['REAL'][model_type] = res
        print(f"static={res['static']:.1%}, adapted={res['adapted']:.1%}, "
              f"Δ={res['adapted']-res['static']:+.1%}")
    except Exception as e:
        print(f"failed: {e}")
        all_results['REAL'][model_type] = {'static': 0, 'adapted': 0}

# Test on each SYNTHETIC variant
for variant_name, (X_syn, y_syn) in drift_variants.items():
    print(f"\n--- Testing on SYNTHETIC ({variant_name}) ---")
    
    # Split synthetic data same way as real
    X_adapt_syn, y_adapt_syn, X_test_syn, y_test_syn = split_adaptation_by_user(X_syn, y_syn)
    
    all_results[variant_name] = {}
    for model_type in ["KNN", "SVM", "CNN", "LSTM"]:
        print(f"  {model_type}...", end=" ")
        try:
            res = run_adaptation_pipeline(X_s1, y_s1, X_adapt_syn, y_adapt_syn,
                                           X_test_syn, y_test_syn, model_type)
            all_results[variant_name][model_type] = res
            print(f"static={res['static']:.1%}, adapted={res['adapted']:.1%}, "
                  f"Δ={res['adapted']-res['static']:+.1%}")
        except Exception as e:
            print(f"failed: {e}")
            all_results[variant_name][model_type] = {'static': 0, 'adapted': 0}

# =============================================================================
# STEP 5: Create Comparison Visualization
# =============================================================================
print("\n" + "=" * 70)
print("STEP 5: Create Comparison Visualization")
print("=" * 70)

# Prepare data for plotting
models = ["KNN", "SVM", "CNN", "LSTM"]
drift_types = [
    "REAL",
    # Show magmatch first among synthetic variants for emphasis in plots
    "calibrated_magmatch",
    "calibrated",
    "calibrated_tuned",
    "calibrated_seq_friendly",
    "calibrated_seq_friendly_v2",
    "calibrated_seq_friendly_v3",
    "light",
    "heavy",
    "gaussian_only",
    "mean_shift_only",
]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Static accuracy comparison
ax1 = axes[0, 0]
x = np.arange(len(drift_types))
width = 0.2
for i, model in enumerate(models):
    static_accs = [all_results[dt].get(model, {}).get('static', 0) for dt in drift_types]
    ax1.bar(x + i*width, static_accs, width, label=model, alpha=0.8)
ax1.set_ylabel("Accuracy", fontweight="bold")
ax1.set_title("Static Accuracy (No Adaptation)", fontweight="bold")
ax1.set_xticks(x + width*1.5)
ax1.set_xticklabels(drift_types, rotation=45, ha="right")
ax1.legend()
ax1.set_ylim(0, 1)
ax1.grid(axis="y", alpha=0.3)

# Plot 2: Adapted accuracy comparison
ax2 = axes[0, 1]
for i, model in enumerate(models):
    adapted_accs = [all_results[dt].get(model, {}).get('adapted', 0) for dt in drift_types]
    ax2.bar(x + i*width, adapted_accs, width, label=model, alpha=0.8)
ax2.set_ylabel("Accuracy", fontweight="bold")
ax2.set_title("Adapted Accuracy (With Fine-tuning)", fontweight="bold")
ax2.set_xticks(x + width*1.5)
ax2.set_xticklabels(drift_types, rotation=45, ha="right")
ax2.legend()
ax2.set_ylim(0, 1)
ax2.grid(axis="y", alpha=0.3)

# Plot 3: Improvement comparison
ax3 = axes[1, 0]
for i, model in enumerate(models):
    improvements = [
        all_results[dt].get(model, {}).get('adapted', 0) - 
        all_results[dt].get(model, {}).get('static', 0)
        for dt in drift_types
    ]
    ax3.bar(x + i*width, improvements, width, label=model, alpha=0.8)
ax3.set_ylabel("Improvement", fontweight="bold")
ax3.set_title("Adaptation Improvement (Adapted - Static)", fontweight="bold")
ax3.set_xticks(x + width*1.5)
ax3.set_xticklabels(drift_types, rotation=45, ha="right")
ax3.legend()
ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax3.grid(axis="y", alpha=0.3)

# Plot 4: Gap from Real (key metric!)
ax4 = axes[1, 1]
real_adapted = {m: all_results['REAL'].get(m, {}).get('adapted', 0) for m in models}
gaps = {}
for dt in drift_types[1:]:  # Skip 'REAL'
    gaps[dt] = []
    for model in models:
        syn_adapted = all_results[dt].get(model, {}).get('adapted', 0)
        gap = abs(syn_adapted - real_adapted[model])
        gaps[dt].append(gap)

x_gap = np.arange(len(drift_types) - 1)
for i, model in enumerate(models):
    gap_values = [gaps[dt][i] for dt in drift_types[1:]]
    ax4.bar(x_gap + i*width, gap_values, width, label=model, alpha=0.8)
ax4.set_ylabel("Absolute Gap from Real", fontweight="bold")
ax4.set_title("How Close is Synthetic to Real? (Lower = Better)", fontweight="bold")
ax4.set_xticks(x_gap + width*1.5)
ax4.set_xticklabels(drift_types[1:], rotation=45, ha="right")
ax4.legend()
ax4.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("real_vs_synthetic_comparison.png", dpi=150, bbox_inches="tight")
print("✅ Saved: real_vs_synthetic_comparison.png")

# =============================================================================
# STEP 6: Summary Report
# =============================================================================
print("\n" + "=" * 70)
print("📊 REAL vs SYNTHETIC DRIFT - SUMMARY REPORT")
print("=" * 70)

# Calculate average gap for each synthetic variant
print("\n🎯 SYNTHETIC DRIFT QUALITY (avg gap from real):")
print("-" * 50)
for variant in drift_types[1:]:
    avg_gap = np.mean([
        abs(all_results[variant].get(m, {}).get('adapted', 0) - 
            all_results['REAL'].get(m, {}).get('adapted', 0))
        for m in models
    ])
    quality = "✅ GOOD" if avg_gap < 0.10 else "⚠️ OK" if avg_gap < 0.20 else "❌ POOR"
    print(f"  {variant:20s}: {avg_gap:.1%} gap {quality}")

print("\n📈 BEST SYNTHETIC MATCH:")
best_variant = min(drift_types[1:], key=lambda v: np.mean([
    abs(all_results[v].get(m, {}).get('adapted', 0) - 
        all_results['REAL'].get(m, {}).get('adapted', 0))
    for m in models
]))
print(f"  → {best_variant}")

# Difficulty metric: how far synthetic is from S1 vs how far S2 is from S1
try:
    n_cmp = min(len(X_s1), len(X_s2))
    real_mag = float(np.nanmean(np.abs(X_s2[:n_cmp] - X_s1[:n_cmp]))) + 1e-8
    print("\n🧪 DIFFICULTY METRIC (relative to real |S2-S1|):")
    print("-" * 50)
    for variant in drift_types[1:]:
        X_syn, _ = (drift_variants.get(variant) if variant != 'REAL' else (X_s2, y_s2)) or (None, None)
        if X_syn is None:
            continue
        syn_mag = float(np.nanmean(np.abs(X_syn[:n_cmp] - X_s1[:n_cmp])))
        rel = syn_mag / real_mag
        print(f"  {variant:20s}: {rel:6.3f}× of real")
except Exception as _e:
    pass

print("\n📊 DETAILED COMPARISON (Real vs Best Synthetic):")
print("-" * 50)
print(f"{'Model':<8} {'Real Static':>12} {'Real Adapted':>13} {'Syn Static':>12} {'Syn Adapted':>13}")
print("-" * 50)
for model in models:
    r_stat = all_results['REAL'].get(model, {}).get('static', 0)
    r_adapt = all_results['REAL'].get(model, {}).get('adapted', 0)
    s_stat = all_results[best_variant].get(model, {}).get('static', 0)
    s_adapt = all_results[best_variant].get(model, {}).get('adapted', 0)
    print(f"{model:<8} {r_stat:>12.1%} {r_adapt:>13.1%} {s_stat:>12.1%} {s_adapt:>13.1%}")

print("\n" + "=" * 70)
print("✅ EXPERIMENT COMPLETE!")
print("=" * 70)

# Save results to file
results_df = []
for drift_type in drift_types:
    for model in models:
        res = all_results.get(drift_type, {}).get(model, {})
        results_df.append({
            'drift_type': drift_type,
            'model': model,
            'static': res.get('static', 0),
            'adapted': res.get('adapted', 0),
            'improvement': res.get('adapted', 0) - res.get('static', 0)
        })

pd.DataFrame(results_df).to_csv("real_vs_synthetic_results.csv", index=False)
print("\n✅ Saved results to: real_vs_synthetic_results.csv")

