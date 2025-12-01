"""
Advanced Drift Strategies for Gaze Authentication
==================================================
Testing multiple drift approaches to better match real behavioral changes.

Strategies:
1. Gaussian Noise (Baseline) - Current approach
2. Feature-Specific Drift - Different rates for different features
3. Distribution Shift - Change means/variances, not just add noise
4. Temporal Decay - Gradual drift over time
5. Combined Strategy - Mix of above

Author: Sathya & Arya Kulkarni
Date: Nov 30, 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.preprocessing import StandardScaler


def apply_gaussian_drift(features: np.ndarray, magnitude: float = 0.15) -> np.ndarray:
    """
    Strategy 1: Uniform Gaussian Noise (Current approach - BASELINE)

    All features get same drift magnitude.
    Problem: Doesn't reflect real behavioral change.
    """
    noise = np.random.normal(
        0, magnitude * np.std(features, axis=0), size=features.shape
    )
    return features + noise


def apply_feature_specific_drift(
    features: np.ndarray, magnitude: float = 0.15, feature_names: List[str] = None
) -> np.ndarray:
    """
    Strategy 2: Feature-Specific Drift

    Different features drift at different rates based on behavioral research:
    - Fixations: Moderate drift (fatigue affects concentration)
    - Saccades: Low drift (motor patterns stable)
    - Velocity: High drift (speed changes with familiarity)
    - Statistical: Moderate drift
    """
    drifted = features.copy()
    n_features = features.shape[1]

    # Default feature groupings (assuming 46 features)
    # Fixations (0-6): 7 features
    # Saccades (7-15): 9 features
    # Scanpath (16-20): 5 features
    # Velocity (21-28): 8 features
    # Statistical (29-42): 14 features
    # Temporal (43-45): 3 features

    # Feature-specific drift rates (calibrated)
    drift_rates = np.ones(n_features) * magnitude

    if n_features >= 46:
        # Fixations: 1.2x baseline (fatigue effect)
        drift_rates[0:7] = magnitude * 1.2

        # Saccades: 0.7x baseline (stable motor patterns)
        drift_rates[7:16] = magnitude * 0.7

        # Scanpath: 1.0x baseline (moderate)
        drift_rates[16:21] = magnitude * 1.0

        # Velocity: 1.5x baseline (speed changes with practice)
        drift_rates[21:29] = magnitude * 1.5

        # Statistical: 1.1x baseline
        drift_rates[29:43] = magnitude * 1.1

        # Temporal: 1.3x baseline (blink rate, dwell time change)
        drift_rates[43:46] = magnitude * 1.3

    # Apply feature-specific noise
    for i in range(n_features):
        feature_std = np.std(features[:, i])
        if feature_std > 0:
            noise = np.random.normal(
                0, drift_rates[i] * feature_std, size=features.shape[0]
            )
            drifted[:, i] += noise

    return drifted


def apply_distribution_shift_drift(
    features: np.ndarray, magnitude: float = 0.15
) -> np.ndarray:
    """
    Strategy 3: Distribution Shift

    Real drift changes distributions, not just adds noise:
    - Mean shift: User gets faster/slower
    - Variance increase: User gets less consistent
    - Skewness: Distribution shape changes
    """
    drifted = features.copy()
    n_features = features.shape[1]

    for i in range(n_features):
        feature_col = drifted[:, i]
        feature_mean = np.mean(feature_col)
        feature_std = np.std(feature_col)

        if feature_std > 0:
            # 1. Mean shift (10% of drift magnitude)
            mean_shift = (
                np.random.uniform(-magnitude * 0.1, magnitude * 0.1) * feature_mean
            )

            # 2. Variance increase (drift makes behavior less consistent)
            variance_multiplier = 1 + magnitude * 0.5  # 7.5% more variable at mag=0.15

            # 3. Add noise with new variance
            noise = np.random.normal(
                mean_shift,
                variance_multiplier * feature_std * magnitude,
                size=features.shape[0],
            )

            drifted[:, i] = feature_col + noise

    return drifted


def apply_temporal_decay_drift(
    features: np.ndarray,
    magnitude: float = 0.15,
    days_elapsed: int = 7,
    half_life: float = 14.0,
) -> np.ndarray:
    """
    Strategy 4: Temporal Decay Model

    Drift increases gradually over time (exponential decay curve).

    Args:
        days_elapsed: Days between S1 and S2
        half_life: Days for drift to reach 50% of maximum
    """
    # Exponential decay: drift_factor = 1 - exp(-t / half_life)
    drift_factor = 1 - np.exp(-days_elapsed / half_life)

    # Apply scaled noise
    effective_magnitude = magnitude * drift_factor
    noise = np.random.normal(
        0, effective_magnitude * np.std(features, axis=0), size=features.shape
    )

    return features + noise


def apply_combined_drift(
    features: np.ndarray, magnitude: float = 0.15, days_elapsed: int = 7
) -> np.ndarray:
    """
    Strategy 5: Combined Approach

    Combines multiple strategies for realistic drift:
    1. Feature-specific rates
    2. Distribution shift
    3. Temporal decay
    """
    # Step 1: Temporal scaling
    half_life = 14.0
    drift_factor = 1 - np.exp(-days_elapsed / half_life)
    effective_magnitude = magnitude * drift_factor

    # Step 2: Feature-specific drift
    drifted = apply_feature_specific_drift(features, effective_magnitude)

    # Step 3: Add distribution shift (lighter, as we already have feature-specific)
    n_features = drifted.shape[1]
    for i in range(n_features):
        feature_mean = np.mean(drifted[:, i])
        # Small mean shift (5% of drift magnitude)
        mean_shift = (
            np.random.uniform(-effective_magnitude * 0.05, effective_magnitude * 0.05)
            * feature_mean
        )
        drifted[:, i] += mean_shift

    return drifted


def generate_drifted_session(
    S1_features: np.ndarray,
    drift_strategy: str = "gaussian",
    magnitude: float = 0.15,
    days_elapsed: int = 7,
) -> np.ndarray:
    """
    Main function: Generate drifted S2 from S1 using specified strategy.

    Args:
        S1_features: Original Session 1 features
        drift_strategy: One of ['gaussian', 'feature_specific', 'distribution_shift',
                                'temporal_decay', 'combined']
        magnitude: Drift magnitude (0.0-1.0)
        days_elapsed: Days between S1 and S2 (for temporal models)

    Returns:
        S2_synthetic: Drifted features
    """
    strategies = {
        "gaussian": apply_gaussian_drift,
        "feature_specific": apply_feature_specific_drift,
        "distribution_shift": apply_distribution_shift_drift,
        "temporal_decay": lambda f, m: apply_temporal_decay_drift(f, m, days_elapsed),
        "combined": lambda f, m: apply_combined_drift(f, m, days_elapsed),
    }

    if drift_strategy not in strategies:
        raise ValueError(
            f"Unknown strategy: {drift_strategy}. "
            f"Choose from {list(strategies.keys())}"
        )

    drift_func = strategies[drift_strategy]
    return drift_func(S1_features, magnitude)


if __name__ == "__main__":
    # Quick test
    print("Testing drift strategies...")

    # Create dummy features
    np.random.seed(42)
    S1 = np.random.randn(100, 46)  # 100 samples, 46 features

    print("\nOriginal S1 stats:")
    print(f"  Mean: {np.mean(S1):.4f}, Std: {np.std(S1):.4f}")

    for strategy in [
        "gaussian",
        "feature_specific",
        "distribution_shift",
        "temporal_decay",
        "combined",
    ]:
        S2 = generate_drifted_session(S1, strategy, magnitude=0.15, days_elapsed=7)
        print(f"\n{strategy.upper()}:")
        print(f"  Mean: {np.mean(S2):.4f}, Std: {np.std(S2):.4f}")
        print(f"  Difference from S1: {np.mean(np.abs(S2 - S1)):.4f}")

    print("\n✅ All strategies working!")
