"""
Calibrated Synthetic Drift Generator

This module creates synthetic drift that mimics REAL drift observed in GazebaseVR.
The key insight: we measure actual S1→S2 changes and apply those same statistics.

Goal: synthetic_drift(S1) ≈ S2 (statistically similar)
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


def _make_psd(cov: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Return a symmetric positive semi-definite matrix close to cov.

    Strategy:
    - Symmetrize and replace NaNs/Infs with 0
    - Eigen clip to eps
    - If still not PD for multivariate normal, add diagonal jitter progressively
    """
    if cov is None:
        return cov
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    cov_sym = 0.5 * (cov + cov.T)
    w, v = np.linalg.eigh(cov_sym)
    w_clipped = np.clip(w, eps, None)
    cov_psd = (v * w_clipped) @ v.T
    cov_psd = 0.5 * (cov_psd + cov_psd.T)

    # Try Cholesky; if fails, add jitter until it works
    jitter = eps
    max_tries = 5
    for _ in range(max_tries):
        try:
            _ = np.linalg.cholesky(cov_psd + np.eye(cov_psd.shape[0]) * jitter)
            return cov_psd + np.eye(cov_psd.shape[0]) * jitter
        except np.linalg.LinAlgError:
            jitter *= 10
    # Final fallback
    return cov_psd + np.eye(cov_psd.shape[0]) * jitter


@dataclass
class DriftProfile:
    """Captures the statistical characteristics of real drift."""
    feature_names: list
    # Per-feature drift statistics (global)
    mean_shift: np.ndarray      # How much the mean changed (S2 - S1)
    std_ratio: np.ndarray       # How much std changed (S2_std / S1_std)
    correlation_change: float   # Change in feature correlations
    # Global statistics
    overall_magnitude: float    # Overall drift strength
    user_variability: float     # How much drift varies per user (scalar summary)
    # Per-user drift direction (more realistic shift directions)
    user_mean_shifts: Dict[int, np.ndarray]


class DriftAnalyzer:
    """Analyze real drift characteristics from Session 1 → Session 2."""
    
    def __init__(self):
        self.profile: Optional[DriftProfile] = None
    
    def analyze(self, X_s1: np.ndarray, y_s1: np.ndarray,
                X_s2: np.ndarray, y_s2: np.ndarray) -> DriftProfile:
        """
        Compute drift statistics between Session 1 and Session 2.
        
        Args:
            X_s1: Features from Session 1 (N1, features)
            y_s1: Labels from Session 1
            X_s2: Features from Session 2 (N2, features)
            y_s2: Labels from Session 2
            
        Returns:
            DriftProfile with measured drift characteristics
        """
        n_features = X_s1.shape[1]
        
        # Per-feature statistics
        s1_mean = np.nanmean(X_s1, axis=0)
        s1_std = np.nanstd(X_s1, axis=0) + 1e-8
        s2_mean = np.nanmean(X_s2, axis=0)
        s2_std = np.nanstd(X_s2, axis=0) + 1e-8
        
        mean_shift = s2_mean - s1_mean
        std_ratio = s2_std / s1_std
        
        # Correlation change (simplified: correlation of feature means)
        s1_corr = np.corrcoef(X_s1.T) if X_s1.shape[0] > 1 else np.eye(n_features)
        s2_corr = np.corrcoef(X_s2.T) if X_s2.shape[0] > 1 else np.eye(n_features)
        
        # Handle NaN in correlation matrices
        s1_corr = np.nan_to_num(s1_corr, nan=0.0)
        s2_corr = np.nan_to_num(s2_corr, nan=0.0)
        
        correlation_change = np.mean(np.abs(s2_corr - s1_corr))
        
        # Overall magnitude (normalized mean shift)
        overall_magnitude = np.mean(np.abs(mean_shift) / s1_std)

        # User variability and per-user mean shift
        user_drifts: list = []
        user_mean_shifts: Dict[int, np.ndarray] = {}
        common_users = set(np.unique(y_s1)) & set(np.unique(y_s2))
        for user in common_users:
            u_s1 = X_s1[y_s1 == user]
            u_s2 = X_s2[y_s2 == user]
            if len(u_s1) > 0 and len(u_s2) > 0:
                u_shift = (np.nanmean(u_s2, axis=0) - np.nanmean(u_s1, axis=0))
                user_mean_shifts[int(user)] = u_shift
                u_drift = float(np.mean(np.abs(u_shift)))
                user_drifts.append(u_drift)

        user_variability = float(np.std(user_drifts)) if user_drifts else 0.0

        self.profile = DriftProfile(
            feature_names=[f"feature_{i}" for i in range(n_features)],
            mean_shift=mean_shift,
            std_ratio=std_ratio,
            correlation_change=correlation_change,
            overall_magnitude=overall_magnitude,
            user_variability=user_variability,
            user_mean_shifts=user_mean_shifts,
        )

        return self.profile
    
    def print_summary(self):
        """Print human-readable drift summary."""
        if self.profile is None:
            print("No drift profile analyzed yet.")
            return
            
        p = self.profile
        print("\n" + "=" * 60)
        print("REAL DRIFT PROFILE (S1 → S2)")
        print("=" * 60)
        print(f"Overall magnitude: {p.overall_magnitude:.4f}")
        print(f"User variability: {p.user_variability:.4f}")
        print(f"Correlation change: {p.correlation_change:.4f}")
        print(f"\nTop 5 features with largest mean shift:")
        
        shift_indices = np.argsort(np.abs(p.mean_shift))[::-1][:5]
        for idx in shift_indices:
            print(f"  Feature {idx}: shift={p.mean_shift[idx]:.4f}, "
                  f"std_ratio={p.std_ratio[idx]:.4f}")
        print("=" * 60)


class CalibratedSyntheticDrift:
    """
    Generate synthetic drift calibrated to match real drift statistics.
    
    This is the KEY to making synthetic drift useful:
    - Measure real drift → extract statistics
    - Apply those same statistics to generate synthetic drift
    - Result: synthetic drift ≈ real drift (statistically)
    """
    
    def __init__(self, profile: Optional[DriftProfile] = None):
        self.profile = profile
        self.rng = np.random.default_rng(42)
    
    def set_profile(self, profile: DriftProfile):
        """Set the drift profile to match."""
        self.profile = profile
    
    def apply(self, X: np.ndarray, y: np.ndarray, 
              strength: float = 1.0,
              add_noise: bool = True,
              per_user_variation: bool = True,
              calibration_scale: float = 1.0,
              temporal_decay: Optional[str] = None,
              period_index: int = 1,
              total_periods: int = 1,
              correlated_noise: bool = False,
              noise_scale_factor: float = 1.0,
              mean_shift_only: bool = False,
              per_user_constant_noise: bool = False,
              s1_s2_delta_cov: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply calibrated synthetic drift to features.
        
        Args:
            X: Original features (N, features)
            y: User labels
            strength: Multiplier for drift intensity (1.0 = match real drift)
            add_noise: Add random noise component
            per_user_variation: Apply different drift per user
        
        Returns:
            X_drifted: Features with synthetic drift applied
        """
        if self.profile is None:
            raise ValueError("No drift profile set. Call set_profile() first.")

        X_drifted = X.copy()
        n_features = X.shape[1]

        # Ensure profile arrays match feature count and apply calibration scaling
        mean_shift_global = self.profile.mean_shift[:n_features] * calibration_scale
        std_ratio = 1.0 + (self.profile.std_ratio[:n_features] - 1.0) * calibration_scale
        # Safety clamp and finite cleanup to avoid invalid power operations
        std_ratio = np.nan_to_num(std_ratio, nan=1.0, posinf=1.0, neginf=1.0)
        std_ratio = np.clip(std_ratio, 0.1, 10.0)

        # Temporal decay schedule
        if temporal_decay:
            t = max(1, period_index)
            T = max(1, total_periods)
            if temporal_decay == "early_strong_poly":
                alpha = min(1.0, 0.5 + 0.3 * (t / T) + 0.2 * (t / T) ** 2)
            elif temporal_decay == "linear":
                alpha = min(1.0, t / T)
            else:
                alpha = 1.0
        else:
            alpha = 1.0

        if per_user_variation:
            for user in np.unique(y):
                user_mask = (y == user)
                # Use a modest per-user variation to avoid extreme exponents
                user_factor = 1.0 + self.rng.normal(0.0, 0.15)
                user_factor = float(np.clip(user_factor, 0.6, 1.4))

                # Prefer per-user mean shift direction when available
                user_shift = self.profile.user_mean_shifts.get(int(user), mean_shift_global)
                user_shift = user_shift[:n_features] * calibration_scale

                # Mean shift
                X_drifted[user_mask] += user_shift * (strength * user_factor * alpha)

                # Variance change around user mean (skip when mean_shift_only)
                if not mean_shift_only:
                    user_mean = np.nanmean(X_drifted[user_mask], axis=0)
                    centered = X_drifted[user_mask] - user_mean
                    scaled = centered * (std_ratio ** (strength * user_factor * alpha))
                    X_drifted[user_mask] = user_mean + scaled
        else:
            X_drifted += mean_shift_global * (strength * alpha)
            if not mean_shift_only:
                global_mean = np.nanmean(X_drifted, axis=0)
                centered = X_drifted - global_mean
                scaled = centered * (std_ratio ** (strength * alpha))
                X_drifted = global_mean + scaled

        # Noise component
        if add_noise:
            noise_scale = self.profile.overall_magnitude * 0.1 * strength * alpha * noise_scale_factor
            cov = None
            if correlated_noise and s1_s2_delta_cov is not None:
                cov = s1_s2_delta_cov[:n_features, :n_features]
                cov = _make_psd(cov)

            if per_user_constant_noise and per_user_variation:
                # Add a single constant noise vector per user (sequence-friendly)
                for user in np.unique(y):
                    user_mask = (y == user)
                    if cov is not None:
                        noise_vec = self.rng.multivariate_normal(np.zeros(n_features), cov)
                    else:
                        noise_vec = self.rng.normal(0, 1.0, size=n_features)
                    X_drifted[user_mask] += noise_vec * noise_scale
            else:
                if cov is not None:
                    mean = np.zeros(n_features)
                    # Use a stable eigen-based sampler and ignore validity warnings
                    correlated = self.rng.multivariate_normal(
                        mean, cov, size=X_drifted.shape[0], check_valid='ignore', method='eigh'
                    )
                    X_drifted += correlated * noise_scale
                else:
                    noise = self.rng.normal(0, noise_scale, X_drifted.shape)
                    X_drifted += noise

        return X_drifted
    
    def generate_synthetic_session2(self, X_s1: np.ndarray, y_s1: np.ndarray,
                                     strength: float = 1.0,
                                     calibration_scale: float = 1.0,
                                     temporal_decay: Optional[str] = None,
                                     period_index: int = 1,
                                     total_periods: int = 1,
                                     correlated_noise: bool = False,
                                     noise_scale_factor: float = 1.0,
                                     mean_shift_only: bool = False,
                                     per_user_constant_noise: bool = False,
                                     s1_s2_delta_cov: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a synthetic Session 2 from Session 1 data.
        
        This creates data that should statistically resemble real Session 2.
        
        Args:
            X_s1: Session 1 features
            y_s1: Session 1 labels
            strength: Drift strength (1.0 = match real drift)
            
        Returns:
            X_synthetic_s2, y_synthetic_s2
        """
        X_synthetic = self.apply(
            X_s1, y_s1,
            strength=strength,
            add_noise=True,
            per_user_variation=True,
            calibration_scale=calibration_scale,
            temporal_decay=temporal_decay,
            period_index=period_index,
            total_periods=total_periods,
            correlated_noise=correlated_noise,
            noise_scale_factor=noise_scale_factor,
            mean_shift_only=mean_shift_only,
            per_user_constant_noise=per_user_constant_noise,
            s1_s2_delta_cov=s1_s2_delta_cov,
        )
        return X_synthetic, y_s1.copy()


def create_drift_variants(
    X: np.ndarray,
    y: np.ndarray,
    profile: DriftProfile,
    s1_s2_delta_cov: Optional[np.ndarray] = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Create multiple synthetic drift variants for comparison.
    
    Returns dict with different drift strategies:
    - 'calibrated': Matches real drift statistics
    - 'light': 50% of real drift
    - 'heavy': 150% of real drift
    - 'gaussian_only': Simple Gaussian noise (baseline)
    - 'mean_shift_only': Only mean shift, no variance change
    """
    generator = CalibratedSyntheticDrift(profile)
    
    variants = {}
    
    # Prefer S1->S2 delta covariance when provided (more realistic correlated noise)
    fallback_cov = np.cov((X - np.nanmean(X, axis=0)).T)
    cov_to_use = s1_s2_delta_cov if s1_s2_delta_cov is not None else fallback_cov

    # Calibrated (matches real drift)
    X_cal, y_cal = generator.generate_synthetic_session2(
        X, y, strength=1.0, calibration_scale=1.0,
        temporal_decay="early_strong_poly", period_index=1, total_periods=2,
        correlated_noise=True,
        s1_s2_delta_cov=cov_to_use,
    )
    variants['calibrated'] = (X_cal, y_cal)
    
    # Light drift (50%)
    X_light, y_light = generator.generate_synthetic_session2(
        X, y, strength=0.5, calibration_scale=0.7,
        temporal_decay="early_strong_poly", period_index=1, total_periods=3,
        correlated_noise=True,
        s1_s2_delta_cov=cov_to_use,
    )
    variants['light'] = (X_light, y_light)
    
    # Heavy drift (150%)
    X_heavy, y_heavy = generator.generate_synthetic_session2(
        X, y, strength=1.2, calibration_scale=0.9,
        temporal_decay="early_strong_poly", period_index=2, total_periods=3,
        correlated_noise=True,
        s1_s2_delta_cov=cov_to_use,
    )
    variants['heavy'] = (X_heavy, y_heavy)
    
    # Gaussian noise only (baseline)
    rng = np.random.default_rng(42)
    noise_scale = profile.overall_magnitude * np.nanstd(X, axis=0)
    X_gaussian = X + rng.normal(0, noise_scale * 0.5, X.shape)
    variants['gaussian_only'] = (X_gaussian, y.copy())
    
    # Mean shift only (no variance change)
    X_mean_only = X + profile.mean_shift[:X.shape[1]]
    variants['mean_shift_only'] = (X_mean_only, y.copy())
    
    return variants


# Convenience function
def analyze_and_generate_drift(X_s1: np.ndarray, y_s1: np.ndarray,
                               X_s2: np.ndarray, y_s2: np.ndarray,
                               verbose: bool = True) -> Tuple[DriftProfile, CalibratedSyntheticDrift]:
    """
    One-liner to analyze real drift and create calibrated generator.
    
    Usage:
        profile, generator = analyze_and_generate_drift(X_s1, y_s1, X_s2, y_s2)
        X_synthetic = generator.apply(X_s1, y_s1)
    """
    analyzer = DriftAnalyzer()
    profile = analyzer.analyze(X_s1, y_s1, X_s2, y_s2)
    
    if verbose:
        analyzer.print_summary()
    
    generator = CalibratedSyntheticDrift(profile)
    return profile, generator

