"""
Advanced Synthetic Drift Strategies

These go beyond simple noise-based approaches to try to better 
replicate the complexity of real temporal drift.

Strategies:
1. Feature-Specific Drift - Different features drift at different rates
2. User-Specific Drift - Each user has their own drift pattern
3. Distribution Matching - Transform S1 to match S2's exact distribution
4. Temporal Decay - Drift that simulates gradual change over time
5. Adversarial Perturbation - Small perturbations that maximally confuse the model
6. Mixup Interpolation - Blend S1 and S2 features
"""

import numpy as np
from scipy import stats
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class AdvancedDriftProfile:
    """Extended drift profile with per-feature and per-user statistics."""
    n_features: int
    n_users: int
    
    # Per-feature statistics
    feature_drift_rates: np.ndarray  # How much each feature drifts
    feature_importance: np.ndarray   # Which features matter most for classification
    
    # Per-user statistics  
    user_drift_magnitudes: Dict[int, float]  # How much each user drifts
    user_drift_directions: Dict[int, np.ndarray]  # Direction of drift per user
    
    # Distribution statistics
    s1_means: np.ndarray
    s1_stds: np.ndarray
    s2_means: np.ndarray
    s2_stds: np.ndarray


class AdvancedDriftAnalyzer:
    """Analyze drift with more granularity."""
    
    def analyze(self, X_s1: np.ndarray, y_s1: np.ndarray,
                X_s2: np.ndarray, y_s2: np.ndarray) -> AdvancedDriftProfile:
        """Extract detailed drift characteristics."""
        n_features = X_s1.shape[1]
        
        # Global statistics
        s1_means = np.nanmean(X_s1, axis=0)
        s1_stds = np.nanstd(X_s1, axis=0) + 1e-8
        s2_means = np.nanmean(X_s2, axis=0)
        s2_stds = np.nanstd(X_s2, axis=0) + 1e-8
        
        # Per-feature drift rates (normalized)
        feature_drift_rates = np.abs(s2_means - s1_means) / s1_stds
        
        # Feature importance (based on variance explained)
        feature_importance = s1_stds / np.sum(s1_stds)
        
        # Per-user drift
        common_users = list(set(np.unique(y_s1)) & set(np.unique(y_s2)))
        user_drift_magnitudes = {}
        user_drift_directions = {}
        
        for user in common_users:
            u_s1 = X_s1[y_s1 == user]
            u_s2 = X_s2[y_s2 == user]
            if len(u_s1) > 0 and len(u_s2) > 0:
                u_s1_mean = np.nanmean(u_s1, axis=0)
                u_s2_mean = np.nanmean(u_s2, axis=0)
                drift_vec = u_s2_mean - u_s1_mean
                user_drift_magnitudes[user] = np.linalg.norm(drift_vec)
                user_drift_directions[user] = drift_vec / (np.linalg.norm(drift_vec) + 1e-8)
        
        return AdvancedDriftProfile(
            n_features=n_features,
            n_users=len(common_users),
            feature_drift_rates=feature_drift_rates,
            feature_importance=feature_importance,
            user_drift_magnitudes=user_drift_magnitudes,
            user_drift_directions=user_drift_directions,
            s1_means=s1_means,
            s1_stds=s1_stds,
            s2_means=s2_means,
            s2_stds=s2_stds
        )


class AdvancedSyntheticDrift:
    """Generate synthetic drift using advanced strategies."""
    
    def __init__(self, profile: AdvancedDriftProfile, seed: int = 42):
        self.profile = profile
        self.rng = np.random.default_rng(seed)
    
    def feature_specific_drift(self, X: np.ndarray, y: np.ndarray, 
                                strength: float = 1.0) -> np.ndarray:
        """
        Apply different drift rates to different features.
        Features that drifted more in real data drift more here.
        """
        X_out = X.copy()
        
        # Scale drift by per-feature rates
        drift_rates = self.profile.feature_drift_rates * strength
        
        for i in range(X.shape[1]):
            # Apply drift proportional to feature's real drift rate
            shift = (self.profile.s2_means[i] - self.profile.s1_means[i]) * drift_rates[i]
            X_out[:, i] += shift
            
            # Add feature-specific noise
            noise = self.rng.normal(0, self.profile.s1_stds[i] * drift_rates[i] * 0.1, X.shape[0])
            X_out[:, i] += noise
        
        return X_out
    
    def user_specific_drift(self, X: np.ndarray, y: np.ndarray,
                            strength: float = 1.0) -> np.ndarray:
        """
        Apply user-specific drift patterns.
        Each user drifts according to their actual drift direction and magnitude.
        """
        X_out = X.copy()
        
        for user in np.unique(y):
            mask = y == user
            
            if user in self.profile.user_drift_directions:
                direction = self.profile.user_drift_directions[user]
                magnitude = self.profile.user_drift_magnitudes[user]
                
                # Apply this user's specific drift
                drift = direction * magnitude * strength
                X_out[mask] += drift
                
                # Add user-specific noise
                noise_scale = magnitude * 0.1 * strength
                noise = self.rng.normal(0, noise_scale, X_out[mask].shape)
                X_out[mask] += noise
            else:
                # For unknown users, use average drift
                avg_drift = self.profile.s2_means - self.profile.s1_means
                X_out[mask] += avg_drift * strength * 0.5
        
        return X_out
    
    def distribution_matching(self, X: np.ndarray, y: np.ndarray,
                               strength: float = 1.0) -> np.ndarray:
        """
        Transform S1 features to exactly match S2's distribution.
        Uses z-score normalization then rescaling.
        """
        X_out = X.copy()
        
        for i in range(X.shape[1]):
            # Normalize to z-scores using S1 stats
            z = (X[:, i] - self.profile.s1_means[i]) / self.profile.s1_stds[i]
            
            # Rescale to S2 distribution
            target_mean = (1 - strength) * self.profile.s1_means[i] + strength * self.profile.s2_means[i]
            target_std = (1 - strength) * self.profile.s1_stds[i] + strength * self.profile.s2_stds[i]
            
            X_out[:, i] = z * target_std + target_mean
        
        return X_out
    
    def temporal_decay(self, X: np.ndarray, y: np.ndarray,
                       decay_rate: float = 0.1) -> np.ndarray:
        """
        Simulate gradual drift over time.
        Earlier samples drift less, later samples drift more.
        """
        X_out = X.copy()
        n_samples = len(X)
        
        # Create time-dependent drift factor
        time_factors = np.exp(decay_rate * np.arange(n_samples) / n_samples) - 1
        time_factors = time_factors / time_factors.max()  # Normalize to [0, 1]
        
        # Apply drift that increases over time
        drift = self.profile.s2_means - self.profile.s1_means
        
        for i in range(n_samples):
            X_out[i] += drift * time_factors[i]
        
        # Add increasing noise over time
        for i in range(n_samples):
            noise = self.rng.normal(0, self.profile.s1_stds * time_factors[i] * 0.1)
            X_out[i] += noise
        
        return X_out
    
    def adversarial_perturbation(self, X: np.ndarray, y: np.ndarray,
                                  epsilon: float = 0.1) -> np.ndarray:
        """
        Add small perturbations in the direction that maximally confuses classifiers.
        Uses the direction of highest feature variance as proxy for decision boundary.
        """
        X_out = X.copy()
        
        # For each user, perturb in direction of highest confusion
        for user in np.unique(y):
            mask = y == user
            user_X = X[mask]
            
            # Find direction of highest variance (likely decision boundary direction)
            if len(user_X) > 1:
                cov = np.cov(user_X.T)
                cov = np.nan_to_num(cov, nan=0.0)
                
                try:
                    eigenvalues, eigenvectors = np.linalg.eigh(cov)
                    # Perturb along principal component
                    direction = eigenvectors[:, -1]  # Highest variance direction
                    perturbation = direction * epsilon * self.profile.s1_stds
                    
                    # Randomly flip direction per sample
                    signs = self.rng.choice([-1, 1], size=len(user_X))
                    X_out[mask] += perturbation * signs[:, np.newaxis]
                except:
                    # Fallback to random noise
                    X_out[mask] += self.rng.normal(0, epsilon, user_X.shape)
        
        return X_out
    
    def mixup_interpolation(self, X_s1: np.ndarray, y_s1: np.ndarray,
                            X_s2: np.ndarray, y_s2: np.ndarray,
                            alpha: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Blend S1 and S2 features using mixup.
        This directly uses real S2 data, which should be the closest match.
        """
        X_out = []
        y_out = []
        
        for user in np.unique(y_s1):
            s1_mask = y_s1 == user
            s2_mask = y_s2 == user
            
            user_s1 = X_s1[s1_mask]
            user_s2 = X_s2[s2_mask]
            
            if len(user_s1) > 0 and len(user_s2) > 0:
                # Sample from both sessions and blend
                n_samples = len(user_s1)
                s2_indices = self.rng.choice(len(user_s2), n_samples, replace=True)
                
                # Random mixing coefficients
                lambdas = self.rng.beta(alpha, alpha, n_samples)
                
                mixed = user_s1 * lambdas[:, np.newaxis] + user_s2[s2_indices] * (1 - lambdas[:, np.newaxis])
                X_out.append(mixed)
                y_out.append(np.full(n_samples, user))
            elif len(user_s1) > 0:
                X_out.append(user_s1)
                y_out.append(np.full(len(user_s1), user))
        
        return np.concatenate(X_out), np.concatenate(y_out)


def create_advanced_drift_variants(X_s1: np.ndarray, y_s1: np.ndarray,
                                    X_s2: np.ndarray, y_s2: np.ndarray
                                    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Create all advanced synthetic drift variants.
    
    Returns dict mapping strategy name to (X_synthetic, y_synthetic).
    """
    # Analyze drift
    analyzer = AdvancedDriftAnalyzer()
    profile = analyzer.analyze(X_s1, y_s1, X_s2, y_s2)
    
    generator = AdvancedSyntheticDrift(profile)
    
    variants = {}
    
    # Feature-specific drift
    X_fs = generator.feature_specific_drift(X_s1, y_s1, strength=1.0)
    variants['feature_specific'] = (X_fs, y_s1.copy())
    
    # User-specific drift
    X_us = generator.user_specific_drift(X_s1, y_s1, strength=1.0)
    variants['user_specific'] = (X_us, y_s1.copy())
    
    # Distribution matching (full match)
    X_dm = generator.distribution_matching(X_s1, y_s1, strength=1.0)
    variants['distribution_match'] = (X_dm, y_s1.copy())
    
    # Distribution matching (partial - 70%)
    X_dm70 = generator.distribution_matching(X_s1, y_s1, strength=0.7)
    variants['distribution_70pct'] = (X_dm70, y_s1.copy())
    
    # Temporal decay
    X_td = generator.temporal_decay(X_s1, y_s1, decay_rate=0.5)
    variants['temporal_decay'] = (X_td, y_s1.copy())
    
    # Adversarial perturbation
    X_adv = generator.adversarial_perturbation(X_s1, y_s1, epsilon=0.2)
    variants['adversarial'] = (X_adv, y_s1.copy())
    
    # Mixup (50% blend with real S2)
    X_mix, y_mix = generator.mixup_interpolation(X_s1, y_s1, X_s2, y_s2, alpha=0.5)
    variants['mixup_50'] = (X_mix, y_mix)
    
    # Mixup (80% S2 - almost real)
    X_mix80, y_mix80 = generator.mixup_interpolation(X_s1, y_s1, X_s2, y_s2, alpha=0.2)
    variants['mixup_80'] = (X_mix80, y_mix80)
    
    return variants


def print_drift_profile_summary(profile: AdvancedDriftProfile):
    """Print summary of advanced drift profile."""
    print("\n" + "=" * 60)
    print("ADVANCED DRIFT PROFILE")
    print("=" * 60)
    print(f"Features: {profile.n_features}")
    print(f"Users analyzed: {profile.n_users}")
    
    print(f"\nTop 5 drifting features:")
    top_features = np.argsort(profile.feature_drift_rates)[::-1][:5]
    for idx in top_features:
        print(f"  Feature {idx}: drift_rate={profile.feature_drift_rates[idx]:.4f}")
    
    print(f"\nUser drift variability:")
    if profile.user_drift_magnitudes:
        magnitudes = list(profile.user_drift_magnitudes.values())
        print(f"  Mean: {np.mean(magnitudes):.4f}")
        print(f"  Std:  {np.std(magnitudes):.4f}")
        print(f"  Min:  {np.min(magnitudes):.4f}")
        print(f"  Max:  {np.max(magnitudes):.4f}")
    print("=" * 60)

