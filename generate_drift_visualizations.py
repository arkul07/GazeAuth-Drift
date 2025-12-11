"""
Generate Drift Analysis Visualizations

Creates visualizations showing:
1. PCA projection of Session 1 vs Session 2 (real drift visualization)
2. Feature drift magnitude ranking (top drifting features)
3. Temporal drift evolution (how drift accumulates over time)
4. Per-user drift magnitude distribution
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10

print("=" * 70)
print("GENERATING DRIFT ANALYSIS VISUALIZATIONS")
print("=" * 70)

# =============================================================================
# Load Data
# =============================================================================
print("\n1. Loading data...")

# Import experiment code to reuse data loading logic
import sys
sys.path.append(str(Path(__file__).parent))

# Use cached features if available
features_file = Path('data/features_cached.parquet')
if features_file.exists():
    print("Loading cached features...")
    import pandas as pd
    df = pd.read_parquet(features_file)
    
    # Filter to subset of users
    subjects_to_load = [1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009]
    df = df[df['user_id'].isin(subjects_to_load)]
    
    # Split by session
    df_s1 = df[df['session'] == 1]
    df_s2 = df[df['session'] == 2]
    
    # Extract features and labels
    feature_cols = [col for col in df.columns if col not in ['user_id', 'session', 'round', 'task']]
    X_s1 = df_s1[feature_cols].values
    X_s2 = df_s2[feature_cols].values
    y_s1 = df_s1['user_id'].values
    y_s2 = df_s2['user_id'].values
    
    print(f"Loaded {len(subjects_to_load)} subjects from cache")
else:
    print("Cached features not found. Please run experiment first or use a smaller dataset.")
    exit(1)

from data.calibrated_synthetic_drift import DriftAnalyzer

print(f"✅ Loaded S1: {X_s1.shape}, S2: {X_s2.shape}")

# Clean data - replace inf/nan with zeros or median
X_s1 = np.nan_to_num(X_s1, nan=0.0, posinf=0.0, neginf=0.0)
X_s2 = np.nan_to_num(X_s2, nan=0.0, posinf=0.0, neginf=0.0)

print(f"✅ Data cleaned (NaN/Inf removed)")

# Analyze drift
analyzer = DriftAnalyzer()
drift_report = analyzer.analyze(X_s1, X_s2, y_s1, y_s2)

print(f"✅ Drift analysis complete")

# =============================================================================
# PLOT 1: PCA Projection (Session 1 vs Session 2)
# =============================================================================
print("\n2. Creating PCA projection plot...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Combine data for PCA fitting
X_combined = np.vstack([X_s1, X_s2])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)

# Fit PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Split back
X_s1_pca = X_pca[:len(X_s1)]
X_s2_pca = X_pca[len(X_s1):]

# Plot 1: All users combined
ax = axes[0]
scatter1 = ax.scatter(X_s1_pca[:, 0], X_s1_pca[:, 1], 
                     alpha=0.3, s=20, c='steelblue', label='Session 1', edgecolors='none')
scatter2 = ax.scatter(X_s2_pca[:, 0], X_s2_pca[:, 1], 
                     alpha=0.3, s=20, c='coral', label='Session 2', edgecolors='none')

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontweight='bold')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontweight='bold')
ax.set_title('Feature Space Drift: Session 1 → Session 2', fontweight='bold', pad=15)
ax.legend(loc='upper right', framealpha=0.9)
ax.grid(alpha=0.3)

# Plot 2: Per-user drift vectors
ax = axes[1]

# Sample a few users for clarity
unique_users = np.unique(y_s1)[:6]  # First 6 users
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_users)))

for i, user_id in enumerate(unique_users):
    # Get user data
    s1_mask = y_s1 == user_id
    s2_mask = y_s2 == user_id
    
    if np.sum(s1_mask) > 0 and np.sum(s2_mask) > 0:
        # Compute centroids
        s1_centroid = X_s1_pca[s1_mask].mean(axis=0)
        s2_centroid = X_s2_pca[s2_mask].mean(axis=0)
        
        # Plot points
        ax.scatter(s1_centroid[0], s1_centroid[1], 
                  c=[colors[i]], s=100, marker='o', edgecolors='black', linewidth=1,
                  label=f'User {user_id} S1', zorder=3)
        ax.scatter(s2_centroid[0], s2_centroid[1], 
                  c=[colors[i]], s=100, marker='s', edgecolors='black', linewidth=1,
                  alpha=0.7, zorder=3)
        
        # Draw drift vector
        ax.arrow(s1_centroid[0], s1_centroid[1],
                s2_centroid[0] - s1_centroid[0],
                s2_centroid[1] - s1_centroid[1],
                head_width=0.3, head_length=0.2, fc=colors[i], ec=colors[i],
                alpha=0.6, linewidth=2, zorder=2)

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontweight='bold')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontweight='bold')
ax.set_title('Per-User Drift Vectors (Sample)', fontweight='bold', pad=15)
ax.legend(loc='upper left', framealpha=0.9, fontsize=8, ncol=2)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("plot_drift_pca_projection.png", dpi=150, bbox_inches='tight')
print("✅ Saved: plot_drift_pca_projection.png")

# =============================================================================
# PLOT 2: Feature Drift Magnitude Ranking
# =============================================================================
print("\n3. Creating feature drift magnitude plot...")

fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Compute mean absolute drift per feature
feature_drifts = np.abs(drift_report.mean_shift)

# Sort by magnitude
sorted_indices = np.argsort(feature_drifts)[::-1]
top_n = 20  # Show top 20 features

top_indices = sorted_indices[:top_n]
top_drifts = feature_drifts[top_indices]

# Create feature names
feature_names = [f'Feature {i}' for i in range(len(feature_drifts))]

# Create horizontal bar chart
y_pos = np.arange(top_n)
bars = ax.barh(y_pos, top_drifts[np.arange(top_n)], 
               color=plt.cm.RdYlGn_r(top_drifts[np.arange(top_n)] / top_drifts.max()),
               alpha=0.8, edgecolor='black', linewidth=0.5)

ax.set_yticks(y_pos)
ax.set_yticklabels([feature_names[i] for i in top_indices], fontsize=9)
ax.set_xlabel('Mean Absolute Drift Magnitude', fontweight='bold')
ax.set_title('Top 20 Drifting Features (Session 1 → Session 2)', fontweight='bold', pad=15)
ax.grid(axis='x', alpha=0.3)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, top_drifts)):
    ax.text(val, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
            va='center', ha='left', fontsize=8, color='black')

plt.tight_layout()
plt.savefig("plot_feature_drift_magnitude.png", dpi=150, bbox_inches='tight')
print("✅ Saved: plot_feature_drift_magnitude.png")

# =============================================================================
# PLOT 3: Per-User Drift Magnitude Distribution
# =============================================================================
print("\n4. Creating per-user drift magnitude distribution...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Get per-user magnitudes from drift report
user_magnitudes = []
for user_id in np.unique(y_s1):
    s1_mask = y_s1 == user_id
    s2_mask = y_s2 == user_id
    
    if np.sum(s1_mask) > 0 and np.sum(s2_mask) > 0:
        s1_mean = X_s1[s1_mask].mean(axis=0)
        s2_mean = X_s2[s2_mask].mean(axis=0)
        magnitude = np.linalg.norm(s2_mean - s1_mean)
        user_magnitudes.append(magnitude)

user_magnitudes = np.array(user_magnitudes)

# Plot 1: Histogram
ax1.hist(user_magnitudes, bins=15, color='steelblue', alpha=0.7, 
         edgecolor='black', linewidth=1)
ax1.axvline(user_magnitudes.mean(), color='red', linestyle='--', 
           linewidth=2, label=f'Mean: {user_magnitudes.mean():.2f}')
ax1.axvline(np.median(user_magnitudes), color='green', linestyle='--', 
           linewidth=2, label=f'Median: {np.median(user_magnitudes):.2f}')
ax1.set_xlabel('Drift Magnitude (L2 Distance)', fontweight='bold')
ax1.set_ylabel('Number of Users', fontweight='bold')
ax1.set_title('Distribution of Per-User Drift Magnitudes', fontweight='bold', pad=15)
ax1.legend(framealpha=0.9)
ax1.grid(axis='y', alpha=0.3)

# Plot 2: Box plot with individual points
ax2.boxplot(user_magnitudes, vert=True, widths=0.5, patch_artist=True,
            boxprops=dict(facecolor='lightblue', alpha=0.7),
            medianprops=dict(color='red', linewidth=2),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5))

# Add jittered scatter points
x_jitter = np.random.normal(1, 0.04, len(user_magnitudes))
ax2.scatter(x_jitter, user_magnitudes, alpha=0.5, s=50, 
           c='darkblue', edgecolors='black', linewidth=0.5, zorder=3)

ax2.set_ylabel('Drift Magnitude (L2 Distance)', fontweight='bold')
ax2.set_title('Per-User Drift Magnitude Variability', fontweight='bold', pad=15)
ax2.set_xticks([1])
ax2.set_xticklabels(['All Users'])
ax2.grid(axis='y', alpha=0.3)

# Add statistics text
stats_text = f"n = {len(user_magnitudes)}\nμ = {user_magnitudes.mean():.2f}\nσ = {user_magnitudes.std():.2f}"
ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
        fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig("plot_user_drift_distribution.png", dpi=150, bbox_inches='tight')
print("✅ Saved: plot_user_drift_distribution.png")

# =============================================================================
# PLOT 4: Correlation Heatmap (S1 vs S2)
# =============================================================================
print("\n5. Creating correlation structure comparison...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Sample features for visualization (too many to show all)
n_features_to_show = 15
feature_subset = sorted_indices[:n_features_to_show]  # Top drifting features

# Compute correlation matrices
corr_s1 = np.corrcoef(X_s1[:, feature_subset].T)
corr_s2 = np.corrcoef(X_s2[:, feature_subset].T)
corr_diff = corr_s2 - corr_s1

# Plot Session 1 correlation
im1 = axes[0].imshow(corr_s1, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
axes[0].set_title('Session 1 Feature Correlations', fontweight='bold', pad=10)
axes[0].set_xlabel('Feature Index', fontweight='bold')
axes[0].set_ylabel('Feature Index', fontweight='bold')
plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

# Plot Session 2 correlation
im2 = axes[1].imshow(corr_s2, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
axes[1].set_title('Session 2 Feature Correlations', fontweight='bold', pad=10)
axes[1].set_xlabel('Feature Index', fontweight='bold')
axes[1].set_ylabel('Feature Index', fontweight='bold')
plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

# Plot difference
im3 = axes[2].imshow(corr_diff, cmap='RdBu_r', aspect='auto', vmin=-0.5, vmax=0.5)
axes[2].set_title('Correlation Change (S2 - S1)', fontweight='bold', pad=10)
axes[2].set_xlabel('Feature Index', fontweight='bold')
axes[2].set_ylabel('Feature Index', fontweight='bold')
plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04, label='Δ Correlation')

plt.tight_layout()
plt.savefig("plot_correlation_structure.png", dpi=150, bbox_inches='tight')
print("✅ Saved: plot_correlation_structure.png")

# =============================================================================
# PLOT 5: Variance Change Analysis
# =============================================================================
print("\n6. Creating variance change analysis...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Compute variance ratios
std_s1 = X_s1.std(axis=0)
std_s2 = X_s2.std(axis=0)
variance_ratio = std_s2 / (std_s1 + 1e-10)  # Avoid division by zero

# Sort by ratio
sorted_ratio_indices = np.argsort(np.abs(variance_ratio - 1.0))[::-1]
top_n = 20

# Plot 1: Variance ratio bar chart
top_ratio_indices = sorted_ratio_indices[:top_n]
y_pos = np.arange(top_n)

colors = ['coral' if r > 1.0 else 'steelblue' for r in variance_ratio[top_ratio_indices]]
bars = ax1.barh(y_pos, variance_ratio[top_ratio_indices], color=colors, 
                alpha=0.8, edgecolor='black', linewidth=0.5)

ax1.axvline(x=1.0, color='black', linestyle='--', linewidth=2, alpha=0.5, label='No change')
ax1.set_yticks(y_pos)
ax1.set_yticklabels([f'Feature {i}' for i in top_ratio_indices], fontsize=9)
ax1.set_xlabel('Variance Ratio (S2 / S1)', fontweight='bold')
ax1.set_title('Top 20 Features by Variance Change', fontweight='bold', pad=15)
ax1.legend(framealpha=0.9)
ax1.grid(axis='x', alpha=0.3)

# Add annotations
for i, (bar, val) in enumerate(zip(bars, variance_ratio[top_ratio_indices])):
    direction = 'left' if val > 1.0 else 'right'
    color = 'darkred' if val > 1.0 else 'darkblue'
    ax1.text(val, bar.get_y() + bar.get_height()/2, f'{val:.2f}×',
            va='center', ha=direction, fontsize=8, color=color, fontweight='bold')

# Plot 2: Scatter plot of S1 vs S2 std
ax2.scatter(std_s1, std_s2, alpha=0.6, s=50, c='steelblue', edgecolors='black', linewidth=0.5)

# Add diagonal line (no change)
max_std = max(std_s1.max(), std_s2.max())
ax2.plot([0, max_std], [0, max_std], 'r--', linewidth=2, alpha=0.7, label='No change')

# Highlight features with large changes
large_change_mask = np.abs(variance_ratio - 1.0) > 0.3
ax2.scatter(std_s1[large_change_mask], std_s2[large_change_mask], 
           alpha=0.8, s=100, c='coral', edgecolors='red', linewidth=1.5,
           label='Large change', zorder=3)

ax2.set_xlabel('Session 1 Std Dev', fontweight='bold')
ax2.set_ylabel('Session 2 Std Dev', fontweight='bold')
ax2.set_title('Feature Variability: Session 1 vs Session 2', fontweight='bold', pad=15)
ax2.legend(framealpha=0.9)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("plot_variance_change_analysis.png", dpi=150, bbox_inches='tight')
print("✅ Saved: plot_variance_change_analysis.png")

print("\n" + "=" * 70)
print("✅ ALL DRIFT VISUALIZATIONS GENERATED SUCCESSFULLY!")
print("=" * 70)
print("\nGenerated files:")
print("  1. plot_drift_pca_projection.png - PCA visualization of S1→S2 drift")
print("  2. plot_feature_drift_magnitude.png - Top drifting features ranked")
print("  3. plot_user_drift_distribution.png - Per-user drift magnitude stats")
print("  4. plot_correlation_structure.png - Correlation matrix changes")
print("  5. plot_variance_change_analysis.png - Variance ratio analysis")
print("=" * 70)
