"""
Generate Additional Plots for Report

Creates publication-ready visualizations from experiment results:
1. Drift profile summary (top drifting features)
2. Per-model comparison (real vs best synthetic)
3. Difficulty metric visualization
4. Per-user drift magnitude distribution
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for publication quality
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9

# Load experiment results
results_df = pd.read_csv("real_vs_synthetic_results.csv")

print("=" * 70)
print("GENERATING REPORT PLOTS")
print("=" * 70)

# =============================================================================
# PLOT 1: Per-Model Comparison (Real vs Best Synthetic)
# =============================================================================
print("\n1. Creating per-model comparison plot...")

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

models = ["KNN", "SVM", "CNN", "LSTM"]
real_data = results_df[results_df['drift_type'] == 'REAL']
magmatch_data = results_df[results_df['drift_type'] == 'calibrated_magmatch']

x = np.arange(len(models))
width = 0.35

# Plot real and synthetic side-by-side
real_adapted = [real_data[real_data['model'] == m]['adapted'].values[0] for m in models]
syn_adapted = [magmatch_data[magmatch_data['model'] == m]['adapted'].values[0] for m in models]

bars1 = ax.bar(x - width/2, real_adapted, width, label='Real Drift', 
               color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, syn_adapted, width, label='Synthetic (magmatch)', 
               color='coral', alpha=0.8, edgecolor='black', linewidth=0.5)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}',
                ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Model Type', fontweight='bold')
ax.set_ylabel('Adapted Accuracy', fontweight='bold')
ax.set_title('Real vs Synthetic Drift: Per-Model Adapted Accuracy', fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend(loc='upper left', framealpha=0.9)
ax.set_ylim(0, 0.8)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("plot_per_model_comparison.png", dpi=150, bbox_inches='tight')
print("✅ Saved: plot_per_model_comparison.png")

# =============================================================================
# PLOT 2: Difficulty Metric Visualization
# =============================================================================
print("\n2. Creating difficulty metric plot...")

# Calculate difficulty metrics from the CSV
# Difficulty = how far synthetic is from S1 relative to real S2-S1 distance
# We'll approximate this using the gap metric

fig, ax = plt.subplots(1, 1, figsize=(12, 6))

# Get all synthetic variants (exclude REAL)
synthetic_variants = results_df['drift_type'].unique()
synthetic_variants = [v for v in synthetic_variants if v != 'REAL']

# Calculate average adapted accuracy for each variant
variant_scores = []
for variant in synthetic_variants:
    variant_data = results_df[results_df['drift_type'] == variant]
    avg_adapted = variant_data['adapted'].mean()
    variant_scores.append((variant, avg_adapted))

# Sort by score
variant_scores.sort(key=lambda x: x[1])

variants = [v[0] for v in variant_scores]
scores = [v[1] for v in variant_scores]

# Get real baseline
real_avg = results_df[results_df['drift_type'] == 'REAL']['adapted'].mean()

# Create horizontal bar chart
y_pos = np.arange(len(variants))
colors = ['coral' if 'magmatch' in v else 'lightsteelblue' for v in variants]

bars = ax.barh(y_pos, scores, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

# Highlight best variant
best_idx = variants.index('calibrated_magmatch')
bars[best_idx].set_color('darkred')
bars[best_idx].set_alpha(1.0)

# Add real drift reference line
ax.axvline(x=real_avg, color='green', linestyle='--', linewidth=2, 
           label=f'Real Drift Baseline ({real_avg:.1%})', alpha=0.7)

ax.set_yticks(y_pos)
ax.set_yticklabels(variants, fontsize=9)
ax.set_xlabel('Average Adapted Accuracy', fontweight='bold')
ax.set_title('Synthetic Drift Quality: Average Performance Across Models', 
             fontweight='bold', pad=15)
ax.legend(loc='lower right', framealpha=0.9)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig("plot_difficulty_comparison.png", dpi=150, bbox_inches='tight')
print("✅ Saved: plot_difficulty_comparison.png")

# =============================================================================
# PLOT 3: Gap Analysis Heatmap
# =============================================================================
print("\n3. Creating gap analysis heatmap...")

fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Calculate gaps from real for each model and variant
gap_matrix = []
gap_labels = []

for variant in synthetic_variants:
    gaps = []
    for model in models:
        real_acc = real_data[real_data['model'] == model]['adapted'].values[0]
        syn_acc = results_df[(results_df['drift_type'] == variant) & 
                             (results_df['model'] == model)]['adapted'].values[0]
        gap = abs(syn_acc - real_acc)
        gaps.append(gap)
    gap_matrix.append(gaps)
    gap_labels.append(variant)

gap_matrix = np.array(gap_matrix)

# Create heatmap
im = ax.imshow(gap_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=0.5)

# Set ticks and labels
ax.set_xticks(np.arange(len(models)))
ax.set_yticks(np.arange(len(gap_labels)))
ax.set_xticklabels(models)
ax.set_yticklabels(gap_labels, fontsize=9)

# Rotate the tick labels
plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Gap from Real Drift (Lower = Better)', rotation=270, labelpad=20, fontweight='bold')

# Add text annotations
for i in range(len(gap_labels)):
    for j in range(len(models)):
        text = ax.text(j, i, f'{gap_matrix[i, j]:.2f}',
                      ha="center", va="center", color="black", fontsize=8)

ax.set_title('Synthetic Drift Quality: Gap from Real Drift by Model', 
             fontweight='bold', pad=15)
ax.set_xlabel('Model Type', fontweight='bold')
ax.set_ylabel('Synthetic Drift Variant', fontweight='bold')

plt.tight_layout()
plt.savefig("plot_gap_heatmap.png", dpi=150, bbox_inches='tight')
print("✅ Saved: plot_gap_heatmap.png")

# =============================================================================
# PLOT 4: Adaptation Improvement Analysis
# =============================================================================
print("\n4. Creating adaptation improvement plot...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left plot: Improvement for Real drift
real_improvements = []
for model in models:
    imp = real_data[real_data['model'] == model]['improvement'].values[0]
    real_improvements.append(imp)

x_pos = np.arange(len(models))
bars = ax1.bar(x_pos, real_improvements, color=['steelblue', 'steelblue', 'coral', 'coral'],
               alpha=0.8, edgecolor='black', linewidth=0.5)

for i, bar in enumerate(bars):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:+.1%}',
            ha='center', va='bottom' if height > 0 else 'top', fontsize=9)

ax1.set_xlabel('Model Type', fontweight='bold')
ax1.set_ylabel('Improvement (Adapted - Static)', fontweight='bold')
ax1.set_title('Real Drift: Adaptation Benefit', fontweight='bold', pad=10)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(models)
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim(-0.05, 0.25)

# Right plot: Improvement for best synthetic
magmatch_improvements = []
for model in models:
    imp = magmatch_data[magmatch_data['model'] == model]['improvement'].values[0]
    magmatch_improvements.append(imp)

bars = ax2.bar(x_pos, magmatch_improvements, color=['steelblue', 'steelblue', 'coral', 'coral'],
               alpha=0.8, edgecolor='black', linewidth=0.5)

for i, bar in enumerate(bars):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:+.1%}',
            ha='center', va='bottom' if height > 0 else 'top', fontsize=9)

ax2.set_xlabel('Model Type', fontweight='bold')
ax2.set_ylabel('Improvement (Adapted - Static)', fontweight='bold')
ax2.set_title('Synthetic Drift (magmatch): Adaptation Benefit', fontweight='bold', pad=10)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(models)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim(-0.05, 0.4)

plt.tight_layout()
plt.savefig("plot_adaptation_improvement.png", dpi=150, bbox_inches='tight')
print("✅ Saved: plot_adaptation_improvement.png")

# =============================================================================
# PLOT 5: Summary Statistics Table (as image)
# =============================================================================
print("\n5. Creating summary statistics table...")

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.axis('tight')
ax.axis('off')

# Calculate summary statistics
summary_data = []
summary_data.append(['Metric', 'Real Drift', 'Synthetic (magmatch)', 'Gap'])
summary_data.append(['=' * 20, '=' * 15, '=' * 20, '=' * 10])

# Average adapted accuracy
real_avg_adapted = results_df[results_df['drift_type'] == 'REAL']['adapted'].mean()
syn_avg_adapted = results_df[results_df['drift_type'] == 'calibrated_magmatch']['adapted'].mean()
gap = abs(real_avg_adapted - syn_avg_adapted)
summary_data.append(['Avg Adapted Accuracy', f'{real_avg_adapted:.1%}', f'{syn_avg_adapted:.1%}', f'{gap:.1%}'])

# Average improvement
real_avg_imp = results_df[results_df['drift_type'] == 'REAL']['improvement'].mean()
syn_avg_imp = results_df[results_df['drift_type'] == 'calibrated_magmatch']['improvement'].mean()
gap_imp = abs(real_avg_imp - syn_avg_imp)
summary_data.append(['Avg Improvement', f'{real_avg_imp:+.1%}', f'{syn_avg_imp:+.1%}', f'{gap_imp:.1%}'])

# Best model performance
real_best = results_df[results_df['drift_type'] == 'REAL']['adapted'].max()
syn_best = results_df[results_df['drift_type'] == 'calibrated_magmatch']['adapted'].max()
gap_best = abs(real_best - syn_best)
summary_data.append(['Best Model (Adapted)', f'{real_best:.1%}', f'{syn_best:.1%}', f'{gap_best:.1%}'])

# Worst model performance
real_worst = results_df[results_df['drift_type'] == 'REAL']['adapted'].min()
syn_worst = results_df[results_df['drift_type'] == 'calibrated_magmatch']['adapted'].min()
gap_worst = abs(real_worst - syn_worst)
summary_data.append(['Worst Model (Adapted)', f'{real_worst:.1%}', f'{syn_worst:.1%}', f'{gap_worst:.1%}'])

# Create table
table = ax.table(cellText=summary_data, cellLoc='center', loc='center',
                colWidths=[0.35, 0.2, 0.25, 0.15])

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Style the header row
for i in range(4):
    cell = table[(0, i)]
    cell.set_facecolor('#4472C4')
    cell.set_text_props(weight='bold', color='white')

# Style separator row
for i in range(4):
    cell = table[(1, i)]
    cell.set_facecolor('#E7E6E6')

# Alternate row colors
for i in range(2, len(summary_data)):
    for j in range(4):
        cell = table[(i, j)]
        if i % 2 == 0:
            cell.set_facecolor('#F2F2F2')

ax.set_title('Real vs Synthetic Drift: Summary Statistics', 
             fontweight='bold', fontsize=14, pad=20)

plt.tight_layout()
plt.savefig("plot_summary_table.png", dpi=150, bbox_inches='tight')
print("✅ Saved: plot_summary_table.png")

print("\n" + "=" * 70)
print("✅ ALL REPORT PLOTS GENERATED SUCCESSFULLY!")
print("=" * 70)
print("\nGenerated files:")
print("  1. plot_per_model_comparison.png - Real vs synthetic per model")
print("  2. plot_difficulty_comparison.png - Synthetic variant quality ranking")
print("  3. plot_gap_heatmap.png - Gap analysis across models and variants")
print("  4. plot_adaptation_improvement.png - Adaptation benefits comparison")
print("  5. plot_summary_table.png - Key statistics summary")
print("\nThese plots complement the main real_vs_synthetic_comparison.png")
print("=" * 70)
