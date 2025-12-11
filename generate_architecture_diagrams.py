"""
Generate Model Architecture Diagrams

Creates publication-quality architecture diagrams for CNN and LSTM models.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 9

print("=" * 70)
print("GENERATING MODEL ARCHITECTURE DIAGRAMS")
print("=" * 70)

# =============================================================================
# CNN Architecture Diagram
# =============================================================================
print("\n1. Creating CNN architecture diagram...")

fig, ax = plt.subplots(1, 1, figsize=(14, 6))
ax.set_xlim(0, 14)
ax.set_ylim(0, 6)
ax.axis('off')

# Layer specifications
layers = [
    {"name": "Input", "shape": "(10, 40)", "color": "#E8F4F8", "x": 0.5, "width": 1.2},
    {"name": "Conv1D", "shape": "(10, 64)\nfilters=64\nkernel=3", "color": "#B3D9E8", "x": 2.2, "width": 1.4},
    {"name": "MaxPool", "shape": "(5, 64)\npool=2", "color": "#7FB3D5", "x": 4.1, "width": 1.2},
    {"name": "Conv1D", "shape": "(5, 128)\nfilters=128\nkernel=3", "color": "#4A90B8", "x": 5.8, "width": 1.4},
    {"name": "MaxPool", "shape": "(2, 128)\npool=2", "color": "#2E5F8C", "x": 7.7, "width": 1.2},
    {"name": "GlobalAvg\nPool", "shape": "(128,)", "color": "#1A3E5C", "x": 9.4, "width": 1.2},
    {"name": "Dense", "shape": "(64,)\nDropout=0.3", "color": "#FFD966", "x": 11.1, "width": 1.2},
    {"name": "Output", "shape": "(n_users,)\nSoftmax", "color": "#FF6B6B", "x": 12.8, "width": 1.2},
]

# Draw layers
for i, layer in enumerate(layers):
    # Draw box
    box = FancyBboxPatch(
        (layer["x"], 2), layer["width"], 2,
        boxstyle="round,pad=0.1", 
        edgecolor="black", facecolor=layer["color"],
        linewidth=2, zorder=2
    )
    ax.add_patch(box)
    
    # Add text
    ax.text(layer["x"] + layer["width"]/2, 3.5, layer["name"], 
            ha='center', va='center', fontweight='bold', fontsize=10, zorder=3)
    ax.text(layer["x"] + layer["width"]/2, 2.7, layer["shape"], 
            ha='center', va='center', fontsize=8, zorder=3)
    
    # Draw arrows
    if i < len(layers) - 1:
        arrow = FancyArrowPatch(
            (layer["x"] + layer["width"], 3),
            (layers[i+1]["x"], 3),
            arrowstyle='->', mutation_scale=20, 
            linewidth=2, color='black', zorder=1
        )
        ax.add_patch(arrow)

# Add title
ax.text(7, 5.2, 'CNN Architecture for Gaze Authentication', 
        ha='center', va='center', fontsize=14, fontweight='bold')

# Add legend
legend_text = "Temporal sequence → Convolutional feature extraction → Classification"
ax.text(7, 0.8, legend_text, ha='center', va='center', 
        fontsize=10, style='italic', color='darkblue')

plt.tight_layout()
plt.savefig("plot_cnn_architecture.png", dpi=150, bbox_inches='tight')
print("✅ Saved: plot_cnn_architecture.png")

# =============================================================================
# LSTM Architecture Diagram
# =============================================================================
print("\n2. Creating LSTM architecture diagram...")

fig, ax = plt.subplots(1, 1, figsize=(14, 6))
ax.set_xlim(0, 14)
ax.set_ylim(0, 6)
ax.axis('off')

# Layer specifications
layers = [
    {"name": "Input", "shape": "(10, 40)", "color": "#E8F4F8", "x": 0.5, "width": 1.2},
    {"name": "LSTM", "shape": "(10, 64)\nunits=64\nreturn_seq=True", "color": "#D4A5FF", "x": 2.5, "width": 1.6},
    {"name": "Dropout", "shape": "(10, 64)\nrate=0.3", "color": "#C085FF", "x": 4.7, "width": 1.2},
    {"name": "LSTM", "shape": "(64,)\nunits=64\nreturn_seq=False", "color": "#A855FF", "x": 6.5, "width": 1.6},
    {"name": "Dropout", "shape": "(64,)\nrate=0.3", "color": "#8A3FD5", "x": 8.7, "width": 1.2},
    {"name": "Dense", "shape": "(32,)\nReLU", "color": "#FFD966", "x": 10.5, "width": 1.2},
    {"name": "Output", "shape": "(n_users,)\nSoftmax", "color": "#FF6B6B", "x": 12.3, "width": 1.2},
]

# Draw layers
for i, layer in enumerate(layers):
    # Draw box
    box = FancyBboxPatch(
        (layer["x"], 2), layer["width"], 2,
        boxstyle="round,pad=0.1", 
        edgecolor="black", facecolor=layer["color"],
        linewidth=2, zorder=2
    )
    ax.add_patch(box)
    
    # Add text
    ax.text(layer["x"] + layer["width"]/2, 3.5, layer["name"], 
            ha='center', va='center', fontweight='bold', fontsize=10, zorder=3)
    ax.text(layer["x"] + layer["width"]/2, 2.7, layer["shape"], 
            ha='center', va='center', fontsize=8, zorder=3)
    
    # Draw arrows
    if i < len(layers) - 1:
        arrow = FancyArrowPatch(
            (layer["x"] + layer["width"], 3),
            (layers[i+1]["x"], 3),
            arrowstyle='->', mutation_scale=20, 
            linewidth=2, color='black', zorder=1
        )
        ax.add_patch(arrow)

# Add title
ax.text(7, 5.2, 'LSTM Architecture for Gaze Authentication', 
        ha='center', va='center', fontsize=14, fontweight='bold')

# Add legend
legend_text = "Temporal sequence → Recurrent processing → Dense classification"
ax.text(7, 0.8, legend_text, ha='center', va='center', 
        fontsize=10, style='italic', color='darkblue')

plt.tight_layout()
plt.savefig("plot_lstm_architecture.png", dpi=150, bbox_inches='tight')
print("✅ Saved: plot_lstm_architecture.png")

# =============================================================================
# System Architecture Overview
# =============================================================================
print("\n3. Creating system architecture overview...")

fig, ax = plt.subplots(1, 1, figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')

# Module specifications
modules = [
    # Row 1: Data input
    {"name": "GazebaseVR\nRaw Data", "x": 1, "y": 8.5, "width": 2, "height": 1, "color": "#E8F4F8"},
    
    # Row 2: Core pipeline
    {"name": "1. Data Loader", "x": 0.5, "y": 7, "width": 2, "height": 0.8, "color": "#B3E5FC"},
    {"name": "2. Feature\nExtractor", "x": 3, "y": 7, "width": 2, "height": 0.8, "color": "#81D4FA"},
    {"name": "3. Drift\nAnalyzer", "x": 5.5, "y": 7, "width": 2, "height": 0.8, "color": "#4FC3F7"},
    {"name": "4. Synthetic\nGenerator", "x": 8, "y": 7, "width": 2, "height": 0.8, "color": "#29B6F6"},
    
    # Row 3: Training and adaptation
    {"name": "5. Model\nTraining", "x": 1.75, "y": 5.3, "width": 2, "height": 0.8, "color": "#FFE082"},
    {"name": "6. Adaptation\nModule", "x": 6.75, "y": 5.3, "width": 2, "height": 0.8, "color": "#FFD54F"},
    
    # Row 4: Models
    {"name": "KNN", "x": 0.5, "y": 3.8, "width": 1.3, "height": 0.6, "color": "#C5E1A5"},
    {"name": "SVM", "x": 2.2, "y": 3.8, "width": 1.3, "height": 0.6, "color": "#C5E1A5"},
    {"name": "CNN", "x": 6.2, "y": 3.8, "width": 1.3, "height": 0.6, "color": "#AED581"},
    {"name": "LSTM", "x": 7.9, "y": 3.8, "width": 1.3, "height": 0.6, "color": "#AED581"},
    
    # Row 5: Evaluation
    {"name": "7. Evaluation\nModule", "x": 3.5, "y": 2.3, "width": 3, "height": 0.8, "color": "#FFAB91"},
    
    # Row 6: Output
    {"name": "Performance\nMetrics & Plots", "x": 3.5, "y": 0.8, "width": 3, "height": 0.8, "color": "#FF8A65"},
]

# Additional data boxes
data_boxes = [
    {"name": "Session 1\nTrain Data", "x": 11, "y": 7, "width": 1.8, "height": 0.6, "color": "#E1BEE7"},
    {"name": "Session 2\nReal Drift", "x": 11, "y": 6.2, "width": 1.8, "height": 0.6, "color": "#CE93D8"},
    {"name": "Synthetic\nDrift Variants", "x": 11, "y": 5.4, "width": 1.8, "height": 0.6, "color": "#BA68C8"},
    {"name": "Drift\nParameters", "x": 13.2, "y": 7, "width": 1.8, "height": 0.6, "color": "#AB47BC"},
]

# Draw modules
for module in modules:
    box = FancyBboxPatch(
        (module["x"], module["y"]), module["width"], module["height"],
        boxstyle="round,pad=0.05", 
        edgecolor="black", facecolor=module["color"],
        linewidth=2, zorder=2
    )
    ax.add_patch(box)
    ax.text(module["x"] + module["width"]/2, module["y"] + module["height"]/2, 
            module["name"], ha='center', va='center', 
            fontweight='bold', fontsize=9, zorder=3)

# Draw data boxes
for box_spec in data_boxes:
    box = FancyBboxPatch(
        (box_spec["x"], box_spec["y"]), box_spec["width"], box_spec["height"],
        boxstyle="round,pad=0.05", 
        edgecolor="darkviolet", facecolor=box_spec["color"],
        linewidth=1.5, linestyle='--', zorder=2
    )
    ax.add_patch(box)
    ax.text(box_spec["x"] + box_spec["width"]/2, box_spec["y"] + box_spec["height"]/2, 
            box_spec["name"], ha='center', va='center', 
            fontsize=8, zorder=3)

# Draw arrows
arrows = [
    # Input to loader
    ((2, 8.5), (1.5, 7.8)),
    # Loader to feature extractor
    ((2.5, 7.4), (3, 7.4)),
    # Feature extractor to drift analyzer
    ((5, 7.4), (5.5, 7.4)),
    # Drift analyzer to synthetic generator
    ((7.5, 7.4), (8, 7.4)),
    # Feature extractor to training
    ((4, 7), (2.75, 6.1)),
    # Synthetic generator to adaptation
    ((9, 7), (7.75, 6.1)),
    # Training to models (KNN, SVM)
    ((2.75, 5.3), (1.15, 4.4)),
    ((2.75, 5.3), (2.85, 4.4)),
    # Adaptation to models (CNN, LSTM)
    ((7.75, 5.3), (6.85, 4.4)),
    ((7.75, 5.3), (8.55, 4.4)),
    # Models to evaluation
    ((1.15, 3.8), (4.5, 3.1)),
    ((2.85, 3.8), (4.8, 3.1)),
    ((6.85, 3.8), (5.5, 3.1)),
    ((8.55, 3.8), (5.8, 3.1)),
    # Evaluation to output
    ((5, 2.3), (5, 1.6)),
    # Data boxes to modules
    ((11, 7.3), (10, 7.4)),  # S1 to generator
    ((11, 5.7), (8.9, 6.3)),  # Synthetic to adaptation
    ((13.2, 7.3), (10, 7.4)),  # Params to generator
]

for start, end in arrows:
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle='->', mutation_scale=15, 
        linewidth=1.5, color='black', alpha=0.6, zorder=1
    )
    ax.add_patch(arrow)

# Add title
ax.text(8, 9.5, 'Drift-Aware Gaze Authentication System Architecture', 
        ha='center', va='center', fontsize=16, fontweight='bold')

# Add legend
legend_elements = [
    mpatches.Rectangle((0, 0), 1, 1, fc="#B3E5FC", ec="black", label="Data Processing"),
    mpatches.Rectangle((0, 0), 1, 1, fc="#FFD54F", ec="black", label="Model Training"),
    mpatches.Rectangle((0, 0), 1, 1, fc="#AED581", ec="black", label="ML Models"),
    mpatches.Rectangle((0, 0), 1, 1, fc="#FF8A65", ec="black", label="Analysis & Output"),
    mpatches.Rectangle((0, 0), 1, 1, fc="#BA68C8", ec="darkviolet", label="Data Artifacts"),
]
ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9, fontsize=9)

plt.tight_layout()
plt.savefig("plot_system_architecture.png", dpi=150, bbox_inches='tight')
print("✅ Saved: plot_system_architecture.png")

print("\n" + "=" * 70)
print("✅ ALL ARCHITECTURE DIAGRAMS GENERATED SUCCESSFULLY!")
print("=" * 70)
print("\nGenerated files:")
print("  1. plot_cnn_architecture.png - CNN model architecture")
print("  2. plot_lstm_architecture.png - LSTM model architecture")
print("  3. plot_system_architecture.png - Complete system overview")
print("=" * 70)
