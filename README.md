# Drift-Aware Gaze Authentication for AR/VR

## Project Overview

This project implements a **drift-aware continuous authentication system** for AR/VR using gaze biometrics. We address the challenge of temporal drift in behavioral biometrics by:
1. Analyzing real drift patterns from GazeBaseVR dataset (Session 1 → Session 2)
2. Generating calibrated synthetic drift that mimics real drift characteristics
3. Evaluating 4 models (KNN, SVM, CNN, LSTM) under static and adapted conditions

## Quick Start - Verify Results

### Prerequisites
```bash
# Activate virtual environment
source .venv/bin/activate  # or your venv path

# Install dependencies (if not already done)
pip install -r requirements.txt
```

### Reproduce Paper Results (Takes ~10-15 minutes)
```bash
# Run the main experiment with 8 subjects
python experiment_real_vs_synthetic.py --max-subjects 8 --seed 42
```

**Expected Output:**
- Real drift adapted accuracy: KNN 35.2%, SVM 40.7%, CNN 50.8%, LSTM 48.2%
- Best synthetic (calibrated_magmatch) adapted: KNN 59.2%, SVM 61.2%, CNN 55.1%, LSTM 30.6%
- Average gap from real: 16.6 percentage points
- Difficulty ratio: 0.956× of real drift
- Results saved to: `real_vs_synthetic_results.csv`
- Visualization saved to: `real_vs_synthetic_comparison.png`

### Generate All Visualizations
```bash
# Architecture diagrams (system, CNN, LSTM)
python generate_architecture_diagrams.py

# Drift analysis plots (PCA, feature magnitude, variance, correlation, user distribution)
python generate_drift_visualizations.py

# Performance comparison plots (summary table, per-model, gap heatmap, difficulty, adaptation)
python generate_report_plots.py
```

All plots are saved to the `images/` folder.

## Dataset: GazeBaseVR

File format: `S_1002_S1_5_RAN.csv`
- `1002` = Subject ID
- `S1` = Session 1, `S2` = Session 2 (2-3 weeks later)
- `5_RAN` = Task type (RAN=Random, TEX=Text, PUR=Pursuit)

**8 Subjects Used:** S_1002, S_1003, S_1004, S_1005, S_1007, S_1008, S_1010, S_1011

**Key Challenge:** Session 2 data shows significant drift from Session 1 due to learning, fatigue, headset fit changes, etc.

## System Architecture

Our pipeline consists of 7 modules:
1. **Data Loader** - Parses GazeBaseVR CSV files
2. **Feature Extractor** - Extracts 40+ features from 5-second windows (1s overlap)
3. **Drift Analyzer** - Computes S1→S2 mean shifts, variance ratios, covariance changes
4. **Synthetic Drift Generator** - Applies calibrated transformations to simulate drift
5. **Model Training** - Trains KNN/SVM/CNN/LSTM on Session 1
6. **Adaptation Module** - Fine-tunes with mixed replay (50% S1 + 50% S2)
7. **Evaluation** - Computes static/adapted accuracy, gap from real, difficulty ratio

## Feature Extraction (40 Features)

Extracted from 5-second sliding windows:
- **Gaze coordinates**: mean, std, min, max, median (H/V)
- **Pupil diameter**: mean, std, min, max, median
- **Fixation duration**: mean, std, min, max, median
- **Saccade velocity**: mean, std, min, max, median
- **Inter-saccade interval**: mean, std, min, max, median
- **Blink rate** and **gaze dispersion**

## Synthetic Drift Variants

We generate multiple synthetic drift variants:
- **calibrated_magmatch** (best): Magnitude-matched to real drift, gap=16.6%, difficulty=0.956×
- **calibrated**: Global mean drift parameters
- **calibrated_magmatch_per_user**: Per-user magnitude matching
- **light/heavy**: Scaled drift intensity
- **gaussian_only**: Unstructured noise only
- **mean_shift_only**: Mean shifts without variance changes

## Key Results

### Real Drift Performance (Adapted Accuracy)
- **KNN**: 35.2% (static: 29.6%, improvement: +5.6%)
- **SVM**: 40.7% (static: 32.2%, improvement: +8.5%)
- **CNN**: 50.8% (static: 38.2%, improvement: +12.6%)
- **LSTM**: 48.2% (static: 40.7%, improvement: +7.5%)
- **Average**: 43.7%

### Best Synthetic Drift (calibrated_magmatch)
- **KNN**: 59.2% (static: 34.2%, improvement: +25.0%)
- **SVM**: 61.2% (static: 29.1%, improvement: +32.1%)
- **CNN**: 55.1% (static: 11.7%, improvement: +43.4%)
- **LSTM**: 30.6% (static: 14.3%, improvement: +16.3%)
- **Average**: 51.5%

### Key Findings
1. **Asymmetric difficulty**: Synthetic drift is easier for shallow models (KNN/SVM) but harder for LSTM
2. **Gap from real**: 16.6 percentage points average (best among all variants)
3. **Difficulty ratio**: 0.956× of real drift magnitude
4. **Adaptation benefit**: Mixed replay fine-tuning improves all models significantly

## Project Structure

```
gaze_auth_project/
├── data/
│   ├── gazebase_loader.py              # CSV parsing and data loading
│   ├── calibrated_synthetic_drift.py   # Main synthetic drift generator
│   ├── advanced_synthetic_drift.py     # Additional drift variants
│   └── raw/                            # GazeBaseVR CSV files
├── pipeline/
│   ├── feature_extractor.py            # 40-feature extraction
│   ├── drift_monitor.py                # Drift detection
│   └── decision_module.py              # Authentication decision logic
├── models/
│   ├── baselines.py                    # KNN, SVM classifiers
│   └── temporal/                       # CNN, LSTM models
├── experiment_real_vs_synthetic.py     # Main experiment script
├── generate_architecture_diagrams.py   # System/model architecture plots
├── generate_drift_visualizations.py    # Drift analysis plots
├── generate_report_plots.py            # Performance comparison plots
├── real_vs_synthetic_results.csv       # Experiment results
└── images/                             # All generated visualizations
```

## Generated Visualizations (14 plots)

### Architecture Diagrams
- `plot_system_architecture.png` - End-to-end pipeline
- `plot_cnn_architecture.png` - CNN model structure
- `plot_lstm_architecture.png` - LSTM model structure

### Drift Analysis
- `plot_drift_pca_projection.png` - PCA visualization of S1→S2 drift
- `plot_feature_drift_magnitude.png` - Per-feature drift intensity
- `plot_variance_change_analysis.png` - Variance ratio distributions
- `plot_correlation_structure.png` - Covariance change heatmap
- `plot_user_drift_distribution.png` - Per-user drift magnitudes

### Performance Comparisons
- `plot_summary_table.png` - Static vs adapted accuracy table
- `plot_per_model_comparison.png` - Real vs synthetic per model
- `plot_gap_heatmap.png` - Gap from real for all variants
- `plot_difficulty_comparison.png` - Difficulty ratios
- `plot_adaptation_improvement.png` - Improvement from adaptation
- `real_vs_synthetic_comparison.png` - Overall comparison

## Model Architectures

### KNN
- k=5 neighbors
- Euclidean distance in 40-dimensional feature space

### SVM
- RBF kernel, C=1.0, γ=scale
- Multi-class classification

### CNN
- Input: sequences of 10 windows × 40 features
- 2× Conv1D layers (64, 128 filters, kernel=3)
- Max pooling + global average pooling
- Dense(64) + softmax output

### LSTM
- Input: sequences of 10 windows × 40 features
- 2× LSTM layers (64, 64 units)
- Dense(32) + softmax output
- Fine-tuning LR: 0.0001, epochs: 15

## Adaptation Strategy

**Mixed Replay Fine-Tuning:**
1. Train initial model on Session 1 only
2. Create replay buffer with 50% Session 1 windows
3. Mix with 50% Session 2 windows
4. Fine-tune with reduced learning rate (0.0001)
5. Prevents catastrophic forgetting while adapting to drift

## Citation

If you use this work, please cite:
```
Kulkarni, A., & Kumaraguru, S. (2025). Drift-Aware Continuous Authentication 
in AR/VR Using Simulated Gaze Biometrics. CS 228 Course Project, 
San Jose State University.
```

## References

- GazeBaseVR: Masse et al. (2023) - Large-scale VR eye-tracking dataset
- Eye movement biometrics: Galdi et al. (2016) - Critical survey
- Concept drift in biometrics: Carmona-Duarte et al. (2020)

---

**Project Status:** ✅ Complete  
**Experiment Results:** Verified and reproducible  
**Visualizations:** 14 publication-quality plots generated
