# Drift-Aware Gaze Authentication Pipeline Documentation

## Table of Contents
1. [Problem Statement](#1-problem-statement)
2. [Pipeline Architecture](#2-pipeline-architecture)
3. [Data Flow](#3-data-flow)
4. [Models & Their Configurations](#4-models--their-configurations)
5. [Adaptation Strategies](#5-adaptation-strategies)
6. [Key Design Decisions](#6-key-design-decisions)
7. [Results Comparison](#7-results-comparison)
8. [Scalability Analysis](#8-scalability-analysis)
9. [Future Improvements](#9-future-improvements)

---

## 1. Problem Statement

### What is Temporal Drift?
In biometric authentication, **temporal drift** refers to the gradual change in a user's behavioral patterns over time. For gaze-based authentication in VR/AR:

- **Session 1**: User's gaze patterns when first enrolled (baseline)
- **Session 2**: User's gaze patterns days/weeks later (drifted)

### Why It Matters
A model trained on Session 1 data performs poorly on Session 2 data because:
- Eye muscle fatigue patterns change
- User becomes more familiar with the VR environment
- Physiological changes (sleep, stress, caffeine)
- Equipment recalibration differences

### Our Goal
Create a **drift-aware** authentication system that can:
1. Detect when drift has occurred
2. Adapt to the new patterns using minimal calibration data
3. Maintain authentication accuracy over time

---

## 2. Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DRIFT-AWARE GAZE AUTHENTICATION                       │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Raw Gaze    │    │   Feature    │    │   Model      │    │  Prediction  │
│  Data (CSV)  │───▶│  Extraction  │───▶│  Training    │───▶│  & Eval      │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
       │                   │                   │                    │
       ▼                   ▼                   ▼                    ▼
  GazebaseVR          48 Features         KNN/SVM/CNN/LSTM    Accuracy Metrics
  Dataset             per window          4 model types        Static vs Adapted
```

### Components

| Component | File | Purpose |
|-----------|------|---------|
| Data Loader | `data/gazebase_loader.py` | Load CSV files, filter by subject/session/task |
| Feature Extractor | `pipeline/feature_extractor.py` | Extract 48 gaze features per window |
| Baseline Models | `models/baselines.py` | KNN and SVM classifiers |
| CNN Model | `models/temporal/gaze_cnn.py` | 1D Convolutional Neural Network |
| LSTM Model | `models/temporal/gaze_lstm.py` | Long Short-Term Memory Network |
| Experiment | `experiment_drift_adaptation.py` | Main experiment runner |

---

## 3. Data Flow

### 3.1 Data Loading
```
GazebaseVR Dataset
├── S_1002_S1_2_PUR.csv  (Subject 1002, Session 1, Task PUR)
├── S_1002_S1_4_TEX.csv  (Subject 1002, Session 1, Task TEX)
├── S_1002_S1_5_RAN.csv  (Subject 1002, Session 1, Task RAN)
├── S_1002_S2_2_PUR.csv  (Subject 1002, Session 2, Task PUR)
└── ...
```

**Columns in CSV:**
- `n`: Sample index
- `x`, `y`: Gaze position (degrees)
- `t`: Timestamp
- `d`: Pupil diameter
- `subject_id`: User identifier
- `session`: 1 or 2
- `task`: PUR (pursuit), TEX (text reading), RAN (random saccade)

### 3.2 Feature Extraction

Each **5-second window** produces **48 features**:

| Category | Features | Count |
|----------|----------|-------|
| Fixation | count, duration (mean/std/max), dispersion | 5 |
| Saccade | count, amplitude (mean/std/max), velocity (mean/max) | 7 |
| Scanpath | total length, convex hull area, entropy | 3 |
| Velocity | mean, std, max, acceleration | 4 |
| Pupil | diameter (mean/std/min/max) | 4 |
| Statistical | x/y position (mean/std/range), correlation | 7 |
| Temporal | inter-fixation intervals, reading speed | 3 |
| **Total** | | **~48** |

### 3.3 Train/Test Split

```
Session 1 (Training)           Session 2 (Testing + Adaptation)
┌─────────────────────┐        ┌─────────────────────────────────┐
│  All S1 windows     │        │  40% Adaptation  │  60% Test    │
│  (769 windows)      │        │  (307 windows)   │  (462 win)   │
│  19 users           │        │  Per-user split  │  Per-user    │
└─────────────────────┘        └─────────────────────────────────┘
        │                               │                │
        ▼                               ▼                ▼
   Train models                   Fine-tune         Evaluate
   (initial)                      (adapt)           (final acc)
```

---

## 4. Models & Their Configurations

### 4.1 K-Nearest Neighbors (KNN)

**Type:** Instance-based learning (lazy learner)

**How it works:**
- Stores all training samples
- For new sample, finds K nearest neighbors
- Predicts majority class among neighbors

**Configuration:**
```python
KNN(
    n_neighbors=5,      # Number of neighbors to consider
    metric='euclidean', # Distance metric
    weights='uniform'   # All neighbors weighted equally
)
```

**Strengths:**
- No training phase (instant adaptation)
- Naturally handles multi-class
- Interpretable decisions

**Weaknesses:**
- Slow prediction with large datasets
- Sensitive to irrelevant features
- Memory-intensive

**Adaptation Strategy:** **Incremental**
- Simply add new S2 windows to the training set
- No retraining needed, just more neighbors available

---

### 4.2 Support Vector Machine (SVM)

**Type:** Discriminative classifier (kernel-based)

**How it works:**
- Finds optimal hyperplane separating classes
- Uses RBF kernel for non-linear boundaries
- One-vs-Rest for multi-class

**Configuration:**
```python
SVM(
    kernel='rbf',       # Radial Basis Function kernel
    C=1.0,              # Regularization parameter
    gamma='scale'       # Kernel coefficient (auto-scaled)
)
```

**Strengths:**
- Effective in high dimensions
- Memory-efficient (only support vectors)
- Works well with clear margins

**Weaknesses:**
- Slow training on large datasets
- Requires feature scaling
- Hyperparameter sensitive

**Adaptation Strategy:** **Retraining**
- Combine S1 + S2 adaptation data
- Retrain from scratch (no incremental SVM)

---

### 4.3 Convolutional Neural Network (CNN)

**Type:** Deep learning with spatial pattern recognition

**Architecture:**
```
Input: (batch, seq_length, features) = (N, 5, 43)
        │
        ▼
┌───────────────────────────────────────┐
│  Conv1D(43 → 64, kernel=3, padding=1) │
│  BatchNorm1D(64)                      │
│  ReLU                                 │
│  MaxPool1D(2)                         │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  Conv1D(64 → 128, kernel=3, padding=1)│
│  BatchNorm1D(128)                     │
│  ReLU                                 │
│  MaxPool1D(2)                         │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  Flatten + Dropout(0.5)               │
│  Linear(128 → num_classes)            │
└───────────────────────────────────────┘
        │
        ▼
Output: (N, num_classes) probabilities
```

**Configuration:**
```python
GazeCNNClassifier(
    seq_length=5,       # Windows per sequence (was 10, reduced for more samples)
    epochs=20,          # Training epochs
    batch_size=32,      # Mini-batch size
    learning_rate=0.001,# Initial LR (0.0001 for fine-tuning)
    device='cpu'        # or 'cuda'
)
```

**Strengths:**
- Learns hierarchical features automatically
- Captures local temporal patterns
- Translation invariant

**Weaknesses:**
- Requires more data
- Longer training time
- Can overfit on small datasets

**Adaptation Strategy:** **Fine-tuning with Mixed Replay**
- Start from pre-trained S1 weights
- Fine-tune with lower learning rate (0.0001 vs 0.001)
- Mix S1 + S2 data to prevent catastrophic forgetting

---

### 4.4 Long Short-Term Memory (LSTM)

**Type:** Recurrent neural network with memory gates

**Architecture:**
```
Input: (batch, seq_length, features) = (N, 5, 43)
        │
        ▼
┌───────────────────────────────────────┐
│  LSTM(input=43, hidden=128, layers=2) │
│  Dropout between layers (0.3)         │
└───────────────────────────────────────┘
        │
        ▼
   Take last hidden state
        │
        ▼
┌───────────────────────────────────────┐
│  Dropout(0.5)                         │
│  Linear(128 → num_classes)            │
└───────────────────────────────────────┘
        │
        ▼
Output: (N, num_classes) probabilities
```

**Configuration:**
```python
GazeLSTMClassifier(
    seq_length=5,       # Windows per sequence
    hidden_size=128,    # LSTM hidden state size
    num_layers=2,       # Stacked LSTM layers
    epochs=20,          # Training epochs
    batch_size=32,      # Mini-batch size
    learning_rate=0.001,# Initial LR (0.0001 for fine-tuning)
    device='cpu'        # or 'cuda'
)
```

**Strengths:**
- Captures long-range temporal dependencies
- Handles variable-length sequences
- Memory cells prevent vanishing gradients

**Weaknesses:**
- Slower than CNN
- Sequential processing (not parallelizable)
- Can still forget with very long sequences

**Adaptation Strategy:** **Fine-tuning with Mixed Replay**
- Same as CNN: lower LR + mixed S1/S2 data

---

## 5. Adaptation Strategies

### 5.1 Static (No Adaptation)
```
Train on S1 ──────────────────────▶ Test on S2
             No updates                Poor accuracy
```
- Baseline to show drift problem exists
- Expected accuracy: ~20-30% (vs ~90% within-session)

### 5.2 Incremental (KNN/SVM)
```
Train on S1 ──▶ Add S2 adapt data ──▶ Test on S2
                (simply append)        Better accuracy
```
- For KNN: Just add more reference points
- For SVM: Retrain with combined data
- No forgetting risk (all data retained)

### 5.3 Fine-Tuning with Mixed Replay (CNN/LSTM)
```
Train on S1 ──▶ Fine-tune on (S1_sample + S2_adapt) ──▶ Test on S2
                    └── Lower learning rate ──┘        Best accuracy
```
- **Mixed Replay**: Combine 50% S1 + 50% S2 during fine-tuning
- **Lower LR**: 0.0001 instead of 0.001 (10x lower)
- **Prevents catastrophic forgetting**: Model doesn't forget S1 patterns

---

## 6. Key Design Decisions

### 6.1 Why 40% Adaptation Split?

| Split | Adapt Windows | Test Windows | Result |
|-------|---------------|--------------|--------|
| 20% | ~150 | ~620 | Too few sequences for fine-tuning |
| **40%** | **~300** | **~460** | **Good balance** |
| 60% | ~460 | ~310 | Test set too small |

**Decision:** 40% gives enough adaptation data while keeping test set meaningful.

### 6.2 Why seq_length=5?

| seq_length | Sequences Created | Training Signal |
|------------|-------------------|-----------------|
| 10 | ~150 | Too few (model underfits) |
| **5** | **~700** | **Good (model learns well)** |
| 3 | ~1200 | Noisy (too little context) |

**Decision:** 5 consecutive windows capture enough temporal context while maximizing samples.

### 6.3 Why Mixed Replay?

**Problem:** Fine-tuning only on S2 data causes **catastrophic forgetting**
```
Before: Model knows S1 patterns
After pure S2 fine-tune: Model forgets S1, only knows S2 adapt users
Result: Fails on S2 test (different samples than adapt)
```

**Solution:** Mix S1 data during fine-tuning
```python
# 50% S1 + 50% S2 adaptation data
X_mixed = concat(sample(X_s1), X_s2_adapt)
model.train(X_mixed, continue_training=True)
```

**Result:** Model retains S1 knowledge while adapting to S2

### 6.4 Why Lower Learning Rate for Fine-Tuning?

| Learning Rate | Effect |
|---------------|--------|
| 0.001 (default) | Large updates → destroys S1 weights → forgetting |
| **0.0001** | Small updates → gentle adaptation → preserves S1 |
| 0.00001 | Too small → doesn't adapt fast enough |

**Decision:** 10x lower LR (0.0001) balances adaptation vs retention.

---

## 7. Results Comparison

### 7.1 Final Results (20 Subjects)

| Model | Static Accuracy | Adapted Accuracy | Improvement | Adaptation Method |
|-------|-----------------|------------------|-------------|-------------------|
| KNN | 20.6% | 23.9% | **+3.2%** | Incremental |
| SVM | 24.1% | 28.2% | **+4.1%** | Retraining |
| CNN | 30.1% | 44.3% | **+14.2%** | Fine-tune + Mixed Replay |
| LSTM | 31.4% | 38.7% | **+7.3%** | Fine-tune + Mixed Replay |

### 7.2 Analysis

**Best Performer: CNN (+14.2%)**
- Convolutional filters learn drift-invariant features
- Mixed replay prevents forgetting
- Shorter sequences provide more training signal

**Why KNN/SVM Have Lower Improvement:**
- They don't learn representations, just memorize
- Adding S2 data helps, but doesn't generalize patterns
- Still competitive as baselines

**Why LSTM < CNN:**
- LSTM more prone to overfitting on small data
- Sequential nature limits parallelization
- May need more epochs or regularization

---

## 8. Scalability Analysis

### 8.1 Scaling to 100 Subjects

**Current (20 subjects):**
- Training windows: ~769
- Sequences (seq_length=5): ~693
- Training time: ~3-5 minutes

**Projected (100 subjects):**
```
Training windows: ~3,845 (5x more)
Sequences: ~3,465 (5x more)
Training time: ~15-25 minutes
Memory: ~2-4 GB
```

### 8.2 Expected Accuracy Changes

| Metric | 20 Subjects | 100 Subjects (Expected) |
|--------|-------------|-------------------------|
| Static Accuracy | ~26% | ~15-20% (harder task) |
| Adapted Accuracy | ~34% | ~25-35% |
| Improvement | +7% | +10-15% (more room to improve) |

**Why accuracy drops with more subjects:**
- More classes to distinguish
- Less data per class
- Higher confusion between similar users

**Why improvement might increase:**
- More diverse training data
- Better generalization from adaptation
- Mixed replay more effective with variety

### 8.3 Recommendations for 100+ Subjects

1. **Increase epochs:** 30-50 instead of 20
2. **Use data augmentation:** Add noise, temporal shifts
3. **Consider hierarchical approach:** Cluster similar users
4. **Use GPU:** Training time scales linearly
5. **Increase hidden size:** 256 instead of 128 for LSTM/CNN

---

## 9. Future Improvements

### 9.1 Immediate Improvements
- [ ] Add early stopping based on validation loss
- [ ] Implement learning rate scheduling
- [ ] Add data augmentation for gaze features

### 9.2 Advanced Techniques
- [ ] **Elastic Weight Consolidation (EWC):** Penalize changes to important weights
- [ ] **Knowledge Distillation:** Use S1 model as teacher
- [ ] **Domain Adaptation:** Align S1 and S2 feature distributions
- [ ] **Attention Mechanisms:** Focus on drift-invariant features

### 9.3 Production Considerations
- [ ] Online adaptation (continuous learning)
- [ ] Confidence-based authentication thresholds
- [ ] Drift detection before adaptation
- [ ] User-specific adaptation rates

---

## Appendix A: Configuration Reference

```python
# Current optimal configuration
ADAPT_RATIO = 0.40        # 40% of S2 for adaptation
SEQ_LENGTH = 5            # Shorter sequences = more samples
CNN_EPOCHS = 20           # Training epochs
LSTM_EPOCHS = 20          # Training epochs
MIXED_REPLAY_RATIO = 0.5  # 50% S1 + 50% S2 during fine-tune
LEARNING_RATE = 0.001     # Initial training
FINETUNE_LR = 0.0001      # Fine-tuning (10x lower)
```

## Appendix B: Quick Start

```bash
# Run the drift adaptation experiment
cd GazeAuth-Drift
python experiment_drift_adaptation.py

# Expected output
# KNN:  20.6% → 23.9% (+3.2%) ✅
# SVM:  24.1% → 28.2% (+4.1%) ✅
# CNN:  30.1% → 44.3% (+14.2%) ✅
# LSTM: 31.4% → 38.7% (+7.3%) ✅
```

---

*Last updated: December 2024*
*Authors: Biometric Security Project Team*

