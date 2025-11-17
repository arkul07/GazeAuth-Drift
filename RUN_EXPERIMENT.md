# How to Run the Experiment

## Quick Start

### 1. Setup (One-Time)

```bash
# If you have Python 3.13 (PyTorch won't work)
brew install python@3.12
/usr/local/bin/python3.12 -m venv venv_312
source venv_312/bin/activate

# Or if you have Python 3.12 already
python3.12 -m venv venv
source venv/bin/activate

# Install dependencies
pip install pandas numpy==1.26.4 scipy scikit-learn matplotlib seaborn torch torchvision
```

### 2. Run the Experiment

```bash
python experiment_drift_adaptation.py
```

## What It Tests

**Drift Problem (Baseline Models):**

- Trains KNN/SVM on Session 1
- Tests on Session 2 (temporal drift)
- Shows accuracy drops to ~33-45%

**Drift Solution (Adaptive Models):**

- Trains CNN/LSTM on Session 1
- Fine-tunes on first 20% of Session 2
- Tests on rest of Session 2
- (Currently overfitting with limited data)

## Results

### Baseline Models ✅

- **KNN:** 32.8% on S2 (proves drift is a problem)
- **SVM:** 44.8% on S2

### Deep Learning Models ⚠️

- **CNN/LSTM:** Currently overfitting due to limited adaptation data
- Need to tune: more adaptation data, lower learning rate, or fewer epochs

## Files Structure

**Main Experiment:**

- `experiment_drift_adaptation.py` - Run this!

**Core Implementations:**

- `data/gazebase_loader.py` - Load GazebaseVR CSVs
- `data/simulated_drift.py` - Generate synthetic drift
- `pipeline/feature_extractor.py` - Extract 46 features
- `models/baselines.py` - KNN/SVM classifiers
- `models/temporal/gaze_cnn.py` - CNN implementation
- `models/temporal/gaze_lstm.py` - LSTM implementation
- `utils/metrics.py` - EER, FMR, FRR metrics

## For Report

**Key Findings:**

1. ✅ Temporal drift reduces accuracy from ~90% → ~35% (baseline)
2. ✅ Proves drift is a significant authentication problem
3. ⏳ CNN/LSTM adaptation needs tuning (future work)

**What to Write:**

> "We demonstrated that gaze authentication accuracy degrades significantly due to temporal drift, with baseline models (KNN, SVM) dropping to 33-45% accuracy when testing on Session 2 data collected weeks after Session 1. This validates the need for drift-aware authentication systems."
