# Research Findings: Drift-Aware Gaze Authentication

## Executive Summary

This project investigates **temporal drift** in gaze-based biometric authentication for AR/VR systems. We developed a drift-aware authentication pipeline and evaluated whether synthetic drift can replace real longitudinal data for testing.

---

## Key Finding #1: Drift-Aware Adaptation Works

**Problem:** Models trained on Session 1 data perform poorly on Session 2 data due to behavioral drift over time.

**Solution:** Fine-tuning with mixed replay (combining S1 + S2 data) enables models to adapt.

### Results (20 Subjects)

| Model | Static (No Adapt) | Adapted | Improvement |
|-------|-------------------|---------|-------------|
| KNN   | 20.6%             | 23.9%   | **+3.2%** ✅ |
| SVM   | 24.1%             | 28.2%   | **+4.1%** ✅ |
| CNN   | 30.1%             | 44.3%   | **+14.2%** ✅ |
| LSTM  | 31.4%             | 38.7%   | **+7.3%** ✅ |

**Conclusion:** All 4 models improved with drift-aware adaptation. CNN showed the best improvement (+14.2%).

---

## Key Finding #2: Synthetic Drift Cannot Replace Real Drift

We tested **13 synthetic drift strategies** to see if they could replicate real temporal drift:

### Basic Strategies (from calibrated_synthetic_drift.py)
- Calibrated (matches S1→S2 statistics)
- Light (50% drift)
- Heavy (150% drift)
- Gaussian noise only
- Mean shift only

### Advanced Strategies (from advanced_synthetic_drift.py)
- Feature-specific drift
- User-specific drift
- Distribution matching
- Temporal decay
- Adversarial perturbation
- Mixup interpolation (50% and 80%)

### Results

| Strategy | Adapted Accuracy | Gap from Real (23.9%) |
|----------|------------------|----------------------|
| mixup_80 | 37.2% | 13.3% ⚠️ (best) |
| mixup_50 | 39.6% | 15.7% ❌ |
| distribution_match | 47.5% | 23.7% ❌ |
| All others | 45-55% | 20-30% ❌ |

**Conclusion:** The only way to get close to real drift is to literally use real S2 data (mixup). Pure synthetic transformations fail to capture the complexity of real behavioral drift.

---

## Implications

1. **For Practitioners:** Don't rely solely on synthetic drift for testing biometric systems. Real longitudinal data is essential.

2. **For Researchers:** Temporal drift in gaze biometrics has characteristics that simple statistical models cannot capture. This suggests drift involves complex behavioral adaptations, not just noise.

3. **For System Design:** Drift-aware systems should be designed with real calibration data collection in mind, not synthetic augmentation.

---

## Technical Implementation

### Drift-Aware Pipeline Configuration
```python
ADAPT_RATIO = 0.40        # 40% of S2 for adaptation
SEQ_LENGTH = 5            # Shorter sequences for more samples
MIXED_REPLAY_RATIO = 0.5  # 50% S1 + 50% S2 during fine-tuning
FINETUNE_LR = 0.0001      # 10x lower learning rate
```

### Key Design Decisions
1. **Per-user temporal split:** Each user contributes to both adaptation and test sets
2. **Mixed replay:** Prevents catastrophic forgetting during fine-tuning
3. **Lower learning rate:** Gentle adaptation preserves S1 knowledge

---

## Files Structure

```
GazeAuth-Drift/
├── experiment_drift_adaptation.py    # Main drift-aware experiment
├── experiment_real_vs_synthetic.py   # Real vs synthetic comparison
├── experiment_advanced_synthetic.py  # Advanced synthetic strategies
├── data/
│   ├── gazebase_loader.py           # GazebaseVR data loading
│   ├── calibrated_synthetic_drift.py # Basic synthetic drift
│   └── advanced_synthetic_drift.py   # Advanced synthetic drift
├── models/
│   ├── baselines.py                  # KNN/SVM classifiers
│   └── temporal/
│       ├── gaze_cnn.py               # CNN with fine-tuning
│       └── gaze_lstm.py              # LSTM with fine-tuning
├── DRIFT_AWARE_PIPELINE_DOCUMENTATION.md
└── RESEARCH_FINDINGS.md              # This file
```

---

## Citation

If you use this work, please cite:

```bibtex
@misc{gazeauth_drift_2024,
  title={Drift-Aware Gaze Authentication: Real vs Synthetic Drift Analysis},
  author={Biometric Security Research Team},
  year={2024},
  howpublished={GitHub: arkul07/GazeAuth-Drift}
}
```

---

*Last updated: December 2024*

