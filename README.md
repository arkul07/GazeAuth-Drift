# Gaze-Only Continuous Authentication System

A research project focused on developing robust gaze-based continuous authentication systems for AR/VR applications with emphasis on handling temporal drift and long-term reliability.

## Project Overview

This project implements a comprehensive framework for gaze-based continuous authentication that includes:

- **Data Processing**: Loading and preprocessing GazebaseVR dataset
- **Feature Extraction**: Behavioral gaze feature calculation
- **Baseline Models**: Non-temporal classifiers (KNN, SVM)
- **Temporal Models**: CNN and LSTM for sequence processing
- **Drift Detection**: Statistical methods for temporal drift detection
- **Continuous Authentication**: EWMA-based decision making
- **Simulation Environment**: Longitudinal testing with drift scenarios

## Project Structure

```
gaze_auth_project/
├── data/                          # Data loading and simulation
│   ├── gazebase_loader.py        # GazebaseVR data loading functions
│   └── simulated_drift.py        # Simulated drift data generation
├── models/                        # Machine learning models
│   ├── baselines.py              # KNN/SVM baseline models
│   └── temporal/                 # Temporal sequence models
│       ├── gaze_cnn.py           # CNN for gaze sequences
│       └── gaze_lstm.py          # LSTM for gaze sequences
├── pipeline/                      # Core processing pipeline
│   ├── feature_extractor.py      # Gaze feature extraction
│   ├── decision_module.py        # Continuous authentication logic
│   └── drift_monitor.py          # Drift detection and handling
├── simulation/                    # Simulation environment
│   └── simulator.py              # Continuous authentication simulator
├── utils/                         # Utility functions
│   └── metrics.py                # Evaluation metrics (EER, FMR, FRR)
├── main.py                       # Main entry point
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Key Features

### Gaze Feature Extraction
- Fixation duration distributions
- Saccade amplitude and velocity patterns
- Scanpath entropy and complexity measures
- Velocity-based behavioral features

### Baseline Models
- K-Nearest Neighbors (KNN) classifier
- Support Vector Machine (SVM) classifier
- Equal Error Rate (EER) evaluation

### Temporal Models
- Convolutional Neural Networks (CNN) for gaze sequences
- Long Short-Term Memory (LSTM) networks
- Sequence-based authentication

### Drift Handling
- Statistical drift detection methods
- Model adaptation strategies
- Longitudinal performance monitoring

### Continuous Authentication
- Exponentially Weighted Moving Average (EWMA)
- Real-time confidence scoring
- Time-to-detection metrics

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd gaze_auth_project
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**:
   ```bash
   python main.py --help
   ```

## Usage

### Running Baseline Experiments
```bash
python main.py --experiment baseline --data_path /path/to/gaze_data
```

### Running Temporal Experiments
```bash
python main.py --experiment temporal --data_path /path/to/gaze_data
```

### Running Drift Experiments
```bash
python main.py --experiment drift --data_path /path/to/gaze_data
```

### Running Simulation Experiments
```bash
python main.py --experiment simulation --data_path /path/to/gaze_data
```

## Configuration

Create a `config.json` file to customize experiment parameters:

```json
{
  "data": {
    "window_size_sec": 5,
    "overlap_sec": 1
  },
  "models": {
    "baseline": {
      "knn": {"n_neighbors": 5},
      "svm": {"kernel": "rbf", "C": 1.0}
    },
    "temporal": {
      "cnn": {"sequence_length": 100},
      "lstm": {"hidden_dim": 128, "num_layers": 2}
    }
  },
  "drift": {
    "detection_method": "statistical",
    "adaptation_strategy": "retrain"
  }
}
```

## Research Focus

This project specifically addresses:

1. **Temporal Drift**: How gaze patterns change over time
2. **Long-term Reliability**: Maintaining authentication accuracy over extended periods
3. **Continuous Authentication**: Real-time decision making without user interruption
4. **AR/VR Applications**: Optimized for immersive environments

## Contributing

This is a research project. For questions or contributions, please contact the research team.

## License

This project is for research purposes. Please cite appropriately if used in academic work.

## Citation

```bibtex
@article{gaze_auth_2024,
  title={Gaze-Only Continuous Authentication for AR/VR with Temporal Drift Handling},
  author={Research Team},
  journal={Biometric Security with AI},
  year={2024}
}
```
