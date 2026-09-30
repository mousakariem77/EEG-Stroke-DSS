# 05 — PROJECT ARCHITECTURE

> Cấu trúc folder project cho việc re-implement paper và mở rộng thành Medical Decision Support System.

---

## PROJECT STRUCTURE

```
explainable-eeg-stroke-dss/
│
├── README.md                          # Project overview, setup guide, quickstart
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment (alternative)
├── setup.py                           # Package setup (optional)
├── .gitignore                         # Git ignore rules
├── Makefile                           # Common commands (train, test, evaluate)
│
├── configs/                           # Configuration files
│   ├── default.yaml                   # Default hyperparameters & settings
│   ├── preprocessing.yaml             # EEG preprocessing config
│   ├── feature_extraction.yaml        # Feature extraction parameters
│   ├── esn.yaml                       # ESN hyperparameters
│   ├── ensemble.yaml                  # Ensemble configuration
│   ├── xai.yaml                       # XAI (SHAP/LIME) settings
│   └── dss.yaml                       # DSS / recommendation settings
│
├── data/                              # Data directory (NOT in git)
│   ├── raw/                           # Raw EEG files as received
│   │   └── .gitkeep
│   ├── processed/                     # Preprocessed EEG data
│   │   └── .gitkeep
│   ├── features/                      # Extracted feature matrices
│   │   └── .gitkeep
│   └── metadata/                      # Patient demographics, labels
│       └── .gitkeep
│
├── src/                               # Source code — main package
│   ├── __init__.py
│   │
│   ├── data/                          # Module M1: Data loading & management
│   │   ├── __init__.py
│   │   ├── loader.py                  # EEG data loading (multiple formats)
│   │   ├── dataset.py                 # Dataset class (PyTorch-style or custom)
│   │   ├── validator.py               # Data validation & sanity checks
│   │   └── splitter.py                # Train/test splitting strategies
│   │
│   ├── preprocessing/                 # Module M2: EEG preprocessing
│   │   ├── __init__.py
│   │   ├── filters.py                 # Band-pass filtering (0.5–30 Hz)
│   │   ├── artifacts.py               # Artifact detection & removal
│   │   ├── epochs.py                  # Epoching (4-second segments)
│   │   └── pipeline.py                # Preprocessing pipeline orchestrator
│   │
│   ├── features/                      # Module M3 & M4: Feature extraction & selection
│   │   ├── __init__.py
│   │   ├── psd.py                     # Power Spectral Density computation
│   │   ├── frequency_bands.py         # Frequency band definitions & RP calculation
│   │   ├── ratios.py                  # DAR, DTR ratio computation
│   │   ├── selection.py               # Feature selection algorithms
│   │   ├── normalization.py           # Feature normalization (MinMax)
│   │   └── pipeline.py               # Feature extraction pipeline
│   │
│   ├── models/                        # Module M5 & M6: ESN & Ensemble
│   │   ├── __init__.py
│   │   ├── esn.py                     # Echo State Network implementation
│   │   ├── reservoir.py               # Reservoir initialization & dynamics
│   │   ├── readout.py                 # Output layer (ridge regression)
│   │   ├── ensemble.py                # Ensemble ESN (E-ESN) with bagging
│   │   ├── baselines.py               # Baseline classifiers (SVM, RF, LR, KNN)
│   │   └── utils.py                   # Model utilities (save/load, etc.)
│   │
│   ├── evaluation/                    # Module M7: Evaluation
│   │   ├── __init__.py
│   │   ├── metrics.py                 # Classification metrics computation
│   │   ├── visualization.py           # Confusion matrix, ROC curves, etc.
│   │   └── comparison.py              # Model comparison tables
│   │
│   ├── explainability/                # Module M8 & M9: XAI
│   │   ├── __init__.py
│   │   ├── shap_explainer.py          # SHAP global explanation
│   │   ├── lime_explainer.py          # LIME local explanation
│   │   ├── visualizations.py          # XAI-specific visualizations
│   │   └── utils.py                   # XAI utilities
│   │
│   ├── dss/                           # Module M10 & M11: Decision Support System
│   │   ├── __init__.py
│   │   ├── recommendation.py          # Recommendation engine (rule-based)
│   │   ├── patient.py                 # Patient data model
│   │   └── engine.py                  # DSS orchestration engine
│   │
│   ├── pipeline/                      # Module M12: Pipeline orchestration
│   │   ├── __init__.py
│   │   ├── train.py                   # Full training pipeline
│   │   ├── predict.py                 # Inference pipeline
│   │   └── experiment.py              # Experiment runner & tracking
│   │
│   └── utils/                         # Shared utilities
│       ├── __init__.py
│       ├── config.py                  # Configuration loader (YAML)
│       ├── logging.py                 # Logging setup
│       ├── seed.py                    # Random seed management
│       └── io.py                      # File I/O helpers
│
├── app/                               # DSS Web Application
│   ├── __init__.py
│   ├── main.py                        # App entry point (Streamlit / Flask)
│   ├── pages/                         # UI pages/views
│   │   ├── patient_input.py           # Patient data input form
│   │   ├── prediction.py              # Prediction results display
│   │   ├── explanation.py             # SHAP/LIME visualization page
│   │   └── recommendation.py          # Recommendation display
│   ├── components/                    # Reusable UI components
│   │   ├── header.py
│   │   ├── sidebar.py
│   │   └── charts.py
│   ├── static/                        # Static assets (CSS, JS, images)
│   │   └── style.css
│   └── templates/                     # HTML templates (if Flask)
│       └── base.html
│
├── notebooks/                         # Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb      # EDA on raw EEG data
│   ├── 02_preprocessing_demo.ipynb    # Preprocessing pipeline demo
│   ├── 03_feature_analysis.ipynb      # Feature extraction & visualization
│   ├── 04_baseline_models.ipynb       # Baseline classifiers
│   ├── 05_esn_experiments.ipynb       # ESN hyperparameter experiments
│   ├── 06_ensemble_esn.ipynb          # E-ESN training & evaluation
│   ├── 07_shap_analysis.ipynb         # SHAP global explanation
│   ├── 08_lime_analysis.ipynb         # LIME local explanation
│   └── 09_full_pipeline.ipynb         # End-to-end pipeline demo
│
├── tests/                             # Test suite
│   ├── __init__.py
│   ├── conftest.py                    # Pytest fixtures
│   ├── test_data/                     # Test data files
│   │   └── sample_eeg.csv
│   ├── test_preprocessing.py
│   ├── test_features.py
│   ├── test_esn.py
│   ├── test_ensemble.py
│   ├── test_shap.py
│   ├── test_lime.py
│   ├── test_recommendation.py
│   └── test_pipeline.py
│
├── experiments/                       # Experiment results (NOT in git)
│   ├── .gitkeep
│   ├── logs/                          # Training logs
│   ├── models/                        # Saved trained models
│   ├── results/                       # Evaluation results
│   └── figures/                       # Generated figures
│
├── document/                          # Paper & references
│   └── [14]-2024-15-Enhancing accuracy and interpretability in EEG-based medical decision for stroke prediction.pdf
│
├── analysis/                          # Project analysis documents
│   ├── 01-paper-analysis.md
│   ├── 02-module-analysis.md
│   ├── 03-reproducibility-analysis.md
│   ├── 04-implementation-plan.md
│   └── 05-project-architecture.md
│
├── scripts/                           # Standalone scripts
│   ├── download_data.py               # Script to download/prepare dataset
│   ├── train_model.py                 # CLI for model training
│   ├── evaluate_model.py              # CLI for model evaluation
│   ├── run_xai.py                     # CLI for XAI analysis
│   └── run_dss.py                     # CLI to launch DSS app
│
└── docs/                              # Extended documentation
    ├── setup.md                       # Detailed setup instructions
    ├── architecture.md                # System architecture documentation
    ├── api.md                         # API reference
    ├── experiments.md                 # Experiment tracking guide
    └── clinical_notes.md              # Clinical context & domain notes
```

---

## DESIGN PRINCIPLES

### Separation of Concerns
- **src/**: Core logic — no UI, no scripts, no notebooks
- **app/**: UI-only — uses src/ as backend
- **scripts/**: CLI entry points — thin wrappers around src/
- **notebooks/**: Exploration — imports from src/

### Configuration-Driven
- Tất cả hyperparameters trong `configs/` (YAML)
- Không hard-code values trong source code
- Dễ dàng experiment với different settings

### Reproducibility-First
- Random seed management centralized (`src/utils/seed.py`)
- Experiment tracking (`src/pipeline/experiment.py`)
- All results saved to `experiments/`

### Modularity
- Mỗi module independent (có thể test riêng)
- Clear interfaces giữa modules
- Pipeline orchestrator connects modules

### Extensibility
- Easy to add new models (`src/models/`)
- Easy to add new XAI methods (`src/explainability/`)
- Easy to add new features (`src/features/`)
- DSS architecture supports future extensions (new diseases, new modalities)

---

## KEY DEPENDENCIES (requirements.txt)

```
# Core
numpy>=1.24.0
scipy>=1.10.0
pandas>=2.0.0
scikit-learn>=1.3.0

# EEG Processing
mne>=1.5.0

# ESN / Reservoir Computing
reservoirpy>=0.3.0

# Explainability
shap>=0.42.0
lime>=0.2.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0

# Web App (choose one)
streamlit>=1.28.0
# flask>=3.0.0
# fastapi>=0.104.0

# Configuration
pyyaml>=6.0

# Experiment Tracking
# mlflow>=2.8.0  # optional

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0

# Utilities
tqdm>=4.66.0
joblib>=1.3.0
```

---

## DATA FLOW DIAGRAM

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐
│  Raw EEG    │────▶│ Preprocessing│────▶│   Feature     │
│  (data/raw/)│     │  (filters,   │     │  Extraction   │
│             │     │  artifacts,  │     │  (PSD, RP,    │
│             │     │  epoching)   │     │   DAR, DTR)   │
└─────────────┘     └──────────────┘     └───────┬───────┘
                                                  │
                                                  ▼
┌─────────────┐     ┌──────────────┐     ┌───────────────┐
│  Patient    │────▶│  Feature     │◀────│  Feature      │
│  Metadata   │     │  Matrix      │     │  Selection    │
│  (data/     │     │  (data/      │     │  + Normalize  │
│  metadata/) │     │  features/)  │     │               │
└─────────────┘     └──────┬───────┘     └───────────────┘
                           │
                     ┌─────┴──────┐
                     ▼            ▼
              ┌───────────┐ ┌───────────┐
              │  Train    │ │   Test    │
              │  Set      │ │   Set     │
              └─────┬─────┘ └─────┬─────┘
                    │             │
                    ▼             │
              ┌───────────┐      │
              │  E-ESN    │      │
              │  Training │      │
              │  (7 ESNs) │      │
              └─────┬─────┘      │
                    │             │
                    ▼             ▼
              ┌───────────────────────┐
              │  E-ESN Prediction     │
              │  (experiments/models/)│
              └───────┬───────────────┘
                      │
           ┌──────────┼──────────┐
           ▼          ▼          ▼
     ┌──────────┐ ┌───────┐ ┌──────────┐
     │  SHAP    │ │ LIME  │ │Evaluation│
     │  Global  │ │ Local │ │ Metrics  │
     └────┬─────┘ └───┬───┘ └────┬─────┘
          │           │          │
          └─────┬─────┘          │
                ▼                ▼
          ┌───────────┐    ┌──────────┐
          │Recommend  │    │ Results  │
          │Engine     │    │ Report   │
          └─────┬─────┘    └──────────┘
                │
                ▼
          ┌───────────┐
          │  DSS UI   │
          │  (app/)   │
          └───────────┘
```

---

## CONFIG STRUCTURE (configs/default.yaml)

```yaml
# Default configuration for Explainable EEG Stroke DSS

seed: 42

data:
  raw_path: "data/raw/"
  processed_path: "data/processed/"
  features_path: "data/features/"
  metadata_path: "data/metadata/"

preprocessing:
  filter:
    low_freq: 0.5        # Hz
    high_freq: 30.0       # Hz
    filter_type: "fir"    # fir or iir
  artifact:
    amplitude_threshold: 100  # μV
    method: "threshold"       # threshold, ica, autoreject
  epoch:
    duration: 4.0         # seconds
    overlap: 0.0          # seconds (paper doesn't specify)

features:
  psd:
    method: "welch"
    window: "hann"
    nperseg: null         # auto (epoch_length * sfreq)
  frequency_bands:
    delta: [0.5, 4.0]
    theta: [4.0, 8.0]
    alpha: [8.0, 13.0]
    beta: [13.0, 30.0]
  normalization: "minmax" # minmax [0, 1]

model:
  esn:
    n_neurons: 200
    spectral_radius: 0.95
    sparsity: 0.0         # 0 = fully connected
    noise: 0.001          # regularization
    leaking_rate: 1.0     # NEEDS TUNING — not specified in paper
    input_scaling: 1.0    # NEEDS TUNING — not specified in paper
    activation: "tanh"
    solver: "ridge"       # ridge regression for Wout
  ensemble:
    n_estimators: 7
    method: "bagging"
    aggregation: "soft"   # soft voting (average probabilities)

evaluation:
  split:
    method: "stratified_kfold"  # or "loocv", "holdout"
    n_splits: 5
    test_size: 0.2        # for holdout
  metrics:
    - accuracy
    - precision
    - recall
    - f1
    - specificity
    - npv

xai:
  shap:
    explainer: "kernel"   # kernel (model-agnostic)
    n_samples: 100        # background samples
  lime:
    n_features: 8
    n_samples: 5000       # perturbation samples

dss:
  recommendations:
    high_risk_threshold: 0.7
    tests:
      - "MRI"
      - "CBC"
      - "BMP"
      - "Coagulation Profile"
      - "Lipid Profile"
      - "D-Dimer"

experiment:
  output_dir: "experiments/"
  save_model: true
  save_predictions: true
  save_explanations: true
```

---

## NOTES ON SCALABILITY

### Mở rộng sang neurological disorders khác
- Thêm dataset mới vào `data/raw/`
- Thêm config mới (ví dụ: `configs/epilepsy.yaml`)
- Feature extraction module có thể reuse (PSD approach general cho EEG)
- Model architecture có thể reuse (ESN general cho time series)
- XAI module hoàn toàn reusable
- DSS cần extend recommendation engine cho từng bệnh

### Mở rộng sang multi-channel EEG
- Modify `src/data/loader.py` cho multi-channel formats
- Extend `src/features/` cho spatial features (coherence, connectivity)
- ESN input dimension thay đổi → reservoir có thể lớn hơn
- SHAP/LIME vẫn hoạt động (model-agnostic)

### Production deployment
- Containerize với Docker
- API-first design (FastAPI)
- Model serving (TorchServe hoặc custom)
- Database cho patient records
- Authentication/Authorization
- HIPAA compliance considerations
