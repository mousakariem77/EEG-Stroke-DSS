# Explainable EEG Stroke Decision Support System

Re-implementation of the paper:
> **"Enhancing accuracy and interpretability in EEG-based medical decision making using an explainable ensemble learning framework application for stroke prediction"**
> Bouazizi & Ltifi, Decision Support Systems 178 (2024)

## Overview

Multi-level framework for EEG-based stroke prediction combining:
- **Ensemble Echo State Networks (E-ESN)** for classification
- **SHAP + LIME** for model explainability
- **Medical DSS** for physician decision support

## Quick Start

### 1. Setup environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Download dataset
Download **EPoC Data 14.02.17.xlsx** from:
https://datadryad.org/dataset/doi:10.5061/dryad.h6986

Place the file in `data/raw/`

### 3. Run pipeline
```bash
# Coming soon
python scripts/train_model.py
```

## Dataset

- **Source**: Aminov et al. (2017) — "Acute single channel EEG predictors of cognitive function after stroke" (PLoS One)
- **Repository**: [Dryad — doi:10.5061/dryad.h6986](https://doi.org/10.5061/dryad.h6986)
- **License**: CC0 (Public Domain)
- **Details**: 24 participants, single-channel EEG (FP1), resting state, within 72h of first-ever stroke
- **Device**: Single-channel wireless EEG (NeuroSky MindWave or similar), ~512 Hz sampling
- **Features**: Relative Power (Delta, Theta, Alpha, Beta), DAR, DTR + demographics

## Project Structure

```
├── configs/          # Hyperparameters & settings (YAML)
├── data/             # Dataset (not in git)
├── src/              # Source code
│   ├── data/         # Data loading
│   ├── preprocessing/# EEG preprocessing
│   ├── features/     # Feature extraction & selection
│   ├── models/       # ESN & Ensemble
│   ├── evaluation/   # Metrics & comparison
│   ├── explainability/ # SHAP & LIME
│   ├── dss/          # Decision Support System logic
│   ├── pipeline/     # Pipeline orchestration
│   └── utils/        # Shared utilities
├── app/              # DSS Web Application
├── notebooks/        # Jupyter exploration
├── tests/            # Test suite
├── experiments/      # Results & saved models
├── analysis/         # Paper analysis & planning docs
├── scripts/          # CLI entry points
└── docs/             # Documentation
```

## Analysis Documents

See `analysis/` for detailed paper analysis:
- `01-paper-analysis.md` — Paper overview, pipeline, model, XAI, DSS
- `02-module-analysis.md` — 12 modules decomposition
- `03-reproducibility-analysis.md` — Dataset gaps, contradictions, risks
- `04-implementation-plan.md` — 9 phases + top 10 decisions
- `05-project-architecture.md` — Full project structure & config

## References

1. Bouazizi, S., & Ltifi, H. (2024). Enhancing accuracy and interpretability in EEG-based medical decision making. *Decision Support Systems*, 178, 114126.
2. Aminov, A., et al. (2017). Acute single channel EEG predictors of cognitive function after stroke. *PLoS One*, 12, e0185841.
