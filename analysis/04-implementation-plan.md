# 04 — IMPLEMENTATION PLAN

> Re-implementation strategy theo từng Phase, kế hoạch triển khai tuần tự, và top 10 việc cần giải quyết trước khi code.

---

## PHASE 0: Research & Setup

### Mục tiêu
Hoàn tất nghiên cứu, thu thập tài liệu, setup development environment.

### Các task chính
- [ ] Đọc paper gốc Aminov et al. (2017) [2] để hiểu dataset chi tiết
- [ ] Liên hệ authors để request dataset (nếu cần)
- [ ] Tìm hiểu thêm về ESN: đọc Jaeger (2001) [24], Lukoševičius & Jaeger (2009) [37]
- [ ] Survey ESN libraries: reservoirpy, pyESN — so sánh API, features
- [ ] Setup Python environment, install dependencies
- [ ] Setup version control (git)
- [ ] Setup experiment tracking (MLflow hoặc custom)
- [ ] Xác định và resolve tất cả ambiguities trong paper (xem 03-reproducibility-analysis.md)

### Kết quả cần đạt
- Dataset secured hoặc alternative dataset identified
- Development environment sẵn sàng
- All libraries installed và tested
- Decision log cho các ambiguities

### Điều kiện chuyển Phase
- ✅ Có dataset (hoặc plan B rõ ràng)
- ✅ Environment works
- ✅ Đã hiểu rõ ESN theory

---

## PHASE 1: Data Pipeline

### Mục tiêu
Build Module M1 (Data Loading) + M2 (Preprocessing) + M3 (Feature Extraction) + M4 (Feature Selection).

### Các task chính
- [ ] Implement data loader cho EEG dataset
- [ ] Implement bandpass filter (0.5–30 Hz)
- [ ] Implement artifact rejection (automated alternative to manual)
- [ ] Implement amplitude thresholding (±100 μV)
- [ ] Implement epoching (4-second epochs)
- [ ] Implement PSD feature extraction (FFT/Welch)
- [ ] Calculate Relative Power (RP Delta, Theta, Alpha, Beta)
- [ ] Calculate DAR, DTR ratios
- [ ] Implement feature selection
- [ ] Implement MinMax normalization [0, 1]
- [ ] Combine EEG features + categorical features
- [ ] Write unit tests cho mỗi step
- [ ] Visualize intermediate results (PSD plots, feature distributions)

### Kết quả cần đạt
- Feature matrix shape: (n_samples, n_features)
- Feature distributions match paper's Fig. 4 (approximately)
- All preprocessing steps verified visually
- Reproducible pipeline (same input → same output)

### Điều kiện chuyển Phase
- ✅ Feature matrix generated successfully
- ✅ Feature distributions make sense (no NaN, no extreme values)
- ✅ Pipeline runs end-to-end from raw data → features
- ✅ Unit tests pass

---

## PHASE 2: Baseline Models

### Mục tiêu
Implement baseline classifiers để validate data pipeline trước khi tackle ESN.

### Các task chính
- [ ] Implement train/test split (try multiple strategies: 80/20, LOOCV, 5-fold CV)
- [ ] Train baseline models: SVM, Random Forest, Logistic Regression, KNN
- [ ] Evaluate baselines: accuracy, precision, recall, F1, confusion matrix
- [ ] Compare baseline results với related works (Table 3 trong paper)
- [ ] If baselines reasonable → pipeline is correct → proceed to ESN
- [ ] If baselines terrible → debug data pipeline first

### Kết quả cần đạt
- Baseline accuracy trong range hợp lý (70–95% dựa trên related works)
- Validation rằng data pipeline tạo ra features có discriminative power
- Understanding of data characteristics (imbalance, separability)

### Điều kiện chuyển Phase
- ✅ Ít nhất 1 baseline model đạt accuracy > 70%
- ✅ No data leakage detected
- ✅ Feature importance từ baselines align roughly với paper's top features

---

## PHASE 3: Single ESN

### Mục tiêu
Implement và tune một ESN classifier đơn lẻ.

### Các task chính
- [ ] Chọn ESN library hoặc implement from scratch
- [ ] Implement single ESN với paper's hyperparameters (200 neurons, SR=0.95, etc.)
- [ ] Configure ESN cho classification task (output layer adaptation)
- [ ] Experiment với missing hyperparameters:
  - Leaking rate: sweep [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
  - Input scaling: sweep [0.01, 0.1, 0.5, 1.0]
  - Reservoir activation: tanh (default)
  - Wout solver: ridge regression with alpha=0.001
- [ ] Train single ESN, evaluate metrics
- [ ] Compare với baselines
- [ ] Visualize reservoir states để verify proper dynamics
- [ ] Hyperparameter sensitivity analysis

### Kết quả cần đạt
- Working single ESN classifier
- Performance comparable to or better than baselines
- Understanding of ESN sensitivity to hyperparameters
- Best leaking rate + input scaling identified

### Điều kiện chuyển Phase
- ✅ Single ESN accuracy > 80% (reasonable for single model)
- ✅ Hyperparameters finalized
- ✅ ESN implementation verified

---

## PHASE 4: Ensemble ESN (E-ESN)

### Mục tiêu
Build ensemble của 7 ESN với bagging.

### Các task chính
- [ ] Implement bagging mechanism:
  - Bootstrap sampling
  - 7 ESN with different random seeds
  - Prediction aggregation (averaging)
- [ ] Train E-ESN ensemble
- [ ] Evaluate: accuracy, PPV, NPV, sensitivity, specificity, F1
- [ ] Generate confusion matrix
- [ ] Compare với paper's reported metrics (Table 2)
- [ ] Ablation study: vary number of ESNs (3, 5, 7, 9)
- [ ] Ablation study: vary bootstrap ratio
- [ ] Record running time

### Kết quả cần đạt
- E-ESN accuracy target: ~96.5% (match paper)
- Confusion matrix comparable to paper's Fig. 5
- Improvement over single ESN demonstrated
- Running time recorded

### Điều kiện chuyển Phase
- ✅ E-ESN accuracy > 90% (close to paper)
- ✅ Improvement over single ESN confirmed
- ✅ Model saved and serializable

---

## PHASE 5: Explainable AI (XAI)

### Mục tiêu
Implement SHAP (global) + LIME (local) explanations cho E-ESN.

### Các task chính
- [ ] Implement SHAP explanation:
  - Setup KernelSHAP explainer with E-ESN predict function
  - Compute SHAP values cho all test instances
  - Generate summary plot (beeswarm)
  - Generate bar plot (mean absolute SHAP)
  - Verify top features match paper: DTR, Theta, RP Delta, RP Alpha, age, beta, education, gender
- [ ] Implement LIME explanation:
  - Setup LimeTabularExplainer
  - Generate explanations cho sample instances (1 stroke, 1 control)
  - Generate bar chart + feature table visualizations
- [ ] Validate explanations:
  - Do top features make clinical sense?
  - Are explanations consistent across runs?
- [ ] Save explanation results

### Kết quả cần đạt
- SHAP summary plot similar to paper's Fig. 6
- LIME explanations similar to paper's Fig. 7
- Feature importance ranking reasonably aligned with paper
- Reusable explain() functions

### Điều kiện chuyển Phase
- ✅ SHAP top features mostly match paper
- ✅ LIME produces coherent explanations
- ✅ Both visualizations generated successfully

---

## PHASE 6: Comprehensive Evaluation

### Mục tiêu
Full evaluation suite — compare all models, validate results.

### Các task chính
- [ ] Compile results: baselines vs single ESN vs E-ESN
- [ ] Create comparison table matching paper's Table 3
- [ ] Statistical significance testing (if applicable with small sample)
- [ ] Cross-validation analysis for robustness
- [ ] Document any discrepancies vs paper's reported results
- [ ] Generate all publication-quality figures

### Kết quả cần đạt
- Comprehensive results table
- All figures comparable to paper
- Clear documentation of differences vs paper

### Điều kiện chuyển Phase
- ✅ All metrics documented
- ✅ Discrepancies explained
- ✅ Results are reproducible (multiple runs)

---

## PHASE 7: Decision Support System (DSS)

### Mục tiêu
Build prototype Medical DSS interface.

### Các task chính
- [ ] Design recommendation engine (rule-based):
  - Map stroke probability → recommended tests
  - Define probability thresholds
  - Generate recommendation text
- [ ] Build DSS prototype:
  - Patient info input form
  - EEG data upload/input
  - Prediction display (probability + class)
  - SHAP visualization panel (button click)
  - LIME visualization panel (button click)
  - Recommendations display
  - Confirm/Cancel actions
- [ ] Connect UI with trained model backend
- [ ] Test end-to-end flow

### Kết quả cần đạt
- Working prototype matching paper's Fig. 8 layout
- Full flow: input → prediction → explanation → recommendation
- Screenshots comparable to paper's UI

### Điều kiện chuyển Phase
- ✅ DSS prototype functional end-to-end
- ✅ All UI components working
- ✅ Model predictions match direct API calls

---

## PHASE 8: Testing & Documentation

### Mục tiêu
Comprehensive testing, documentation, và packaging.

### Các task chính
- [ ] Write comprehensive unit tests
- [ ] Integration tests (end-to-end pipeline)
- [ ] Documentation:
  - README.md
  - Setup guide
  - Architecture documentation
  - API documentation
  - User guide cho DSS
- [ ] Code cleanup và refactoring
- [ ] Create reproducibility checklist
- [ ] Write experiment report

### Kết quả cần đạt
- Test coverage > 80%
- Complete documentation
- Clean, maintainable codebase
- Reproducibility checklist

### Điều kiện chuyển Phase
- ✅ All tests pass
- ✅ Documentation complete
- ✅ New developer can setup and run project from README

---

## MASTER IMPLEMENTATION PLAN — TUẦN TỰ

### Thứ tự thực hiện

```
Phase 0 (Research & Setup)         ← START HERE
    │
    ▼
Phase 1 (Data Pipeline)           ← Foundation — CRITICAL
    │
    ▼
Phase 2 (Baseline Models)         ← Validation checkpoint
    │
    ▼
Phase 3 (Single ESN)              ← Core ML — HARDEST
    │
    ▼
Phase 4 (Ensemble ESN)            ← Build on Phase 3
    │
    ├─────────────────────┐
    ▼                     ▼
Phase 5 (XAI)       Phase 6 (Evaluation)   ← CAN DO IN PARALLEL
    │                     │
    └─────────┬───────────┘
              ▼
Phase 7 (DSS)                     ← Integration
    │
    ▼
Phase 8 (Testing & Docs)          ← Finalization
```

### Module dependencies cụ thể

```
M1 (Data Loading)         → standalone, phải làm đầu tiên
M2 (Preprocessing)        → cần M1
M3 (Feature Extraction)   → cần M2
M4 (Feature Selection)    → cần M3 + M1 (categorical data)
M5 (Single ESN)           → cần M4
M6 (Ensemble ESN)         → cần M5
M7 (Evaluation)           → cần M6 + M4 (test data)
M8 (SHAP)                 → cần M6 + M4 (có thể parallel với M9)
M9 (LIME)                 → cần M6 + M4 (có thể parallel với M8)
M10 (Recommendation)      → cần M6 (có thể parallel với M8/M9)
M11 (DSS UI)              → cần M6 + M8 + M9 + M10 (phải làm cuối)
M12 (Pipeline)             → cross-cutting, xây dần
```

### Phần có thể làm song song

- M8 (SHAP) và M9 (LIME) → cùng cần trained E-ESN, không phụ thuộc nhau
- M7 (Evaluation) và M8/M9 → cùng level
- M10 (Recommendation) có thể bắt đầu thiết kế trong khi M5/M6 đang development
- M12 (Pipeline) xây dần song song với tất cả phases

### Quyết định PHẢI chốt trước khi code

1. **Dataset**: Có lấy được data gốc không? Nếu không → plan B là gì?
2. **ESN Library**: Dùng reservoirpy hay pyESN hay implement from scratch?
3. **Train/test strategy**: LOOCV? k-fold? Hold-out?
4. **Missing hyperparameters**: Leaking rate? Input scaling? → Phải quyết định hoặc grid search
5. **Feature set cuối cùng**: 4 features (EEG only) + demographics separately? Hay 8 features gộp chung?
6. **Artifact rejection**: Manual → automated (ICA/autoreject)?
7. **UI technology**: Streamlit (prototype) hay Flask/React (production)?
8. **Evaluation strategy**: Reproduce exact numbers hay reproduce methodology?
9. **Frequency band definitions**: Standard boundaries hay tùy chỉnh?
10. **Confusion matrix discrepancy**: Dùng per-epoch hay per-participant prediction?

---

## TOP 10 VIỆC CẦN GIẢI QUYẾT TRƯỚC KHI BẮT ĐẦU CODE

### 1. 🔴 DATASET ACCESS
**Vấn đề**: Paper dùng dataset từ Aminov et al. (2017). Data "available on request". Nếu không lấy được → không reproduce được.
**Action**: Đọc paper gốc [2], contact authors, search PLoS One supplementary materials. Chuẩn bị plan B (alternative dataset).

### 2. 🔴 ĐỌC PAPER GỐC AMINOV ET AL. (2017)
**Vấn đề**: Paper hiện tại lấy dataset từ reference [2] nhưng không mô tả đầy đủ. Cần đọc paper gốc để biết: sampling rate, recording duration, electrode details, data format, số stroke vs control.
**Action**: Tìm và đọc "Acute single channel EEG predictors of cognitive function after stroke" — PLoS One 2017.

### 3. 🟡 CHỌN ESN LIBRARY
**Vấn đề**: Cần quyết định dùng thư viện nào hay implement từ đầu. Mỗi thư viện có default parameters khác nhau.
**Action**: So sánh reservoirpy vs pyESN vs custom implementation. Test cơ bản với toy data.

### 4. 🟡 XỬ LÝ HYPERPARAMETERS THIẾU
**Vấn đề**: Paper thiếu leaking rate, input scaling, activation function, Wout solver. Đây là các tham số ảnh hưởng lớn đến ESN performance.
**Action**: Lên kế hoạch grid search cho leaking rate [0.1, 0.3, 0.5, 0.7, 0.9, 1.0] và input scaling [0.01, 0.1, 0.5, 1.0]. Document decisions.

### 5. 🟡 GIẢI QUYẾT MÂU THUẪN FEATURES
**Vấn đề**: Paper nói "4 optimal features" nhưng SHAP shows 8 features. Cần xác định chính xác feature set.
**Action**: Assumption → dùng tất cả EEG features + categorical. Document clearly.

### 6. 🟡 TRAIN/TEST STRATEGY
**Vấn đề**: Paper không nói cách split. Với 24 mẫu, strategy choice ảnh hưởng rất lớn.
**Action**: Implement multiple strategies (LOOCV, 5-fold stratified CV, 80/20 hold-out). Report tất cả.

### 7. 🟢 SETUP DEVELOPMENT ENVIRONMENT
**Vấn đề**: Cần unified environment với tất cả dependencies.
**Action**: Tạo requirements.txt / environment.yml. Install: numpy, scipy, mne, scikit-learn, reservoirpy, shap, lime, matplotlib, plotly.

### 8. 🟢 ARTIFACT REJECTION STRATEGY
**Vấn đề**: Paper dùng "manual inspection" — không reproducible. Cần automated alternative.
**Action**: Dùng MNE autoreject hoặc amplitude-based rejection. Document deviation from paper.

### 9. 🟢 CONFUSION MATRIX DISCREPANCY
**Vấn đề**: 24 participants ≠ ~39 predictions trong confusion matrix. Cần hiểu paper dùng per-epoch hay per-participant.
**Action**: Read paper gốc [2] cho epoch counts. Thử cả hai approaches.

### 10. 🟢 EXPERIMENT TRACKING SETUP
**Vấn đề**: Cần track multiple experiments (hyperparameter tuning, ablation studies).
**Action**: Setup MLflow hoặc simple CSV logger. Define metrics to track.

---

## LEGEND
- 🔴 **BLOCKER** — phải giải quyết trước khi bắt đầu code
- 🟡 **HIGH PRIORITY** — cần giải quyết sớm, ảnh hưởng design
- 🟢 **IMPORTANT** — cần giải quyết nhưng có thể song song với development
