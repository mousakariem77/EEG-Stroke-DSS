# 03 — REPRODUCIBILITY ANALYSIS

> Phân tích chi tiết dataset, khả năng reproduce, và các thông tin thiếu hụt cần thiết để build lại hệ thống.

---

## 1. DATASET ANALYSIS

### 1.1 Dataset là gì?

> **FACT FROM PAPER**: Dataset từ nghiên cứu của Aminov et al. (2017):
> - **Reference**: [2] A. Aminov, J.M. Rogers, S.J. Johnstone, S. Middleton, P.H. Wilson, "Acute single channel EEG predictors of cognitive function after stroke", PLoS One, 2017.
> - Paper hiện tại KHÔNG tạo dataset mới — sử dụng lại dataset có sẵn.

### 1.2 Bao nhiêu người / samples?

> **FACT FROM PAPER**: 
- **24 participants** tổng cộng
- Bao gồm cả stroke patients và control subjects
- Paper KHÔNG nói rõ bao nhiêu stroke vs bao nhiêu control

> **INFERENCE**: Từ confusion matrix (Fig. 5) và metrics, có thể suy ngược:
- Sensitivity = 96.42% = TP/(TP+FN) → nếu TP=27, FN=1 → 28 positive predictions
- Specificity = 81.81% = TN/(TN+FP) → nếu TN=9, FP=2 → 11 negative predictions
- Nhưng tổng 28+11 = 39 ≠ 24 participants

> **QUAN TRỌNG**: Số trong confusion matrix (~39 samples) KHÔNG khớp với 24 participants. Có thể paper dùng:
> - Multiple epochs per participant → mỗi epoch là 1 sample
> - Cross-validation tạo multiple predictions
> - Hoặc có sự mâu thuẫn trong paper

### 1.3 EEG được thu thập như thế nào?

> **FACT FROM PAPER**:
- **Thiết bị**: Single pre-frontal electrode (1 kênh duy nhất)
- **Thời điểm**: Within 72 hours of stroke
- **Loại**: Resting state EEG
- **Đánh giá cognitive**: MoCA test tại ~90 ngày sau đột quỵ
- **Reference study** [2] tìm thấy delta/theta ratio và admission stroke severity dự đoán tốt nhất MoCA scores

> **PAPER KHÔNG NÓI**:
- Sampling rate
- Recording duration
- Electrode placement cụ thể (chỉ nói "pre-frontal")
- Reference electrode
- Equipment/device brand
- Recording protocol cụ thể

### 1.4 Features được sử dụng

> **FACT FROM PAPER**: Hai nhóm features:

**EEG-derived features (Module M3)**:
- RP Delta (Relative Power — Delta band)
- RP Theta (Relative Power — Theta band)
- RP Alpha (Relative Power — Alpha band)
- RP Beta (Relative Power — Beta band)
- DAR (Delta/Alpha Ratio)
- DTR (Delta/Theta Ratio)

**Categorical/demographic features**:
- Age
- Gender
- Years of education

**Tổng features theo SHAP analysis**: 8 features (DTR, Theta, RP Delta, RP Alpha, age, beta, years of education, gender)

> **MÂU THUẪN**: Paper nói "4 optimal features" ở Section 4.3.2 nhưng SHAP shows 8 features. Xem phân tích chi tiết ở mục 4.

### 1.5 Cách train/test

> **PAPER KHÔNG NÓI RÕ**:
- Train/test split ratio (70/30? 80/20?)
- Có dùng cross-validation không?
- Stratified split hay random?
- Random seed

> **FACT FROM PAPER**: 
- Confusion matrix visible trong Fig. 5
- Paper chỉ nói "the trained Ensemble ESN must then be evaluated on a hold-out test set"

---

## 2. ĐÁNH GIÁ KHẢ NĂNG REPRODUCE

### 2.1 Thông tin ĐỦ để reproduce

- ✅ Framework architecture tổng thể (4 modules)
- ✅ Preprocessing steps: filter range, threshold, epoch length
- ✅ Feature types: RP bands, ratios, demographics
- ✅ ESN hyperparameters: neurons, spectral radius, sparsity, noise (Table 1)
- ✅ Ensemble: 7 ESNs, bagging, averaging
- ✅ XAI: SHAP (global) + LIME (local)
- ✅ Target metrics: Accuracy 96.5%, F1 94.73%, etc.
- ✅ Top features ranked by SHAP importance

### 2.2 Thông tin THIẾU (critical gaps)

#### Về Dataset
- ❌ Link download dataset
- ❌ Sampling rate
- ❌ Recording duration
- ❌ Số stroke vs control
- ❌ File format
- ❌ Electrode placement details

#### Về Preprocessing
- ❌ Filter type (FIR/IIR)
- ❌ Filter order
- ❌ Epoch overlap strategy
- ❌ Manual artifact rejection criteria

#### Về Feature Extraction
- ❌ FFT window function
- ❌ Welch method parameters (segment length, overlap)
- ❌ Exact frequency band boundaries
- ❌ Cách aggregate features từ multiple epochs per subject

#### Về Model
- ❌ **Leaking rate** (rất quan trọng cho ESN — ảnh hưởng mạnh đến performance)
- ❌ **Input scaling** (ảnh hưởng đến non-linearity)
- ❌ Activation function cụ thể
- ❌ Wout solving method (ridge regression? pseudo-inverse? regularization parameter?)
- ❌ Có dùng Wback (feedback) không?
- ❌ Warmup/washout period
- ❌ Train/test split strategy

#### Về Ensemble
- ❌ Bootstrap sample size
- ❌ Soft voting vs hard voting

#### Về XAI
- ❌ SHAP explainer type (KernelSHAP assumed)
- ❌ Number of perturbation samples
- ❌ Background dataset for SHAP
- ❌ LIME configuration

#### Về DSS
- ❌ Recommendation logic
- ❌ Technology stack
- ❌ UI interaction flow details

---

## 3. SOURCE DATASET — AMINOV ET AL. (2017)

### 3.1 Tìm dataset gốc

> **INFERENCE/RECOMMENDATION**: Cần truy cập paper gốc [2] để tìm dataset details:
- Paper: "Acute single channel EEG predictors of cognitive function after stroke" — PLoS One, 2017
- DOI: 10.1371/journal.pone.0185841
- PLoS One thường yêu cầu authors share data → có thể available hoặc "on request"

### 3.2 Thông tin cần tìm từ paper gốc

1. Sampling rate
2. Recording duration
3. Electrode type & placement
4. Number of stroke vs control subjects
5. Data format
6. Any supplementary data files
7. EEG device used
8. Inclusion/exclusion criteria

### 3.3 Contingency plan nếu không lấy được data

> **INFERENCE/RECOMMENDATION**:

**Option A**: Contact authors
- Samar Bouazizi: samar.bouazizi@enis.tn
- Hela Ltifi: hela.ltifi@ieee.org
- Paper nói "Data will be made available on request"

**Option B**: Tìm dataset tương tự
- Temple University Hospital EEG Corpus
- BCI Competition datasets
- PhysioNet stroke EEG datasets
- EEGMAT dataset

**Option C**: Simulate synthetic data
- Generate synthetic EEG data matching paper's feature distributions
- Use published statistics (means, SDs) from paper's figures
- Useful cho testing pipeline nhưng kết quả sẽ khác

---

## 4. MÂU THUẪN VÀ AMBIGUITY TRONG PAPER

### 4.1 Số features: "4 optimal" vs "8 features" trong SHAP

**Paper nói**:
> "four optimal features are extracted through this approach" (Section 4.3.2)

**Nhưng SHAP analysis cho thấy**:
> "Fig. 6 illustrates the top eight factors with the greatest effects: DTR, Theta, RP Delta, RP Alpha, age, beta, years of education, and gender" (Section 4.5.1)

**Giải thích khả dĩ nhất**: 
> **INFERENCE**: "4 optimal features" chỉ EEG-based features từ adapted PSD (có thể là RP Delta, RP Theta, RP Alpha, RP Beta hoặc một subset). Categorical features (age, gender, education) được thêm riêng. DTR có thể được tính như derived feature. Tổng cộng 8+ features.

### 4.2 Confusion matrix vs sample size

**Paper nói**: 24 participants

**Confusion matrix** (Fig. 5 + derived from metrics):
- Total predictions ≈ 39 (tính từ sensitivity + specificity)
- 39 ≠ 24

**Giải thích khả dĩ**:
> **INFERENCE**: Paper có thể dùng mỗi epoch (4 giây) như 1 sample riêng biệt thay vì aggregate per participant. Hoặc dùng cross-validation → tổng predictions nhiều hơn số participants.

### 4.3 "Sparsity = 0" interpretation

> **FACT FROM PAPER**: Table 1 — Sparsity = 0

**Ambiguity**: "Sparsity = 0" có thể nghĩa là:
- Reservoir fully connected (không có sparsity) — common interpretation
- Hoặc density = 0 (reservoir không có connections) — rõ ràng sai

> **INFERENCE**: Sparsity = 0 → fully connected reservoir. Một số libraries dùng "connectivity" thay vì "sparsity" → connectivity = 1.0.

### 4.4 "Noise = 0.001"

**Ambiguity**: "Noise" có thể là:
- Regularization noise added to reservoir states
- Ridge regression regularization parameter (alpha)
- Input noise for robustness

> **INFERENCE**: Nhiều khả năng là ridge regression regularization parameter cho Wout training.

---

## 5. PHÂN BIỆT RÕ: FACT / UNKNOWN / INFERENCE

### Bảng tổng hợp

**CONFIRMED BY PAPER (FACT)**:
- Framework architecture: 4 modules
- Dataset: 24 participants, single pre-frontal EEG, 72h post-stroke
- Preprocessing: bandpass 0.5–30 Hz, manual artifact removal, ±100 μV threshold
- Features: PSD-based (RP Delta/Theta/Alpha/Beta, DAR, DTR) + demographics
- Epoch: 4 seconds, FFT resolution 0.25 Hz
- ESN: 200 neurons, spectral radius 0.95, sparsity 0, noise 0.001
- Ensemble: 7 ESNs, bagging, averaging
- XAI: SHAP (global) + LIME (local)
- Performance: Accuracy 96.5%, F1 94.73%, Sensitivity 96.42%, Specificity 81.81%
- DSS: physician interface with prediction + explanation + recommendation
- Running time: 9.35 minutes
- Top features: DTR, Theta, RP Delta, RP Alpha, age, beta, education, gender

**NOT PROVIDED (UNKNOWN)**:
- Dataset access/download link
- Sampling rate
- Number of stroke vs control
- Train/test split ratio
- Cross-validation strategy
- ESN leaking rate
- ESN input scaling
- Activation functions
- Wout solving method
- Bootstrap sample size
- SHAP/LIME configuration
- Recommendation logic
- UI technology
- Feature aggregation per subject
- Filter type/order
- FFT window function

**INFERRED (RECOMMENDATION)**:
- Leaking rate default: 1.0 hoặc cần tuning (0.1–1.0)
- Input scaling: needs tuning (0.01–1.0)
- Activation: tanh (standard ESN)
- Wout: ridge regression with alpha=0.001
- SHAP: KernelSHAP (model-agnostic)
- Frequency bands: standard (Delta 0.5–4, Theta 4–8, Alpha 8–13, Beta 13–30)
- Filter: Butterworth or FIR bandpass
- Feature aggregation: average across epochs per participant
- Train/test: 80/20 stratified or LOOCV (due to small sample)
- Bootstrap: sample size = training size (sklearn default)
- Soft voting (probability averaging)

---

## 6. YẾU TỐ GÂY SAI LỆCH KẾT QUẢ KHI REPRODUCE

### 6.1 High-Impact Factors (sẽ gây khác biệt lớn)

1. **Dataset không giống** → nếu dùng dataset khác → kết quả hoàn toàn khác
2. **ESN random initialization** → mỗi lần chạy khác nhau, đặc biệt trên 24 mẫu
3. **Train/test split** → với 24 mẫu, 1 sample khác biệt = 4% accuracy change
4. **Leaking rate** → giá trị này không được report, nhưng ảnh hưởng rất lớn
5. **Preprocessing artifacts** → manual vs automated → features khác → accuracy khác

### 6.2 Medium-Impact Factors

6. **Input scaling** → ảnh hưởng non-linearity trong reservoir
7. **Feature selection** → nếu chọn features khác → model khác
8. **FFT parameters** → window function, overlap → PSD estimates khác
9. **Bootstrap sampling** → random subsets khác → ensemble predictions khác

### 6.3 Low-Impact Factors

10. **SHAP/LIME configuration** → không ảnh hưởng model accuracy, chỉ explanation quality
11. **Normalization method** → MinMax là khá chuẩn
12. **Metrics calculation** → standard formulas

---

## 7. KẾ HOẠCH ĐẢM BẢO REPRODUCIBILITY

### 7.1 Random Seed Management
- Cố định global random seed
- Seed cho mỗi ESN trong ensemble
- Seed cho train/test split
- Seed cho bootstrap sampling
- Log tất cả seeds

### 7.2 Configuration as Code
- Tất cả hyperparameters trong config file
- Version control cho configs
- Experiment tracking (MLflow hoặc custom logging)

### 7.3 Ablation Studies
- Test với nhiều leaking rate values
- Test với nhiều input scaling values
- Test với nhiều train/test splits
- Compare LOOCV vs hold-out vs k-fold

### 7.4 Baseline Comparisons
- Implement baseline classifiers (SVM, RF, LR) trước ESN
- So sánh để validate pipeline correctness
- Nếu baselines cho kết quả hợp lý → pipeline đúng → ESN kết quả đáng tin hơn
