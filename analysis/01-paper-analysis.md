# 01 — PAPER ANALYSIS

> **Paper**: *Enhancing accuracy and interpretability in EEG-based medical decision making using an explainable ensemble learning framework application for stroke prediction*
> **Authors**: Samar Bouazizi, Hela Ltifi
> **Journal**: Decision Support Systems 178 (2024) 114126
> **Published**: November 2023 (online)

---

## 1. PAPER OVERVIEW

### 1.1 Vấn đề thực tế

Đột quỵ (stroke) là nguyên nhân tử vong thứ hai trên thế giới (theo WHO). Việc phát hiện sớm đột quỵ là cực kỳ quan trọng để can thiệp kịp thời. Các phương pháp chẩn đoán dựa trên hình ảnh y khoa (MRI, CT) tốn kém, mất thời gian, và cần thiết bị cồng kềnh. EEG là phương pháp không xâm lấn, rẻ tiền, đo liên tục hoạt động não — có tiềm năng lớn cho phát hiện sớm đột quỵ.

### 1.2 Vấn đề nghiên cứu

- Các mô hình ML hiện tại cho EEG data phần lớn là "black box" — bác sĩ không thể hiểu tại sao model đưa ra quyết định
- ESN (Echo State Network) đã chứng minh hiệu quả trong xử lý dữ liệu chuỗi thời gian như EEG, nhưng thiếu khả năng giải thích
- Chưa có nghiên cứu nào kết hợp ESN ensemble + XAI cho EEG-based stroke prediction

### 1.3 Research Gap

> **FACT FROM PAPER**: "a critical gap persists in the development of interpretable models that can effectively support clinical decision-making" (Section 1)

- Thiếu mô hình vừa chính xác VỪA có thể giải thích cho EEG-based medical DSS
- Chưa có nghiên cứu sử dụng ESN cho stroke prediction từ EEG
- Chưa có framework end-to-end: từ EEG preprocessing → prediction → explanation → decision support

### 1.4 Đóng góp chính của paper (5 đóng góp)

> **FACT FROM PAPER** (Section 6 — Conclusion):

1. **Multi-level framework** cho EEG-based medical DSS — giải quyết thách thức của dữ liệu EEG phức tạp thông qua cách tiếp cận phân tầng
2. **Novel feature selection algorithm** — trích xuất features dựa trên frequency band có khả năng phân biệt (discriminative) tốt nhất
3. **Ensemble ESN (E-ESN)** — kết hợp nhiều ESN với ensemble learning để giảm variance và overfitting
4. **Explainable AI tích hợp** — nhúng SHAP (global) + LIME (local) vào E-ESN để cung cấp model explanation
5. **Medical Decision-Making module** — kết nối ML classification + interpretive insights + diagnostic expertise

### 1.5 Ý tưởng cốt lõi

Framework 4 module:
- **Module 1 (Pre-treatment)**: Preprocessing + Feature Extraction + Feature Selection
- **Module 2 (E-ESN)**: Ensemble của 7 ESN với bagging
- **Module 3 (XAI)**: SHAP (global) + LIME (local) explanations
- **Module 4 (DSS)**: Giao diện cho physician — prediction + explanation + recommendation

---

## 2. END-TO-END PIPELINE

### Flow tổng thể

```
Raw EEG Data (single-channel, pre-frontal)
    │
    ▼
[Module 1a] EEG Preprocessing
    │   - Band-pass filter 0.5–30 Hz
    │   - Manual artifact inspection & removal
    │   - Amplitude thresholding (±100 μV)
    │
    ▼
[Module 1b] Feature Extraction (Adapted PSD)
    │   - FFT on 4-second artifact-free epochs (1/4 Hz resolution)
    │   - Calculate Relative Power: RP Delta, RP Theta, RP Alpha, RP Beta
    │   - Calculate ratios: DAR (Delta/Alpha), DTR (Delta/Theta)
    │   - Normalize features to [0, 1]
    │
    ▼
[Module 1c] Feature Selection
    │   - Filter-based feature importance
    │   - Select 4 optimal features (from paper's Algorithm 1)
    │
    ▼
Selected Features + Categorical Data (age, gender, education)
    │
    ▼
[Module 2] Ensemble ESN Training
    │   - 7 individual ESNs
    │   - Each: 200 neurons, spectral radius 0.95, sparsity 0, noise 0.001
    │   - Different random initializations
    │   - Bagging: each ESN trained on different bootstrap subset
    │   - Aggregation: averaging predictions
    │
    ▼
E-ESN Prediction (Stroke vs Control)
    │
    ├──────────────────────────┐
    ▼                          ▼
[Module 3a]              [Module 3b]
SHAP (Global)            LIME (Local)
    │                          │
    │  Mean SHAP values        │  Per-instance explanation
    │  Feature importance      │  Feature contribution bars
    │  ranking across          │
    │  all predictions         │
    │                          │
    ▼                          ▼
[Module 4] Medical DSS Interface
    │   - Patient info display
    │   - EEG data display
    │   - Stroke probability (e.g., 95%)
    │   - SHAP button → global explanation
    │   - LIME button → local explanation
    │   - Recommendations (MRI, CBC, BMP, etc.)
    │   - Physician confirm/cancel
    │
    ▼
Clinical Decision
```

### Mối quan hệ giữa các bước

- **Module 1 → Module 2**: Features đã chọn + categorical data là input cho E-ESN
- **Module 2 → Module 3**: E-ESN phải được train xong trước khi chạy SHAP/LIME
- **Module 2 + Module 3 → Module 4**: DSS cần cả prediction lẫn explanations
- **Module 1 độc lập**: Preprocessing/feature extraction có thể phát triển và test riêng

---

## 3. MODEL — ESN & Ensemble ESN

### 3.1 Vì sao paper chọn ESN?

> **FACT FROM PAPER**:

- ESN là một dạng RNN hiệu quả, phù hợp cho dữ liệu chuỗi thời gian như EEG
- Training đơn giản hơn RNN thông thường — chỉ cần giải bài toán linear regression cho output weights
- Reservoir (lớp ẩn) có cấu trúc ngẫu nhiên, không cần backpropagation through time
- Có khả năng bắt temporal dynamics trong EEG
- Chưa có nghiên cứu nào dùng ESN cho stroke prediction từ EEG → đây là novelty

### 3.2 Cách ESN hoạt động

> **FACT FROM PAPER** (Section 2.2):

**Kiến trúc**: Input Layer → Reservoir Layer → Output Layer (Readout)

**State update equations**:
```
x(i+1) = f(Win * u(i+1) + W * x(i) + Wback * y(i))    ... (1)
y(i+1) = g(Wout * x(i+1))                                ... (2)
```

Trong đó:
- `x(i)`: trạng thái reservoir tại thời điểm i
- `u(i)`: input tại thời điểm i
- `y(i)`: output tại thời điểm i
- `Win`: ma trận trọng số input → reservoir (RANDOM, CỐ ĐỊNH)
- `W`: ma trận trọng số trong reservoir (RANDOM, CỐ ĐỊNH)
- `Wback`: ma trận feedback output → reservoir (tùy chọn)
- `Wout`: ma trận trọng số reservoir → output (ĐÂY LÀ DUY NHẤT ĐƯỢC TRAIN)
- `f`: activation function (thường là tanh)
- `g`: activation function output

**Điểm mấu chốt**: Chỉ có `Wout` được học. Tất cả trọng số khác (`Win`, `W`) được khởi tạo ngẫu nhiên và giữ cố định → training cực nhanh.

**Tham số quan trọng**:
- **Spectral radius**: phải < 1 để đảm bảo echo state property (stability)
- **Reservoir size** (số neurons): quyết định complexity mà ESN xử lý được
- **Connectivity rate**: ảnh hưởng đến memory capacity
- **Sparsity**: mức thưa thớt của kết nối trong reservoir

### 3.3 Ensemble ESN (E-ESN)

> **FACT FROM PAPER** (Section 3.2 & 4.4):

- **Số lượng ESN trong ensemble**: 7
- **Mỗi ESN**: cùng hyperparameters nhưng random initializations khác nhau
- **Phương pháp ensemble**: Bagging
  - Mỗi ESN được train trên bootstrap subset khác nhau của training data
  - Predictions được aggregate bằng cách **averaging**
- **Lý do dùng bagging**: giảm variance, chống overfitting, tăng generalizability

### 3.4 Hyperparameters được paper cung cấp

> **FACT FROM PAPER** (Table 1):

- Neurons: 200
- Spectral radius: 0.95
- Sparsity: 0 (fully connected reservoir)
- Noise (regularization): 0.001
- Số ESN trong ensemble: 7
- Running time: 9.35 phút

### 3.5 Kết quả hiệu năng

> **FACT FROM PAPER** (Table 2):

- Accuracy: 96.50%
- PPV (Precision): 93.10%
- NPV: 90%
- Sensitivity (Recall): 96.42%
- Specificity: 81.81%
- F1-score: 94.73%

### 3.6 Thư viện/model có sẵn

> **INFERENCE/RECOMMENDATION**:

- **reservoirpy** (Python): thư viện phổ biến nhất cho ESN, hỗ trợ reservoir computing
- **pyESN**: thư viện nhỏ gọn cho ESN
- **scikit-learn**: có thể dùng BaggingClassifier để wrap ESN
- Không có pretrained ESN model phù hợp cho EEG stroke prediction → phải training từ đầu
- Bagging có thể tự implement hoặc dùng sklearn wrapper

### 3.7 Thông tin paper KHÔNG cung cấp về model

> **PAPER KHÔNG NÓI**:

- Activation function cụ thể của reservoir và output (chỉ nói chung là "nonlinear")
- Cách chia train/test split chính xác (tỷ lệ bao nhiêu)
- Có sử dụng cross-validation không?
- Input scaling factor
- Leaking rate (quan trọng cho ESN)
- Phương pháp giải Wout (ridge regression? pseudo-inverse?)
- Có sử dụng Wback (feedback connection) không?
- Bootstrap sampling ratio cho bagging
- Có warmup/washout period cho reservoir không?

---

## 4. EXPLAINABLE AI (XAI)

### 4.1 Global Explanation — SHAP

> **FACT FROM PAPER** (Section 4.5.1):

- Sử dụng SHAP (SHapley Additive exPlanations) cho global model explanation
- Tính mean SHAP values từ E-ESN classifier
- Kết quả: **Top 8 features quan trọng nhất** (theo thứ tự):
  1. DTR (Delta/Theta Ratio)
  2. Theta
  3. RP Delta
  4. RP Alpha
  5. Age
  6. Beta
  7. Years of education
  8. Gender
- Hiển thị dưới dạng:
  - SHAP value impact per class (Fig. 6a)
  - Average feature impact on model output magnitude (Fig. 6b)

**Cách kết nối với model**: SHAP được tính dựa trên predictions của E-ESN classifier đã train. SHAP values cho biết mức đóng góp (positive/negative) của từng feature vào prediction.

### 4.2 Local Explanation — LIME

> **FACT FROM PAPER** (Section 4.5.2):

- Sử dụng LIME cho local model explanation — giải thích từng prediction riêng lẻ
- LIME xấp xỉ model phức tạp (E-ESN) bằng một model đơn giản hơn, dễ hiểu hơn
- Không cần truy cập internal parameters của E-ESN
- Output: bar charts + feature value tables cho mỗi instance
- Paper minh họa 2 ví dụ:
  - Instance 1: predicted "Control" (healthy) — hiển thị features ảnh hưởng
  - Instance 2: predicted "Stroke" — hiển thị features ảnh hưởng

**Cách kết nối với model**: LIME coi E-ESN như black-box, chỉ cần hàm predict(). LIME perturbates input xung quanh instance cần giải thích và fit surrogate model.

### 4.3 Điều paper KHÔNG nói rõ về XAI

> **PAPER KHÔNG NÓI**:

- Loại SHAP explainer cụ thể (KernelSHAP? TreeSHAP? — vì ESN không phải tree-based → nhiều khả năng là KernelSHAP)
- Số lượng perturbation samples cho LIME/SHAP
- Cách handle ensemble khi giải thích: SHAP/LIME áp dụng cho ensemble prediction hay cho từng ESN?
- Thời gian tính toán SHAP/LIME
- Evaluation metric cho quality of explanations (fidelity, stability, etc.)

---

## 5. DECISION SUPPORT SYSTEM (DSS)

### 5.1 Người dùng

> **FACT FROM PAPER**: Physician (bác sĩ)

### 5.2 Input

> **FACT FROM PAPER** (Section 4.6, Fig. 8):

- Patient information: name, age, gender, years of education
- EEG data: alpha, beta waves, etc.
- Categorical data được nhập vào hệ thống

### 5.3 Prediction hiển thị gì

> **FACT FROM PAPER**:

- Stroke risk percentage (ví dụ: "95% risk of stroke")
- Binary classification: Stroke vs Control

### 5.4 Explanation hiển thị gì

> **FACT FROM PAPER**:

- Nút "LIME" → hiển thị local explanation (feature importance cho instance đang xét)
- Nút "SHAP" → hiển thị global explanation (SHAP values)

### 5.5 Recommendation hoạt động thế nào

> **FACT FROM PAPER**:

- Dựa trên predicted stroke risk, hệ thống đề xuất các xét nghiệm chẩn đoán cụ thể:
  - MRI (Magnetic Resonance Imaging)
  - CBC (Complete Blood Count) test
  - BMP (Basic Metabolic Panel) test
  - Coagulation Profile test
  - Lipid Profile test
  - D-Dimer test

### 5.6 Physician tương tác ra sao

> **FACT FROM PAPER**:

- Xem prediction + explanation
- Chọn giữa các recommendations
- Confirm hoặc Cancel recommendations

### 5.7 Phần paper KHÔNG mô tả chi tiết

> **PAPER KHÔNG NÓI**:

- Logic cụ thể để quyết định recommend test nào (rule-based? threshold-based?)
- UI được build bằng technology gì
- Có lưu lịch sử không
- Có nhiều hơn một bệnh nhân cùng lúc không
- Workflow khi physician cancel recommendation
- Cách EEG data được upload/nhập vào hệ thống
- Có real-time processing không

> **INFERENCE/RECOMMENDATION**: Phần DSS trong paper chủ yếu là conceptual design + mockup UI. Khi build lại, phần logic recommendation, UI/UX flow, và data management sẽ phải tự thiết kế hoàn toàn.

---

## 6. LIMITATIONS & RISKS

### 6.1 Dataset

> **FACT FROM PAPER**: Chỉ có 24 participants → sample size cực nhỏ
> **Risk**: Kết quả 96.5% accuracy trên 24 người rất khó generalize. Variance cực cao với sample nhỏ.

### 6.2 Sample Size & Generalization

> **FACT FROM PAPER**: Paper tự thừa nhận "While only 24 participants were included, limiting generalizability"
> **Risk khi reproduce**: Với 24 samples, sự khác biệt nhỏ trong preprocessing hoặc random seed có thể tạo ra kết quả rất khác nhau.

### 6.3 Data Leakage Potential

> **PAPER KHÔNG NÓI** rõ cách train/test split
> **Risk**: Với 24 samples, nếu không dùng proper cross-validation, rất dễ data leak. Paper không đề cập cross-validation. Nếu chỉ dùng simple train/test split trên 24 mẫu → test set chỉ có vài mẫu → kết quả không đáng tin cậy thống kê.

### 6.4 EEG Preprocessing

> **FACT FROM PAPER**: Có bước "manual inspection" để identify artifacts
> **Risk khi reproduce**: Manual inspection là subjective — 2 người khác nhau có thể loại bỏ artifacts khác nhau → kết quả khác nhau.

### 6.5 Model

> **Risk**: Paper thiếu nhiều hyperparameters quan trọng cho ESN (leaking rate, input scaling, output solving method). Giá trị mặc định khác nhau giữa các thư viện → kết quả có thể khác đáng kể.

### 6.6 Single-channel EEG

> **FACT FROM PAPER**: Dữ liệu từ dataset [2] — single pre-frontal electrode
> **Risk**: Một kênh EEG duy nhất cung cấp thông tin rất hạn chế so với multi-channel. Kết quả tốt có thể do overfitting trên dataset nhỏ.

### 6.7 Explainability

> **Risk**: SHAP/LIME explanations chưa được đánh giá bằng bất kỳ metric nào (fidelity, stability). Không có user study với bác sĩ thực tế.

### 6.8 Clinical / DSS

> **Risk**: DSS interface chỉ là mockup, chưa được deploy hay test với bác sĩ. Recommendation logic không được mô tả → phải tự thiết kế.

### 6.9 Reproducibility tổng thể

**Nguyên nhân chính có thể khiến kết quả khác paper:**
- Random seed khác nhau (ESN rất nhạy với initialization)
- Preprocessing khác (đặc biệt manual artifact removal)
- Thiếu thông tin hyperparameters (leaking rate, input scaling)
- Train/test split khác
- Thư viện ESN khác nhau (implementation details)
- Feature normalization order có thể khác
- Bootstrap sampling cho bagging khác nhau

---

## 7. THÔNG TIN QUAN TRỌNG TỪ FIGURES & ALGORITHMS

### Algorithm 1 — Adapted PSD Feature Selection

> **FACT FROM PAPER**: Paper trình bày Algorithm 1 cho adapted PSD
> **Nội dung chính** (suy từ mô tả):
> - Input: Artifact-free EEG epochs (4 seconds)
> - FFT với resolution 1/4 Hz
> - Tính PSD cho từng frequency band
> - Tính Relative Power (RP) cho Delta, Theta, Alpha, Beta
> - Tính DAR (Delta/Alpha Ratio) và DTR (Delta/Theta Ratio)
> - Normalize features về [0, 1]
> - Chọn 4 features tối ưu

> **PAPER KHÔNG NÓI**: Tiêu chí cụ thể để chọn 4 features nào (filter metric nào?)

### Algorithm 2 — E-ESN

> **FACT FROM PAPER**: Paper trình bày Algorithm 2 cho E-ESN training
> **Nội dung chính** (suy từ mô tả):
> - 7 ESN với cùng hyperparameters
> - Random initialization khác nhau
> - Bagging: bootstrap sampling
> - Aggregation: averaging

### Fig. 3 — Participant demographics

> **FACT FROM PAPER**: Visualizations về gender và age của 24 participants

### Fig. 4 — Frequency distributions

> **FACT FROM PAPER**: Distribution plots cho Delta, Theta, Alpha, Beta frequencies

### Fig. 5 — Confusion matrix

> **FACT FROM PAPER**: Confusion matrix của E-ESN prediction

### Fig. 6 — SHAP explanations

> **FACT FROM PAPER**: (a) SHAP value impact per class, (b) Average feature impact magnitude

### Fig. 7 — LIME explanations

> **FACT FROM PAPER**: 2 LIME examples — 1 Control case + 1 Stroke case

### Fig. 8 — DSS User Interface

> **FACT FROM PAPER**: Screenshot/mockup của DSS interface cho physician
