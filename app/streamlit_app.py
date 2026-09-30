"""
Explainable EEG Stroke Decision Support System — Web Application
Hệ thống Hỗ trợ Quyết định Đột quỵ dựa trên EEG

Dành cho bác sĩ và nhà nghiên cứu lâm sàng.
"""
import sys
sys.path.insert(0, r"d:\Personal\Explainable EEG Stroke DSS")

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import shap
import lime.lime_tabular
import io
import json
import tempfile
import os

from src.data.loader import (
    load_dataset, encode_labels, encode_categorical_features,
    EEG_FEATURES, DEMOGRAPHIC_FEATURES, LABEL_MAP_INV
)
from src.features.normalization import FeatureNormalizer
from src.models.esn import EchoStateNetwork
from src.models.ensemble import EnsembleESN
from src.models.baselines import compute_metrics
from src.dss.recommendation import RecommendationEngine, DIAGNOSTIC_TESTS
from src.utils.seed import set_global_seed
from src.reports.pdf_generator import generate_report

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Hệ thống Hỗ trợ Chẩn đoán Đột quỵ qua EEG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        padding: 2rem 2.5rem; border-radius: 16px; margin-bottom: 1.5rem;
        color: #ffffff !important; box-shadow: 0 8px 32px rgba(30,58,95,0.25);
    }
    .main-header,
    .main-header * {
        color: #ffffff !important;
    }
    .main-header h1 {
        margin: 0;
        font-size: 1.9rem;
        font-weight: 700;
        color: #ffffff !important;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.92;
        font-size: 1rem;
        color: #ffffff !important;
    }

    /* Guide / onboarding card */
    .guide-card {
        background: linear-gradient(135deg, #e8f4fd 0%, #d1ecf1 100%);
        border-left: 5px solid #2d6a9f; padding: 1.2rem 1.5rem;
        border-radius: 0 12px 12px 0; margin-bottom: 1rem;
    }
    .guide-card h4 { margin: 0 0 0.5rem 0; color: #1e3a5f; font-size: 1rem; }
    .guide-card p  { margin: 0; color: #2c5282; font-size: 0.9rem; line-height: 1.6; }

    /* Content-type badges */
    .badge-static {
        display: inline-block; background: #ebf8ff; color: #2b6cb0;
        border: 1px solid #bee3f8; border-radius: 20px; padding: 3px 12px;
        font-size: 0.75rem; font-weight: 600; margin-bottom: 0.5rem;
    }
    .badge-live {
        display: inline-block; background: #f0fff4; color: #276749;
        border: 1px solid #9ae6b4; border-radius: 20px; padding: 3px 12px;
        font-size: 0.75rem; font-weight: 600; margin-bottom: 0.5rem;
    }
    .badge-research {
        display: inline-block; background: #faf5ff; color: #6b46c1;
        border: 1px solid #d6bcfa; border-radius: 20px; padding: 3px 12px;
        font-size: 0.75rem; font-weight: 600; margin-bottom: 0.5rem;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        padding: 1.25rem; border-radius: 12px; text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06); border-top: 3px solid #2d6a9f;
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); }
    .metric-value { font-size: 2rem; font-weight: 700; color: #1e3a5f; }
    .metric-label { font-size: 0.78rem; color: #718096; text-transform: uppercase;
                    letter-spacing: 1px; margin-top: 0.25rem; }
    .metric-sub   { font-size: 0.7rem; color: #a0aec0; margin-top: 0.2rem; font-style: italic; }

    /* Risk levels */
    .risk-high {
        background: linear-gradient(135deg, #c53030, #9b2c2c); color: white;
        padding: 1.2rem; border-radius: 12px; text-align: center;
        font-size: 1.2rem; font-weight: 700;
    }
    .risk-medium {
        background: linear-gradient(135deg, #dd6b20, #c05621); color: white;
        padding: 1.2rem; border-radius: 12px; text-align: center;
        font-size: 1.2rem; font-weight: 700;
    }
    .risk-low {
        background: linear-gradient(135deg, #276749, #2f855a); color: white;
        padding: 1.2rem; border-radius: 12px; text-align: center;
        font-size: 1.2rem; font-weight: 700;
    }

    /* Test / recommendation cards */
    .test-card {
        background: white; border-left: 4px solid #2d6a9f;
        padding: 0.9rem 1.2rem; margin: 0.4rem 0;
        border-radius: 0 8px 8px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .test-name { font-weight: 600; color: #1e3a5f; font-size: 0.95rem; }
    .test-desc  { color: #718096; font-size: 0.82rem; margin-top: 0.15rem; }

    /* Explanation callout */
    .explain-box {
        background: #fffbeb; border-left: 4px solid #d69e2e;
        padding: 0.8rem 1.1rem; border-radius: 0 8px 8px 0;
        margin: 0.7rem 0; font-size: 0.88rem; color: #744210;
    }

    /* Section label */
    .section-label {
        font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 2px; color: #718096; margin-bottom: 0.3rem;
    }

    /* Divider */
    .section-divider {
        height: 2px;
        background: linear-gradient(90deg, #2d6a9f, #90cdf4, transparent);
        border: none; margin: 1.8rem 0; border-radius: 2px;
    }

    /* Feature bar */
    .feature-bar { height: 22px; border-radius: 10px; display: inline-block; transition: width 0.5s ease; }

    /* Sidebar */
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a2742 0%, #1e3a5f 100%);
    }
    div[data-testid="stSidebar"] .stMarkdown { color: #e2e8f0; }
    div[data-testid="stSidebar"] label { color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPER FUNCTIONS — badges and callout boxes
# ============================================================
def badge(kind: str, text: str) -> str:
    return f'<span class="badge-{kind}">{text}</span>'


def explain_box(text: str) -> str:
    return f'<div class="explain-box">💡 {text}</div>'


# Glossary of EEG features in Vietnamese
EEG_FEATURE_EXPLAIN = {
    "Delta": "Sóng Delta (0.5-4 Hz) — tăng khi có tổn thương não, giấc ngủ sâu",
    "Theta": "Sóng Theta (4-8 Hz) — liên quan buồn ngủ, thiếu oxy não",
    "Alpha": "Sóng Alpha (8-12 Hz) — giảm khi có tổn thương hoặc căng thẳng",
    "Beta": "Sóng Beta (12-30 Hz) — hoạt động nhận thức, tập trung",
    "RP Delta": "Công suất tương đối Delta (%) — tăng cao khi não bất thường",
    "RP Theta": "Công suất tương đối Theta (%) — liên quan thiếu máu não",
    "RP Alpha": "Công suất tương đối Alpha (%) — giảm khi tổn thương",
    "RP Beta": "Công suất tương đối Beta (%)",
    "DTR": "Tỉ số Delta/Theta — tăng khi tổn thương não nghiêm trọng",
    "DAR": "Tỉ số (Delta+Theta)/(Alpha+Beta) — chỉ số nhạy nhất với bất thường não",
    "Total Power": "Tổng công suất phổ EEG",
    "Epoch": "Số đoạn EEG được phân tích",
}


# ============================================================
# CACHED DATA LOADING & MODEL TRAINING
# ============================================================
@st.cache_data
def load_data():
    """Load and prepare dataset."""
    df = load_dataset()
    df_encoded = encode_labels(df)
    df_encoded = encode_categorical_features(df_encoded)
    
    feature_cols = [f for f in EEG_FEATURES if f in df_encoded.columns]
    X = df_encoded[feature_cols].values.astype(np.float64)
    y = df_encoded['label'].values.astype(np.int64)
    
    normalizer = FeatureNormalizer()
    X_norm = normalizer.fit_transform(X)
    
    return df, df_encoded, X, X_norm, y, feature_cols, normalizer


@st.cache_resource
def train_model(seed=42):
    """Train the E-ESN model."""
    set_global_seed(seed)
    _, _, _, X_norm, y, feature_names, _ = load_data()
    
    eesn = EnsembleESN(
        n_estimators=7, n_neurons=200, spectral_radius=0.95,
        sparsity=0.0, noise=0.01, leaking_rate=0.7, input_scaling=0.5,
        aggregation='soft', bootstrap=False, random_state=seed
    )
    eesn.fit(X_norm, y)
    return eesn


@st.cache_data
def compute_loocv_metrics():
    """Compute LOOCV metrics for display."""
    _, _, _, X_norm, y, _, _ = load_data()
    
    y_pred = np.zeros_like(y)
    for i in range(len(y)):
        mask = np.ones(len(y), dtype=bool)
        mask[i] = False
        
        esn = EchoStateNetwork(
            n_neurons=200, spectral_radius=0.95, sparsity=0.0,
            noise=0.01, leaking_rate=0.7, input_scaling=0.5,
            random_state=42
        )
        esn.fit(X_norm[mask], y[mask])
        y_pred[i] = esn.predict(X_norm[i:i+1])[0]
    
    return compute_metrics(y, y_pred)


@st.cache_data(show_spinner="🔬 Computing SHAP values for all patients (first time only)...")
def compute_all_shap_values():
    """
    Compute SHAP values for all 38 patients using KernelSHAP.
    Cached after first computation (~1-2 min).
    """
    _, _, _, X_norm, y, feature_names, _ = load_data()
    _model = train_model()

    def predict_fn(X):
        return _model.predict_proba(X)[:, 1]

    bg = shap.kmeans(X_norm, 10)
    explainer = shap.KernelExplainer(predict_fn, bg)
    sv = explainer.shap_values(X_norm, nsamples=100)
    sv = np.array(sv)
    if sv.ndim == 3:
        # (n_classes, n_samples, n_features) → take stroke class
        sv = sv[1] if sv.shape[0] == 2 else sv[:, :, 1]

    base_val = explainer.expected_value
    if hasattr(base_val, '__len__'):
        base_val = float(np.array(base_val).flat[0])
    else:
        base_val = float(base_val)

    return sv, base_val


@st.cache_resource
def get_lime_explainer():
    """Create and cache a LIME tabular explainer."""
    _, _, _, X_norm, y, feature_names, _ = load_data()
    return lime.lime_tabular.LimeTabularExplainer(
        training_data=X_norm,
        feature_names=feature_names,
        class_names=['Control', 'Stroke'],
        mode='classification',
        discretize_continuous=True,
        random_state=42
    )


@st.cache_data
def load_full_evaluation():
    """Load full evaluation results from JSON."""
    results_path = os.path.join(
        r"d:\Personal\Explainable EEG Stroke DSS",
        "experiments", "results", "full_evaluation.json"
    )
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            return json.load(f)
    return None


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:0.5rem 0 1rem 0;">
        <span style="font-size:2.5rem;">🧠</span><br>
        <span style="color:#90cdf4; font-weight:700; font-size:1rem;">EEG Stroke DSS</span><br>
        <span style="color:#718096; font-size:0.75rem;">Hỗ trợ Chẩn đoán Đột quỵ</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    user_role = st.radio(
        "Bạn là:",
        ["Bac si / Nhan vien y te", "Nha nghien cuu"],
        format_func=lambda x: (
            "\U0001f469\u200d\u2695\ufe0f Bác sĩ / Nhân viên y tế"
            if "Bac" in x else "\U0001f52c Nhà nghiên cứu"
        ),
        index=0
    )

    st.markdown("---")

    if "Bac" in user_role:
        page_options = [
            "\U0001f3e5 Phân tích Bệnh nhân",
            "\U0001f4a1 Tại sao AI đưa ra kết quả?",
            "\U0001f4ca Thông tin Hệ thống",
            "\u2139\ufe0f Giới thiệu",
        ]
        st.markdown(
            "<small style='color:#90cdf4'>💡 Bắt đầu từ <b>Phân tích Bệnh nhân</b></small>",
            unsafe_allow_html=True
        )
    else:
        page_options = [
            "\U0001f3e5 Phân tích Bệnh nhân",
            "\U0001f4a1 Tại sao AI đưa ra kết quả?",
            "\U0001f4ca Thông tin Hệ thống",
            "\U0001f4c8 Đánh giá Chuyên sâu",
            "\u2139\ufe0f Giới thiệu",
        ]
        st.markdown(
            "<small style='color:#90cdf4'>💡 Xem <b>Đánh giá Chuyên sâu</b> để so sánh mô hình</small>",
            unsafe_allow_html=True
        )

    page = st.radio("Chọn trang", page_options, label_visibility="collapsed")

    st.markdown("---")
    st.markdown("""
    <div style="color:#718096; font-size:0.75rem; line-height:1.6;">
        <b style="color:#90cdf4;">Thông tin hệ thống</b><br>
        Mô hình AI: E-ESN (7 mạng nơ-ron)<br>
        Dữ liệu: 38 bệnh nhân nghiên cứu<br>
        Điện cực EEG: FP1 (trán trước)<br>
        Nguồn: Bouazizi & Ltifi (2024)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        "<div style='background:#7b341e;border-radius:8px;padding:0.6rem;"
        "color:#fbd38d;font-size:0.72rem;line-height:1.5;'>"
        "<b>⚠️ CẢNH BÁO</b><br>"
        "Chỉ dùng cho <b>nghiên cứu</b>. "
        "KHÔNG thay thế quyết định lâm sàng."
        "</div>",
        unsafe_allow_html=True
    )


# ============================================================
# LOAD DATA & MODEL
# ============================================================
df_raw, df_enc, X, X_norm, y, feature_names, normalizer = load_data()
model = train_model()


# ============================================================
# PAGE ROUTING  (new Vietnamese UX)
# ============================================================

PAGE_PATIENT  = "\U0001f3e5 Ph\u00e2n t\u00edch B\u1ec7nh nh\u00e2n"
PAGE_EXPLAIN  = "\U0001f4a1 T\u1ea1i sao AI \u0111\u01b0a ra k\u1ebft qu\u1ea3?"
PAGE_OVERVIEW = "\U0001f4ca Th\u00f4ng tin H\u1ec7 th\u1ed1ng"
PAGE_EVAL     = "\U0001f4c8 \u0110\u00e1nh gi\u00e1 Chuy\u00ean s\u00e2u"
PAGE_ABOUT    = "\u2139\ufe0f Gi\u1edbi thi\u1ec7u"

FIG_DIR = r"d:\Personal\Explainable EEG Stroke DSS\experiments\figures"


# ============================================================
# PAGE: PHAN TICH BENH NHAN  (doctor main page)
# ============================================================
if page == PAGE_PATIENT:
    st.markdown("""
    <div class="main-header">
        <h1>\U0001f3e5 Ph\u00e2n t\u00edch & Ch\u1ea9n \u0111o\u00e1n B\u1ec7nh nh\u00e2n</h1>
        <p>Ch\u1ecdn b\u1ec7nh nh\u00e2n t\u1eeb d\u1eef li\u1ec7u nghi\u00ean c\u1ee9u \u0111\u1ec3 xem k\u1ebft qu\u1ea3 d\u1ef1 \u0111o\u00e1n \u0111\u1ed9t qu\u1ef5,
           m\u1ee9c \u0111\u1ed9 r\u1ee7i ro v\u00e0 kuy\u1ebfn ngh\u1ecb l\u00e2m s\u00e0ng t\u1eeb AI</p>
    </div>
    """, unsafe_allow_html=True)

    # Onboarding guide (shown until dismissed)
    if 'guide_dismissed' not in st.session_state:
        st.markdown("""
        <div class="guide-card">
            <h4>\U0001f4d6 H\u01b0\u1edbng d\u1eabn s\u1eed d\u1ee5ng nhanh</h4>
            <p>
            <b>B\u01b0\u1edbc 1:</b> Ch\u1ecdn b\u1ec7nh nh\u00e2n t\u1eeb danh s\u00e1ch b\u00ean d\u01b0\u1edbi.<br>
            <b>B\u01b0\u1edbc 2:</b> Xem k\u1ebft qu\u1ea3 d\u1ef1 \u0111o\u00e1n AI v\u00e0 m\u1ee9c \u0111\u1ed9 r\u1ee7i ro \u0111\u1ed9t qu\u1ef5.<br>
            <b>B\u01b0\u1edbc 3:</b> \u0110\u1ecdc b\u1ea3ng ch\u1ec9 s\u1ed1 EEG v\u00e0 c\u00e1c x\u00e9t nghi\u1ec7m \u0111\u01b0\u1ee3c kuy\u1ebfn ngh\u1ecb.<br>
            <b>B\u01b0\u1edbc 4:</b> T\u1ea3i b\u00e1o c\u00e1o PDF \u0111\u1ec3 l\u01b0u tr\u1eef ho\u1eb7c tham kh\u1ea3o.<br><br>
            <i>L\u01b0u \u00fd: D\u1eef li\u1ec7u hi\u1ec3n th\u1ecb l\u00e0 t\u1eeb nghi\u00ean c\u1ee9u 38 b\u1ec7nh nh\u00e2n \u0111\u00e3 \u0111\u01b0\u1ee3c thu th\u1eadp s\u1eb5n.</i>
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("\u2715 \u0110\u00e3 hi\u1ec3u, \u1ea9n h\u01b0\u1edbng d\u1eabn", key="dismiss_guide"):
            st.session_state['guide_dismissed'] = True
            st.rerun()

    st.markdown(
        badge("static", "\U0001f4c2 D\u1eef li\u1ec7u Nghi\u00ean c\u1ee9u \u2014 38 b\u1ec7nh nh\u00e2n \u0111\u00e3 thu th\u1eadp"),
        unsafe_allow_html=True
    )
    st.markdown("#### \U0001f464 Ch\u1ecdn b\u1ec7nh nh\u00e2n \u0111\u1ec3 ph\u00e2n t\u00edch")

    col_sel, col_info = st.columns([2, 1])
    with col_sel:
        patient_idx = st.selectbox(
            "Ch\u1ecdn b\u1ec7nh nh\u00e2n",
            range(len(y)),
            format_func=lambda i: (
                f"B\u1ec7nh nh\u00e2n #{i+1:02d}  \u2014  "
                f"{'\U0001f534 \u0110\u1ed9t qu\u1ef5' if y[i]==1 else '\U0001f7e2 B\u00ecnh th\u01b0\u1eddng'}"
            ),
            help="Danh s\u00e1ch 38 b\u1ec7nh nh\u00e2n trong b\u1ed9 d\u1eef li\u1ec7u nghi\u00ean c\u1ee9u EPoC"
        )

    patient_features = X_norm[patient_idx:patient_idx+1]
    patient_raw      = X[patient_idx]
    true_label       = y[patient_idx]

    with col_info:
        true_name_vi = "\u0110\u1ed9t qu\u1ef5" if true_label == 1 else "B\u00ecnh th\u01b0\u1eddng"
        icon_vi = "\U0001f534" if true_label == 1 else "\U0001f7e2"
        st.markdown(f"""
        <div style="background:#f7fafc;border-radius:10px;padding:1rem;margin-top:1.6rem;border:1px solid #e2e8f0;">
            <div style="font-size:0.75rem;color:#718096;text-transform:uppercase;letter-spacing:1px;">Ch\u1ea9n \u0111o\u00e1n th\u1ef1c t\u1ebf</div>
            <div style="font-size:1.4rem;font-weight:700;color:#1e3a5f;margin-top:0.3rem;">
                {icon_vi} {true_name_vi}
            </div>
            <div style="font-size:0.72rem;color:#a0aec0;margin-top:0.2rem;">
                Theo h\u1ed3 s\u01a1 b\u1ec7nh nh\u00e2n trong nghi\u00ean c\u1ee9u
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── AI PREDICTION ──
    st.markdown(
        badge("live", "\u26a1 K\u1ebft qu\u1ea3 TR\u1ef0C TI\u1ebeP \u2014 AI ph\u00e2n t\u00edch ngay l\u00fac n\u00e0y"),
        unsafe_allow_html=True
    )
    st.markdown("#### \U0001f916 K\u1ebft qu\u1ea3 d\u1ef1 \u0111o\u00e1n t\u1eeb AI")

    proba       = model.predict_proba(patient_features)[0]
    stroke_prob = proba[1]
    pred_en     = "Stroke" if np.argmax(proba) == 1 else "Control"
    pred_vi     = "\u0110\u1ed9t qu\u1ef5" if pred_en == "Stroke" else "B\u00ecnh th\u01b0\u1eddng"

    rec_engine = RecommendationEngine()
    risk_level = rec_engine.classify_risk(stroke_prob)
    risk_vi    = {"high": "CAO", "medium": "TRUNG B\u00ccNH", "low": "TH\u1ea4P"}.get(risk_level, risk_level.upper())

    col_pred, col_risk, col_prob = st.columns(3)

    with col_pred:
        st.markdown("<div class='section-label'>K\u1ebft lu\u1eadn AI</div>", unsafe_allow_html=True)
        if pred_en == "Stroke":
            st.error(f"\u26a0\ufe0f **{pred_vi}**")
        else:
            st.success(f"\u2705 **{pred_vi}**")
        match_en  = pred_en == ("Stroke" if true_label == 1 else "Control")
        match_lbl = "\u2705 \u0110\u00fang" if match_en else "\u274c Sai"
        st.caption(f"Ch\u1ea9n \u0111o\u00e1n th\u1ef1c t\u1ebf: **{true_name_vi}** \u2014 {match_lbl}")

    with col_risk:
        st.markdown("<div class='section-label'>M\u1ee9c \u0111\u1ed9 r\u1ee7i ro</div>", unsafe_allow_html=True)
        st.markdown(
            f'<div class="risk-{risk_level}">{risk_vi}</div>',
            unsafe_allow_html=True
        )

    with col_prob:
        st.markdown("<div class='section-label'>X\u00e1c su\u1ea5t \u0111\u1ed9t qu\u1ef5</div>", unsafe_allow_html=True)
        bar_color = "#c53030" if stroke_prob >= 0.7 else ("#dd6b20" if stroke_prob >= 0.4 else "#276749")
        st.markdown(f"""
        <div style="margin-top:0.3rem;">
            <div style="font-size:2.2rem;font-weight:700;color:{bar_color};">{stroke_prob:.1%}</div>
            <div style="background:#e2e8f0;border-radius:8px;height:12px;margin-top:0.4rem;">
                <div style="background:{bar_color};width:{stroke_prob*100:.1f}%;height:12px;border-radius:8px;"></div>
            </div>
            <div style="font-size:0.75rem;color:#718096;margin-top:0.3rem;">
                Ng\u01b0\u1ee1ng c\u1ea3nh b\u00e1o: 40% (trung b\u00ecnh), 70% (cao)
            </div>
        </div>
        """, unsafe_allow_html=True)

    if stroke_prob >= 0.7:
        expl = f"AI \u0111\u00e1nh gi\u00e1 b\u1ec7nh nh\u00e2n n\u00e0y c\u00f3 <b>r\u1ee7i ro cao \u0111\u1ed9t qu\u1ef5 ({stroke_prob:.0%})</b>. C\u1ea7n x\u00e9m x\u00e9t c\u00e1c x\u00e9t nghi\u1ec7m ch\u1ea9n \u0111o\u00e1n b\u1ed5 sung ngay l\u1eadp t\u1ee9c."
    elif stroke_prob >= 0.4:
        expl = f"AI \u0111\u00e1nh gi\u00e1 b\u1ec7nh nh\u00e2n n\u00e0y c\u00f3 <b>r\u1ee7i ro trung b\u00ecnh \u0111\u1ed9t qu\u1ef5 ({stroke_prob:.0%})</b>. N\u00ean theo d\u00f5i th\u00eam v\u00e0 c\u00e2n nh\u1eafc ki\u1ec3m tra b\u1ed5 sung."
    else:
        expl = f"AI \u0111\u00e1nh gi\u00e1 b\u1ec7nh nh\u00e2n n\u00e0y c\u00f3 <b>r\u1ee7i ro th\u1ea5p \u0111\u1ed9t qu\u1ef5 ({stroke_prob:.0%})</b>. C\u00e1c ch\u1ec9 s\u1ed1 EEG trong gi\u1edbi h\u1ea1n b\u00ecnh th\u01b0\u1eddng."
    st.markdown(explain_box(expl), unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── EEG FEATURES ──
    st.markdown(
        badge("static", "\U0001f4ca D\u1eef li\u1ec7u EEG t\u1eeb h\u1ed3 s\u01a1 b\u1ec7nh nh\u00e2n \u0111\u01b0\u1ee3c ch\u1ecdn"),
        unsafe_allow_html=True
    )
    st.markdown("#### \U0001f4cb Ch\u1ec9 s\u1ed1 EEG c\u1ee7a B\u1ec7nh nh\u00e2n")
    st.markdown(
        explain_box(
            "C\u00e1c ch\u1ec9 s\u1ed1 d\u01b0\u1edbi \u0111\u00e2y \u0111\u01b0\u1ee3c t\u00ednh to\u00e1n t\u1eeb t\u00edn hi\u1ec7u EEG 1 k\u00eanh (FP1, v\u00f9ng tr\u00e1n). "
            "AI s\u1eed d\u1ee5ng 12 ch\u1ec9 s\u1ed1 n\u00e0y \u0111\u1ec3 \u0111\u01b0a ra d\u1ef1 \u0111o\u00e1n."
        ),
        unsafe_allow_html=True
    )
    feat_rows = []
    for i, fname in enumerate(feature_names):
        feat_rows.append({
            "Ch\u1ec9 s\u1ed1": fname,
            "Gi\u00e1 tr\u1ecb \u0111o \u0111\u01b0\u1ee3c": f"{patient_raw[i]:.4f}",
            "Gi\u00e1 tr\u1ecb chu\u1ea9n h\u00f3a": f"{patient_features[0][i]:.4f}",
            "\u00dd ngh\u0129a": EEG_FEATURE_EXPLAIN.get(fname, "\u2014"),
        })
    with st.expander("\U0001f4ca Xem chi ti\u1ebft 12 ch\u1ec9 s\u1ed1 EEG (click \u0111\u1ec3 m\u1edf)", expanded=False):
        st.dataframe(pd.DataFrame(feat_rows), use_container_width=True, hide_index=True)
        st.caption(
            "Gi\u00e1 tr\u1ecb chu\u1ea9n h\u00f3a [0,1] d\u00f9ng \u0111\u01b0a v\u00e0o m\u00f4 h\u00ecnh AI. "
            "Gi\u00e1 tr\u1ecb \u0111o \u0111\u01b0\u1ee3c l\u00e0 \u0111\u01a1n v\u1ecb g\u1ed1c t\u1eeb ph\u00e2n t\u00edch ph\u1ed5 EEG (\u00b5V\u00b2/Hz)."
        )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── RECOMMENDATIONS ──
    st.markdown(
        badge("live", "\u26a1 Kuy\u1ebfn ngh\u1ecb \u0111\u01b0\u1ee3c t\u1ea1o t\u1ef1 \u0111\u1ed9ng d\u1ef1a tr\u00ean k\u1ebft qu\u1ea3 AI"),
        unsafe_allow_html=True
    )
    st.markdown("#### \U0001f48a Kuy\u1ebfn ngh\u1ecb L\u00e2m s\u00e0ng")

    rec = rec_engine.get_recommendations(stroke_prob, pred_en)
    st.info(f"**H\u00e0nh \u0111\u1ed9ng \u0111\u1ec1 xu\u1ea5t:** {rec['action']}", icon="\U0001f4cb")

    if rec.get('recommended_tests'):
        st.markdown("**C\u00e1c x\u00e9t nghi\u1ec7m \u0111\u01b0\u1ee3c kuy\u1ebfn ngh\u1ecb:**")
        for test in rec['recommended_tests']:
            st.markdown(f"""
            <div class="test-card">
                <div class="test-name">\U0001f539 {test['full_name']}</div>
                <div class="test-desc">{test['description']}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("Kh\u00f4ng c\u1ea7n x\u00e9t nghi\u1ec7m b\u1ed5 sung kh\u1ea9n c\u1ea5p theo k\u1ebft qu\u1ea3 hi\u1ec7n t\u1ea1i.")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── PDF REPORT ──
    st.markdown(
        badge("live", "\u26a1 B\u00e1o c\u00e1o \u0111\u01b0\u1ee3c t\u1ea1o ngay cho b\u1ec7nh nh\u00e2n \u0111ang ch\u1ecdn"),
        unsafe_allow_html=True
    )
    st.markdown("#### \U0001f4c4 T\u1ea3i B\u00e1o c\u00e1o L\u00e2m s\u00e0ng (PDF)")
    st.markdown(
        explain_box(
            "B\u00e1o c\u00e1o PDF bao g\u1ed3m: th\u00f4ng tin b\u1ec7nh nh\u00e2n, k\u1ebft qu\u1ea3 AI, bi\u1ec3u \u0111\u1ed3 gi\u1ea3i th\u00edch LIME "
            "v\u00e0 c\u00e1c kuy\u1ebfn ngh\u1ecb l\u00e2m s\u00e0ng \u2014 ph\u00f9 h\u1ee3p l\u01b0u tr\u1eef h\u1ed3 s\u01a1."
        ),
        unsafe_allow_html=True
    )

    if st.button("\U0001f4c4 T\u1ea1o b\u00e1o c\u00e1o PDF", type="primary"):
        with st.spinner("Đang tạo báo cáo... (khoảng 10-20 giây)"):
            _lime_expl = get_lime_explainer()
            _lime_exp  = _lime_expl.explain_instance(
                patient_features[0], model.predict_proba,
                num_features=len(feature_names), num_samples=5000
            )
            lime_contribs = _lime_exp.as_list()
            _lime_fig = _lime_exp.as_pyplot_figure()
            _lime_fig.set_size_inches(11, 5)
            _lime_tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            _lime_tmp_name = _lime_tmp.name
            _lime_tmp.close()
            _lime_fig.savefig(_lime_tmp_name, dpi=100, bbox_inches='tight')
            plt.close(_lime_fig)

            feat_raw_d  = {fn: float(patient_raw[i]) for i, fn in enumerate(feature_names)}
            feat_norm_d = {fn: float(patient_features[0][i]) for i, fn in enumerate(feature_names)}
            # PDF uses English only (fpdf2 built-in fonts are Latin-1 only)
            true_label_en = "Stroke" if true_label == 1 else "Control"

            pdf_bytes = generate_report(
                patient_id=patient_idx,
                prediction=pred_en,
                stroke_probability=stroke_prob,
                risk_level=risk_level,
                features=feat_raw_d,
                normalized_features=feat_norm_d,
                recommendations=rec,
                lime_contributions=lime_contribs,
                true_label=true_label_en,
                lime_fig_path=_lime_tmp_name
            )
            try:
                os.unlink(_lime_tmp_name)
            except OSError:
                pass
            # Save PDF to disk (reliable) AND to session for download button
            _reports_dir = os.path.join(
                r"d:\Personal\Explainable EEG Stroke DSS", "reports"
            )
            os.makedirs(_reports_dir, exist_ok=True)
            _pdf_filename = f"stroke_report_patient_{patient_idx + 1:02d}.pdf"
            _pdf_path = os.path.join(_reports_dir, _pdf_filename)
            with open(_pdf_path, 'wb') as _f:
                _f.write(pdf_bytes)
            st.session_state['report_pdf']        = pdf_bytes
            st.session_state['report_patient_id'] = patient_idx
            st.session_state['report_saved_path'] = _pdf_path
            st.session_state['report_filename']   = _pdf_filename

    if 'report_pdf' in st.session_state:
        pid       = st.session_state.get('report_patient_id', 0)
        saved_path = st.session_state.get('report_saved_path', '')
        fname      = st.session_state.get('report_filename', f"stroke_report_patient_{pid+1:02d}.pdf")

        st.success(f"\u2705 B\u00e1o c\u00e1o \u0111\u00e3 \u0111\u01b0\u1ee3c t\u1ea1o th\u00e0nh c\u00f4ng!", icon="\U0001f4c4")

        col_dl, col_path = st.columns([1, 2])
        with col_dl:
            st.download_button(
                label="\u2b07\ufe0f Download PDF Report",
                data=st.session_state['report_pdf'],
                file_name=fname,
                mime="application/pdf",
                key="pdf_dl_btn"
            )
        with col_path:
            if saved_path:
                st.info(
                    f"\U0001f4c1 **\u0110\u01b0\u1eddng d\u1eabn file tr\u00ean m\u00e1y t\u00ednh:**\n\n`{saved_path}`",
                    icon="\U0001f4be"
                )
                st.caption(
                    "N\u1ebfu n\u00fat t\u1ea3i kh\u00f4ng ho\u1ea1t \u0111\u1ed9ng, h\u00e3y m\u1edf th\u01b0 m\u1ee5c **reports\\** trong th\u01b0 m\u1ee5c d\u1ef1 \u00e1n."
                )



# ============================================================
# PAGE: TAI SAO AI DUA RA KET QUA?  (Explainability)
# ============================================================
elif page == PAGE_EXPLAIN:
    st.markdown("""
    <div class="main-header">
        <h1>\U0001f4a1 T\u1ea1i sao AI \u0111\u01b0a ra k\u1ebft qu\u1ea3 \u0111\u00f3?</h1>
        <p>Xem chi ti\u1ebft nh\u1eefng y\u1ebfu t\u1ed1 EEG n\u00e0o \u1ea3nh h\u01b0\u1edfng \u0111\u1ebfn quy\u1ebft \u0111\u1ecbnh c\u1ee7a AI,
           c\u1ea3 to\u00e0n b\u1ed9 d\u1eef li\u1ec7u l\u1eabn t\u1eebng b\u1ec7nh nh\u00e2n c\u1ee5 th\u1ec3</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="guide-card">
        <h4>\U0001f4d6 Trang n\u00e0y d\u00f9ng \u0111\u1ec3 l\u00e0m g\u00ec?</h4>
        <p>
        AI kh\u00f4ng ph\u1ea3i "h\u1ed9p \u0111en" \u2014 trang n\u00e0y gi\u00fap b\u1ea1n hi\u1ec3u <b>t\u1ea1i sao</b> AI \u0111\u01b0a ra m\u1ed9t k\u1ebft qu\u1ea3 c\u1ee5 th\u1ec3.<br>
        \u2022 <b>SHAP</b>: Nh\u00ecn t\u1ed5ng th\u1ec3 \u2014 ch\u1ec9 s\u1ed1 n\u00e0o quan tr\u1ecdng nh\u1ea5t tr\u00ean to\u00e0n b\u1ed9 38 b\u1ec7nh nh\u00e2n?<br>
        \u2022 <b>LIME</b>: Nh\u00ecn c\u00e1 nh\u00e2n \u2014 ch\u1ec9 s\u1ed1 n\u00e0o \u0111\u00e3 d\u1eabn AI \u0111\u1ebfn k\u1ebft qu\u1ea3 cho <i>b\u1ec7nh nh\u00e2n n\u00e0y c\u1ee5 th\u1ec3</i>?
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab_shap, tab_lime = st.tabs([
        "\U0001f310 SHAP \u2014 T\u1ea7m quan tr\u1ecdng t\u1ed5ng th\u1ec3",
        "\U0001f50d LIME \u2014 Gi\u1ea3i th\u00edch t\u1eebng b\u1ec7nh nh\u00e2n"
    ])

    # ── SHAP TAB ──
    with tab_shap:
        st.markdown(
            badge("live", "\u26a1 T\u00ednh to\u00e1n TR\u1ef0C TI\u1ebeP t\u1eeb m\u00f4 h\u00ecnh AI (l\u1ea7n \u0111\u1ea7u ~1-2 ph\u00fat)"),
            unsafe_allow_html=True
        )
        st.markdown("### \U0001f310 Ch\u1ec9 s\u1ed1 EEG n\u00e0o quan tr\u1ecdng nh\u1ea5t v\u1edbi AI?")
        st.markdown(
            explain_box(
                "SHAP \u0111o l\u01b0\u1eddng m\u1ee9c \u0111\u1ed9 \u0111\u00f3ng g\u00f3p c\u1ee7a t\u1eebng ch\u1ec9 s\u1ed1 EEG v\u00e0o quy\u1ebft \u0111\u1ecbnh c\u1ee7a AI. "
                "Ch\u1ec9 s\u1ed1 c\u00f3 thanh d\u00e0i h\u01a1n = \u1ea3nh h\u01b0\u1edfng nhi\u1ec1u h\u01a1n \u0111\u1ebfn k\u1ebft qu\u1ea3 d\u1ef1 \u0111o\u00e1n."
            ),
            unsafe_allow_html=True
        )

        shap_computed = False
        try:
            shap_values_all, shap_base_value = compute_all_shap_values()
            shap_computed = True
        except Exception as e:
            st.warning(f"\u26a0\ufe0f Kh\u00f4ng th\u1ec3 t\u00ednh SHAP: {e}")

        if shap_computed:
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                fig_summary = plt.figure(figsize=(8, 6))
                shap.summary_plot(shap_values_all, X_norm, feature_names=feature_names, show=False)
                plt.title("Ph\u00e2n b\u1ed1 \u1ea3nh h\u01b0\u1edfng t\u1eebng ch\u1ec9 s\u1ed1 (SHAP)", fontsize=11)
                plt.tight_layout()
                st.pyplot(fig_summary)
                plt.close('all')
                st.caption("\U0001f534 \u0110\u1ecf = ch\u1ec9 s\u1ed1 cao \u2192 t\u0103ng nguy c\u01a1 \u0111\u1ed9t qu\u1ef5  |  \U0001f535 Xanh = ch\u1ec9 s\u1ed1 th\u1ea5p \u2192 gi\u1ea3m nguy c\u01a1")

            with col_s2:
                mean_abs  = np.abs(shap_values_all).mean(axis=0)
                sorted_idx = np.argsort(mean_abs)
                fig_bar, ax_bar = plt.subplots(figsize=(8, 6))
                median_val = np.median(mean_abs)
                colors_bar = ['#2d6a9f' if mean_abs[i] >= median_val else '#bee3f8' for i in sorted_idx]
                ax_bar.barh(range(len(feature_names)), mean_abs[sorted_idx], color=colors_bar, height=0.65)
                ax_bar.set_yticks(range(len(feature_names)))
                ax_bar.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=10)
                ax_bar.set_xlabel("M\u1ee9c \u0111\u1ed9 \u1ea3nh h\u01b0\u1edfng trung b\u00ecnh", fontsize=10)
                ax_bar.set_title("X\u1ebfp h\u1ea1ng t\u1ea7m quan tr\u1ecdng c\u00e1c ch\u1ec9 s\u1ed1", fontsize=11)
                ax_bar.spines['top'].set_visible(False)
                ax_bar.spines['right'].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig_bar)
                plt.close()
                st.caption("\U0001f535 \u0110\u1eadm = quan tr\u1ecdng h\u01a1n trung b\u00ecnh  |  \U0001f535 Nh\u1ea1t = \u00edt quan tr\u1ecdng h\u01a1n")

            # Top 5 table
            st.markdown("#### Top 5 ch\u1ec9 s\u1ed1 \u1ea3nh h\u01b0\u1edfng nhi\u1ec1u nh\u1ea5t")
            top5_idx = np.argsort(mean_abs)[::-1][:5]
            top5_rows = []
            for rank, idx in enumerate(top5_idx, 1):
                top5_rows.append({
                    "H\u1ea1ng": f"#{rank}",
                    "Ch\u1ec9 s\u1ed1": feature_names[idx],
                    "M\u1ee9c \u1ea3nh h\u01b0\u1edfng": f"{mean_abs[idx]:.4f}",
                    "\u00dd ngh\u0129a l\u00e2m s\u00e0ng": EEG_FEATURE_EXPLAIN.get(feature_names[idx], "\u2014"),
                })
            st.dataframe(pd.DataFrame(top5_rows), use_container_width=True, hide_index=True)

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

            # Per-patient waterfall
            st.markdown(
                badge("live", "\u26a1 Ch\u1ecdn b\u1ec7nh nh\u00e2n \u0111\u1ec3 xem ph\u00e2n t\u00edch ri\u00eang"),
                unsafe_allow_html=True
            )
            st.markdown("#### \U0001f3af Ph\u00e2n t\u00edch cho t\u1eebng b\u1ec7nh nh\u00e2n c\u1ee5 th\u1ec3")
            st.markdown(
                explain_box(
                    "D\u01b0\u1edbi \u0111\u00e2y cho th\u1ea5y: v\u1edbi b\u1ec7nh nh\u00e2n \u0111\u01b0\u1ee3c ch\u1ecdn, "
                    "ch\u1ec9 s\u1ed1 n\u00e0o \u0111\u00e3 \u0111\u1ea9y AI v\u1ec1 ph\u00eda '\u0110\u1ed9t qu\u1ef5' (\U0001f534) "
                    "hay '\u0110B\u00ecnh th\u01b0\u1eddng' (\U0001f535)."
                ),
                unsafe_allow_html=True
            )

            patient_shap = st.selectbox(
                "Ch\u1ecdn b\u1ec7nh nh\u00e2n \u0111\u1ec3 ph\u00e2n t\u00edch SHAP",
                range(len(y)),
                format_func=lambda i: f"B\u1ec7nh nh\u00e2n #{i+1:02d} \u2014 {'\U0001f534 \u0110\u1ed9t qu\u1ef5' if y[i]==1 else '\U0001f7e2 B\u00ecnh th\u01b0\u1eddng'}",
                key="shap_patient"
            )

            sv_patient   = shap_values_all[patient_shap]
            sorted_idx_p = np.argsort(np.abs(sv_patient))
            fig_wf, ax_wf = plt.subplots(figsize=(10, 6))
            vals_wf  = sv_patient[sorted_idx_p]
            names_wf = [feature_names[i] for i in sorted_idx_p]
            colors_wf = ['#c53030' if v > 0 else '#2d6a9f' for v in vals_wf]

            ax_wf.barh(range(len(names_wf)), vals_wf, color=colors_wf, height=0.65, edgecolor='white', linewidth=0.5)
            ax_wf.set_yticks(range(len(names_wf)))
            ax_wf.set_yticklabels(names_wf, fontsize=10)
            ax_wf.set_xlabel("M\u1ee9c \u0111\u1ed3ng g\u00f3p v\u00e0o quy\u1ebft \u0111\u1ecbnh (+ = \u0111\u1ea9y v\u1ec1 \u0110\u1ed9t qu\u1ef5, - = \u0111\u1ea9y v\u1ec1 B\u00ecnh th\u01b0\u1eddng)", fontsize=10)
            ax_wf.axvline(x=0, color='black', linewidth=0.8, alpha=0.5)
            ax_wf.set_title(f"B\u1ec7nh nh\u00e2n #{patient_shap+1} \u2014 \u0110\u00f3ng g\u00f3p t\u1eebng ch\u1ec9 s\u1ed1 EEG", fontsize=12, fontweight='bold')
            for i, v in enumerate(vals_wf):
                offset = max(abs(v) * 0.08, 0.002)
                ax_wf.text(v + (offset if v >= 0 else -offset), i, f'{v:+.4f}',
                           ha='left' if v >= 0 else 'right', va='center', fontsize=9)
            ax_wf.spines['top'].set_visible(False)
            ax_wf.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig_wf)
            plt.close()

            col_leg1, col_leg2, col_leg3 = st.columns(3)
            final_score = shap_base_value + sv_patient.sum()
            with col_leg1:
                st.markdown("\U0001f534 **Thanh \u0111\u1ecf** \u2192 \u0110\u1ea9y v\u1ec1 **\u0110\u1ed9t qu\u1ef5**")
            with col_leg2:
                st.markdown("\U0001f535 **Thanh xanh** \u2192 \u0110\u1ea9y v\u1ec1 **B\u00ecnh th\u01b0\u1eddng**")
            with col_leg3:
                pred_lbl = "\u0110\u1ed9t qu\u1ef5" if final_score > 0.5 else "B\u00ecnh th\u01b0\u1eddng"
                st.markdown(f"\U0001f3af **K\u1ebft qu\u1ea3 cu\u1ed1i:** {pred_lbl} ({final_score:.2f})")
        else:
            st.info("Xem \u1ea3nh k\u1ebft qu\u1ea3 t\u1eeb nghi\u00ean c\u1ee9u t\u1ea1i trang **Th\u00f4ng tin H\u1ec7 th\u1ed1ng**.")

    # ── LIME TAB ──
    with tab_lime:
        st.markdown(
            badge("live", "\u26a1 T\u00ednh to\u00e1n TR\u1ef0C TI\u1ebeP \u2014 b\u1ea5m n\u00fat \u0111\u1ec3 b\u1eaft \u0111\u1ea7u (~5 gi\u00e2y)"),
            unsafe_allow_html=True
        )
        st.markdown("### \U0001f50d Gi\u1ea3i th\u00edch ri\u00eang cho t\u1eebng b\u1ec7nh nh\u00e2n (LIME)")
        st.markdown(
            explain_box(
                "LIME ph\u00e2n t\u00edch xung quanh b\u1ec7nh nh\u00e2n \u0111\u01b0\u1ee3c ch\u1ecdn \u0111\u1ec3 t\u00ecm ra: "
                "v\u1edbi <i>b\u1ec7nh nh\u00e2n n\u00e0y c\u1ee5 th\u1ec3</i>, nh\u1eefng \u0111i\u1ec1u ki\u1ec7n EEG n\u00e0o \u0111\u00e3 "
                "d\u1eabn AI \u0111\u1ebfn quy\u1ebft \u0111\u1ecbnh \u0111\u00f3. K\u1ebft qu\u1ea3 s\u1ebd kh\u00e1c nhau cho t\u1eebng b\u1ec7nh nh\u00e2n."
            ),
            unsafe_allow_html=True
        )

        patient_lime = st.selectbox(
            "Ch\u1ecdn b\u1ec7nh nh\u00e2n \u0111\u1ec3 gi\u1ea3i th\u00edch LIME",
            range(len(y)),
            format_func=lambda i: f"B\u1ec7nh nh\u00e2n #{i+1:02d} \u2014 {'\U0001f534 \u0110\u1ed9t qu\u1ef5' if y[i]==1 else '\U0001f7e2 B\u00ecnh th\u01b0\u1eddng'}",
            key="lime_patient"
        )

        if st.button("\U0001f50d Gi\u1ea3i th\u00edch cho b\u1ec7nh nh\u00e2n n\u00e0y", type="primary"):
            with st.spinner("Đang phân tích (~5-10 giây)..."):
                _lime_expl = get_lime_explainer()
                _exp   = _lime_expl.explain_instance(
                    X_norm[patient_lime], model.predict_proba,
                    num_features=len(feature_names), num_samples=5000
                )
                _proba = model.predict_proba(X_norm[patient_lime:patient_lime+1])[0]
                _fig   = _exp.as_pyplot_figure()
                _fig.set_size_inches(12, 6)
                _buf   = io.BytesIO()
                _fig.savefig(_buf, format='png', dpi=120, bbox_inches='tight')
                plt.close(_fig)
                st.session_state['lime_result'] = {
                    'exp_list':    _exp.as_list(),
                    'patient_idx': patient_lime,
                    'proba':       _proba.tolist(),
                    'fig_bytes':   _buf.getvalue(),
                }

        if 'lime_result' in st.session_state:
            lr = st.session_state['lime_result']
            if lr['patient_idx'] != patient_lime:
                st.info("\U0001f4cc B\u1ec7nh nh\u00e2n \u0111\u00e3 thay \u0111\u1ed5i. B\u1ea5m **Gi\u1ea3i th\u00edch cho b\u1ec7nh nh\u00e2n n\u00e0y** \u0111\u1ec3 c\u1eadp nh\u1eadt.")

            proba_arr = np.array(lr['proba'])
            pred_cls  = "\u0110\u1ed9t qu\u1ef5" if np.argmax(proba_arr) == 1 else "B\u00ecnh th\u01b0\u1eddng"
            lr_pidx   = lr['patient_idx']

            col_lp, col_lr, col_lt = st.columns(3)
            with col_lp:
                if pred_cls == "\u0110\u1ed9t qu\u1ef5":
                    st.error(f"**K\u1ebft qu\u1ea3 AI: {pred_cls}**", icon="\u26a0\ufe0f")
                else:
                    st.success(f"**K\u1ebft qu\u1ea3 AI: {pred_cls}**", icon="\u2705")
            with col_lr:
                st.metric("X\u00e1c su\u1ea5t \u0111\u1ed9t qu\u1ef5", f"{proba_arr[1]:.1%}")
            with col_lt:
                true_cls  = '\u0110\u1ed9t qu\u1ef5' if y[lr_pidx] == 1 else 'B\u00ecnh th\u01b0\u1eddng'
                match_ico = "\u2705" if pred_cls == true_cls else "\u274c"
                st.metric("Ch\u1ea9n \u0111o\u00e1n th\u1ef1c t\u1ebf", f"{true_cls} {match_ico}")

            st.image(
                lr['fig_bytes'],
                caption=f"Bi\u1ec3u \u0111\u1ed3 LIME \u2014 B\u1ec7nh nh\u00e2n #{lr_pidx+1}: c\u00e1c \u0111i\u1ec1u ki\u1ec7n EEG d\u1eabn \u0111\u1ebfn quy\u1ebft \u0111\u1ecbnh c\u1ee7a AI",
                use_column_width=True
            )

            top_stroke  = sorted([(n, w) for n, w in lr['exp_list'] if w > 0], key=lambda x: -x[1])
            top_ctrl    = sorted([(n, w) for n, w in lr['exp_list'] if w < 0], key=lambda x: x[1])

            st.markdown("#### \U0001f4ac Gi\u1ea3i th\u00edch b\u1eb1ng ng\u00f4n ng\u1eef l\u00e2m s\u00e0ng")
            col_ts, col_tc = st.columns(2)
            with col_ts:
                st.markdown("**\U0001f534 Y\u1ebfu t\u1ed1 cho th\u1ea5y nguy c\u01a1 \u0110\u1ed9t qu\u1ef5:**")
                for name, w in top_stroke[:4]:
                    st.markdown(f"- {name} _(\u1ea3nh h\u01b0\u1edfng: {abs(w):.4f})_")
                if not top_stroke:
                    st.markdown("- *(Kh\u00f4ng c\u00f3)*")
            with col_tc:
                st.markdown("**\U0001f7e2 Y\u1ebfu t\u1ed1 cho th\u1ea5y B\u00ecnh th\u01b0\u1eddng:**")
                for name, w in top_ctrl[:4]:
                    st.markdown(f"- {name} _(\u1ea3nh h\u01b0\u1edfng: {abs(w):.4f})_")
                if not top_ctrl:
                    st.markdown("- *(Kh\u00f4ng c\u00f3)*")

            with st.expander("\U0001f4cb B\u1ea3ng chi ti\u1ebft t\u1ea5t c\u1ea3 y\u1ebfu t\u1ed1"):
                cont_df = pd.DataFrame(lr['exp_list'], columns=['\u0110i\u1ec1u ki\u1ec7n EEG', 'Tr\u1ecdng s\u1ed1'])
                cont_df['Chi\u1ec1u h\u01b0\u1edbng'] = cont_df['Tr\u1ecdng s\u1ed1'].apply(
                    lambda w: '\U0001f534 \u2192 \u0110\u1ed9t qu\u1ef5' if w > 0 else '\U0001f7e2 \u2192 B\u00ecnh th\u01b0\u1eddng'
                )
                cont_df['M\u1ee9c \u1ea3nh h\u01b0\u1edfng'] = cont_df['Tr\u1ecdng s\u1ed1'].abs()
                cont_df = cont_df.sort_values('M\u1ee9c \u1ea3nh h\u01b0\u1edfng', ascending=False)
                st.dataframe(cont_df, use_container_width=True, hide_index=True)
        else:
            st.info(
                "\U0001f446 Ch\u1ecdn b\u1ec7nh nh\u00e2n r\u1ed3i b\u1ea5m **Gi\u1ea3i th\u00edch cho b\u1ec7nh nh\u00e2n n\u00e0y** \u0111\u1ec3 xem k\u1ebft qu\u1ea3."
            )


# ============================================================
# PAGE: THONG TIN HE THONG  (System Overview)
# ============================================================
elif page == PAGE_OVERVIEW:
    st.markdown("""
    <div class="main-header">
        <h1>\U0001f4ca Th\u00f4ng tin & Hi\u1ec7u su\u1ea5t H\u1ec7 th\u1ed1ng</h1>
        <p>T\u1ed5ng quan v\u1ec1 AI, d\u1eef li\u1ec7u nghi\u00ean c\u1ee9u, v\u00e0 c\u00e1c s\u1ed1 li\u1ec7u \u0111\u00e1nh gi\u00e1 hi\u1ec7u su\u1ea5t</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="guide-card">
        <h4>\U0001f52c \u0110\u00e2y l\u00e0 g\u00ec?</h4>
        <p>
        H\u1ec7 th\u1ed1ng s\u1eed d\u1ee5ng AI (m\u1ea1ng n\u01a1-ron ESN) \u0111\u1ec3 ph\u00e2n t\u00edch t\u00edn hi\u1ec7u n\u00e3o (EEG) nh\u1eb1m
        ph\u00e1t hi\u1ec7n nguy c\u01a1 \u0111\u1ed9t qu\u1ef5 s\u1edbm. To\u00e0n b\u1ed9 s\u1ed1 li\u1ec7u tr\u00ean trang n\u00e0y l\u00e0 k\u1ebft qu\u1ea3
        \u0111\u00e1nh gi\u00e1 tr\u00ean <b>38 b\u1ec7nh nh\u00e2n</b> trong nghi\u00ean c\u1ee9u g\u1ed1c (Bouazizi & Ltifi, 2024).
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        badge("static", "\U0001f4ca S\u1ed1 li\u1ec7u t\u0129nh \u2014 K\u1ebft qu\u1ea3 \u0111\u00e1nh gi\u00e1 t\u1eeb 38 b\u1ec7nh nh\u00e2n nghi\u00ean c\u1ee9u"),
        unsafe_allow_html=True
    )

    # Metrics
    st.markdown("#### \U0001f3af Hi\u1ec7u su\u1ea5t AI tr\u00ean b\u1ed9 d\u1eef li\u1ec7u nghi\u00ean c\u1ee9u")
    st.caption(
        "Ph\u01b0\u01a1ng ph\u00e1p ki\u1ec3m tra: LOOCV (Leave-One-Out Cross-Validation) \u2014 "
        "l\u1ea7n l\u01b0\u1ee3t d\u00f9ng 37 BN hu\u1ea5n luy\u1ec7n, 1 BN ki\u1ec3m tra, l\u1eb7p 38 l\u1ea7n"
    )

    metrics = compute_loocv_metrics()

    metric_items = [
        ("\u0110\u1ed9 ch\u00ednh x\u00e1c\nt\u1ed5ng th\u1ec3", metrics['accuracy'],  "D\u1ef1 \u0111o\u00e1n \u0111\u00fang bao nhi\u00eau % t\u1ed5ng s\u1ed1 BN"),
        ("Ph\u00e1t hi\u1ec7n\n\u0110\u1ed9t qu\u1ef5",        metrics['sensitivity'], "Ph\u00e1t hi\u1ec7n \u0111\u00fang bao nhi\u00eau % ca \u0111\u1ed9t qu\u1ef5"),
        ("X\u00e1c nh\u1eadn\nB\u00ecnh th\u01b0\u1eddng",       metrics['specificity'], "X\u00e1c nh\u1eadn \u0111\u00fang bao nhi\u00eau % ca b\u00ecnh th\u01b0\u1eddng"),
        ("D\u01b0\u01a1ng t\u00ednh\n\u0111\u00fang (PPV)",     metrics['ppv'],         "Khi b\u00e1o \u0110\u1ed9t qu\u1ef5: x\u00e1c su\u1ea5t \u0111\u00fang"),
        ("\u00c2m t\u00ednh\n\u0111\u00fang (NPV)",       metrics['npv'],         "Khi b\u00e1o B\u00ecnh th\u01b0\u1eddng: x\u00e1c su\u1ea5t \u0111\u00fang"),
        ("Th\u1eb9m F1\n(c\u00e2n b\u1eb1ng)",        metrics['f1'],          "Ch\u1ec9 s\u1ed1 k\u1ebft h\u1ee3p t\u1ed5ng th\u1ec3"),
    ]

    cols = st.columns(6)
    for col, (label, value, sub) in zip(cols, metric_items):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{value:.1%}</div>
                <div class="metric-label">{label}</div>
                <div class="metric-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    left_col, right_col = st.columns(2)
    with left_col:
        st.markdown("#### \U0001f4cb Ma tr\u1eadn k\u1ebft qu\u1ea3 d\u1ef1 \u0111o\u00e1n")
        st.caption("Hi\u1ec3n th\u1ecb s\u1ed1 l\u01b0\u1ee3ng d\u1ef1 \u0111o\u00e1n \u0111\u00fang/sai tr\u00ean 38 b\u1ec7nh nh\u00e2n nghi\u00ean c\u1ee9u")
        cm = metrics['confusion_matrix']
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        im = ax_cm.imshow(cm, cmap='Blues')
        for i in range(2):
            for j in range(2):
                color  = 'white' if cm[i, j] > cm.max()/2 else 'black'
                label_cm = "\u0110\u00fang \u2705" if i == j else "Sai \u274c"
                ax_cm.text(j, i, f"{cm[i,j]}\n({label_cm})", ha='center', va='center',
                           fontsize=13, fontweight='bold', color=color)
        ax_cm.set_xticks([0, 1])
        ax_cm.set_yticks([0, 1])
        ax_cm.set_xticklabels(['D\u1ef1 \u0111o\u00e1n:\nB\u00ecnh th\u01b0\u1eddng', 'D\u1ef1 \u0111o\u00e1n:\n\u0110\u1ed9t qu\u1ef5'], fontsize=10)
        ax_cm.set_yticklabels(['Th\u1ef1c t\u1ebf:\nB\u00ecnh th\u01b0\u1eddng', 'Th\u1ef1c t\u1ebf:\n\u0110\u1ed9t qu\u1ef5'], fontsize=10)
        ax_cm.set_title('Ma tr\u1eadn K\u1ebft qu\u1ea3 D\u1ef1 \u0111o\u00e1n (LOOCV)', fontsize=11, fontweight='bold')
        plt.colorbar(im, ax=ax_cm, shrink=0.8)
        plt.tight_layout()
        st.pyplot(fig_cm)
        plt.close()

    with right_col:
        st.markdown("#### \U0001f4cb Th\u1ed1ng k\u00ea d\u1eef li\u1ec7u nghi\u00ean c\u1ee9u")
        st.markdown(f"""
        | Th\u00f4ng tin | Gi\u00e1 tr\u1ecb |
        |-----------|---------|
        | T\u1ed5ng s\u1ed1 b\u1ec7nh nh\u00e2n | {len(y)} ng\u01b0\u1eddi |
        | B\u1ec7nh nh\u00e2n \u0110\u1ed9t qu\u1ef5 | {np.sum(y==1)} ng\u01b0\u1eddi |
        | B\u1ec7nh nh\u00e2n B\u00ecnh th\u01b0\u1eddng | {np.sum(y==0)} ng\u01b0\u1eddi |
        | S\u1ed1 ch\u1ec9 s\u1ed1 EEG ph\u00e2n t\u00edch | {len(feature_names)} ch\u1ec9 s\u1ed1 |
        | \u0110i\u1ec7n c\u1ef1c \u0111o | FP1 (v\u00f9ng tr\u00e1n tr\u01b0\u1edbc) |
        | Th\u1eddi \u0111i\u1ec3m \u0111o | Trong v\u00f2ng 72h sau nh\u1eadp vi\u1ec7n |
        | Tr\u1ea1ng th\u00e1i khi \u0111o | Ngh\u1ec9 ng\u01a1i, m\u1eaft nh\u1eafm |
        | Ngu\u1ed3n d\u1eef li\u1ec7u | Aminov et al. (2017) |
        """)

        st.markdown("#### \U0001f9e0 V\u1ec1 m\u00f4 h\u00ecnh AI")
        st.markdown("""
        | Th\u00e0nh ph\u1ea7n | Chi ti\u1ebft |
        |------------|----------|
        | Lo\u1ea1i AI | Echo State Network (ESN) |
        | S\u1ed1 m\u00f4 h\u00ecnh k\u1ebft h\u1ee3p | 7 ESN (b\u1ea7u ch\u1ecdn \u0111a s\u1ed1) |
        | S\u1ed1 n\u01a1-ron m\u1ed7i ESN | 200 n\u01a1-ron |
        | Ph\u01b0\u01a1ng ph\u00e1p ki\u1ec3m tra | LOOCV (38 l\u1ea7n) |
        """)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown(
        badge("static", "\U0001f4ca Bi\u1ec3u \u0111\u1ed3 t\u1eeb k\u1ebft qu\u1ea3 nghi\u00ean c\u1ee9u \u0111\u00e3 ch\u1ea1y tr\u01b0\u1edbc"),
        unsafe_allow_html=True
    )
    st.markdown("#### \U0001f4c8 \u0110\u01b0\u1eddng cong ROC & So s\u00e1nh v\u1edbi c\u00e1c AI kh\u00e1c")
    st.markdown(
        explain_box(
            "\u0110\u01b0\u1eddng cong ROC cho th\u1ea5y kh\u1ea3 n\u0103ng ph\u00e2n bi\u1ec7t \u0111\u1ed9t qu\u1ef5/b\u00ecnh th\u01b0\u1eddng. "
            "\u0110\u01b0\u1eddng cong c\u00e0ng 'ph\u00ecnh' v\u1ec1 g\u00f3c tr\u00ean tr\u00e1i \u2192 AI c\u00e0ng t\u1ed1t. "
            "AUC = 1.0 l\u00e0 ho\u00e0n h\u1ea3o, AUC = 0.5 l\u00e0 t\u01b0\u01a1ng \u0111\u01b0\u01a1ng \u0111o\u00e1n ng\u1eabu nhi\u00ean."
        ),
        unsafe_allow_html=True
    )

    col_roc, col_comp = st.columns(2)
    roc_path  = os.path.join(FIG_DIR, "roc_curves.png")
    comp_path = os.path.join(FIG_DIR, "model_comparison.png")
    if os.path.exists(roc_path):
        with col_roc:
            st.image(roc_path, caption="\u0110\u01b0\u1eddng cong ROC \u2014 So s\u00e1nh c\u00e1c m\u00f4 h\u00ecnh AI (LOOCV)", use_column_width=True)
    if os.path.exists(comp_path):
        with col_comp:
            st.image(comp_path, caption="So s\u00e1nh hi\u1ec7u su\u1ea5t c\u00e1c m\u00f4 h\u00ecnh AI tr\u00ean c\u00f9ng d\u1eef li\u1ec7u", use_column_width=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### \U0001f50d Ch\u1ec9 s\u1ed1 EEG n\u00e0o \u0111\u01b0\u1ee3c AI coi tr\u1ecdng nh\u1ea5t?")
    fs_path = os.path.join(FIG_DIR, "feature_selection.png")
    if os.path.exists(fs_path):
        st.image(fs_path, caption="\u0110i\u1ec3m t\u1ea7m quan tr\u1ecdng c\u1ee7a t\u1eebng ch\u1ec9 s\u1ed1 (ANOVA + Mutual Information)", use_column_width=True)

    st.markdown("""
    > **4 ch\u1ec9 s\u1ed1 quan tr\u1ecdng nh\u1ea5t** (\u0111\u01b0\u1ee3c t\u00f4 s\u00e1ng):
    > - **DAR** \u2014 T\u1ec9 s\u1ed1 (Delta+Theta)/(Alpha+Beta): nh\u1ea1y nh\u1ea5t v\u1edbi t\u1ed5n th\u01b0\u01a1ng n\u00e3o
    > - **Epoch** \u2014 S\u1ed1 \u0111o\u1ea1n EEG \u0111\u01b0\u1ee3c ph\u00e2n t\u00edch
    > - **RP Delta** \u2014 C\u00f4ng su\u1ea5t t\u01b0\u01a1ng \u0111\u1ed1i s\u00f3ng Delta: t\u0103ng cao khi c\u00f3 b\u1ea5t th\u01b0\u1eddng n\u00e3o
    > - **RP Theta** \u2014 C\u00f4ng su\u1ea5t t\u01b0\u01a1ng \u0111\u1ed1i s\u00f3ng Theta: li\u00ean quan \u0111\u1ebfn thi\u1ebfu m\u00e1u n\u00e3o
    """)


# ============================================================
# PAGE: DANH GIA CHUYEN SAU (Researchers only)
# ============================================================
elif page == PAGE_EVAL:
    st.markdown("""
    <div class="main-header">
        <h1>\U0001f4c8 \u0110\u00e1nh gi\u00e1 Chuy\u00ean s\u00e2u \u2014 T\u00e1i hi\u1ec7n Nghi\u00ean c\u1ee9u</h1>
        <p>So s\u00e1nh to\u00e0n di\u1ec7n t\u1ea5t c\u1ea3 m\u00f4 h\u00ecnh \u00d7 ph\u01b0\u01a1ng ph\u00e1p ki\u1ec3m tra \u00d7 b\u1ed9 features \u2014
           theo \u0111\u00fang giao th\u1ee9c b\u00e0i b\u00e1o Bouazizi & Ltifi (2024)</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        badge("research", "\U0001f52c D\u00e0nh cho Nh\u00e0 nghi\u00ean c\u1ee9u \u2014 S\u1ed1 li\u1ec7u khoa h\u1ecdc chuy\u00ean s\u00e2u"),
        unsafe_allow_html=True
    )
    st.markdown("""
    <div class="guide-card">
        <h4>\U0001f4d6 Trang n\u00e0y d\u00f9ng \u0111\u1ec3 l\u00e0m g\u00ec?</h4>
        <p>
        So s\u00e1nh \u0111\u1ea7y \u0111\u1ee7 10 m\u00f4 h\u00ecnh AI \u00d7 3 ph\u01b0\u01a1ng ph\u00e1p ki\u1ec3m tra ch\u00e9o (LOOCV, 5-Fold, 10-Fold)
        \u00d7 2 b\u1ed9 ch\u1ec9 s\u1ed1 (12 EEG / 15 EEG+nh\u00e2n kh\u1ea9u h\u1ecdc) \u2014 ph\u1ee5c v\u1ee5 nghi\u00ean c\u1ee9u v\u00e0 t\u00e1i hi\u1ec7n b\u00e0i b\u00e1o.
        </p>
    </div>
    """, unsafe_allow_html=True)

    eval_data = load_full_evaluation()

    if eval_data is None:
        st.error("Ch\u01b0a c\u00f3 k\u1ebft qu\u1ea3 \u0111\u00e1nh gi\u00e1. Ch\u1ea1y: `python scripts/run_full_evaluation.py`")
    else:
        # Feature distributions
        st.markdown("### \U0001f4ca Ph\u00e2n b\u1ed1 Ch\u1ec9 s\u1ed1 EEG theo Nh\u00f3m b\u1ec7nh nh\u00e2n")
        dist_path = os.path.join(FIG_DIR, "feature_distributions.png")
        if os.path.exists(dist_path):
            st.image(dist_path, use_column_width=True)
            st.caption(
                "Boxplot ph\u00e2n b\u1ed1 t\u1eebng ch\u1ec9 s\u1ed1 EEG: "
                "\U0001f535 Xanh = B\u1ec7nh nh\u00e2n b\u00ecnh th\u01b0\u1eddng  |  \U0001f534 \u0110\u1ecf = B\u1ec7nh nh\u00e2n \u0111\u1ed9t qu\u1ef5."
            )

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # Comprehensive table
        st.markdown("### \U0001f4cb B\u1ea3ng so s\u00e1nh: T\u1ea5t c\u1ea3 m\u00f4 h\u00ecnh \u00d7 Ph\u01b0\u01a1ng ph\u00e1p ki\u1ec3m tra")

        with st.expander("\U0001f4d6 Gi\u1ea3i th\u00edch c\u00e1c thu\u1eadt ng\u1eef trong b\u1ea3ng"):
            st.markdown("""
            | Thu\u1eadt ng\u1eef | \u00dd ngh\u0129a |
            |-----------|---------|
            | **LOOCV** | Leave-One-Out CV: D\u00f9ng 37 BN train, 1 BN test, l\u1eb7p 38 l\u1ea7n |
            | **5-Fold CV** | Chia 38 BN th\u00e0nh 5 nh\u00f3m, l\u1ea7n l\u01b0\u1ee3t ki\u1ec3m tra |
            | **10-Fold CV** | Chia 38 BN th\u00e0nh 10 nh\u00f3m, l\u1ea7n l\u01b0\u1ee3t ki\u1ec3m tra |
            | **ESN** | Echo State Network \u2014 1 m\u1ea1ng n\u01a1-ron \u0111\u01a1n |
            | **E-ESN (7)** | Ensemble ESN \u2014 K\u1ebft h\u1ee3p 7 ESN (m\u00f4 h\u00ecnh \u0111\u1ec1 xu\u1ea5t trong b\u00e0i b\u00e1o) |
            | **SVM** | Support Vector Machine |
            | **RF** | Random Forest |
            | **Sensitivity** | T\u1ef7 l\u1ec7 ph\u00e1t hi\u1ec7n \u0111\u00fang ca \u0111\u1ed9t qu\u1ef5 |
            | **Specificity** | T\u1ef7 l\u1ec7 x\u00e1c nh\u1eadn \u0111\u00fang ca b\u00ecnh th\u01b0\u1eddng |
            """)

        fs_tab_labels = list(eval_data['models'].keys())
        fs_tabs = st.tabs([
            "EEG Only (12 ch\u1ec9 s\u1ed1)",
            "To\u00e0n b\u1ed9 (15 ch\u1ec9 s\u1ed1, bao g\u1ed3m tu\u1ed5i/gi\u1edbi)"
        ])

        for tab, fs_name in zip(fs_tabs, fs_tab_labels):
            with tab:
                rows = []
                for cv_name in ['LOOCV', '5-Fold', '10-Fold']:
                    cv_data = eval_data['models'][fs_name].get(cv_name, {})
                    for model_name, m in cv_data.items():
                        rows.append({
                            'M\u00f4 h\u00ecnh': model_name,
                            'Ph\u01b0\u01a1ng ph\u00e1p CV': cv_name,
                            'Accuracy': f"{m['accuracy']:.1%}",
                            'F1': f"{m['f1']:.1%}",
                            'Sensitivity': f"{m['sensitivity']:.1%}",
                            'Specificity': f"{m['specificity']:.1%}",
                            'PPV': f"{m['ppv']:.1%}",
                            'NPV': f"{m['npv']:.1%}",
                        })
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=700)
                    st.markdown("**M\u00f4 h\u00ecnh t\u1ed1t nh\u1ea5t theo t\u1eebng ph\u01b0\u01a1ng ph\u00e1p CV:**")
                    for cv in ['LOOCV', '5-Fold', '10-Fold']:
                        cv_rows = [(r['M\u00f4 h\u00ecnh'], float(r['Accuracy'].strip('%')) / 100)
                                   for r in rows if r['Ph\u01b0\u01a1ng ph\u00e1p CV'] == cv]
                        if cv_rows:
                            best = max(cv_rows, key=lambda x: x[1])
                            st.markdown(f"- **{cv}**: {best[0]} \u2014 Accuracy = {best[1]:.1%}")

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # Charts
        st.markdown("### \U0001f4ca Bi\u1ec3u \u0111\u1ed3 So s\u00e1nh")
        col_e, col_a = st.columns(2)
        eeg_c = os.path.join(FIG_DIR, "full_comparison_eeg.png")
        all_c = os.path.join(FIG_DIR, "full_comparison_all.png")
        if os.path.exists(eeg_c):
            with col_e:
                st.image(eeg_c, caption="So s\u00e1nh v\u1edbi 12 ch\u1ec9 s\u1ed1 EEG", use_column_width=True)
        if os.path.exists(all_c):
            with col_a:
                st.image(all_c, caption="So s\u00e1nh v\u1edbi 15 ch\u1ec9 s\u1ed1 (EEG + nh\u00e2n kh\u1ea9u h\u1ecdc)", use_column_width=True)

        cv_c = os.path.join(FIG_DIR, "cv_comparison.png")
        if os.path.exists(cv_c):
            st.image(cv_c, caption="So s\u00e1nh LOOCV / 5-Fold / 10-Fold theo m\u00f4 h\u00ecnh", use_column_width=True)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # Feature selection
        st.markdown("### \U0001f50d Th\u00ed nghi\u1ec7m L\u1ef1a ch\u1ecdn Ch\u1ec9 s\u1ed1")
        fs_data = eval_data.get('feature_selection', {})
        if fs_data:
            fs_rows = []
            for k_label, data in fs_data.items():
                features = data.get('features', [])
                esn  = data.get('ESN', {})
                eesn = data.get('E-ESN', {})
                fs_rows.append({
                    'S\u1ed1 ch\u1ec9 s\u1ed1 (k)': k_label,
                    'Ch\u1ec9 s\u1ed1 \u0111\u01b0\u1ee3c ch\u1ecdn': ', '.join(features[:6]) + ('...' if len(features) > 6 else ''),
                    'ESN Accuracy': f"{esn.get('accuracy', 0):.1%}",
                    'E-ESN Accuracy': f"{eesn.get('accuracy', 0):.1%}",
                })
            st.dataframe(pd.DataFrame(fs_rows), use_container_width=True, hide_index=True)

            fs_c = os.path.join(FIG_DIR, "feature_selection_comparison.png")
            if os.path.exists(fs_c):
                st.image(fs_c, use_column_width=True)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # Paper comparison
        st.markdown("### \U0001f4dd So s\u00e1nh v\u1edbi K\u1ebft qu\u1ea3 trong B\u00e0i b\u00e1o g\u1ed1c")
        eeg_loocv = eval_data['models'].get('EEG Only (12)', {}).get('LOOCV', {})
        esn_res   = eeg_loocv.get('ESN', {})
        eesn_res  = eeg_loocv.get('E-ESN (7)', {})

        comp_rows = [
            {
                'Ngu\u1ed3n': '\U0001f4c4 B\u00e0i b\u00e1o (Bouazizi & Ltifi, 2024)',
                'M\u00f4 h\u00ecnh': 'E-ESN \u2014 7 m\u00f4 h\u00ecnh, LOOCV',
                'Accuracy': '96.50%', 'F1': '94.73%',
                'Sensitivity': '96.42%', 'Specificity': '81.81%',
            },
            {
                'Ngu\u1ed3n': '\U0001f4bb Tri\u1ec3n khai n\u00e0y',
                'M\u00f4 h\u00ecnh': 'ESN \u2014 LOOCV (12 ch\u1ec9 s\u1ed1 EEG)',
                'Accuracy': f"{esn_res.get('accuracy', 0):.1%}",
                'F1': f"{esn_res.get('f1', 0):.1%}",
                'Sensitivity': f"{esn_res.get('sensitivity', 0):.1%}",
                'Specificity': f"{esn_res.get('specificity', 0):.1%}",
            },
            {
                'Ngu\u1ed3n': '\U0001f4bb Tri\u1ec3n khai n\u00e0y',
                'M\u00f4 h\u00ecnh': 'E-ESN 7 \u2014 LOOCV (12 ch\u1ec9 s\u1ed1 EEG)',
                'Accuracy': f"{eesn_res.get('accuracy', 0):.1%}",
                'F1': f"{eesn_res.get('f1', 0):.1%}",
                'Sensitivity': f"{eesn_res.get('sensitivity', 0):.1%}",
                'Specificity': f"{eesn_res.get('specificity', 0):.1%}",
            },
        ]
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)
        st.markdown("""
        > **Ghi ch\u00fa v\u1ec1 s\u1ef1 ch\u00eanh l\u1ec7ch ~10%**: C\u00f3 th\u1ec3 do (1) chi ti\u1ebft kh\u1edfi t\u1ea1o reservoir ESN,
        > (2) chu\u1ea9n h\u00f3a d\u1eef li\u1ec7u tr\u01b0\u1edbc/sau CV, (3) t\u1eadp d\u1eef li\u1ec7u nh\u1ecf (n=38).
        """)


# ============================================================
# PAGE: GIOI THIEU  (About)
# ============================================================
elif page == PAGE_ABOUT:
    st.markdown("""
    <div class="main-header">
        <h1>\u2139\ufe0f Gi\u1edbi thi\u1ec7u H\u1ec7 th\u1ed1ng</h1>
        <p>Th\u00f4ng tin v\u1ec1 m\u1ee5c \u0111\u00edch, ngu\u1ed3n g\u1ed1c, v\u00e0 gi\u1edbi h\u1ea1n c\u1ee7a h\u1ec7 th\u1ed1ng</p>
    </div>
    """, unsafe_allow_html=True)

    col_about, col_tech = st.columns([3, 2])

    with col_about:
        st.markdown("### \U0001f3af M\u1ee5c \u0111\u00edch c\u1ee7a h\u1ec7 th\u1ed1ng")
        st.markdown("""
        H\u1ec7 th\u1ed1ng n\u00e0y l\u00e0 **c\u00f4ng c\u1ee5 nghi\u00ean c\u1ee9u v\u00e0 gi\u00e1o d\u1ee5c** nh\u1eb1m:
        - Minh h\u1ecda \u1ee9ng d\u1ee5ng AI trong ph\u00e2n t\u00edch t\u00edn hi\u1ec7u EEG
        - H\u1ed7 tr\u1ee3 nghi\u00ean c\u1ee9u ph\u00e1t hi\u1ec7n s\u1edbm \u0111\u1ed9t qu\u1ef5
        - Th\u1ec3 hi\u1ec7n t\u00ednh minh b\u1ea1ch c\u1ee7a AI qua gi\u1ea3i th\u00edch SHAP/LIME

        ### \U0001f4da Ngu\u1ed3n g\u1ed1c nghi\u00ean c\u1ee9u

        **B\u00e0i b\u00e1o g\u1ed1c:**
        > *"Enhancing accuracy and interpretability in EEG-based medical
        > decision making using an explainable ensemble learning framework
        > application for stroke prediction"*
        >
        > Bouazizi, S., & Ltifi, H. (2024). *Decision Support Systems*, 178, 114126.

        **D\u1eef li\u1ec7u EEG:**
        > *"Acute single channel EEG predictors of cognitive function after stroke"*
        >
        > Aminov, A., et al. (2017). *PLoS One*, 12, e0185841.
        > [T\u1ea3i v\u1ec1 t\u1ea1i Dryad](https://doi.org/10.5061/dryad.h6986) \u2014 Gi\u1ea5y ph\u00e9p CC0

        ### \U0001f4d6 Thu\u1eadt ng\u1eef chuy\u00ean ng\u00e0nh

        | Thu\u1eadt ng\u1eef | \u00dd ngh\u0129a |
        |-----------|---------|
        | EEG | \u0110i\u1ec7n n\u00e3o \u0111\u1ed3 \u2014 \u0111o ho\u1ea1t \u0111\u1ed9ng \u0111i\u1ec7n n\u00e3o |
        | ESN | Echo State Network \u2014 lo\u1ea1i m\u1ea1ng n\u01a1-ron t\u00e1i ph\u00e1t |
        | SHAP | Ph\u01b0\u01a1ng ph\u00e1p gi\u1ea3i th\u00edch t\u1ea7m quan tr\u1ecdng t\u1ed5ng th\u1ec3 |
        | LIME | Ph\u01b0\u01a1ng ph\u00e1p gi\u1ea3i th\u00edch t\u1eebng d\u1ef1 \u0111o\u00e1n c\u00e1 bi\u1ec7t |
        | LOOCV | Ki\u1ec3m tra ch\u00e9o b\u1ecf-m\u1ed9t-ra \u2014 ph\u00f9 h\u1ee3p t\u1eadp nh\u1ecf |
        | DAR | (Delta+Theta) / (Alpha+Beta) \u2014 ch\u1ec9 s\u1ed1 nh\u1ea1y nh\u1ea5t |
        | DTR | Delta / Theta \u2014 ch\u1ec9 s\u1ed1 t\u1ed5n th\u01b0\u01a1ng n\u00e3o |
        | RP | Relative Power \u2014 c\u00f4ng su\u1ea5t t\u01b0\u01a1ng \u0111\u1ed1i c\u1ee7a t\u1eebng b\u0103ng t\u1ea7n |
        | PPV | Positive Predictive Value \u2014 \u0111\u1ed9 tin c\u1eady k\u1ebft qu\u1ea3 D\u01b0\u01a1ng t\u00ednh |
        | NPV | Negative Predictive Value \u2014 \u0111\u1ed9 tin c\u1eady k\u1ebft qu\u1ea3 \u00c2m t\u00ednh |
        """)

    with col_tech:
        st.markdown("### \u2699\ufe0f C\u00f4ng ngh\u1ec7 s\u1eed d\u1ee5ng")
        st.markdown("""
        | Th\u00e0nh ph\u1ea7n | C\u00f4ng ngh\u1ec7 |
        |------------|-----------|
        | M\u00f4 h\u00ecnh AI | Echo State Network |
        | Ensemble | 7 ESN, b\u1ea7u ch\u1ecdn m\u1ec1m |
        | Gi\u1ea3i th\u00edch to\u00e0n c\u1ee5c | KernelSHAP |
        | Gi\u1ea3i th\u00edch c\u1ee5c b\u1ed9 | LIME |
        | T\u01b0 v\u1ea5n l\u00e2m s\u00e0ng | Quy t\u1eafc d\u1ef1a tr\u00ean r\u1ee7i ro |
        | B\u00e1o c\u00e1o | PDF (fpdf2) |
        | Giao di\u1ec7n | Streamlit |
        | Ng\u00f4n ng\u1eef | Python 3.12 |
        """)

        st.markdown("### \U0001f527 Tham s\u1ed1 k\u1ef9 thu\u1eadt m\u00f4 h\u00ecnh")
        st.markdown("""
        | Tham s\u1ed1 | Gi\u00e1 tr\u1ecb |
        |---------|---------|
        | S\u1ed1 n\u01a1-ron | 200 |
        | Spectral Radius | 0.95 |
        | Leaking Rate | 0.7 |
        | Input Scaling | 0.5 |
        | S\u1ed1 ESN trong Ensemble | 7 |
        | Ph\u01b0\u01a1ng ph\u00e1p t\u1ed5ng h\u1ee3p | Soft Voting |
        """)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.error("""
    ⚠️ **CẢNH BÁO QUAN TRỌNG**

    Hệ thống này được phát triển **chỉ cho mục đích nghiên cứu và giáo dục**.

    - ❌ **KHÔNG** phải thiết bị y tế được cấp phép
    - ❌ **KHÔNG** thay thế chẩn đoán của bác sĩ chuyên khoa
    - ❌ **KHÔNG** được sử dụng cho quyết định điều trị thực tế
    - ✅ **CHỈ DÙNG** cho nghiên cứu, học thuật, và thử nghiệm ý tưởng
    """)
    # End of PAGE_ABOUT


# ============================================================
# PAGE: PATIENT ANALYSIS
# ============================================================
elif page == "🔬 Patient Analysis":
    st.markdown("""
    <div class="main-header">
        <h1>🔬 Patient Analysis & Prediction</h1>
        <p>Select a patient or enter custom EEG features for stroke risk assessment</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["📂 Select Patient from Dataset", "✏️ Custom Input"])
    
    with tab1:
        patient_idx = st.selectbox(
            "Select Patient",
            range(len(y)),
            format_func=lambda i: f"Patient #{i+1} — {LABEL_MAP_INV[y[i]].capitalize()} "
                                  f"({'Stroke' if y[i]==1 else 'Control'})"
        )
        
        patient_features = X_norm[patient_idx:patient_idx+1]
        patient_raw = X[patient_idx]
        true_label = y[patient_idx]
    
    with tab2:
        st.markdown("### Enter EEG Features")
        custom_cols = st.columns(4)
        custom_values = []
        
        for i, fname in enumerate(feature_names):
            col = custom_cols[i % 4]
            with col:
                val = st.number_input(
                    fname,
                    value=float(X[:, i].mean()),
                    min_value=float(X[:, i].min()),
                    max_value=float(X[:, i].max()),
                    step=0.01,
                    key=f"custom_{fname}"
                )
                custom_values.append(val)
        
        if st.button("🔍 Predict", type="primary"):
            custom_arr = np.array(custom_values).reshape(1, -1)
            patient_features = normalizer.transform(custom_arr)
            patient_raw = custom_arr[0]
            true_label = None
        else:
            patient_features = X_norm[0:1]
            patient_raw = X[0]
            true_label = y[0]
    
    # ---- PREDICTION ----
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    proba = model.predict_proba(patient_features)[0]
    stroke_prob = proba[1]
    pred_class = "Stroke" if np.argmax(proba) == 1 else "Control"
    
    rec_engine = RecommendationEngine()
    risk_level = rec_engine.classify_risk(stroke_prob)
    
    # Results
    col_pred, col_risk, col_prob = st.columns(3)
    
    with col_pred:
        st.markdown("### 🎯 Prediction")
        if pred_class == "Stroke":
            st.error(f"**{pred_class}**", icon="⚠️")
        else:
            st.success(f"**{pred_class}**", icon="✅")
        
        if true_label is not None:
            true_name = LABEL_MAP_INV[true_label].capitalize()
            match = "✅ Correct" if (pred_class.lower() == true_name.lower()) else "❌ Incorrect"
            st.caption(f"True label: {true_name} {match}")
    
    with col_risk:
        st.markdown("### ⚡ Risk Level")
        risk_css = f"risk-{risk_level}"
        st.markdown(f'<div class="{risk_css}">{risk_level.upper()} RISK</div>',
                   unsafe_allow_html=True)
    
    with col_prob:
        st.markdown("### 📊 Probability")
        fig_gauge, ax_gauge = plt.subplots(figsize=(4, 2.5))
        
        colors = ['#2ed573', '#ffa502', '#ff6b6b']
        thresholds = [0, 0.4, 0.7, 1.0]
        
        for i in range(3):
            theta_start = thresholds[i] * 180
            theta_end = thresholds[i+1] * 180
            wedge = plt.matplotlib.patches.Wedge(
                (0.5, 0), 0.4, theta_start, theta_end,
                width=0.12, facecolor=colors[i], alpha=0.3,
                transform=ax_gauge.transAxes
            )
            ax_gauge.add_patch(wedge)
        
        ax_gauge.barh(0, stroke_prob, height=0.3, 
                     color='#667eea' if stroke_prob < 0.7 else '#ff6b6b',
                     alpha=0.8, zorder=2)
        ax_gauge.barh(0, 1, height=0.3, color='#e2e8f0', alpha=0.3, zorder=1)
        ax_gauge.set_xlim(0, 1)
        ax_gauge.set_ylim(-0.5, 0.5)
        ax_gauge.text(stroke_prob, 0, f'{stroke_prob:.1%}', 
                     ha='center', va='bottom', fontsize=16, fontweight='bold',
                     color='#2d3748')
        ax_gauge.set_yticks([])
        ax_gauge.set_xlabel('Stroke Probability')
        ax_gauge.spines['top'].set_visible(False)
        ax_gauge.spines['right'].set_visible(False)
        ax_gauge.spines['left'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_gauge)
        plt.close()
    
    # Feature values
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### 📋 Patient EEG Features")
    
    feat_df = pd.DataFrame({
        'Feature': feature_names,
        'Value': patient_raw,
        'Normalized': patient_features[0]
    })
    st.dataframe(feat_df, use_container_width=True)
    
    # DSS Recommendations
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### 💊 Diagnostic Recommendations")
    
    rec = rec_engine.get_recommendations(stroke_prob, pred_class)
    
    st.info(f"**Action**: {rec['action']}", icon="📋")
    
    for test in rec['recommended_tests']:
        st.markdown(f"""
        <div class="test-card">
            <div class="test-name">🔹 {test['full_name']}</div>
            <div class="test-desc">{test['description']}</div>
        </div>
        """, unsafe_allow_html=True)

    # ---- PDF REPORT DOWNLOAD ----
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### 📄 Download Clinical Report")
    st.caption("Generate a PDF report with prediction results, LIME explanation, and recommendations.")

    if st.button("📄 Generate PDF Report", type="secondary"):
        with st.spinner("Generating PDF report with LIME explanation..."):
            # Compute LIME explanation for this patient
            _lime_expl = get_lime_explainer()
            _lime_exp = _lime_expl.explain_instance(
                patient_features[0], model.predict_proba,
                num_features=len(feature_names), num_samples=5000
            )
            lime_contribs = _lime_exp.as_list()

            # Save LIME figure to temporary file for PDF embedding
            _lime_fig = _lime_exp.as_pyplot_figure()
            _lime_fig.set_size_inches(11, 5)
            _lime_tmp = tempfile.NamedTemporaryFile(
                suffix='.png', delete=False
            )
            _lime_tmp_name = _lime_tmp.name
            _lime_tmp.close()  # Close handle before writing (Windows)
            _lime_fig.savefig(_lime_tmp_name, dpi=100, bbox_inches='tight')
            plt.close(_lime_fig)

            # Build feature dictionaries
            feat_raw = {fn: float(patient_raw[i])
                        for i, fn in enumerate(feature_names)}
            feat_norm = {fn: float(patient_features[0][i])
                         for i, fn in enumerate(feature_names)}

            true_label_str = None
            if true_label is not None:
                true_label_str = LABEL_MAP_INV[true_label].capitalize()

            pdf_bytes = generate_report(
                patient_id=patient_idx,
                prediction=pred_class,
                stroke_probability=stroke_prob,
                risk_level=risk_level,
                features=feat_raw,
                normalized_features=feat_norm,
                recommendations=rec,
                lime_contributions=lime_contribs,
                true_label=true_label_str,
                lime_fig_path=_lime_tmp_name
            )

            # Clean up temp file
            try:
                os.unlink(_lime_tmp_name)
            except OSError:
                pass  # Windows may still lock the file briefly

            # Store PDF in session state for persistent download button
            st.session_state['report_pdf'] = pdf_bytes
            st.session_state['report_patient_id'] = patient_idx

    # Show download button if report exists
    if 'report_pdf' in st.session_state:
        pid = st.session_state.get('report_patient_id', 0)
        st.download_button(
            label="⬇️ Download PDF Report",
            data=st.session_state['report_pdf'],
            file_name=f"stroke_report_patient_{pid + 1:02d}.pdf",
            mime="application/pdf"
        )
        st.success("✅ Report generated successfully!", icon="📄")


# ============================================================
# PAGE: MODEL EXPLAINABILITY (INTERACTIVE)
# ============================================================
elif page == "📊 Model Explainability":
    st.markdown("""
    <div class="main-header">
        <h1>📊 Model Explainability</h1>
        <p>Understanding how the E-ESN model makes predictions using SHAP and LIME</p>
    </div>
    """, unsafe_allow_html=True)

    tab_shap, tab_lime = st.tabs([
        "🌐 SHAP (Global & Per-Patient)",
        "🔍 LIME (Interactive)"
    ])

    # ──────────────────────────────────────────────────────────
    # TAB: SHAP
    # ──────────────────────────────────────────────────────────
    with tab_shap:
        st.markdown("### SHAP Feature Importance")
        st.markdown("""
        SHAP (SHapley Additive exPlanations) shows which EEG features have the 
        most impact on stroke prediction **across all patients** and for 
        **individual patients**.
        """)

        # Compute SHAP values (cached after first run)
        shap_computed = False
        try:
            shap_values_all, shap_base_value = compute_all_shap_values()
            shap_computed = True
        except Exception as e:
            st.warning(f"⚠️ Could not compute SHAP values: {e}")

        if shap_computed:
            # ── Global Feature Impact ──
            st.markdown("#### 🌍 Global Feature Impact")
            col_s1, col_s2 = st.columns(2)

            with col_s1:
                fig_summary = plt.figure(figsize=(8, 6))
                shap.summary_plot(
                    shap_values_all, X_norm,
                    feature_names=feature_names,
                    show=False
                )
                plt.title("SHAP Summary (Beeswarm)", fontsize=12)
                plt.tight_layout()
                st.pyplot(fig_summary)
                plt.close('all')

            with col_s2:
                mean_abs = np.abs(shap_values_all).mean(axis=0)
                sorted_idx = np.argsort(mean_abs)

                fig_bar, ax_bar = plt.subplots(figsize=(8, 6))
                median_val = np.median(mean_abs)
                colors_bar = [
                    '#667eea' if mean_abs[i] >= median_val else '#c3cfe2'
                    for i in sorted_idx
                ]
                ax_bar.barh(
                    range(len(feature_names)),
                    mean_abs[sorted_idx],
                    color=colors_bar, height=0.7
                )
                ax_bar.set_yticks(range(len(feature_names)))
                ax_bar.set_yticklabels(
                    [feature_names[i] for i in sorted_idx],
                    fontsize=10
                )
                ax_bar.set_xlabel('Mean |SHAP value|', fontsize=11)
                ax_bar.set_title(
                    'SHAP Feature Importance (Bar)', fontsize=12
                )
                ax_bar.spines['top'].set_visible(False)
                ax_bar.spines['right'].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig_bar)
                plt.close()

            st.markdown("""
            > **How to read**: Features with higher SHAP values have more 
            > influence on the model's stroke prediction. In the beeswarm plot, 
            > red dots = high feature value, blue dots = low feature value.
            """)

            # ── Per-Patient SHAP Waterfall ──
            st.markdown(
                '<div class="section-divider"></div>',
                unsafe_allow_html=True
            )
            st.markdown("#### 🎯 Per-Patient SHAP Analysis")
            st.markdown(
                "Select a patient to see how each feature contributed to "
                "*their specific* prediction."
            )

            patient_shap = st.selectbox(
                "Select Patient",
                range(len(y)),
                format_func=lambda i: (
                    f"Patient #{i+1} — "
                    f"{'Stroke' if y[i]==1 else 'Control'}"
                ),
                key="shap_patient"
            )

            sv_patient = shap_values_all[patient_shap]
            sorted_idx_p = np.argsort(np.abs(sv_patient))

            fig_wf, ax_wf = plt.subplots(figsize=(10, 6))
            vals_wf = sv_patient[sorted_idx_p]
            names_wf = [feature_names[i] for i in sorted_idx_p]
            colors_wf = [
                '#ff6b6b' if v > 0 else '#667eea'
                for v in vals_wf
            ]

            ax_wf.barh(
                range(len(names_wf)), vals_wf,
                color=colors_wf, height=0.7,
                edgecolor='white', linewidth=0.5
            )
            ax_wf.set_yticks(range(len(names_wf)))
            ax_wf.set_yticklabels(names_wf, fontsize=10)
            ax_wf.set_xlabel(
                'SHAP Value (impact on stroke probability)',
                fontsize=11
            )
            ax_wf.axvline(x=0, color='black', linewidth=0.8, alpha=0.5)
            ax_wf.set_title(
                f'Patient #{patient_shap+1} — SHAP Feature Contributions',
                fontsize=13, fontweight='bold'
            )

            for i, v in enumerate(vals_wf):
                offset = max(abs(v) * 0.08, 0.002)
                ax_wf.text(
                    v + (offset if v >= 0 else -offset), i,
                    f'{v:+.4f}',
                    ha='left' if v >= 0 else 'right',
                    va='center', fontsize=9
                )

            ax_wf.spines['top'].set_visible(False)
            ax_wf.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig_wf)
            plt.close()

            # Summary metrics for this patient
            col_base, col_sum, col_final = st.columns(3)
            final_score = shap_base_value + sv_patient.sum()

            with col_base:
                st.metric("Base Value", f"{shap_base_value:.4f}")
            with col_sum:
                st.metric("SHAP Sum", f"{sv_patient.sum():+.4f}")
            with col_final:
                pred_label = "Stroke" if final_score > 0.5 else "Control"
                st.metric("Final Score", f"{final_score:.4f}",
                         delta=pred_label)

            col_leg1, col_leg2 = st.columns(2)
            with col_leg1:
                st.markdown("🔴 **Red bars**: Push toward **Stroke**")
            with col_leg2:
                st.markdown("🔵 **Blue bars**: Push toward **Control**")

        else:
            # Fallback: show pre-computed images if SHAP computation failed
            fig_dir = r"d:\Personal\Explainable EEG Stroke DSS\experiments\figures"
            shap_s = os.path.join(fig_dir, "shap_summary.png")
            shap_b = os.path.join(fig_dir, "shap_bar.png")
            if os.path.exists(shap_s):
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    st.image(shap_s, caption="SHAP Summary",
                            use_column_width=True)
                with col_f2:
                    st.image(shap_b, caption="SHAP Bar",
                            use_column_width=True)
            else:
                st.error(
                    "SHAP plots not available. "
                    "Run `scripts/run_pipeline.py` first."
                )

    # ──────────────────────────────────────────────────────────
    # TAB: LIME (Interactive)
    # ──────────────────────────────────────────────────────────
    with tab_lime:
        st.markdown("### LIME Local Explanations")
        st.markdown("""
        LIME explains **individual predictions** by fitting a local interpretable 
        surrogate model around each patient. Select a patient and click 
        **Compute** to generate a real-time explanation.
        """)

        patient_lime = st.selectbox(
            "Select Patient",
            range(len(y)),
            format_func=lambda i: (
                f"Patient #{i+1} — "
                f"{'Stroke' if y[i]==1 else 'Control'}"
            ),
            key="lime_patient"
        )

        if st.button("🔍 Compute LIME Explanation", type="primary"):
            with st.spinner("Computing LIME explanation (~5 seconds)..."):
                _lime_expl = get_lime_explainer()
                _exp = _lime_expl.explain_instance(
                    X_norm[patient_lime], model.predict_proba,
                    num_features=len(feature_names), num_samples=5000
                )

                # Get prediction info
                _proba = model.predict_proba(
                    X_norm[patient_lime:patient_lime+1]
                )[0]

                # Save LIME figure to bytes for display
                _fig = _exp.as_pyplot_figure()
                _fig.set_size_inches(12, 6)
                _buf = io.BytesIO()
                _fig.savefig(
                    _buf, format='png', dpi=120, bbox_inches='tight'
                )
                plt.close(_fig)

                # Store results in session state
                st.session_state['lime_result'] = {
                    'exp_list': _exp.as_list(),
                    'patient_idx': patient_lime,
                    'proba': _proba.tolist(),
                    'fig_bytes': _buf.getvalue(),
                }

        # Display stored LIME results
        if 'lime_result' in st.session_state:
            lr = st.session_state['lime_result']

            # Warn if patient changed
            if lr['patient_idx'] != patient_lime:
                st.info(
                    "📌 Patient changed. Click **Compute LIME Explanation** "
                    "to update."
                )

            proba_arr = np.array(lr['proba'])
            pred_cls = "Stroke" if np.argmax(proba_arr) == 1 else "Control"
            lr_pidx = lr['patient_idx']

            # Prediction summary
            col_lp, col_lr, col_lt = st.columns(3)
            with col_lp:
                if pred_cls == "Stroke":
                    st.error(f"**Prediction: {pred_cls}**", icon="⚠️")
                else:
                    st.success(f"**Prediction: {pred_cls}**", icon="✅")
            with col_lr:
                st.metric("Stroke Probability", f"{proba_arr[1]:.1%}")
            with col_lt:
                true_cls = 'Stroke' if y[lr_pidx] == 1 else 'Control'
                match_icon = "✅" if pred_cls == true_cls else "❌"
                st.metric("True Label", f"{true_cls} {match_icon}")

            # LIME figure
            st.image(
                lr['fig_bytes'],
                caption=(
                    f"LIME Explanation — Patient #{lr_pidx+1}"
                ),
                use_column_width=True
            )

            # Feature contributions table
            st.markdown("#### Feature Contributions")
            cont_df = pd.DataFrame(
                lr['exp_list'],
                columns=['Feature Condition', 'Weight']
            )
            cont_df['Direction'] = cont_df['Weight'].apply(
                lambda w: '🔴 → Stroke' if w > 0 else '🟢 → Control'
            )
            cont_df['|Impact|'] = cont_df['Weight'].abs()
            cont_df = cont_df.sort_values('|Impact|', ascending=False)
            st.dataframe(cont_df, use_container_width=True)

            # Clinical interpretation
            st.markdown(
                '<div class="section-divider"></div>',
                unsafe_allow_html=True
            )
            st.markdown("#### 💡 Clinical Interpretation")

            top_stroke = [
                (name, w) for name, w in lr['exp_list'] if w > 0
            ]
            top_control = [
                (name, w) for name, w in lr['exp_list'] if w < 0
            ]
            top_stroke.sort(key=lambda x: x[1], reverse=True)
            top_control.sort(key=lambda x: x[1])

            col_ts, col_tc = st.columns(2)
            with col_ts:
                st.markdown("**Factors pushing toward Stroke:**")
                for name, w in top_stroke[:5]:
                    st.markdown(
                        f"- 🔴 {name} (impact: {w:+.4f})"
                    )
                if not top_stroke:
                    st.markdown("- *None*")
            with col_tc:
                st.markdown("**Factors pushing toward Control:**")
                for name, w in top_control[:5]:
                    st.markdown(
                        f"- 🟢 {name} (impact: {w:+.4f})"
                    )
                if not top_control:
                    st.markdown("- *None*")

        else:
            st.info(
                "👆 Select a patient and click "
                "**Compute LIME Explanation** to see results."
            )


# ============================================================
# PAGE: ABOUT
# ============================================================
# ============================================================
# PAGE: FULL EVALUATION (Paper Replication)
# ============================================================
elif page == "📈 Full Evaluation":
    st.markdown("""
    <div class="main-header">
        <h1>📈 Comprehensive Model Evaluation</h1>
        <p>Full comparison of all models, CV methods, feature sets, and feature selection — matching the paper</p>
    </div>
    """, unsafe_allow_html=True)

    eval_data = load_full_evaluation()

    if eval_data is None:
        st.error(
            "Evaluation results not found. "
            "Run `python scripts/run_full_evaluation.py` first."
        )
    else:
        fig_dir = r"d:\Personal\Explainable EEG Stroke DSS\experiments\figures"

        # ── Feature Distribution (Paper Fig. 3) ──
        st.markdown("### 📊 Feature Distribution by Class (Paper Fig. 3)")
        dist_path = os.path.join(fig_dir, "feature_distributions.png")
        if os.path.exists(dist_path):
            st.image(dist_path, use_column_width=True)
            st.caption(
                "Boxplots show the distribution of each feature for Control (blue) vs Stroke (red) patients."
            )

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # ── Comprehensive Comparison Table ──
        st.markdown("### 📋 All Models x All CV Methods (Paper Table 4-5)")

        fs_tab_labels = list(eval_data['models'].keys())
        fs_tabs = st.tabs(fs_tab_labels)

        for tab, fs_name in zip(fs_tabs, fs_tab_labels):
            with tab:
                rows = []
                for cv_name in ['LOOCV', '5-Fold', '10-Fold']:
                    cv_data = eval_data['models'][fs_name].get(cv_name, {})
                    for model_name, m in cv_data.items():
                        rows.append({
                            'Model': model_name,
                            'CV': cv_name,
                            'Accuracy': f"{m['accuracy']:.1%}",
                            'F1': f"{m['f1']:.1%}",
                            'Sensitivity': f"{m['sensitivity']:.1%}",
                            'Specificity': f"{m['specificity']:.1%}",
                            'PPV': f"{m['ppv']:.1%}",
                            'NPV': f"{m['npv']:.1%}",
                        })

                if rows:
                    df_table = pd.DataFrame(rows)
                    st.dataframe(
                        df_table,
                        use_container_width=True,
                        hide_index=True,
                        height=700
                    )

                    # Highlight best model per CV method
                    st.markdown("**Best model per CV method:**")
                    for cv in ['LOOCV', '5-Fold', '10-Fold']:
                        cv_rows = [
                            (r['Model'], float(r['Accuracy'].strip('%')) / 100)
                            for r in rows if r['CV'] == cv
                        ]
                        if cv_rows:
                            best = max(cv_rows, key=lambda x: x[1])
                            st.markdown(
                                f"- **{cv}**: {best[0]} ({best[1]:.1%})"
                            )

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # ── Model Comparison Chart ──
        st.markdown("### 📊 Model Comparison Charts")
        col_eeg, col_all = st.columns(2)
        eeg_comp = os.path.join(fig_dir, "full_comparison_eeg.png")
        all_comp = os.path.join(fig_dir, "full_comparison_all.png")
        if os.path.exists(eeg_comp):
            with col_eeg:
                st.image(eeg_comp, caption="EEG Only (12 features)",
                        use_column_width=True)
        if os.path.exists(all_comp):
            with col_all:
                st.image(all_comp, caption="All Features (15 features)",
                        use_column_width=True)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # ── CV Method Comparison ──
        st.markdown("### 🔄 Cross-Validation Comparison")
        cv_comp = os.path.join(fig_dir, "cv_comparison.png")
        if os.path.exists(cv_comp):
            st.image(cv_comp, use_column_width=True)
            st.caption(
                "Comparison of LOOCV, 5-Fold, and 10-Fold CV across key models."
            )

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # ── Feature Selection Experiments ──
        st.markdown("### 🔍 Feature Selection Experiments (ANOVA + MI)")

        fs_data = eval_data.get('feature_selection', {})
        if fs_data:
            # Results table
            fs_rows = []
            for k_label, data in fs_data.items():
                features = data.get('features', [])
                esn = data.get('ESN', {})
                eesn = data.get('E-ESN', {})

                fs_rows.append({
                    'Features': k_label,
                    'Selected': ', '.join(features[:6]) + ('...' if len(features) > 6 else ''),
                    'ESN Accuracy': f"{esn.get('accuracy', 0):.1%}",
                    'ESN F1': f"{esn.get('f1', 0):.1%}",
                    'E-ESN Accuracy': f"{eesn.get('accuracy', 0):.1%}",
                    'E-ESN F1': f"{eesn.get('f1', 0):.1%}",
                })

            st.dataframe(
                pd.DataFrame(fs_rows),
                use_container_width=True,
                hide_index=True
            )

            # Chart
            fs_chart = os.path.join(fig_dir, "feature_selection_comparison.png")
            if os.path.exists(fs_chart):
                st.image(fs_chart, use_column_width=True)

            # Feature ranking table
            st.markdown("#### Feature Ranking (ANOVA + Mutual Information)")
            first_key = list(fs_data.keys())[0]
            ranking = fs_data[first_key].get('ranking', [])
            if ranking:
                rank_df = pd.DataFrame(ranking)
                rank_df['anova_p'] = rank_df['anova_p'].apply(
                    lambda x: f"{x:.6f}" if x > 0.001 else f"{x:.2e}"
                )
                rank_df['combined'] = rank_df['combined'].apply(
                    lambda x: f"{x:.4f}"
                )
                rank_df.columns = ['Feature', 'Rank', 'ANOVA p-value', 'Combined Score']
                st.dataframe(rank_df, use_container_width=True, hide_index=True)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # ── Paper Comparison ──
        st.markdown("### 📝 Comparison with Paper Results")

        # Best ESN result from our evaluation
        eeg_loocv = eval_data['models'].get('EEG Only (12)', {}).get('LOOCV', {})
        esn_res = eeg_loocv.get('ESN', {})
        eesn_res = eeg_loocv.get('E-ESN (7)', {})

        comp_rows = [
            {
                'Model': 'Paper E-ESN (target)',
                'Accuracy': '96.50%', 'F1': '94.73%',
                'Sensitivity': '96.42%', 'Specificity': '81.81%',
                'PPV': '93.10%', 'NPV': '90.00%'
            },
            {
                'Model': 'Our ESN (EEG-only, LOOCV)',
                'Accuracy': f"{esn_res.get('accuracy', 0):.1%}",
                'F1': f"{esn_res.get('f1', 0):.1%}",
                'Sensitivity': f"{esn_res.get('sensitivity', 0):.1%}",
                'Specificity': f"{esn_res.get('specificity', 0):.1%}",
                'PPV': f"{esn_res.get('ppv', 0):.1%}",
                'NPV': f"{esn_res.get('npv', 0):.1%}",
            },
            {
                'Model': 'Our E-ESN (EEG-only, LOOCV)',
                'Accuracy': f"{eesn_res.get('accuracy', 0):.1%}",
                'F1': f"{eesn_res.get('f1', 0):.1%}",
                'Sensitivity': f"{eesn_res.get('sensitivity', 0):.1%}",
                'Specificity': f"{eesn_res.get('specificity', 0):.1%}",
                'PPV': f"{eesn_res.get('ppv', 0):.1%}",
                'NPV': f"{eesn_res.get('npv', 0):.1%}",
            },
        ]
        st.dataframe(
            pd.DataFrame(comp_rows),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("""
        > **Note:** The ~10% accuracy gap vs the paper may be due to:
        > 1. Paper may normalize the entire dataset before CV (potential data leakage)
        > 2. Differences in ESN reservoir initialization details
        > 3. Paper's exact feature preprocessing may differ slightly
        > 4. Small dataset (n=38) makes results sensitive to implementation details
        """)


elif page == "ℹ️ About":
    st.markdown("""
    <div class="main-header">
        <h1>ℹ️ About This System</h1>
        <p>Technical details and references</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### Paper Reference
    
    > **"Enhancing accuracy and interpretability in EEG-based medical decision making 
    > using an explainable ensemble learning framework application for stroke prediction"**
    > 
    > Bouazizi, S., & Ltifi, H. (2024). *Decision Support Systems*, 178, 114126.
    
    ### Dataset
    
    > **"Acute single channel EEG predictors of cognitive function after stroke"**
    > 
    > Aminov, A., et al. (2017). *PLoS One*, 12, e0185841.
    > 
    > [Dryad Repository](https://doi.org/10.5061/dryad.h6986) — CC0 License
    
    ### Technical Stack
    
    | Component | Technology |
    |-----------|-----------|
    | Model | Echo State Network (custom) |
    | Ensemble | Bagging (7 ESNs) |
    | Global XAI | KernelSHAP |
    | Local XAI | LIME |
    | DSS Engine | Rule-based recommendations |
    | Report | PDF generation (fpdf2) |
    | Frontend | Streamlit |
    | Language | Python 3.12 |
    
    ### Optimal Hyperparameters
    
    | Parameter | Value | Source |
    |-----------|-------|--------|
    | Neurons | 200 | Paper |
    | Spectral Radius | 0.95 | Paper |
    | Sparsity | 0.0 (fully connected) | Paper |
    | Noise (regularization) | 0.01 | Tuned |
    | Leaking Rate | 0.7 | Tuned |
    | Input Scaling | 0.5 | Tuned |
    | N Estimators | 7 | Paper |
    | Feature Set | EEG-only (12 features) | Tuned |
    
    ### Disclaimer
    
    > ⚠️ **This system is for research and educational purposes only.** 
    > It is NOT a certified medical device and should NOT be used for 
    > clinical decision-making without proper validation and regulatory approval.
    """)
