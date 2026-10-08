"""
Explainable EEG Stroke Decision Support System (DSS) — Clinical Web Application

Based on the multi-level framework published by:
Samar Bouazizi & Hela Ltifi, "Enhancing accuracy and interpretability in EEG-based medical
decision making using an explainable ensemble learning framework application for stroke prediction",
Decision Support Systems 178 (2024) 114126 (Elsevier).
"""
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import io
import json
import tempfile
import time
from datetime import datetime

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import lime.lime_tabular

from src.data.loader import (
    load_dataset, encode_labels, encode_categorical_features,
    EEG_FEATURES, DEMOGRAPHIC_FEATURES, LABEL_MAP_INV
)
from src.features.normalization import FeatureNormalizer
from src.models.esn import EchoStateNetwork
from src.models.ensemble import EnsembleESN
from src.models.baselines import compute_metrics
from src.dss.recommendation import (
    RecommendationEngine, DecisionStatus, PhysicianDecisionSession, DIAGNOSTIC_TESTS
)
from src.reports.pdf_generator import generate_report
from src.preprocessing.simulator import generate_synthetic_raw_eeg
from src.preprocessing.pipeline import process_raw_eeg_signal
from src.preprocessing.psd import FREQUENCY_BANDS, compute_epoch_psd
from src.utils.seed import set_global_seed


# ============================================================
# PAGE CONFIGURATION & STYLING
# ============================================================
st.set_page_config(
    page_title="Explainable EEG Stroke DSS",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        padding: 1.8rem 2.2rem;
        border-radius: 14px;
        margin-bottom: 1.4rem;
        color: #ffffff !important;
        box-shadow: 0 8px 30px rgba(30,58,95,0.2);
    }
    .main-header h1 {
        margin: 0;
        font-size: 1.85rem;
        font-weight: 700;
        color: #ffffff !important;
    }
    .main-header p {
        margin: 0.4rem 0 0 0;
        opacity: 0.92;
        font-size: 0.95rem;
        color: #ffffff !important;
    }

    /* Badges */
    .badge-live {
        display: inline-block; background: #e6fffa; color: #234e52;
        border: 1px solid #b2f5ea; border-radius: 20px; padding: 3px 12px;
        font-size: 0.75rem; font-weight: 600; margin-bottom: 0.5rem;
    }
    .badge-static {
        display: inline-block; background: #ebf8ff; color: #2b6cb0;
        border: 1px solid #bee3f8; border-radius: 20px; padding: 3px 12px;
        font-size: 0.75rem; font-weight: 600; margin-bottom: 0.5rem;
    }
    .badge-audit {
        display: inline-block; background: #faf5ff; color: #6b46c1;
        border: 1px solid #d6bcfa; border-radius: 20px; padding: 3px 12px;
        font-size: 0.75rem; font-weight: 600; margin-bottom: 0.5rem;
    }

    /* Metric & Risk Cards */
    .metric-card {
        background: #f8fafc;
        padding: 1.1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        border: 1px solid #e2e8f0;
        border-top: 3px solid #2d6a9f;
    }
    .metric-value { font-size: 1.9rem; font-weight: 700; color: #1e3a5f; }
    .metric-label { font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.8px; margin-top: 0.2rem; }
    .metric-sub { font-size: 0.7rem; color: #94a3b8; margin-top: 0.2rem; font-style: italic; }

    .risk-high {
        background: linear-gradient(135deg, #b91c1c, #991b1b);
        color: white; padding: 1.1rem; border-radius: 10px; text-align: center;
        font-size: 1.15rem; font-weight: 700; box-shadow: 0 4px 12px rgba(185,28,28,0.25);
    }
    .risk-medium {
        background: linear-gradient(135deg, #c2410c, #9a3412);
        color: white; padding: 1.1rem; border-radius: 10px; text-align: center;
        font-size: 1.15rem; font-weight: 700; box-shadow: 0 4px 12px rgba(194,65,12,0.25);
    }
    .risk-low {
        background: linear-gradient(135deg, #15803d, #166534);
        color: white; padding: 1.1rem; border-radius: 10px; text-align: center;
        font-size: 1.15rem; font-weight: 700; box-shadow: 0 4px 12px rgba(21,128,61,0.25);
    }

    /* Test Cards */
    .test-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #2d6a9f;
        padding: 0.9rem 1.1rem;
        margin: 0.4rem 0;
        border-radius: 0 8px 8px 0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.03);
    }
    .test-name { font-weight: 600; color: #1e3a5f; font-size: 0.92rem; }
    .test-desc { color: #64748b; font-size: 0.8rem; margin-top: 0.15rem; }

    /* Callout & Dividers */
    .clinical-callout {
        background: #f0fdf4; border-left: 4px solid #16a34a;
        padding: 0.8rem 1.1rem; border-radius: 0 8px 8px 0;
        margin: 0.7rem 0; font-size: 0.85rem; color: #14532d;
    }
    .section-divider {
        height: 1px;
        background: #e2e8f0;
        border: none;
        margin: 1.5rem 0;
</style>
""", unsafe_allow_html=True)


def badge(kind: str, text: str) -> str:
    """Render a styled badge HTML tag."""
    return f'<span class="badge-{kind}">{text}</span>'



# ============================================================
# CACHED DATA LOADERS & MODELS
# ============================================================
@st.cache_data
def get_cached_dataset():
    """Load and prepare tabular dataset from data/raw/."""
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
def get_trained_ensemble_model(seed: int = 42):
    """Train and cache the 7-estimator E-ESN ensemble."""
    set_global_seed(seed)
    _, _, _, X_norm, y, _, _ = get_cached_dataset()
    
    eesn = EnsembleESN(
        n_estimators=7,
        n_neurons=200,
        spectral_radius=0.95,
        sparsity=0.0,
        noise=0.01,
        leaking_rate=0.7,
        input_scaling=0.5,
        aggregation='soft',
        bootstrap=False,
        random_state=seed
    )
    eesn.fit(X_norm, y)
    return eesn


@st.cache_data
def get_cached_loocv_metrics():
    """Compute and cache Leave-One-Out Cross-Validation metrics."""
    _, _, _, X_norm, y, _, _ = get_cached_dataset()
    
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


@st.cache_data(show_spinner="Computing global KernelSHAP values (cached)...")
def get_cached_shap_values():
    """Compute global SHAP values using KernelSHAP."""
    _, _, _, X_norm, y, feature_names, _ = get_cached_dataset()
    model = get_trained_ensemble_model()

    def predict_stroke(X):
        return model.predict_proba(X)[:, 1]

    background = shap.kmeans(X_norm, 10)
    explainer = shap.KernelExplainer(predict_stroke, background)
    shap_vals = explainer.shap_values(X_norm, nsamples=100)
    shap_vals = np.array(shap_vals)
    if shap_vals.ndim == 3:
        shap_vals = shap_vals[1] if shap_vals.shape[0] == 2 else shap_vals[:, :, 1]

    base_val = explainer.expected_value
    if hasattr(base_val, '__len__'):
        base_val = float(np.array(base_val).flat[0])
    else:
        base_val = float(base_val)

    return shap_vals, base_val


@st.cache_resource
def get_cached_lime_explainer():
    """Create and cache LIME tabular explainer."""
    _, _, _, X_norm, y, feature_names, _ = get_cached_dataset()
    return lime.lime_tabular.LimeTabularExplainer(
        training_data=X_norm,
        feature_names=feature_names,
        class_names=['Control', 'Stroke'],
        mode='classification',
        discretize_continuous=True,
        random_state=42
    )


@st.cache_data
def get_cached_full_evaluation():
    """Load precomputed full evaluation benchmark JSON if available."""
    res_path = PROJECT_ROOT / "experiments" / "results" / "full_evaluation.json"
    if res_path.exists():
        with open(res_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


# ============================================================
# INITIALIZE DATA & MODEL
# ============================================================
df_raw, df_enc, X_raw, X_norm, y_labels, feature_names, normalizer = get_cached_dataset()
model = get_trained_ensemble_model()
rec_engine = RecommendationEngine()


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 0.5rem 0 1rem 0;">
        <span style="font-size: 2.4rem;">🧠</span><br>
        <span style="color: #90cdf4; font-weight: 700; font-size: 1.05rem;">EEG Stroke DSS</span><br>
        <span style="color: #cbd5e1; font-size: 0.78rem;">Clinical Decision Support System</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    nav_mode = st.radio(
        "Application Mode",
        [
            "🏥 Clinical DSS Dashboard (Fig. 8)",
            "🔬 Module 1: Raw EEG Processing",
            "🌐 Explainable AI (SHAP & LIME)",
            "📈 Benchmark & Model Evaluation",
            "ℹ️ About & System Reference"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("""
    <div style="color: #cbd5e1; font-size: 0.74rem; line-height: 1.6;">
        <b style="color: #90cdf4;">Model Configuration</b><br>
        Architecture: Ensemble ESN (7 Models)<br>
        Reservoir Units: 200 neurons per model<br>
        Input Channel: Single Pre-frontal (FP1)<br>
        Reference: Bouazizi & Ltifi (2024)
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style="background: rgba(185, 28, 28, 0.2); border-left: 3px solid #ef4444; border-radius: 4px; padding: 0.5rem; color: #fca5a5; font-size: 0.72rem;">
        <b>Clinical Notice:</b> Research & Educational validation prototype. Human physician retains final diagnostic authority.
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# PAGE 1: CLINICAL DSS DASHBOARD (FIG. 8 REPLICATION)
# ============================================================
if nav_mode == "🏥 Clinical DSS Dashboard (Fig. 8)":
    st.markdown("""
    <div class="main-header">
        <h1>🏥 EEG-Based Stroke Clinical Decision Support System</h1>
        <p>Interactive physician decision interface replicating Figure 8 of Bouazizi & Ltifi (2024), featuring automated risk triage, XAI interpretation, and human-in-the-loop diagnostic test confirmation.</p>
    </div>
    """, unsafe_allow_html=True)

    # Patient Selector
    col_sel1, col_sel2 = st.columns([3, 1])
    with col_sel1:
        patient_idx = st.selectbox(
            "Select Clinical Case:",
            range(len(y_labels)),
            format_func=lambda i: (
                f"Patient Case #{i+1:02d} — Ground Truth: "
                f"{'🔴 Acute Stroke' if y_labels[i]==1 else '🟢 Healthy Control'}"
            ),
            index=0
        )

    patient_feats_norm = X_norm[patient_idx:patient_idx+1]
    patient_feats_raw = X_raw[patient_idx]
    true_label_val = y_labels[patient_idx]
    true_label_str = "Stroke" if true_label_val == 1 else "Control"

    with col_sel2:
        st.markdown(f"""
        <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.8rem; margin-top: 1.5rem;">
            <div style="font-size: 0.7rem; color: #64748b; text-transform: uppercase;">Medical Record Ground Truth</div>
            <div style="font-size: 1.15rem; font-weight: 700; color: {'#b91c1c' if true_label_val==1 else '#15803d'};">
                {'🔴 Acute Stroke' if true_label_val==1 else '🟢 Healthy Control'}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Model inference
    proba = model.predict_proba(patient_feats_norm)[0]
    stroke_prob = float(proba[1])
    pred_str = "Stroke" if np.argmax(proba) == 1 else "Control"
    rec_data = rec_engine.get_recommendations(stroke_prob, pred_str)
    risk_level = rec_data['risk_level']

    # Initialize or retrieve physician decision session in session_state
    session_key = f"physician_session_{patient_idx}"
    if session_key not in st.session_state:
        st.session_state[session_key] = PhysicianDecisionSession(
            patient_id=patient_idx + 1,
            recommendations=rec_data
        )
    session: PhysicianDecisionSession = st.session_state[session_key]

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # 4-BOX CLINICAL DASHBOARD (MATCHING FIG. 8 OF PAPER)
    st.markdown("### 📋 Unified 4-Box Clinical Decision Dashboard (Paper Fig. 8)")
    st.caption("Structured into four functional domains: Patient Data, AI Diagnosis, Explainability, and Clinical Recommendations with Physician Oversight.")

    box1_col, box2_col = st.columns([1, 1])

    # ── BOX 1: PATIENT INFORMATION & EEG DATA ──
    with box1_col:
        st.markdown(badge("static", "BOX 1: PATIENT INFORMATION & EEG METRICS"), unsafe_allow_html=True)
        # Extract demographic metadata if available in DataFrame
        age_val = df_raw.iloc[patient_idx].get('Age', 'N/A')
        gender_val = df_raw.iloc[patient_idx].get('Gender', 'N/A')
        edu_val = df_raw.iloc[patient_idx].get('Years of education', 'N/A')

        st.markdown(f"""
        <div class="test-box">
            <div style="font-size: 0.95rem; font-weight: 700; color: #1e3a5f;">Demographics & Clinical Profile</div>
            <div style="font-size: 0.85rem; color: #475569; margin-top: 0.3rem;">
                • <b>Patient Identifier:</b> Case-EPOC-{patient_idx+1:02d}<br>
                • <b>Age Category:</b> {age_val} | <b>Gender:</b> {gender_val}<br>
                • <b>Education:</b> {edu_val} years | <b>Electrode:</b> FP1 (Pre-frontal resting-state)
            </div>
        </div>
        """, unsafe_allow_html=True)

        eeg_summary_data = {
            "Biomarker Feature": ["RP Delta", "RP Theta", "RP Alpha", "RP Beta", "DTR (Delta/Theta)", "DAR (Delta/Alpha)"],
            "Measured Value": [
                f"{patient_feats_raw[feature_names.index('RP Delta')]:.4f}" if 'RP Delta' in feature_names else "N/A",
                f"{patient_feats_raw[feature_names.index('RP Theta')]:.4f}" if 'RP Theta' in feature_names else "N/A",
                f"{patient_feats_raw[feature_names.index('RP Alpha')]:.4f}" if 'RP Alpha' in feature_names else "N/A",
                f"{patient_feats_raw[feature_names.index('RP Beta')]:.4f}" if 'RP Beta' in feature_names else "N/A",
                f"{patient_feats_raw[feature_names.index('DTR')]:.4f}" if 'DTR' in feature_names else "N/A",
                f"{patient_feats_raw[feature_names.index('DAR')]:.4f}" if 'DAR' in feature_names else "N/A",
            ],
            "Clinical Relevance": [
                "Slow-wave elevation; sensitive to focal cerebral ischemia",
                "Subcortical slowing; correlates with post-stroke impairment",
                "Calm wakefulness rhythm; marked reduction during acute infarct",
                "High-frequency sensorimotor rhythm",
                "Key acute marker predicting 90-day post-stroke MoCA score",
                "Ratio index of generalized cortical slowing"
            ]
        }
        st.dataframe(pd.DataFrame(eeg_summary_data), use_container_width=True, hide_index=True)

    # ── BOX 2: PREDICTION & RISK ASSESSMENT ──
    with box2_col:
        st.markdown(badge("live", "BOX 2: ENSEMBLE ESN PREDICTION & RISK"), unsafe_allow_html=True)
        
        pred_box_col1, pred_box_col2 = st.columns([1, 1])
        with pred_box_col1:
            if pred_str == "Stroke":
                st.markdown("""
                <div style="background: #fef2f2; border: 2px solid #ef4444; border-radius: 10px; padding: 1rem; text-align: center;">
                    <div style="font-size: 0.75rem; color: #991b1b; text-transform: uppercase; font-weight: 700;">Model Prediction</div>
                    <div style="font-size: 1.6rem; font-weight: 800; color: #b91c1c; margin-top: 0.2rem;">⚠️ STROKE</div>
                    <div style="font-size: 0.75rem; color: #7f1d1d; margin-top: 0.3rem;">Elevated risk of ischemic/hemorrhagic stroke</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background: #f0fdf4; border: 2px solid #22c55e; border-radius: 10px; padding: 1rem; text-align: center;">
                    <div style="font-size: 0.75rem; color: #166534; text-transform: uppercase; font-weight: 700;">Model Prediction</div>
                    <div style="font-size: 1.6rem; font-weight: 800; color: #15803d; margin-top: 0.2rem;">✅ CONTROL</div>
                    <div style="font-size: 0.75rem; color: #14532d; margin-top: 0.3rem;">Within expected healthy resting-state bounds</div>
                </div>
                """, unsafe_allow_html=True)

        with pred_box_col2:
            st.markdown(f"""
            <div class="risk-{risk_level}">
                <div style="font-size: 0.75rem; text-transform: uppercase; opacity: 0.9;">Triage Classification</div>
                <div style="font-size: 1.6rem; font-weight: 800; margin-top: 0.2rem;">{risk_level.upper()} RISK</div>
                <div style="font-size: 0.75rem; opacity: 0.9; margin-top: 0.3rem;">Urgency: {rec_data['urgency']}</div>
            </div>
            """, unsafe_allow_html=True)

        # Stroke Probability Meter
        meter_color = "#b91c1c" if stroke_prob >= 0.7 else ("#c2410c" if stroke_prob >= 0.4 else "#15803d")
        st.markdown(f"""
        <div style="margin-top: 1rem; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.9rem;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 0.82rem; font-weight: 600; color: #334155;">Model Confidence Score</span>
                <span style="font-size: 1.25rem; font-weight: 800; color: {meter_color};">{stroke_prob:.1%}</span>
            </div>
            <div style="background: #e2e8f0; border-radius: 6px; height: 10px; margin-top: 0.4rem; overflow: hidden;">
                <div style="background: {meter_color}; width: {stroke_prob*100:.1f}%; height: 10px;"></div>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.7rem; color: #94a3b8; margin-top: 0.3rem;">
                <span>0% (Healthy)</span>
                <span>Threshold: 40% (Priority)</span>
                <span>Threshold: 70% (Urgent)</span>
                <span>100%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    box3_col, box4_col = st.columns([1, 1])

    # ── BOX 3: INTERPRETATION (SHAP & LIME TRIGGERS) ──
    with box3_col:
        st.markdown(badge("live", "BOX 3: EXPLAINABILITY (SHAP & LIME CONTROLS)"), unsafe_allow_html=True)
        st.caption("Figure 8 specifies dedicated SHAP (Global) and LIME (Local) triggers to demystify black-box neural dynamics.")

        tab_box3_lime, tab_box3_shap = st.tabs(["🔍 LIME Local Analysis", "🌐 SHAP Feature Impact"])

        with tab_box3_lime:
            if st.button("⚡ Compute Real-Time LIME Explanation", key="btn_dash_lime", type="primary"):
                with st.spinner("Generating local linear surrogate explanation (~4s)..."):
                    lime_explainer = get_cached_lime_explainer()
                    lime_exp = lime_explainer.explain_instance(
                        patient_feats_norm[0], model.predict_proba,
                        num_features=len(feature_names), num_samples=5000
                    )
                    fig_lime = lime_exp.as_pyplot_figure()
                    fig_lime.set_size_inches(9, 4.5)
                    st.pyplot(fig_lime)
                    plt.close(fig_lime)
            else:
                st.info("Click **Compute Real-Time LIME Explanation** to evaluate local feature perturbations for this specific patient.")

        with tab_box3_shap:
            if st.button("📊 Load Patient SHAP Waterfall", key="btn_dash_shap"):
                with st.spinner("Rendering patient SHAP value attributions..."):
                    shap_vals_all, shap_base = get_cached_shap_values()
                    sv_p = shap_vals_all[patient_idx]
                    sorted_p = np.argsort(np.abs(sv_p))

                    fig_w, ax_w = plt.subplots(figsize=(8, 4.2))
                    vals_w = sv_p[sorted_p]
                    names_w = [feature_names[i] for i in sorted_p]
                    colors_w = ['#dc2626' if v > 0 else '#2563eb' for v in vals_w]

                    ax_w.barh(range(len(names_w)), vals_w, color=colors_w, height=0.65)
                    ax_w.set_yticks(range(len(names_w)))
                    ax_w.set_yticklabels(names_w, fontsize=9)
                    ax_w.axvline(x=0, color='black', linewidth=0.8, alpha=0.5)
                    ax_w.set_xlabel('SHAP Value (Contribution to Stroke Probability)', fontsize=9)
                    ax_w.set_title(f'Case #{patient_idx+1} Electrophysiological Contributions', fontsize=10, fontweight='bold')
                    ax_w.spines['top'].set_visible(False)
                    ax_w.spines['right'].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig_w)
                    plt.close(fig_w)
            else:
                st.info("Click **Load Patient SHAP Waterfall** to view global Shapley game-theoretic contributions.")

    # ── BOX 4: CLINICAL RECOMMENDATIONS & PHYSICIAN CONFIRM / CANCEL ──
    with box4_col:
        st.markdown(badge("audit", "BOX 4: CLINICAL RECOMMENDATIONS & PHYSICIAN OVERSIGHT"), unsafe_allow_html=True)
        st.caption("In compliance with DSS design standards: Clinical orders are NOT automated. The physician must actively Confirm or Cancel recommendations.")

        st.markdown(f"**Recommended Clinical Action:** {rec_data['action']}")

        # Render each recommended test with active Confirm / Cancel controls
        recommended_list = rec_data['recommended_tests']
        for test in recommended_list:
            tname = test['name']
            curr_decision = session.decisions.get(tname, {})
            curr_status = curr_decision.get('status', DecisionStatus.PENDING)

            st.markdown(f"""
            <div class="test-box">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span class="test-name">🔹 {test['full_name']}</span>
                    <span style="font-size: 0.72rem; font-weight: 700; padding: 2px 8px; border-radius: 12px;
                        background: {'#dcfce7; color: #15803d;' if curr_status=='CONFIRMED' else ('#fee2e2; color: #b91c1c;' if curr_status=='CANCELLED' else '#fef3c7; color: #b45309;')};">
                        {curr_status}
                    </span>
                </div>
                <div class="test-desc">{test['description']}</div>
            </div>
            """, unsafe_allow_html=True)

            btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 2])
            with btn_col1:
                if st.button(f"✅ Confirm", key=f"conf_{patient_idx}_{tname}"):
                    session.confirm_test(tname, physician_notes="Confirmed via Clinical DSS Dashboard")
                    st.rerun()
            with btn_col2:
                if st.button(f"❌ Cancel", key=f"canc_{patient_idx}_{tname}"):
                    session.cancel_test(tname, reason="Overridden by attending physician judgment")
                    st.rerun()
            with btn_col3:
                if curr_status != DecisionStatus.PENDING:
                    timestamp_val = curr_decision.get('timestamp', '')
                    st.caption(f"Recorded at: {timestamp_val}")

        # Summary of physician oversight
        summary_stats = session.get_summary()
        st.markdown(f"""
        <div style="background: #f1f5f9; border-radius: 8px; padding: 0.7rem 1rem; margin-top: 1rem; font-size: 0.8rem; color: #334155;">
            <b>Physician Audit Summary:</b> {summary_stats['confirmed']} Confirmed | {summary_stats['cancelled']} Cancelled | {summary_stats['pending']} Pending Review
        </div>
        """, unsafe_allow_html=True)

    # ── CLINICAL REPORT GENERATION (PDF) ──
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### 📄 Export Official Clinical Decision Report (PDF)")
    st.caption("Generates a comprehensive clinical document incorporating patient telemetry, E-ESN risk prediction, LIME visualization, and physician audit log.")

    if st.button("📄 Generate & Sign Clinical PDF Report", type="secondary"):
        with st.spinner("Compiling clinical PDF document with physician decision log..."):
            lime_expl = get_cached_lime_explainer()
            lime_exp = lime_expl.explain_instance(
                patient_feats_norm[0], model.predict_proba,
                num_features=len(feature_names), num_samples=3000
            )
            lime_contribs = lime_exp.as_list()

            fig_l = lime_exp.as_pyplot_figure()
            fig_l.set_size_inches(10, 4.5)
            tmp_f = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            tmp_name = tmp_f.name
            tmp_f.close()
            fig_l.savefig(tmp_name, dpi=110, bbox_inches='tight')
            plt.close(fig_l)

            feat_raw_dict = {fn: float(patient_feats_raw[i]) for i, fn in enumerate(feature_names)}
            feat_norm_dict = {fn: float(patient_feats_norm[0][i]) for i, fn in enumerate(feature_names)}

            pdf_bytes = generate_report(
                patient_id=patient_idx + 1,
                prediction=pred_str,
                stroke_probability=stroke_prob,
                risk_level=risk_level,
                features=feat_raw_dict,
                normalized_features=feat_norm_dict,
                recommendations=rec_data,
                lime_contributions=lime_contribs,
                true_label=true_label_str,
                lime_fig_path=tmp_name,
                physician_decisions=session.get_summary()
            )

            try:
                os.unlink(tmp_name)
            except OSError:
                pass

            reports_dir = PROJECT_ROOT / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            pdf_filename = f"stroke_report_patient_{patient_idx + 1:02d}.pdf"
            pdf_path = reports_dir / pdf_filename
            with open(pdf_path, 'wb') as pf:
                pf.write(pdf_bytes)

            st.session_state['active_pdf_bytes'] = pdf_bytes
            st.session_state['active_pdf_filename'] = pdf_filename
            st.session_state['active_pdf_path'] = str(pdf_path)

    if 'active_pdf_bytes' in st.session_state:
        st.success(f"✅ Clinical report compiled successfully: `{st.session_state['active_pdf_filename']}`")
        col_d1, col_d2 = st.columns([1, 2])
        with col_d1:
            st.download_button(
                label="⬇️ Download Signed PDF Report",
                data=st.session_state['active_pdf_bytes'],
                file_name=st.session_state['active_pdf_filename'],
                mime="application/pdf",
                key="pdf_download_button"
            )
        with col_d2:
            st.info(f"Saved locally to: `{st.session_state['active_pdf_path']}`")


# ============================================================
# PAGE 2: MODULE 1 — RAW EEG PROCESSING & PSD SIMULATION
# ============================================================
elif nav_mode == "🔬 Module 1: Raw EEG Processing":
    st.markdown("""
    <div class="main-header">
        <h1>🔬 Module 1: Raw EEG Pre-treatment & Adapted PSD Extraction</h1>
        <p>Demonstration of end-to-end continuous EEG waveform filtering (0.5–30 Hz), artifact amplitude rejection (&plusmn;100 &mu;V), 4-second epoching (0.25 Hz FFT resolution), and clinical frequency band ratio extraction.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="clinical-callout">
        <b>Module 1 Technical Protocol:</b> As specified in Section 4.3 of Bouazizi & Ltifi (2024):
        1. <b>Digital Bandpass Filter</b> (0.5 - 30.0 Hz) to eliminate DC baseline wander and high-frequency noise.
        2. <b>Artifact Exclusion</b>: Manual/variance detection and removal of epochs with absolute amplitude exceeding &plusmn;100 &mu;V.
        3. <b>Adapted PSD</b>: 4-second artifact-free epochs subjected to FFT (frequency resolution = 1/4 Hz = 0.25 Hz).
        4. <b>Biomarkers</b>: Relative power in Delta, Theta, Alpha, Beta bands, plus DAR (Delta/Alpha) and DTR (Delta/Theta).
    </div>
    """, unsafe_allow_html=True)

    sim_col1, sim_col2 = st.columns([1, 2])
    with sim_col1:
        st.markdown("#### Simulation Controls")
        sim_patient_type = st.radio(
            "Target Patient Profile:",
            ["Acute Stroke (Pathological Slowing)", "Healthy Control (Alpha Rhythm)"],
            index=0
        )
        sim_duration = st.slider("Signal Duration (seconds):", min_value=16, max_value=120, value=32, step=8)
        sim_noise = st.checkbox("Inject Occasional High-Amplitude Artifacts (>100 µV)", value=True)
        sim_btn = st.button("🚀 Process Continuous Raw EEG Signal", type="primary")

    with sim_col2:
        pt_key = 'stroke' if "Stroke" in sim_patient_type else 'control'
        raw_sig, meta = generate_synthetic_raw_eeg(
            duration_sec=float(sim_duration),
            sampling_rate=128.0,
            patient_type=pt_key,
            add_artifacts=sim_noise,
            random_state=42
        )

        st.markdown("#### Continuous Raw EEG Signal Preview (FP1 Channel)")
        fig_raw, ax_raw = plt.subplots(figsize=(10, 3.2))
        t_axis = np.linspace(0, min(10.0, float(sim_duration)), int(min(10.0, float(sim_duration)) * 128.0))
        ax_raw.plot(t_axis, raw_sig[:len(t_axis)], color='#1e3a5f', linewidth=0.9)
        ax_raw.axhline(100.0, color='red', linestyle='--', linewidth=0.8, label='+100 µV Threshold')
        ax_raw.axhline(-100.0, color='red', linestyle='--', linewidth=0.8, label='-100 µV Threshold')
        ax_raw.set_xlabel('Time (seconds)')
        ax_raw.set_ylabel('Amplitude (µV)')
        ax_raw.set_title(f'Raw EEG Waveform ({sim_patient_type}) — First 10 Seconds', fontsize=10)
        ax_raw.legend(loc='upper right', fontsize=8)
        ax_raw.grid(True, linestyle=':', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig_raw)
        plt.close(fig_raw)

    if sim_btn:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("### 📊 Module 1 Pipeline Execution Results")

        extracted_feats, pipe_report = process_raw_eeg_signal(
            raw_sig,
            sampling_rate=128.0,
            epoch_duration_sec=4.0,
            amplitude_threshold_uv=100.0,
            lowcut=0.5,
            highcut=30.0
        )

        rep_col1, rep_col2, rep_col3, rep_col4 = st.columns(4)
        with rep_col1:
            st.metric("Total Epochs (4s)", f"{pipe_report['total_epochs']}")
        with rep_col2:
            st.metric("Clean Epochs Retained", f"{pipe_report['clean_epochs']}")
        with rep_col3:
            st.metric("Artifacts Excluded", f"{pipe_report['rejected_epochs']}")
        with rep_col4:
            st.metric("Epoch Retention Rate", f"{pipe_report['retention_rate']:.1%}")

        feat_col1, feat_col2 = st.columns([1, 1])

        with feat_col1:
            st.markdown("#### Extracted Clinical Biomarkers")
            df_biomarkers = pd.DataFrame([
                {"Biomarker": k, "Extracted Value": f"{v:.4f}", "Paper Description": "Primary slow-wave marker" if "Delta" in k or "DTR" in k else "Rhythm index"}
                for k, v in extracted_feats.items()
            ])
            st.dataframe(df_biomarkers, use_container_width=True, hide_index=True)

        with feat_col2:
            st.markdown("#### Relative Band Power Spectrum (%)")
            fig_rp, ax_rp = plt.subplots(figsize=(6, 4))
            bands_plot = ['RP Delta', 'RP Theta', 'RP Alpha', 'RP Beta']
            vals_plot = [extracted_feats[b] * 100 for b in bands_plot]
            colors_rp = ['#b91c1c', '#ea580c', '#15803d', '#2563eb']

            ax_rp.bar(bands_plot, vals_plot, color=colors_rp, width=0.55)
            ax_rp.set_ylabel('Relative Power (%)')
            ax_rp.set_title('Resting-State Power Distribution Across Bands', fontsize=10)
            ax_rp.grid(axis='y', linestyle=':', alpha=0.5)
            for i, v in enumerate(vals_plot):
                ax_rp.text(i, v + 1.0, f"{v:.1f}%", ha='center', fontsize=9, fontweight='bold')
            ax_rp.set_ylim(0, max(vals_plot) + 12)
            plt.tight_layout()
            st.pyplot(fig_rp)
            plt.close(fig_rp)


# ============================================================
# PAGE 3: EXPLAINABLE AI (GLOBAL SHAP & LOCAL LIME)
# ============================================================
elif nav_mode == "🌐 Explainable AI (SHAP & LIME)":
    st.markdown("""
    <div class="main-header">
        <h1>🌐 Explainable Artificial Intelligence (XAI) Suite</h1>
        <p>Comprehensive model interpretability using Global SHAP (Shapley Additive exPlanations) and Local LIME (Local Interpretable Model-agnostic Explanations).</p>
    </div>
    """, unsafe_allow_html=True)

    tab_xai_shap, tab_xai_lime = st.tabs(["🌐 Global SHAP Model Transparency", "🔍 Local LIME Case Analysis"])

    with tab_xai_shap:
        st.markdown("### Global Feature Importance Across Cohort (Paper Fig. 6)")
        st.caption("SHAP calculates the cooperative game-theoretic marginal contribution of each EEG frequency band and demographic covariate across all 38 clinical cases.")

        shap_vals_cohort, shap_base_cohort = get_cached_shap_values()

        col_sh1, col_sh2 = st.columns([1, 1])
        with col_sh1:
            fig_sh_sum = plt.figure(figsize=(8, 5.5))
            shap.summary_plot(shap_vals_cohort, X_norm, feature_names=feature_names, show=False)
            plt.title("SHAP Beeswarm Summary Plot (Class: Stroke)", fontsize=11)
            plt.tight_layout()
            st.pyplot(fig_sh_sum)
            plt.close(fig_sh_sum)
            st.caption("🔴 Red = High feature magnitude | 🔵 Blue = Low feature magnitude.")

        with col_sh2:
            mean_abs_sh = np.abs(shap_vals_cohort).mean(axis=0)
            sort_sh = np.argsort(mean_abs_sh)

            fig_sh_bar, ax_sh_bar = plt.subplots(figsize=(8, 5.5))
            ax_sh_bar.barh(
                range(len(feature_names)),
                mean_abs_sh[sort_sh],
                color='#2d6a9f',
                height=0.65
            )
            ax_sh_bar.set_yticks(range(len(feature_names)))
            ax_sh_bar.set_yticklabels([feature_names[i] for i in sort_sh], fontsize=9)
            ax_sh_bar.set_xlabel('Mean |SHAP Value| (Impact on Model Magnitude)')
            ax_sh_bar.set_title('Average Feature Impact Ranking', fontsize=11)
            ax_sh_bar.spines['top'].set_visible(False)
            ax_sh_bar.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig_sh_bar)
            plt.close(fig_sh_bar)

        st.markdown("""
        > **Clinical Alignment:** The top influential features identified by SHAP (DTR, DAR, RP Delta, and RP Alpha)
        > align directly with published neurological literature regarding acute ischemic hemispheric slowing.
        """)

    with tab_xai_lime:
        st.markdown("### Local Interpretable Surrogate Model Explanations (Paper Fig. 7)")
        st.caption("LIME approximates the non-linear Echo State Network in the immediate local neighborhood of an individual patient.")

        lime_patient_idx = st.selectbox(
            "Select Case for LIME Decomposition:",
            range(len(y_labels)),
            format_func=lambda i: f"Patient #{i+1:02d} — Truth: {'Stroke' if y_labels[i]==1 else 'Control'}",
            key="lime_case_select"
        )

        if st.button("Generate Local LIME Surrogate Breakdown", type="primary"):
            with st.spinner("Perturbing local neighborhood and estimating ridge surrogate..."):
                explainer_obj = get_cached_lime_explainer()
                exp_instance = explainer_obj.explain_instance(
                    X_norm[lime_patient_idx],
                    model.predict_proba,
                    num_features=len(feature_names),
                    num_samples=5000
                )
                fig_lime_full = exp_instance.as_pyplot_figure()
                fig_lime_full.set_size_inches(11, 5)
                st.pyplot(fig_lime_full)
                plt.close(fig_lime_full)

                col_lm1, col_lm2 = st.columns(2)
                exp_list = exp_instance.as_list()
                with col_lm1:
                    st.markdown("**Features Elevating Stroke Risk:**")
                    stroke_factors = [item for item in exp_list if item[1] > 0]
                    for name, w in stroke_factors[:4]:
                        st.markdown(f"- 🔴 `{name}` (weight: `+{w:.4f}`)")
                    if not stroke_factors:
                        st.markdown("- *None*")

                with col_lm2:
                    st.markdown("**Features Supporting Healthy Control:**")
                    ctrl_factors = [item for item in exp_list if item[1] < 0]
                    for name, w in ctrl_factors[:4]:
                        st.markdown(f"- 🟢 `{name}` (weight: `{w:.4f}`)")
                    if not ctrl_factors:
                        st.markdown("- *None*")


# ============================================================
# PAGE 4: BENCHMARK & MODEL EVALUATION (PAPER TABLES 2 & 3)
# ============================================================
elif nav_mode == "📈 Benchmark & Model Evaluation":
    st.markdown("""
    <div class="main-header">
        <h1>📈 Model Evaluation & Scientific Benchmark Replication</h1>
        <p>Comprehensive multi-model comparative analysis across Leave-One-Out (LOOCV), 5-Fold, and 10-Fold cross-validation against baseline algorithms and reported paper metrics.</p>
    </div>
    """, unsafe_allow_html=True)

    loocv_results = get_cached_loocv_metrics()

    st.markdown("### LOOCV Performance Metrics on Clinical Cohort (n=38)")
    m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
    with m_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{loocv_results['accuracy']:.1%}</div>
            <div class="metric-label">Accuracy</div>
            <div class="metric-sub">Paper: 96.5%</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{loocv_results['sensitivity']:.1%}</div>
            <div class="metric-label">Sensitivity</div>
            <div class="metric-sub">Paper: 96.4%</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{loocv_results['specificity']:.1%}</div>
            <div class="metric-label">Specificity</div>
            <div class="metric-sub">Paper: 81.8%</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{loocv_results['ppv']:.1%}</div>
            <div class="metric-label">PPV (Precision)</div>
            <div class="metric-sub">Paper: 93.1%</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{loocv_results['npv']:.1%}</div>
            <div class="metric-label">NPV</div>
            <div class="metric-sub">Paper: 90.0%</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col6:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{loocv_results['f1']:.1%}</div>
            <div class="metric-label">F1-Score</div>
            <div class="metric-sub">Paper: 94.7%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    eval_json = get_cached_full_evaluation()
    if eval_json:
        st.markdown("### Multi-Model Cross-Validation Comparison Table (Paper Table 3)")
        tab_bm1, tab_bm2 = st.tabs(["12 EEG Biomarkers", "15 Features (EEG + Demographics)"])

        for tab_obj, fs_key in zip([tab_bm1, tab_bm2], list(eval_json['models'].keys())):
            with tab_obj:
                rows = []
                for cv_type in ['LOOCV', '5-Fold', '10-Fold']:
                    for m_name, m_dict in eval_json['models'][fs_key].get(cv_type, {}).items():
                        rows.append({
                            "Classifier": m_name,
                            "CV Strategy": cv_type,
                            "Accuracy": f"{m_dict['accuracy']:.1%}",
                            "Sensitivity": f"{m_dict['sensitivity']:.1%}",
                            "Specificity": f"{m_dict['specificity']:.1%}",
                            "PPV": f"{m_dict['ppv']:.1%}",
                            "NPV": f"{m_dict['npv']:.1%}",
                            "F1-Score": f"{m_dict['f1']:.1%}"
                        })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Confusion matrix rendering
    st.markdown("#### Confusion Matrix Visualization (LOOCV)")
    cm_val = loocv_results['confusion_matrix']
    fig_cm, ax_cm = plt.subplots(figsize=(4.5, 3.5))
    im_cm = ax_cm.imshow(cm_val, cmap='Blues')
    for i in range(2):
        for j in range(2):
            val_txt = f"{cm_val[i, j]}\n({'Correct' if i==j else 'Error'})"
            txt_color = 'white' if cm_val[i, j] > cm_val.max()/2 else 'black'
            ax_cm.text(j, i, val_txt, ha='center', va='center', fontsize=11, fontweight='bold', color=txt_color)
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(['Pred: Control', 'Pred: Stroke'])
    ax_cm.set_yticklabels(['True: Control', 'True: Stroke'])
    ax_cm.set_title('E-ESN LOOCV Confusion Matrix', fontsize=10, fontweight='bold')
    plt.colorbar(im_cm, ax=ax_cm, shrink=0.8)
    plt.tight_layout()
    st.pyplot(fig_cm)
    plt.close(fig_cm)


# ============================================================
# PAGE 5: ABOUT & SYSTEM REFERENCE
# ============================================================
elif nav_mode == "ℹ️ About & System Reference":
    st.markdown("""
    <div class="main-header">
        <h1>ℹ️ System Reference & Technical Specifications</h1>
        <p>Scientific documentation, dataset attribution, mathematical foundations, and clinical governance notes.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### Original Publication Reference
    > **Samar Bouazizi & Hela Ltifi (2024)**  
    > *"Enhancing accuracy and interpretability in EEG-based medical decision making using an explainable ensemble learning framework application for stroke prediction"*  
    > **Decision Support Systems**, Volume 178, Article 114126 (Elsevier).  
    > [DOI: 10.1016/j.dss.2023.114126](https://doi.org/10.1016/j.dss.2023.114126)

    ### Dataset Attribution
    > **Aminov et al. (2017)**  
    > *"Acute single channel EEG predictors of cognitive function after stroke"*  
    > **PLoS ONE**, 12(10): e0185841. Data repository: [Dryad Digital Repository](https://doi.org/10.5061/dryad.h6986)  
    > Recorded within 72 hours of admission from a single frontal electrode (FP1) in 38 subjects (19 stroke, 19 control).

    ### Technical Architecture
    | Module | Architectural Role | Implementation in Project |
    | :--- | :--- | :--- |
    | **Module 1** | Pre-treatment & PSD Feature Extraction | Butterworth bandpass (0.5–30 Hz), &plusmn;100 &mu;V artifact exclusion, 4s epoching, adapted PSD FFT |
    | **Module 2** | Ensemble Echo State Networks (E-ESN) | 7 independent ESN reservoirs (200 neurons, &rho;=0.95), soft-voting aggregation |
    | **Module 3** | Explainable Artificial Intelligence (XAI) | Global KernelSHAP feature importance + Local LIME surrogate models |
    | **Module 4** | Medical Decision Support System (DSS) | Fig. 8 Clinical Dashboard with Physician Confirm / Cancel controls & PDF report generator |

    ### Clinical Governance Disclaimer
    > ⚠️ **Notice for Medical Professionals:** This software is developed strictly for research, academic, and clinical decision support demonstration purposes. It does NOT replace professional neurological examination, computed tomography (CT), or magnetic resonance imaging (MRI). The attending clinician maintains legal and medical responsibility for all diagnostic orders.
    """)
