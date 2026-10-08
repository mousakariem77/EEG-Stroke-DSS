import React, { useState } from 'react';
import { Brain, ArrowRight, ArrowLeft, ShieldAlert, CheckCircle, Info, Sparkles, Sliders } from 'lucide-react';

export default function DiagnosisStep({
  prediction,
  shapData,
  limeData,
  onProceed,
  onBack
}) {
  const [activeXaiTab, setActiveXaiTab] = useState('shap');

  const strokeProb = prediction ? prediction.stroke_probability * 100 : 50;
  const riskLevel = prediction?.risk_level || 'Medium';
  const isHigh = riskLevel === 'High';
  const isMed = riskLevel === 'Medium';
  const isLow = riskLevel === 'Low';

  const riskClass = isHigh ? 'high' : isMed ? 'med' : 'low';
  const badgeClass = isHigh ? 'badge-high' : isMed ? 'badge-med' : 'badge-low';

  return (
    <div>
      {/* Page Header */}
      <div className="page-view-header">
        <div>
          <div className="page-view-title">
            <Brain size={20} color="#0284c7" />
            <span>Module 2 &amp; 3: Ensemble Echo State Network (E-ESN) Diagnosis &amp; XAI</span>
            <span className="badge badge-high" style={{ fontSize: '0.68rem', padding: '1px 7px' }}>7 Reservoirs (SR=0.95)</span>
          </div>
          <p className="page-view-sub">
            Sub-second non-linear reservoir dynamics with KernelSHAP waterfall &amp; LIME local attributions
          </p>
        </div>

        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          <button className="btn-secondary" onClick={onBack} style={{ fontSize: '0.78rem', padding: '0.38rem 0.75rem' }}>
            <ArrowLeft size={14} />
            <span>Back to Signal</span>
          </button>
          <button id="proceed-to-step3-top-btn" className="btn-primary" onClick={onProceed} style={{ fontSize: '0.78rem', padding: '0.38rem 0.85rem' }}>
            <span>Proceed to Clinical DSS</span>
            <ArrowRight size={14} />
          </button>
        </div>
      </div>

      {/* 2-Column Grid: AI Probability on Left, Plain-English Clinical Reasoning on Right */}
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(280px, 1fr) 2fr', gap: '1rem', marginBottom: '1rem', alignItems: 'stretch' }}>
        {/* Left: Probability Gauge */}
        <div className="gauge-wrapper" style={{ padding: '0.85rem 1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', color: '#64748b', fontSize: '0.74rem', fontWeight: 600 }}>
            <Brain size={14} color="#0284c7" />
            <span>AI ENSEMBLE ESN PREDICTION</span>
          </div>

          <div className={`gauge-pct-huge ${riskClass}`}>
            {strokeProb.toFixed(1)}%
          </div>

          <div style={{ marginBottom: '0.6rem' }}>
            <span className={`badge ${badgeClass}`} style={{ fontSize: '0.78rem', padding: '0.25rem 0.85rem' }}>
              {isHigh && <ShieldAlert size={14} />}
              {isLow && <CheckCircle size={14} />}
              <span>{prediction?.prediction?.toUpperCase()} ({riskLevel.toUpperCase()} RISK)</span>
            </span>
          </div>

          <div style={{ fontSize: '0.72rem', color: '#64748b', borderTop: '1px solid #e2e8f0', paddingTop: '0.5rem', width: '100%', textAlign: 'center' }}>
            <div>Model Confidence: <strong>{(prediction?.confidence ? prediction.confidence * 100 : 85).toFixed(1)}%</strong></div>
            <div style={{ marginTop: '0.15rem' }}>Architecture: <strong>7 ESN Reservoirs (200 Neurons, SR=0.95)</strong></div>
          </div>
        </div>

        {/* Right: Plain-English Reasoning & Clinical Guidance */}
        <div className="clinical-card" style={{ marginBottom: 0, padding: '0.85rem 1rem', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
          <div>
            <div className="card-header-flex" style={{ marginBottom: '0.55rem', paddingBottom: '0.45rem' }}>
              <div className="card-title-group">
                <div style={{ background: '#fef3c7', color: '#d97706', padding: '5px', borderRadius: '6px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <Sparkles size={16} />
                </div>
                <div>
                  <h3 style={{ fontSize: '0.98rem', fontWeight: 700, color: '#0f172a' }}>Plain-English Clinical AI Explanation</h3>
                  <p style={{ fontSize: '0.74rem', color: '#64748b' }}>Why did the Ensemble ESN model reach this diagnostic conclusion?</p>
                </div>
              </div>
            </div>

            {/* Alert Box */}
            <div className="reasoning-box" style={{ margin: '0.4rem 0', padding: '0.65rem 0.85rem' }}>
              <h4 style={{ fontSize: '0.82rem' }}>
                <Info size={14} />
                <span>Primary Electrophysiological Findings</span>
              </h4>
              <p style={{ fontSize: '0.78rem', lineHeight: 1.45 }}>
                {prediction?.clinical_alert ||
                  "The patient exhibits characteristic acute stroke electrophysiological dynamics in channel FP1. Slow-wave Delta activity is abnormally elevated, while high-frequency Alpha/Beta rhythms show marked attenuation."}
              </p>
            </div>

            {/* Biomarker status indicator */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.65rem', marginTop: '0.55rem' }}>
              <div style={{ background: '#f8fafc', padding: '0.55rem 0.75rem', borderRadius: '7px', border: '1px solid #e2e8f0' }}>
                <span style={{ fontSize: '0.7rem', color: '#64748b', fontWeight: 600 }}>DAR Reference Status</span>
                <div style={{ fontSize: '0.88rem', fontWeight: 700, color: isHigh ? '#dc2626' : '#059669', marginTop: '0.1rem' }}>
                  {prediction?.dar_reference_status || 'DAR: Normal'}
                </div>
              </div>

              <div style={{ background: '#f8fafc', padding: '0.55rem 0.75rem', borderRadius: '7px', border: '1px solid #e2e8f0' }}>
                <span style={{ fontSize: '0.7rem', color: '#64748b', fontWeight: 600 }}>Recommended Next Step</span>
                <div style={{ fontSize: '0.88rem', fontWeight: 700, color: '#0284c7', marginTop: '0.1rem' }}>
                  {isHigh ? 'Urgent Brain MRI (DWI) within 3h' : 'Routine Neuro Assessment'}
                </div>
              </div>
            </div>
          </div>

          <div style={{ fontSize: '0.7rem', color: '#94a3b8', fontStyle: 'italic', marginTop: '0.5rem' }}>
            * Note: EEG screening complements neuroimaging and does NOT replace urgent CT/MRI confirmation.
          </div>
        </div>
      </div>

      {/* Module 3: Explainable AI (XAI) Deep Dive */}
      <div className="clinical-card" style={{ padding: '0.85rem 1rem' }}>
        <div className="card-header-flex" style={{ marginBottom: '0.55rem', paddingBottom: '0.45rem' }}>
          <div className="card-title-group">
            <div style={{ background: '#e0f2fe', color: '#0284c7', padding: '5px', borderRadius: '6px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Brain size={16} />
            </div>
            <div>
              <h2 style={{ fontSize: '0.98rem' }}>Module 3: Transparent Feature Attributions (XAI)</h2>
              <p style={{ fontSize: '0.74rem' }}>Quantitative breakdown of individual EEG biomarker contributions to the stroke decision</p>
            </div>
          </div>

          {/* XAI Tab Toggle */}
          <div style={{ display: 'flex', gap: '0.25rem', background: '#f1f5f9', padding: '2px', borderRadius: '6px' }}>
            <button
              onClick={() => setActiveXaiTab('shap')}
              style={{
                border: 'none',
                padding: '0.25rem 0.65rem',
                borderRadius: '4px',
                fontSize: '0.74rem',
                fontWeight: 600,
                cursor: 'pointer',
                background: activeXaiTab === 'shap' ? '#ffffff' : 'transparent',
                color: activeXaiTab === 'shap' ? '#0284c7' : '#64748b',
                boxShadow: activeXaiTab === 'shap' ? '0 1px 2px rgba(0,0,0,0.06)' : 'none'
              }}
            >
              SHAP Waterfall (KernelSHAP)
            </button>
            <button
              onClick={() => setActiveXaiTab('lime')}
              style={{
                border: 'none',
                padding: '0.25rem 0.65rem',
                borderRadius: '4px',
                fontSize: '0.74rem',
                fontWeight: 600,
                cursor: 'pointer',
                background: activeXaiTab === 'lime' ? '#ffffff' : 'transparent',
                color: activeXaiTab === 'lime' ? '#0284c7' : '#64748b',
                boxShadow: activeXaiTab === 'lime' ? '0 1px 2px rgba(0,0,0,0.06)' : 'none'
              }}
            >
              LIME Local Explanations
            </button>
          </div>
        </div>

        {/* SHAP Tab View */}
        {activeXaiTab === 'shap' && (
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.76rem', color: '#64748b', marginBottom: '0.55rem' }}>
              <span>Base Expected Risk: <strong>{(shapData?.base_value ? shapData.base_value * 100 : 51.7).toFixed(1)}%</strong></span>
              <span>Model Output Risk: <strong>{strokeProb.toFixed(1)}%</strong></span>
            </div>

            {/* Horizontal Bar Chart for SHAP */}
            <div>
              {shapData?.contributions?.slice(0, 7).map((c, i) => {
                const isRisk = c.direction === 'risk_increasing';
                const widthPct = Math.min(100, Math.max(8, c.relative_impact_pct * 2.2));

                return (
                  <div key={i} className="xai-bar-row" style={{ marginBottom: '0.38rem' }}>
                    <div className="xai-label" style={{ fontSize: '0.76rem', width: '160px' }} title={c.feature}>
                      {c.feature} ({c.feature_value})
                    </div>
                    <div className="xai-bar-track" style={{ height: '14px' }}>
                      <div
                        className={`xai-bar-fill ${isRisk ? 'risk' : 'protective'}`}
                        style={{ width: `${widthPct}%` }}
                      ></div>
                    </div>
                    <div className="xai-val" style={{ color: isRisk ? '#dc2626' : '#059669', fontSize: '0.74rem', width: '55px' }}>
                      {c.shap_value > 0 ? `+${c.shap_value.toFixed(3)}` : c.shap_value.toFixed(3)}
                    </div>
                  </div>
                );
              })}
            </div>

            <div style={{ display: 'flex', gap: '1.25rem', marginTop: '0.65rem', fontSize: '0.72rem', color: '#64748b' }}>
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem' }}>
                <span style={{ width: 10, height: 10, background: '#ef4444', borderRadius: '2px' }}></span>
                <span>Red (+): Increases Acute Stroke Risk (Slow-wave Delta, high DAR)</span>
              </span>
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem' }}>
                <span style={{ width: 10, height: 10, background: '#10b981', borderRadius: '2px' }}></span>
                <span>Green (-): Protective / Lowers Risk (Intact Alpha, normal background)</span>
              </span>
            </div>
          </div>
        )}

        {/* LIME Tab View */}
        {activeXaiTab === 'lime' && (
          <div>
            <div style={{ fontSize: '0.76rem', color: '#64748b', marginBottom: '0.55rem' }}>
              Local Interpretable Model-agnostic Explanations (LIME) creates local perturbations to explain non-linear decision boundaries:
            </div>

            <div>
              {limeData?.contributions?.slice(0, 6).map((c, i) => {
                const isRisk = c.direction === 'risk_increasing';
                const widthPct = Math.min(100, Math.max(10, Math.abs(c.weight) * 300));

                return (
                  <div key={i} className="xai-bar-row" style={{ marginBottom: '0.38rem' }}>
                    <div className="xai-label" style={{ width: '180px', fontSize: '0.76rem' }} title={c.rule}>
                      {c.rule}
                    </div>
                    <div className="xai-bar-track" style={{ height: '14px' }}>
                      <div
                        className={`xai-bar-fill ${isRisk ? 'risk' : 'protective'}`}
                        style={{ width: `${widthPct}%` }}
                      ></div>
                    </div>
                    <div className="xai-val" style={{ color: isRisk ? '#dc2626' : '#059669', fontSize: '0.74rem', width: '55px' }}>
                      {c.weight > 0 ? `+${c.weight.toFixed(3)}` : c.weight.toFixed(3)}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Navigation CTAs */}
        <div style={{ marginTop: '0.85rem', display: 'flex', justifyContent: 'space-between', borderTop: '1px solid #f1f5f9', paddingTop: '0.65rem' }}>
          <button className="btn-secondary" onClick={onBack} style={{ fontSize: '0.78rem', padding: '0.38rem 0.75rem' }}>
            <ArrowLeft size={14} />
            <span>Back to Signal Inspection</span>
          </button>
          <button id="proceed-to-step3-btn" className="btn-primary" onClick={onProceed} style={{ fontSize: '0.78rem', padding: '0.38rem 0.85rem' }}>
            <span>Proceed to Step 3: Clinical Decisions &amp; Report</span>
            <ArrowRight size={14} />
          </button>
        </div>
      </div>
    </div>
  );
}
