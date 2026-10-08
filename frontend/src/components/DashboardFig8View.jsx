import React, { useState } from 'react';
import {
  User, Activity, Brain, ShieldAlert, CheckCircle, ShieldCheck,
  ClipboardCheck, Sparkles, FileDown, Eye, Info, CheckCircle2, XCircle
} from 'lucide-react';
import WaveformViewer from './WaveformViewer';

export default function DashboardFig8View({
  patientDetail,
  prediction,
  shapData,
  limeData,
  recommendations,
  sessionState,
  onRecordAction,
  onDownloadPdf,
  downloadingPdf,
  rawSignal,
  filteredSignal
}) {
  const [activeXaiTab, setActiveXaiTab] = useState('lime'); // Paper Fig. 8 highlights LIME primarily
  const [showSignalModal, setShowSignalModal] = useState(false);
  const [actionModalTest, setActionModalTest] = useState(null);
  const [actionModalType, setActionModalType] = useState('CONFIRMED');
  const [modalNotes, setModalNotes] = useState('');
  const [physicianId, setPhysicianId] = useState('Dr. Clinical Neurologist');

  const feat = patientDetail?.features || {};
  const strokeProb = prediction ? prediction.stroke_probability * 100 : 50;
  const isStroke = prediction?.prediction === 'Acute Stroke' || (strokeProb >= 50);
  const riskLevel = prediction?.risk_level || 'Medium';

  const dar = feat.DAR ?? 0;
  const isDarElevated = dar > 3.70;

  const handleOpenAction = (test, action) => {
    setActionModalTest(test);
    setActionModalType(action);
    setModalNotes(action === 'CONFIRMED' ? 'Confirmed acute stroke diagnostic intervention.' : 'Overridden by attending physician clinical judgment.');
  };

  const handleConfirmSubmit = () => {
    if (!actionModalTest) return;
    onRecordAction({
      test_id: actionModalTest.id,
      action: actionModalType,
      notes: modalNotes,
      physician_id: physicianId
    });
    setActionModalTest(null);
  };

  return (
    <div>
      {/* Paper Fig. 8 Sub-header */}
      <div className="page-view-header">
        <div>
          <div className="page-view-title">
            <Brain size={20} color="#0284c7" />
            <span>Unified 4-Box Clinical Decision Dashboard</span>
            <span className="badge badge-teal" style={{ fontSize: '0.68rem', padding: '1px 7px' }}>Paper Fig. 8</span>
          </div>
          <p className="page-view-sub">
            Exact architectural replication of <strong>Figure 8</strong> (Bouazizi &amp; Ltifi, Decision Support Systems 2024)
          </p>
        </div>

        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <button
            className="btn-secondary"
            onClick={() => setShowSignalModal(true)}
            style={{ fontSize: '0.78rem', padding: '0.38rem 0.75rem' }}
          >
            <Activity size={14} color="#0284c7" />
            <span>Inspect 1-Channel Waveform</span>
          </button>

          <button
            id="download-pdf-top-btn"
            className="btn-primary"
            onClick={onDownloadPdf}
            disabled={downloadingPdf}
            style={{ fontSize: '0.78rem', padding: '0.38rem 0.85rem' }}
          >
            <FileDown size={14} />
            <span>{downloadingPdf ? 'Exporting PDF...' : 'Download Clinical PDF'}</span>
          </button>
        </div>
      </div>

      {/* 2x2 Grid Layout — Exact 4 Boxes of Paper Figure 8 */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(460px, 1fr))', gap: '1rem', marginBottom: '1rem', alignItems: 'stretch' }}>
        
        {/* ── BOX 1: PATIENT INFORMATION & EEG DATA (TOP-LEFT) ── */}
        <div className="clinical-card" style={{ marginBottom: 0, display: 'flex', flexDirection: 'column' }}>
          <div className="card-header-flex">
            <div className="card-title-group">
              <div style={{ background: '#e0f2fe', color: '#0284c7', padding: '5px', borderRadius: '6px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <User size={16} />
              </div>
              <div>
                <span className="badge badge-info" style={{ fontSize: '0.64rem', padding: '1px 6px', marginBottom: '1px' }}>
                  BOX 1: PATIENT DATA &amp; TELEMETRY
                </span>
                <h3 style={{ fontSize: '0.98rem', fontWeight: 700, color: '#0f172a' }}>Patient Profile &amp; EEG Parameters</h3>
              </div>
            </div>
            <span className="badge badge-teal" style={{ fontSize: '0.7rem' }}>FP1 Frontal Lead</span>
          </div>

          {/* Demographics Strip */}
          <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: '7px', padding: '0.35rem 0.75rem', marginBottom: '0.5rem', fontSize: '0.76rem', color: '#334155', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>ID: <strong>{patientDetail?.participant_id || 'P_001'}</strong></div>
            <div>Age: <strong>{patientDetail?.age || 'N/A'}</strong></div>
            <div>Gender: <strong>{patientDetail?.gender || 'N/A'}</strong></div>
            <div>Window: <strong>&lt;72h Acute</strong></div>
          </div>

          {/* EEG Biomarker Table matching Fig. 8 */}
          <div style={{ overflowX: 'auto', flex: 1 }}>
            <table className="audit-table" style={{ fontSize: '0.76rem', marginTop: 0 }}>
              <thead>
                <tr>
                  <th style={{ padding: '0.45rem 0.65rem' }}>Biomarker Feature</th>
                  <th style={{ padding: '0.45rem 0.65rem' }}>Measured Value</th>
                  <th style={{ padding: '0.45rem 0.65rem' }}>Clinical Target &amp; Status</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td style={{ padding: '0.38rem 0.65rem', fontWeight: 600 }}>Delta-Alpha Ratio (DAR)</td>
                  <td style={{ padding: '0.38rem 0.65rem', fontFamily: 'monospace', fontWeight: 700, color: isDarElevated ? '#dc2626' : '#059669' }}>
                    {dar.toFixed(2)}
                  </td>
                  <td style={{ padding: '0.38rem 0.65rem' }}>
                    <span className={`badge ${isDarElevated ? 'badge-high' : 'badge-low'}`} style={{ fontSize: '0.68rem', padding: '1px 6px' }}>
                      {isDarElevated ? 'Elevated (>3.70 Ischemia)' : 'Normal (<=3.70)'}
                    </span>
                  </td>
                </tr>
                <tr>
                  <td style={{ padding: '0.38rem 0.65rem', fontWeight: 600 }}>Delta-Theta Ratio (DTR)</td>
                  <td style={{ padding: '0.38rem 0.65rem', fontFamily: 'monospace', fontWeight: 700 }}>{(feat.DTR ?? 0).toFixed(2)}</td>
                  <td style={{ padding: '0.38rem 0.65rem', color: '#64748b' }}>Low-frequency slowing ratio</td>
                </tr>
                <tr>
                  <td style={{ padding: '0.38rem 0.65rem', fontWeight: 600 }}>RP Delta (0.5 – 4.0 Hz)</td>
                  <td style={{ padding: '0.38rem 0.65rem', fontFamily: 'monospace', fontWeight: 700 }}>{((feat['RP Delta'] ?? 0.45) * 100).toFixed(1)}%</td>
                  <td style={{ padding: '0.38rem 0.65rem', color: '#64748b' }}>Slow-wave power fraction</td>
                </tr>
                <tr>
                  <td style={{ padding: '0.38rem 0.65rem', fontWeight: 600 }}>RP Alpha (8.0 – 13.0 Hz)</td>
                  <td style={{ padding: '0.38rem 0.65rem', fontFamily: 'monospace', fontWeight: 700 }}>{((feat['RP Alpha'] ?? 0.20) * 100).toFixed(1)}%</td>
                  <td style={{ padding: '0.38rem 0.65rem', color: '#64748b' }}>Resting wakefulness rhythm</td>
                </tr>
                <tr>
                  <td style={{ padding: '0.38rem 0.65rem', fontWeight: 600 }}>Total Power</td>
                  <td style={{ padding: '0.38rem 0.65rem', fontFamily: 'monospace', fontWeight: 700 }}>{(feat['Total Power'] ?? 0).toFixed(2)} µV²</td>
                  <td style={{ padding: '0.38rem 0.65rem', color: '#64748b' }}>Full-band power (0.5–30 Hz)</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* ── BOX 2: PREDICTION & RISK ASSESSMENT (TOP-RIGHT) ── */}
        <div className="clinical-card" style={{ marginBottom: 0, display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
          <div>
            <div className="card-header-flex">
              <div className="card-title-group">
                <div style={{ background: '#fef2f2', color: '#dc2626', padding: '5px', borderRadius: '6px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <Brain size={16} />
                </div>
                <div>
                  <span className="badge badge-high" style={{ fontSize: '0.64rem', padding: '1px 6px', marginBottom: '1px' }}>
                    BOX 2: E-ESN MODEL PREDICTION
                  </span>
                  <h3 style={{ fontSize: '0.98rem', fontWeight: 700, color: '#0f172a' }}>Stroke Risk Assessment</h3>
                </div>
              </div>
              <span className="badge badge-info" style={{ fontSize: '0.7rem' }}>7 Estimators (SR=0.95)</span>
            </div>

            {/* Prediction & Triage Badges Side-by-Side */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem', margin: '0.5rem 0' }}>
              <div style={{
                background: isStroke ? '#fef2f2' : '#f0fdf4',
                border: `1.5px solid ${isStroke ? '#ef4444' : '#22c55e'}`,
                borderRadius: '8px',
                padding: '0.55rem 0.65rem',
                textAlign: 'center'
              }}>
                <div style={{ fontSize: '0.68rem', textTransform: 'uppercase', fontWeight: 700, color: isStroke ? '#991b1b' : '#166534' }}>
                  Classification
                </div>
                <div style={{ fontSize: '1.08rem', fontWeight: 800, color: isStroke ? '#b91c1c' : '#15803d', marginTop: '0.1rem' }}>
                  {isStroke ? '⚠️ ACUTE STROKE' : '✅ HEALTHY CONTROL'}
                </div>
                <div style={{ fontSize: '0.68rem', color: isStroke ? '#7f1d1d' : '#14532d', marginTop: '0.15rem' }}>
                  {isStroke ? 'Elevated ischemic dynamics' : 'Intact cortical background'}
                </div>
              </div>

              <div style={{
                background: riskLevel === 'High' ? '#fee2e2' : riskLevel === 'Medium' ? '#fef3c7' : '#dcfce7',
                border: `1.5px solid ${riskLevel === 'High' ? '#fca5a5' : riskLevel === 'Medium' ? '#fde68a' : '#86efac'}`,
                borderRadius: '8px',
                padding: '0.55rem 0.65rem',
                textAlign: 'center'
              }}>
                <div style={{ fontSize: '0.68rem', textTransform: 'uppercase', fontWeight: 700, color: '#475569' }}>
                  Triage Risk Tier
                </div>
                <div style={{ fontSize: '1.08rem', fontWeight: 800, color: riskLevel === 'High' ? '#dc2626' : riskLevel === 'Medium' ? '#d97706' : '#16a34a', marginTop: '0.1rem' }}>
                  {riskLevel.toUpperCase()} RISK
                </div>
                <div style={{ fontSize: '0.68rem', color: '#475569', marginTop: '0.15rem' }}>
                  Probability: <strong>{strokeProb.toFixed(1)}%</strong>
                </div>
              </div>
            </div>

            {/* Probability Progress Bar */}
            <div style={{ marginTop: '0.45rem', background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: '7px', padding: '0.55rem 0.75rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.76rem', fontWeight: 600, color: '#334155' }}>
                <span>Model Confidence</span>
                <span style={{ color: isStroke ? '#dc2626' : '#16a34a', fontWeight: 800 }}>{strokeProb.toFixed(1)}%</span>
              </div>
              <div style={{ background: '#e2e8f0', height: '7px', borderRadius: '4px', overflow: 'hidden', marginTop: '0.3rem' }}>
                <div style={{
                  width: `${strokeProb}%`,
                  height: '100%',
                  background: isStroke ? 'linear-gradient(90deg, #f87171, #dc2626)' : 'linear-gradient(90deg, #4ade80, #16a34a)',
                  transition: 'width 0.5s ease'
                }}></div>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.65rem', color: '#94a3b8', marginTop: '0.2rem' }}>
                <span>0% (Control)</span>
                <span>40% (Medium)</span>
                <span>70% (High)</span>
                <span>100% (Stroke)</span>
              </div>
            </div>
          </div>

          <div style={{ fontSize: '0.72rem', color: '#64748b', marginTop: '0.5rem', background: '#f1f5f9', padding: '0.4rem 0.65rem', borderRadius: '6px' }}>
            💡 <strong>Clinical Alert:</strong> {prediction?.clinical_alert || 'Real-time classification based on pre-frontal resting electrophysiology.'}
          </div>
        </div>

        {/* ── BOX 3: MODEL INTERPRETABILITY (XAI) (BOTTOM-LEFT) ── */}
        <div className="clinical-card" style={{ marginBottom: 0, display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
          <div>
            <div className="card-header-flex">
              <div className="card-title-group">
                <div style={{ background: '#fef3c7', color: '#d97706', padding: '5px', borderRadius: '6px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <Sparkles size={16} />
                </div>
                <div>
                  <span className="badge badge-med" style={{ fontSize: '0.64rem', padding: '1px 6px', marginBottom: '1px' }}>
                    BOX 3: EXPLAINABILITY (XAI)
                  </span>
                  <h3 style={{ fontSize: '0.98rem', fontWeight: 700, color: '#0f172a' }}>Decision Attribution (LIME &amp; SHAP)</h3>
                </div>
              </div>

              {/* Toggle XAI Method */}
              <div style={{ display: 'flex', gap: '0.25rem', background: '#f1f5f9', padding: '2px', borderRadius: '6px' }}>
                <button
                  onClick={() => setActiveXaiTab('lime')}
                  style={{
                    border: 'none',
                    padding: '0.25rem 0.55rem',
                    borderRadius: '4px',
                    fontSize: '0.72rem',
                    fontWeight: 600,
                    cursor: 'pointer',
                    background: activeXaiTab === 'lime' ? '#ffffff' : 'transparent',
                    color: activeXaiTab === 'lime' ? '#0284c7' : '#64748b',
                    boxShadow: activeXaiTab === 'lime' ? '0 1px 2px rgba(0,0,0,0.06)' : 'none'
                  }}
                >
                  LIME (Local)
                </button>
                <button
                  onClick={() => setActiveXaiTab('shap')}
                  style={{
                    border: 'none',
                    padding: '0.25rem 0.55rem',
                    borderRadius: '4px',
                    fontSize: '0.72rem',
                    fontWeight: 600,
                    cursor: 'pointer',
                    background: activeXaiTab === 'shap' ? '#ffffff' : 'transparent',
                    color: activeXaiTab === 'shap' ? '#0284c7' : '#64748b',
                    boxShadow: activeXaiTab === 'shap' ? '0 1px 2px rgba(0,0,0,0.06)' : 'none'
                  }}
                >
                  SHAP (Waterfall)
                </button>
              </div>
            </div>

            <p style={{ fontSize: '0.74rem', color: '#64748b', marginBottom: '0.45rem' }}>
              Features pushing prediction toward <strong>Stroke (Red)</strong> or <strong>Control (Green)</strong>:
            </p>

            {/* LIME Tab (Figure 8 default) */}
            {activeXaiTab === 'lime' && (
              <div>
                {limeData?.contributions?.slice(0, 5).map((c, i) => {
                  const isRisk = c.direction === 'risk_increasing';
                  const widthPct = Math.min(100, Math.max(12, Math.abs(c.weight) * 320));
                  return (
                    <div key={i} className="xai-bar-row" style={{ marginBottom: '0.35rem' }}>
                      <div className="xai-label" style={{ width: '160px', fontSize: '0.75rem' }} title={c.rule}>
                        {c.rule}
                      </div>
                      <div className="xai-bar-track" style={{ height: '14px' }}>
                        <div className={`xai-bar-fill ${isRisk ? 'risk' : 'protective'}`} style={{ width: `${widthPct}%` }}></div>
                      </div>
                      <div className="xai-val" style={{ color: isRisk ? '#dc2626' : '#059669', fontSize: '0.74rem', width: '55px' }}>
                        {c.weight > 0 ? `+${c.weight.toFixed(3)}` : c.weight.toFixed(3)}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            {/* SHAP Tab */}
            {activeXaiTab === 'shap' && (
              <div>
                {shapData?.contributions?.slice(0, 5).map((c, i) => {
                  const isRisk = c.direction === 'risk_increasing';
                  const widthPct = Math.min(100, Math.max(12, c.relative_impact_pct * 2.2));
                  return (
                    <div key={i} className="xai-bar-row" style={{ marginBottom: '0.35rem' }}>
                      <div className="xai-label" style={{ width: '160px', fontSize: '0.75rem' }} title={c.feature}>
                        {c.feature} ({c.feature_value})
                      </div>
                      <div className="xai-bar-track" style={{ height: '14px' }}>
                        <div className={`xai-bar-fill ${isRisk ? 'risk' : 'protective'}`} style={{ width: `${widthPct}%` }}></div>
                      </div>
                      <div className="xai-val" style={{ color: isRisk ? '#dc2626' : '#059669', fontSize: '0.74rem', width: '55px' }}>
                        {c.shap_value > 0 ? `+${c.shap_value.toFixed(3)}` : c.shap_value.toFixed(3)}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          <div style={{ marginTop: '0.5rem', fontSize: '0.72rem', color: '#475569', background: '#f8fafc', padding: '0.45rem 0.65rem', borderRadius: '6px', border: '1px solid #e2e8f0' }}>
            <strong>Physiological Rationale:</strong> {shapData?.clinical_summary || 'Elevated slow-wave Delta and pathological DAR elevation dominate the acute stroke prediction.'}
          </div>
        </div>

        {/* ── BOX 4: CLINICAL RECOMMENDATIONS & PHYSICIAN ACTION (BOTTOM-RIGHT) ── */}
        <div className="clinical-card" style={{ marginBottom: 0, display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
          <div>
            <div className="card-header-flex">
              <div className="card-title-group">
                <div style={{ background: '#dcfce7', color: '#16a34a', padding: '5px', borderRadius: '6px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <ClipboardCheck size={16} />
                </div>
                <div>
                  <span className="badge badge-low" style={{ fontSize: '0.64rem', padding: '1px 6px', marginBottom: '1px' }}>
                    BOX 4: CLINICAL DSS &amp; PHYSICIAN REVIEW
                  </span>
                  <h3 style={{ fontSize: '0.98rem', fontWeight: 700, color: '#0f172a' }}>Recommended Interventions</h3>
                </div>
              </div>
              <span className="badge badge-teal" style={{ fontSize: '0.7rem' }}>Human-in-the-Loop</span>
            </div>

            <p style={{ fontSize: '0.74rem', color: '#64748b', marginBottom: '0.45rem' }}>
              Confirm or cancel diagnostic orders per Section 4.6 (MRI, CBC, BMP, Coagulation):
            </p>

            {/* Test Cards List with Confirm/Cancel Buttons */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem', maxHeight: '200px', overflowY: 'auto', paddingRight: '2px' }}>
              {recommendations.map((t) => {
                const isConfirmed = t.status === 'CONFIRMED';
                const isCancelled = t.status === 'CANCELLED';

                return (
                  <div
                    key={t.id}
                    style={{
                      background: isConfirmed ? '#f0fdf4' : isCancelled ? '#f8fafc' : '#ffffff',
                      border: `1px solid ${isConfirmed ? '#86efac' : isCancelled ? '#cbd5e1' : '#e2e8f0'}`,
                      borderRadius: '7px',
                      padding: '0.45rem 0.65rem',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      opacity: isCancelled ? 0.75 : 1
                    }}
                  >
                    <div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                        <span style={{ fontSize: '0.8rem', fontWeight: 700, color: '#0f172a' }}>{t.name}</span>
                        <span className={`badge ${t.urgency === 'CRITICAL' ? 'badge-high' : 'badge-info'}`} style={{ fontSize: '0.62rem', padding: '1px 5px' }}>
                          {t.urgency}
                        </span>
                      </div>
                      <div style={{ fontSize: '0.68rem', color: '#64748b', marginTop: '0.1rem' }}>
                        {t.target_time} | Status: <strong>{t.status}</strong>
                      </div>
                    </div>

                    <div style={{ display: 'flex', gap: '0.3rem' }}>
                      <button
                        className="btn-success"
                        onClick={() => handleOpenAction(t, 'CONFIRMED')}
                        style={{ padding: '0.22rem 0.55rem', fontSize: '0.72rem', background: isConfirmed ? '#059669' : '#10b981' }}
                      >
                        <CheckCircle2 size={12} />
                        <span>{isConfirmed ? 'Confirmed' : 'Confirm'}</span>
                      </button>

                      <button
                        className="btn-secondary"
                        onClick={() => handleOpenAction(t, 'CANCELLED')}
                        style={{ padding: '0.22rem 0.55rem', fontSize: '0.72rem', color: isCancelled ? '#475569' : '#dc2626' }}
                      >
                        <XCircle size={12} />
                        <span>{isCancelled ? 'Cancelled' : 'Cancel'}</span>
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Audit Trail Ticker */}
          <div style={{ borderTop: '1px solid #e2e8f0', paddingTop: '0.5rem', marginTop: '0.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.72rem', color: '#64748b' }}>
            <span>Audit: <strong>{sessionState?.confirmed_count ?? 0} Confirmed</strong> | <strong>{sessionState?.cancelled_count ?? 0} Cancelled</strong></span>
            <span style={{ color: '#0284c7', fontWeight: 600 }}>Recorded in PDF Audit Trail ✓</span>
          </div>
        </div>

      </div>

      {/* Waveform Inspection Modal (Module 1) */}
      {showSignalModal && (
        <div className="modal-backdrop">
          <div className="modal-card" style={{ maxWidth: '850px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <h3 style={{ fontSize: '1.15rem', fontWeight: 700, color: '#0f172a' }}>Module 1: Pre-treatment Signal Processing Lab</h3>
              <button className="btn-secondary" onClick={() => setShowSignalModal(false)}>Close</button>
            </div>
            <WaveformViewer
              rawSignal={rawSignal}
              filteredSignal={filteredSignal}
              samplingRate={128}
              durationSec={8.0}
              title="1-Channel FP1 Resting-State Signal (Zero-Phase Butterworth + 50Hz Notch + Artifact Filter)"
            />
          </div>
        </div>
      )}

      {/* Confirmation & Note Modal */}
      {actionModalTest && (
        <div className="modal-backdrop">
          <div className="modal-card">
            <h3 style={{ fontSize: '1.15rem', fontWeight: 700, color: '#0f172a', marginBottom: '0.4rem' }}>
              {actionModalType === 'CONFIRMED' ? 'Confirm Diagnostic Order' : 'Cancel Clinical Recommendation'}
            </h3>
            <p style={{ fontSize: '0.84rem', color: '#64748b', marginBottom: '1rem' }}>
              Intervention: <strong>{actionModalTest.name}</strong>
            </p>

            <div style={{ marginBottom: '0.85rem' }}>
              <label style={{ display: 'block', fontSize: '0.8rem', fontWeight: 600, color: '#475569', marginBottom: '0.3rem' }}>
                Attending Physician:
              </label>
              <input
                type="text"
                value={physicianId}
                onChange={(e) => setPhysicianId(e.target.value)}
                style={{ width: '100%', padding: '0.5rem', borderRadius: '6px', border: '1px solid #cbd5e1', fontSize: '0.85rem' }}
              />
            </div>

            <div style={{ marginBottom: '1.2rem' }}>
              <label style={{ display: 'block', fontSize: '0.8rem', fontWeight: 600, color: '#475569', marginBottom: '0.3rem' }}>
                Clinical Rationale:
              </label>
              <textarea
                rows={3}
                value={modalNotes}
                onChange={(e) => setModalNotes(e.target.value)}
                style={{ width: '100%', padding: '0.5rem', borderRadius: '6px', border: '1px solid #cbd5e1', fontSize: '0.85rem', fontFamily: 'inherit' }}
              />
            </div>

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.6rem' }}>
              <button className="btn-secondary" onClick={() => setActionModalTest(null)}>Cancel</button>
              <button
                className={actionModalType === 'CONFIRMED' ? 'btn-success' : 'btn-primary'}
                style={{ background: actionModalType === 'CONFIRMED' ? '#10b981' : '#dc2626' }}
                onClick={handleConfirmSubmit}
              >
                Save Action
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
