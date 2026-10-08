import React, { useState, useEffect } from 'react';
import {
  X, User, Activity, Brain, ShieldAlert, CheckCircle2,
  ArrowRight, FileText, Layers, Clock, Sparkles, Loader2
} from 'lucide-react';

export default function PatientDetailModal({
  patient,
  onClose,
  onAnalyze
}) {
  const [detail, setDetail] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!patient) return;
    let isMounted = true;
    setLoading(true);

    fetch(`/api/patients/${patient.id}`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => {
        if (isMounted) {
          setDetail(data);
          setLoading(false);
        }
      })
      .catch((err) => {
        console.warn('Could not fetch full patient detail, using summary:', err);
        if (isMounted) {
          setDetail(patient);
          setLoading(false);
        }
      });

    return () => {
      isMounted = false;
    };
  }, [patient]);

  if (!patient) return null;

  const isStroke = (patient.diagnosis || '').toLowerCase().includes('stroke');
  const clinicalProfile = detail?.clinical_profile || {};
  const features = detail?.features || {};
  const normalizedFeatures = detail?.normalized_features || {};

  // Formatted values from profile or fallback
  const age = detail?.age || patient.age || 'N/A';
  const gender = detail?.gender || patient.gender || 'Unknown';
  const handedness = clinicalProfile['Handedness'] || patient.handedness || 'Right';
  const education = clinicalProfile['Years of education'] ?? patient.education ?? 'N/A';
  const admInterval = clinicalProfile['Admission Recording Interval'] ? `${clinicalProfile['Admission Recording Interval']}h` : 'Within 72h';
  const followUp = clinicalProfile['90 days follow-up'] ? `${clinicalProfile['90 days follow-up']} days` : '90 days';

  // EEG features
  const dar = clinicalProfile['DAR'] ?? features['DAR'] ?? patient.dar ?? 0.0;
  const dtr = clinicalProfile['DTR'] ?? features['DTR'] ?? patient.dtr ?? 0.0;
  const totalPower = clinicalProfile['Total Power'] ?? features['Total Power'] ?? patient.total_power ?? 0.0;
  const epochCount = clinicalProfile['Epoch'] ?? features['Epoch'] ?? patient.epoch ?? 18;

  const delta = clinicalProfile['Delta'] ?? features['Delta'] ?? 'N/A';
  const theta = clinicalProfile['Theta'] ?? features['Theta'] ?? 'N/A';
  const alpha = clinicalProfile['Alpha'] ?? features['Alpha'] ?? 'N/A';
  const beta = clinicalProfile['Beta'] ?? features['Beta'] ?? 'N/A';

  const rpDelta = clinicalProfile['RP Delta'] ?? features['RP Delta'] ?? 0.0;
  const rpTheta = clinicalProfile['RP Theta'] ?? features['RP Theta'] ?? 0.0;
  const rpAlpha = clinicalProfile['RP Alpha'] ?? features['RP Alpha'] ?? 0.0;
  const rpBeta = clinicalProfile['RP Beta'] ?? features['RP Beta'] ?? 0.0;

  // Clinical neurological data (Stroke specifics)
  const nihss = clinicalProfile['NIHSS'] ?? clinicalProfile['NIHSS '];
  const nihssSev = clinicalProfile['NIHSS Severity'];
  const strokeType = clinicalProfile['Stroke Type'];
  const strokeHemisphere = clinicalProfile['Stroke Hemisphere'];
  const strokeLoc = clinicalProfile['Stroke location'];
  const mrs = clinicalProfile['MRS baseline'];
  const mocaScore = clinicalProfile['MoCA 90 days score'];
  const mocaImpairment = clinicalProfile['MoCA_Impairment_Healthy_vs_Impaired'];
  const mocaSev = clinicalProfile['MoCA_Impairment_Mild_vs_Severe'];

  return (
    <div className="modal-backdrop" onClick={onClose} role="dialog" aria-modal="true">
      <div
        className="modal-card"
        style={{ maxWidth: '820px', width: '92%', maxHeight: '88vh', padding: '1.75rem' }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Modal Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', borderBottom: '1px solid #e2e8f0', paddingBottom: '1rem', marginBottom: '1.25rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <div style={{ background: isStroke ? '#fee2e2' : '#dcfce7', color: isStroke ? '#dc2626' : '#16a34a', padding: '10px', borderRadius: '10px' }}>
              {isStroke ? <ShieldAlert size={22} /> : <CheckCircle2 size={22} />}
            </div>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
                <h3 style={{ fontSize: '1.2rem', fontWeight: 800, color: '#0f172a', margin: 0, fontFamily: 'var(--font-mono)' }}>
                  {patient.participant_id}
                </h3>
                <span
                  style={{
                    fontSize: '0.75rem',
                    fontWeight: 700,
                    padding: '2px 9px',
                    borderRadius: '12px',
                    background: isStroke ? '#fee2e2' : '#dcfce7',
                    color: isStroke ? '#dc2626' : '#16a34a'
                  }}
                >
                  {isStroke ? 'Acute Stroke' : 'Healthy Control'}
                </span>
                <span style={{ fontSize: '0.72rem', background: '#f1f5f9', color: '#475569', padding: '2px 8px', borderRadius: '8px', fontWeight: 600 }}>
                  Case #{patient.id + 1} of 38
                </span>
              </div>
              <p style={{ fontSize: '0.8rem', color: '#64748b', margin: '0.25rem 0 0 0' }}>
                Full Clinical Study Admission Record &amp; Electrophysiological Biomarkers (Bouazizi &amp; Ltifi, 2024)
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            aria-label="Close modal"
            style={{
              background: '#f1f5f9',
              border: 'none',
              borderRadius: '8px',
              padding: '6px',
              cursor: 'pointer',
              color: '#64748b',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              transition: 'all 0.15s ease'
            }}
          >
            <X size={18} />
          </button>
        </div>

        {loading ? (
          <div style={{ padding: '3rem', textAlign: 'center', color: '#64748b' }}>
            <Loader2 size={32} className="spin" style={{ margin: '0 auto 0.75rem auto', color: '#0284c7' }} />
            <p style={{ fontSize: '0.86rem' }}>Loading full clinical features...</p>
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
            {/* Section 1: Demographics & Admission */}
            <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: '10px', padding: '1rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', marginBottom: '0.75rem', color: '#0f172a', fontWeight: 700, fontSize: '0.86rem' }}>
                <User size={16} color="#0284c7" />
                <span>Demographics &amp; Admission Profile</span>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: '0.75rem' }}>
                <div style={{ background: '#ffffff', padding: '0.65rem 0.85rem', borderRadius: '8px', border: '1px solid #f1f5f9' }}>
                  <div style={{ fontSize: '0.7rem', color: '#64748b', fontWeight: 600 }}>Age Stratum</div>
                  <div style={{ fontSize: '0.92rem', fontWeight: 700, color: '#0f172a' }}>{age} years</div>
                </div>

                <div style={{ background: '#ffffff', padding: '0.65rem 0.85rem', borderRadius: '8px', border: '1px solid #f1f5f9' }}>
                  <div style={{ fontSize: '0.7rem', color: '#64748b', fontWeight: 600 }}>Biological Sex</div>
                  <div style={{ fontSize: '0.92rem', fontWeight: 700, color: '#0f172a' }}>{gender}</div>
                </div>

                <div style={{ background: '#ffffff', padding: '0.65rem 0.85rem', borderRadius: '8px', border: '1px solid #f1f5f9' }}>
                  <div style={{ fontSize: '0.7rem', color: '#64748b', fontWeight: 600 }}>Handedness</div>
                  <div style={{ fontSize: '0.92rem', fontWeight: 700, color: '#0f172a' }}>{handedness}</div>
                </div>

                <div style={{ background: '#ffffff', padding: '0.65rem 0.85rem', borderRadius: '8px', border: '1px solid #f1f5f9' }}>
                  <div style={{ fontSize: '0.7rem', color: '#64748b', fontWeight: 600 }}>Education</div>
                  <div style={{ fontSize: '0.92rem', fontWeight: 700, color: '#0f172a' }}>{education} years</div>
                </div>

                <div style={{ background: '#ffffff', padding: '0.65rem 0.85rem', borderRadius: '8px', border: '1px solid #f1f5f9' }}>
                  <div style={{ fontSize: '0.7rem', color: '#64748b', fontWeight: 600 }}>Admission Recording Window</div>
                  <div style={{ fontSize: '0.92rem', fontWeight: 700, color: '#0f172a' }}>{admInterval}</div>
                </div>

                <div style={{ background: '#ffffff', padding: '0.65rem 0.85rem', borderRadius: '8px', border: '1px solid #f1f5f9' }}>
                  <div style={{ fontSize: '0.7rem', color: '#64748b', fontWeight: 600 }}>Follow-up Period</div>
                  <div style={{ fontSize: '0.92rem', fontWeight: 700, color: '#0f172a' }}>{followUp}</div>
                </div>
              </div>
            </div>

            {/* Section 2: Quantitative EEG Biomarkers (Module 1 - Adapted PSD) */}
            <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: '10px', padding: '1rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', color: '#0f172a', fontWeight: 700, fontSize: '0.86rem' }}>
                  <Activity size={16} color="#0284c7" />
                  <span>Module 1: Quantitative FP1 EEG Spectral Biomarkers</span>
                </div>
                <span style={{ fontSize: '0.7rem', color: '#0369a1', background: '#e0f2fe', padding: '2px 8px', borderRadius: '6px', fontWeight: 600 }}>
                  Single Channel FP1 (0.5–30 Hz)
                </span>
              </div>

              {/* Highlight Ratios */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '0.75rem', marginBottom: '0.75rem' }}>
                <div style={{ background: '#ffffff', border: '1px solid #cbd5e1', borderRadius: '8px', padding: '0.75rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                    <span style={{ fontSize: '0.75rem', fontWeight: 700, color: '#475569' }}>DAR (Delta / Alpha)</span>
                    <span style={{ fontSize: '0.66rem', color: Number(dar) > 3.7 ? '#dc2626' : '#16a34a', fontWeight: 700 }}>
                      {Number(dar) > 3.7 ? 'Elevated (>3.70)' : 'Normal (<=3.70)'}
                    </span>
                  </div>
                  <div style={{ fontSize: '1.25rem', fontWeight: 800, color: Number(dar) > 3.7 ? '#dc2626' : '#0f172a', fontFamily: 'var(--font-mono)' }}>
                    {typeof dar === 'number' ? dar.toFixed(3) : dar}
                  </div>
                  <div style={{ fontSize: '0.68rem', color: '#64748b', marginTop: '2px' }}>Slow-to-fast power slowing index</div>
                </div>

                <div style={{ background: '#ffffff', border: '1px solid #cbd5e1', borderRadius: '8px', padding: '0.75rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                    <span style={{ fontSize: '0.75rem', fontWeight: 700, color: '#475569' }}>DTR (Delta / Theta)</span>
                    <span style={{ fontSize: '0.66rem', color: '#0284c7', fontWeight: 700 }}>Top Paper Biomarker</span>
                  </div>
                  <div style={{ fontSize: '1.25rem', fontWeight: 800, color: '#0f172a', fontFamily: 'var(--font-mono)' }}>
                    {typeof dtr === 'number' ? dtr.toFixed(3) : dtr}
                  </div>
                  <div style={{ fontSize: '0.68rem', color: '#64748b', marginTop: '2px' }}>Cognitive predictor (MoCA at 90d)</div>
                </div>

                <div style={{ background: '#ffffff', border: '1px solid #cbd5e1', borderRadius: '8px', padding: '0.75rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                    <span style={{ fontSize: '0.75rem', fontWeight: 700, color: '#475569' }}>Total Power &amp; Epochs</span>
                    <span style={{ fontSize: '0.66rem', color: '#64748b' }}>4.0s FFT epochs</span>
                  </div>
                  <div style={{ fontSize: '1.25rem', fontWeight: 800, color: '#0f172a', fontFamily: 'var(--font-mono)' }}>
                    {typeof totalPower === 'number' ? totalPower.toFixed(2) : totalPower} <span style={{ fontSize: '0.76rem', color: '#64748b', fontWeight: 500 }}>µV²</span>
                  </div>
                  <div style={{ fontSize: '0.68rem', color: '#64748b', marginTop: '2px' }}>{epochCount} artifact-free epochs analyzed</div>
                </div>
              </div>

              {/* Sub-band absolute & relative powers */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '0.5rem' }}>
                <div style={{ background: '#ffffff', padding: '0.6rem', borderRadius: '6px', border: '1px solid #e2e8f0', textAlign: 'center' }}>
                  <div style={{ fontSize: '0.7rem', color: '#64748b', fontWeight: 600 }}>Delta (0.5–4Hz)</div>
                  <div style={{ fontSize: '0.85rem', fontWeight: 700, color: '#0f172a', fontFamily: 'var(--font-mono)' }}>{typeof delta === 'number' ? delta.toFixed(3) : delta}</div>
                  <div style={{ fontSize: '0.68rem', color: '#dc2626', fontWeight: 600 }}>RP: {(Number(rpDelta) * 100).toFixed(1)}%</div>
                </div>

                <div style={{ background: '#ffffff', padding: '0.6rem', borderRadius: '6px', border: '1px solid #e2e8f0', textAlign: 'center' }}>
                  <div style={{ fontSize: '0.7rem', color: '#64748b', fontWeight: 600 }}>Theta (4–8Hz)</div>
                  <div style={{ fontSize: '0.85rem', fontWeight: 700, color: '#0f172a', fontFamily: 'var(--font-mono)' }}>{typeof theta === 'number' ? theta.toFixed(3) : theta}</div>
                  <div style={{ fontSize: '0.68rem', color: '#d97706', fontWeight: 600 }}>RP: {(Number(rpTheta) * 100).toFixed(1)}%</div>
                </div>

                <div style={{ background: '#ffffff', padding: '0.6rem', borderRadius: '6px', border: '1px solid #e2e8f0', textAlign: 'center' }}>
                  <div style={{ fontSize: '0.7rem', color: '#64748b', fontWeight: 600 }}>Alpha (8–13Hz)</div>
                  <div style={{ fontSize: '0.85rem', fontWeight: 700, color: '#0f172a', fontFamily: 'var(--font-mono)' }}>{typeof alpha === 'number' ? alpha.toFixed(3) : alpha}</div>
                  <div style={{ fontSize: '0.68rem', color: '#0284c7', fontWeight: 600 }}>RP: {(Number(rpAlpha) * 100).toFixed(1)}%</div>
                </div>

                <div style={{ background: '#ffffff', padding: '0.6rem', borderRadius: '6px', border: '1px solid #e2e8f0', textAlign: 'center' }}>
                  <div style={{ fontSize: '0.7rem', color: '#64748b', fontWeight: 600 }}>Beta (13–30Hz)</div>
                  <div style={{ fontSize: '0.85rem', fontWeight: 700, color: '#0f172a', fontFamily: 'var(--font-mono)' }}>{typeof beta === 'number' ? beta.toFixed(3) : beta}</div>
                  <div style={{ fontSize: '0.68rem', color: '#16a34a', fontWeight: 600 }}>RP: {(Number(rpBeta) * 100).toFixed(1)}%</div>
                </div>
              </div>
            </div>

            {/* Section 3: Clinical Neurological Assessment (if available in record) */}
            {(nihss !== undefined || strokeType || mocaScore !== undefined) && (
              <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: '10px', padding: '1rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', marginBottom: '0.75rem', color: '#0f172a', fontWeight: 700, fontSize: '0.86rem' }}>
                  <FileText size={16} color="#0284c7" />
                  <span>Clinical Stroke Severity &amp; Cognitive Outcomes</span>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: '0.75rem' }}>
                  {strokeType && (
                    <div style={{ background: '#ffffff', padding: '0.65rem 0.85rem', borderRadius: '8px', border: '1px solid #f1f5f9' }}>
                      <div style={{ fontSize: '0.7rem', color: '#64748b', fontWeight: 600 }}>Stroke Classification</div>
                      <div style={{ fontSize: '0.88rem', fontWeight: 700, color: '#0f172a' }}>{strokeType} ({strokeHemisphere || 'N/A'})</div>
                      {strokeLoc && <div style={{ fontSize: '0.68rem', color: '#64748b' }}>Location: {strokeLoc}</div>}
                    </div>
                  )}

                  {nihss !== undefined && nihss !== 'N/A' && (
                    <div style={{ background: '#ffffff', padding: '0.65rem 0.85rem', borderRadius: '8px', border: '1px solid #f1f5f9' }}>
                      <div style={{ fontSize: '0.7rem', color: '#64748b', fontWeight: 600 }}>NIHSS Severity</div>
                      <div style={{ fontSize: '0.88rem', fontWeight: 700, color: '#0f172a' }}>Score {nihss} • {nihssSev || 'Standard'}</div>
                      {mrs !== undefined && mrs !== 'N/A' && <div style={{ fontSize: '0.68rem', color: '#64748b' }}>mRS Baseline: {mrs}</div>}
                    </div>
                  )}

                  {mocaScore !== undefined && mocaScore !== 'N/A' && (
                    <div style={{ background: '#ffffff', padding: '0.65rem 0.85rem', borderRadius: '8px', border: '1px solid #f1f5f9' }}>
                      <div style={{ fontSize: '0.7rem', color: '#64748b', fontWeight: 600 }}>MoCA 90-Day Follow-Up</div>
                      <div style={{ fontSize: '0.88rem', fontWeight: 700, color: '#0f172a' }}>{mocaScore} / 30 pts</div>
                      <div style={{ fontSize: '0.68rem', color: mocaImpairment === 'Impaired' ? '#dc2626' : '#16a34a' }}>
                        {mocaImpairment || 'Evaluated'} ({mocaSev || ''})
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Section 4: AI Model Normalized Features (Inputs to E-ESN) */}
            {Object.keys(normalizedFeatures).length > 0 && (
              <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: '10px', padding: '1rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.65rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', color: '#0f172a', fontWeight: 700, fontSize: '0.86rem' }}>
                    <Brain size={16} color="#0284c7" />
                    <span>Module 2: Normalized Features [0, 1] Vector (Input to Ensemble ESN)</span>
                  </div>
                  <span style={{ fontSize: '0.68rem', color: '#64748b' }}>Min-Max Scaled</span>
                </div>

                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.45rem' }}>
                  {Object.entries(normalizedFeatures).map(([k, v]) => (
                    <div
                      key={k}
                      style={{
                        background: '#ffffff',
                        border: '1px solid #e2e8f0',
                        borderRadius: '6px',
                        padding: '4px 8px',
                        fontSize: '0.72rem',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '6px'
                      }}
                    >
                      <span style={{ color: '#475569', fontWeight: 600 }}>{k}:</span>
                      <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: '#0284c7' }}>
                        {typeof v === 'number' ? v.toFixed(3) : v}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Modal Footer */}
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.75rem', marginTop: '1.5rem', borderTop: '1px solid #e2e8f0', paddingTop: '1rem' }}>
          <button
            onClick={onClose}
            style={{
              padding: '0.45rem 1rem',
              borderRadius: '8px',
              border: '1px solid #cbd5e1',
              background: '#ffffff',
              color: '#475569',
              fontWeight: 600,
              fontSize: '0.82rem',
              cursor: 'pointer'
            }}
          >
            Close
          </button>

          <button
            onClick={() => {
              onAnalyze(patient.id);
              onClose();
            }}
            style={{
              padding: '0.45rem 1.15rem',
              borderRadius: '8px',
              border: 'none',
              background: '#0284c7',
              color: '#ffffff',
              fontWeight: 700,
              fontSize: '0.82rem',
              cursor: 'pointer',
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.45rem',
              boxShadow: '0 2px 4px rgba(2, 132, 199, 0.25)'
            }}
          >
            <span>Analyze in Clinical DSS</span>
            <ArrowRight size={15} />
          </button>
        </div>
      </div>
    </div>
  );
}
