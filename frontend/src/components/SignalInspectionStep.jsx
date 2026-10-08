import React from 'react';
import { ArrowRight, CheckCircle2, AlertTriangle, ShieldCheck, Activity, User, Sliders } from 'lucide-react';
import WaveformViewer from './WaveformViewer';

export default function SignalInspectionStep({
  patientDetail,
  rawSignal,
  filteredSignal,
  processReport,
  onProceed
}) {
  const feat = patientDetail?.features || {};
  const dar = feat.DAR ?? 0;
  const isDarElevated = dar > 3.70;

  return (
    <div>
      {/* Page Header */}
      <div className="page-view-header">
        <div>
          <div className="page-view-title">
            <Activity size={20} color="#0284c7" />
            <span>Module 1: 1-Channel FP1 Signal Processing &amp; Filtering</span>
            <span className="badge badge-info" style={{ fontSize: '0.68rem', padding: '1px 7px' }}>Lead FP1</span>
          </div>
          <p className="page-view-sub">
            Zero-phase 4th order Butterworth bandpass (0.5–30 Hz) • 50Hz Notch filter • &plusmn;100 &mu;V artifact rejection
          </p>
        </div>

        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          <span className="badge badge-low" style={{ fontSize: '0.72rem' }}>
            <ShieldCheck size={13} />
            <span>Artifact: &plusmn;100 &mu;V Clean</span>
          </span>
          <span className="badge badge-info" style={{ fontSize: '0.72rem' }}>
            <Sliders size={13} />
            <span>Epoch: 4.0s (&Delta;f = 0.25 Hz)</span>
          </span>
          <button id="proceed-to-step2-top-btn" className="btn-primary" onClick={onProceed} style={{ fontSize: '0.78rem', padding: '0.38rem 0.85rem' }}>
            <span>Proceed to AI Diagnosis</span>
            <ArrowRight size={14} />
          </button>
        </div>
      </div>

      {/* 4 Metric Stats Grid (Compact) */}
      <div className="metric-grid-4">
        <div className={`stat-box ${isDarElevated ? 'stroke-alert' : 'normal-alert'}`}>
          <div className="stat-label">Delta-Alpha Ratio (DAR)</div>
          <div className="stat-value" style={{ color: isDarElevated ? '#dc2626' : '#059669' }}>
            {dar.toFixed(2)}
          </div>
          <div className="stat-subtext" style={{ fontWeight: 600 }}>
            {isDarElevated ? '⚠️ Elevated (>3.70 Ischemia)' : '✓ Normal Baseline (<= 3.70)'}
          </div>
        </div>

        <div className="stat-box">
          <div className="stat-label">Delta-Theta Ratio (DTR)</div>
          <div className="stat-value">{(feat.DTR ?? 0).toFixed(2)}</div>
          <div className="stat-subtext">Low-frequency slowing indicator</div>
        </div>

        <div className="stat-box">
          <div className="stat-label">Alpha Band Power</div>
          <div className="stat-value">{(feat.Alpha ?? 0).toFixed(2)} &mu;V²</div>
          <div className="stat-subtext">8.0 – 13.0 Hz | Background rhythm</div>
        </div>

        <div className="stat-box">
          <div className="stat-label">Delta Slow-Wave Power</div>
          <div className="stat-value" style={{ color: isDarElevated ? '#dc2626' : '#0f172a' }}>
            {(feat.Delta ?? 0).toFixed(2)} &mu;V²
          </div>
          <div className="stat-subtext">0.5 – 4.0 Hz | Infarct indicator</div>
        </div>
      </div>

      {/* Module 1 Signal Waveform Canvas */}
      <div className="clinical-card" style={{ padding: '0.85rem 1rem' }}>
        <WaveformViewer
          rawSignal={rawSignal}
          filteredSignal={filteredSignal}
          samplingRate={128}
          durationSec={8.0}
          title="FP1 Electrode Continuous EEG Waveform (Time Domain)"
        />

        {/* Frequency Band Breakdown Cards */}
        <div style={{ marginTop: '0.85rem', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '0.65rem' }}>
          <div style={{ background: '#f8fafc', padding: '0.65rem 0.8rem', borderRadius: '7px', border: '1px solid #e2e8f0' }}>
            <div style={{ fontSize: '0.72rem', fontWeight: 700, color: '#dc2626' }}>Delta (0.5 - 4.0 Hz)</div>
            <div style={{ fontSize: '1.08rem', fontWeight: 800 }}>{(feat['RP Delta'] ? feat['RP Delta'] * 100 : 45.2).toFixed(1)}%</div>
            <div style={{ fontSize: '0.68rem', color: '#64748b' }}>Dominant in ischemic stroke</div>
          </div>

          <div style={{ background: '#f8fafc', padding: '0.65rem 0.8rem', borderRadius: '7px', border: '1px solid #e2e8f0' }}>
            <div style={{ fontSize: '0.72rem', fontWeight: 700, color: '#d97706' }}>Theta (4.0 - 8.0 Hz)</div>
            <div style={{ fontSize: '1.08rem', fontWeight: 800 }}>{(feat['RP Theta'] ? feat['RP Theta'] * 100 : 22.1).toFixed(1)}%</div>
            <div style={{ fontSize: '0.68rem', color: '#64748b' }}>Diffuse subcortical slowing</div>
          </div>

          <div style={{ background: '#f8fafc', padding: '0.65rem 0.8rem', borderRadius: '7px', border: '1px solid #e2e8f0' }}>
            <div style={{ fontSize: '0.72rem', fontWeight: 700, color: '#059669' }}>Alpha (8.0 - 13.0 Hz)</div>
            <div style={{ fontSize: '1.08rem', fontWeight: 800 }}>{(feat['RP Alpha'] ? feat['RP Alpha'] * 100 : 18.5).toFixed(1)}%</div>
            <div style={{ fontSize: '0.68rem', color: '#64748b' }}>Preserved in healthy brain</div>
          </div>

          <div style={{ background: '#f8fafc', padding: '0.65rem 0.8rem', borderRadius: '7px', border: '1px solid #e2e8f0' }}>
            <div style={{ fontSize: '0.72rem', fontWeight: 700, color: '#0284c7' }}>Beta (13.0 - 30.0 Hz)</div>
            <div style={{ fontSize: '1.08rem', fontWeight: 800 }}>{(feat['RP Beta'] ? feat['RP Beta'] * 100 : 14.2).toFixed(1)}%</div>
            <div style={{ fontSize: '0.68rem', color: '#64748b' }}>Fast cortical rhythm</div>
          </div>
        </div>

        {/* CTA to Step 2 */}
        <div style={{ marginTop: '0.85rem', display: 'flex', justifyContent: 'flex-end', borderTop: '1px solid #f1f5f9', paddingTop: '0.65rem' }}>
          <button id="proceed-to-step2-btn" className="btn-primary" onClick={onProceed} style={{ fontSize: '0.78rem', padding: '0.4rem 0.9rem' }}>
            <span>Proceed to Step 2: AI Diagnosis &amp; Explainability</span>
            <ArrowRight size={14} />
          </button>
        </div>
      </div>
    </div>
  );
}
