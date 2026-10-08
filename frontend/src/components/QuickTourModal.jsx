import React from 'react';
import { X, Activity, Brain, ShieldCheck, ClipboardCheck, Sparkles, HelpCircle } from 'lucide-react';

export default function QuickTourModal({ onClose }) {
  return (
    <div className="modal-backdrop">
      <div className="modal-card" style={{ maxWidth: '680px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.2rem', borderBottom: '1px solid #e2e8f0', paddingBottom: '0.75rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
            <div style={{ background: '#e0f2fe', color: '#0284c7', padding: '6px', borderRadius: '6px' }}>
              <HelpCircle size={20} />
            </div>
            <div>
              <h2 style={{ fontSize: '1.15rem', fontWeight: 700, color: '#0f172a' }}>Platform User Guide & Architecture</h2>
              <p style={{ fontSize: '0.8rem', color: '#64748b' }}>Bouazizi & Ltifi (Decision Support Systems 2024)</p>
            </div>
          </div>
          <button onClick={onClose} style={{ background: 'transparent', border: 'none', cursor: 'pointer', color: '#64748b' }}>
            <X size={20} />
          </button>
        </div>

        {/* 10-Second Quick Start */}
        <div style={{ background: '#f0fdf4', border: '1px solid #bbf7d0', borderRadius: '8px', padding: '1rem', marginBottom: '1.2rem' }}>
          <h4 style={{ fontSize: '0.88rem', fontWeight: 700, color: '#166534', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <Sparkles size={16} />
            <span>How to Test in 10 Seconds (One-Click)</span>
          </h4>
          <ol style={{ fontSize: '0.82rem', color: '#15803d', marginTop: '0.4rem', paddingLeft: '1.2rem', lineHeight: 1.6 }}>
            <li>Click <strong>"Acute Stroke"</strong> or <strong>"Healthy Control"</strong> button in the top navigation bar.</li>
            <li>In <strong>Step 1</strong>, observe the continuous 1-channel FP1 waveform and filtered frequency band powers.</li>
            <li>Click <strong>Proceed to Step 2</strong> to inspect the Ensemble ESN stroke risk probability and SHAP feature attributions.</li>
            <li>Click <strong>Proceed to Step 3</strong> to confirm or cancel clinical tests and download the signed PDF report.</li>
          </ol>
        </div>

        {/* The 4 Modules Breakdown */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem', marginBottom: '1.5rem' }}>
          <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'flex-start' }}>
            <div style={{ background: '#ccfbf1', color: '#0d9488', padding: '6px', borderRadius: '6px', marginTop: '2px' }}>
              <Activity size={18} />
            </div>
            <div>
              <h5 style={{ fontSize: '0.88rem', fontWeight: 700, color: '#0f172a' }}>Module 1: Pre-treatment (Signal Processing & Adapted PSD)</h5>
              <p style={{ fontSize: '0.8rem', color: '#475569' }}>
                Takes continuous 1-channel FP1 EEG, applies a zero-phase Butterworth filter (0.5–30 Hz), a 50Hz notch filter, and rejects movement artifacts exceeding &plusmn;100 µV. Epoching into 4.0-second segments guarantees exact 0.25 Hz FFT spectral resolution.
              </p>
            </div>
          </div>

          <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'flex-start' }}>
            <div style={{ background: '#e0f2fe', color: '#0284c7', padding: '6px', borderRadius: '6px', marginTop: '2px' }}>
              <Brain size={18} />
            </div>
            <div>
              <h5 style={{ fontSize: '0.88rem', fontWeight: 700, color: '#0f172a' }}>Module 2: AI Classification (Ensemble Echo State Network)</h5>
              <p style={{ fontSize: '0.8rem', color: '#475569' }}>
                Uses an ensemble of 7 Echo State Networks (Reservoir Computing) with 200 random recurrent neurons and a spectral radius of 0.95. Ridge regression linear readout prevents overfitting on small clinical cohorts while capturing temporal electrophysiological dynamics.
              </p>
            </div>
          </div>

          <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'flex-start' }}>
            <div style={{ background: '#fef3c7', color: '#d97706', padding: '6px', borderRadius: '6px', marginTop: '2px' }}>
              <Sparkles size={18} />
            </div>
            <div>
              <h5 style={{ fontSize: '0.88rem', fontWeight: 700, color: '#0f172a' }}>Module 3: Explainable AI (SHAP & LIME)</h5>
              <p style={{ fontSize: '0.8rem', color: '#475569' }}>
                Eliminates the "black-box" dilemma by computing cooperative game theory Shapley values (SHAP) and local linear perturbations (LIME). Shows physicians exactly which frequency bands increased or decreased acute stroke risk.
              </p>
            </div>
          </div>

          <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'flex-start' }}>
            <div style={{ background: '#dcfce7', color: '#16a34a', padding: '6px', borderRadius: '6px', marginTop: '2px' }}>
              <ClipboardCheck size={18} />
            </div>
            <div>
              <h5 style={{ fontSize: '0.88rem', fontWeight: 700, color: '#0f172a' }}>Module 4: Decision Support System (Human-in-the-Loop)</h5>
              <p style={{ fontSize: '0.8rem', color: '#475569' }}>
                Translates risk predictions into recommended clinical pathways (Brain MRI, non-contrast CT, Coagulation panel). The attending physician maintains final authority to Confirm or Cancel each test with full audit trail logging and PDF reporting.
              </p>
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
          <button className="btn-primary" onClick={onClose}>
            Got it, Let's Explore!
          </button>
        </div>
      </div>
    </div>
  );
}
