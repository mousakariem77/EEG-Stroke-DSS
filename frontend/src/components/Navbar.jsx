import React from 'react';
import { Activity, Brain, UserCheck, AlertCircle, HelpCircle, FileText, RefreshCw } from 'lucide-react';

export default function Navbar({
  patients,
  selectedPatientId,
  onSelectPatient,
  onSelectPreset,
  activePreset,
  onOpenTour,
  loading
}) {
  return (
    <header className="clinical-navbar">
      <div className="navbar-inner">
        {/* Brand */}
        <div className="brand-section">
          <div className="brand-icon-box">
            <Brain size={24} />
          </div>
          <div className="brand-titles">
            <h1>Explainable EEG Stroke DSS</h1>
            <p>1-Channel FP1 Resting-State Neuro-Decision Support Platform</p>
          </div>
        </div>

        {/* 1-Click Quick Demo Presets */}
        <div className="demo-preset-group">
          <span style={{ fontSize: '0.75rem', fontWeight: 600, color: '#64748b', padding: '0 6px' }}>
            DEMO PRESETS:
          </span>
          <button
            id="preset-stroke-btn"
            className={`preset-btn ${activePreset === 'stroke' ? 'active-stroke' : ''}`}
            onClick={() => onSelectPreset('stroke')}
            title="Load an acute stroke patient case"
          >
            <AlertCircle size={15} color="#dc2626" />
            <span>Acute Stroke</span>
          </button>

          <button
            id="preset-control-btn"
            className={`preset-btn ${activePreset === 'control' ? 'active-control' : ''}`}
            onClick={() => onSelectPreset('control')}
            title="Load a healthy control patient case"
          >
            <UserCheck size={15} color="#059669" />
            <span>Healthy Control</span>
          </button>

          <button
            id="preset-sim-btn"
            className={`preset-btn ${activePreset === 'sim' ? 'active-sim' : ''}`}
            onClick={() => onSelectPreset('sim')}
            title="Simulate continuous 128Hz raw EEG waveform"
          >
            <Activity size={15} color="#0284c7" />
            <span>Raw Simulator</span>
          </button>
        </div>

        {/* Patient Selector & Tour */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <label htmlFor="patient-select" style={{ fontSize: '0.8rem', fontWeight: 600, color: '#64748b' }}>
              Patient:
            </label>
            <select
              id="patient-select"
              value={selectedPatientId ?? ''}
              onChange={(e) => onSelectPatient(Number(e.target.value))}
              disabled={loading}
              style={{
                padding: '0.45rem 0.8rem',
                borderRadius: '8px',
                border: '1px solid #cbd5e1',
                fontSize: '0.84rem',
                fontWeight: 600,
                color: '#0f172a',
                background: '#ffffff',
                cursor: 'pointer'
              }}
            >
              {patients.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.participant_id} ({p.diagnosis}) - Age {p.age}
                </option>
              ))}
            </select>
          </div>

          <button
            id="quick-tour-btn"
            className="btn-secondary"
            onClick={onOpenTour}
            title="How to use this system"
            style={{ padding: '0.45rem 0.85rem' }}
          >
            <HelpCircle size={16} color="#0284c7" />
            <span>User Guide</span>
          </button>
        </div>
      </div>
    </header>
  );
}
