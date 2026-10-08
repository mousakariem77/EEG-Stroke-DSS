import React from 'react';
import { Menu } from 'lucide-react';

export default function TopBar({
  isCollapsed,
  onToggleSidebar,
  activePage,
  patients,
  selectedPatientId,
  onSelectPatient,
  loading
}) {
  const pageTitles = {
    dashboard: 'Clinical Decision Dashboard (Paper Fig. 8)',
    signal: 'Module 1: 1-Channel FP1 Signal Processing & Filtering',
    diagnosis: 'Module 2: Ensemble Echo State Network (E-ESN) Diagnosis',
    explainability: 'Module 3: Explainable AI Attribution (LIME & SHAP)',
    orders: 'Module 4: Actionable Recommendations & Physician Review',
    cohort: 'Patient Cohort Database (38 Clinical Study Cases)'
  };

  return (
    <header className="top-app-bar">
      {/* 1. Left: Collapse Toggle & Breadcrumb */}
      <div className="top-bar-left">
        {isCollapsed && (
          <button
            className="sidebar-open-btn"
            onClick={onToggleSidebar}
            title="Open Sidebar Navigation"
            aria-label="Open Sidebar Navigation"
          >
            <Menu size={18} />
          </button>
        )}

        <div className="breadcrumb-box">
          <span className="breadcrumb-system">EEG Stroke DSS</span>
          <span className="breadcrumb-divider">/</span>
          <span className="breadcrumb-current">
            {pageTitles[activePage] || 'Clinical Decision Support'}
          </span>
        </div>
      </div>

      {/* 2. Right: Patient Selector */}
      <div className="top-bar-right">
        <div className="top-patient-select-box">
          <label htmlFor="top-patient-select" className="top-patient-label">
            Patient:
          </label>
          <select
            id="top-patient-select"
            className="top-patient-select"
            value={selectedPatientId ?? ''}
            onChange={(e) => onSelectPatient(Number(e.target.value))}
            disabled={loading}
          >
            {patients.map((p) => (
              <option key={p.id} value={p.id}>
                {p.participant_id} • {p.diagnosis} ({p.age}y)
              </option>
            ))}
          </select>
        </div>
      </div>
    </header>
  );
}
