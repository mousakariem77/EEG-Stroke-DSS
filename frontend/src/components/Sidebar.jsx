import React from 'react';
import {
  Brain, LayoutDashboard, Activity, Sparkles, ClipboardCheck,
  Users, HelpCircle, ChevronLeft, ShieldAlert, CheckCircle2
} from 'lucide-react';

export default function Sidebar({
  isCollapsed,
  onToggleCollapse,
  activePage,
  onSelectPage,
  onOpenTour,
  patientDetail,
  prediction,
  patientCount = 38
}) {
  const isStroke = patientDetail?.diagnosis?.toLowerCase().includes('stroke');
  const strokeProb = prediction ? Math.round(prediction.stroke_probability * 100) : null;

  const navItems = [
    {
      id: 'dashboard',
      label: 'Clinical Dashboard',
      badge: 'Fig. 8',
      icon: LayoutDashboard,
      desc: 'Unified 4-box decision dashboard'
    },
    {
      id: 'signal',
      label: 'EEG Waveform (FP1)',
      badge: 'Mod 1',
      icon: Activity,
      desc: '1-ch FP1 signal & bandpass 0.5-30Hz'
    },
    {
      id: 'diagnosis',
      label: 'E-ESN Diagnosis',
      badge: 'Mod 2',
      icon: Brain,
      desc: '7-reservoir Ensemble ESN prediction'
    },
    {
      id: 'explainability',
      label: 'XAI Attribution',
      badge: 'Mod 3',
      icon: Sparkles,
      desc: 'LIME & SHAP explainable AI'
    },
    {
      id: 'orders',
      label: 'Clinical DSS & Orders',
      badge: 'Mod 4',
      icon: ClipboardCheck,
      desc: 'Confirm / cancel tests & audit trail'
    },
    {
      id: 'cohort',
      label: 'Patient Cohort DB',
      badge: String(patientCount || 38),
      icon: Users,
      desc: `${patientCount || 38} clinical study patient records`
    }
  ];

  return (
    <aside
      className={`clinical-sidebar ${isCollapsed ? 'collapsed' : 'expanded'}`}
      aria-label="Clinical Decision Support Navigation"
    >
      {/* 1. Header / Brand & Toggle */}
      <div className="sidebar-brand">
        <div className="brand-logo-row">
          <div
            className="brand-icon-box"
            title={isCollapsed ? "Click to expand sidebar" : "Explainable EEG Stroke DSS (Bouazizi & Ltifi 2024)"}
            onClick={isCollapsed ? onToggleCollapse : undefined}
            style={{ cursor: isCollapsed ? 'pointer' : 'default' }}
          >
            <Brain size={20} color="#ffffff" />
          </div>

          {!isCollapsed && (
            <>
              <div className="brand-info">
                <h1 className="brand-title">EEG Stroke DSS</h1>
                <p className="brand-sub">Bouazizi &amp; Ltifi (2024)</p>
              </div>

              <button
                className="sidebar-collapse-btn"
                onClick={onToggleCollapse}
                title="Collapse sidebar"
                aria-label="Collapse sidebar"
              >
                <ChevronLeft size={15} />
              </button>
            </>
          )}
        </div>
      </div>

      {/* 2. Unified Page Navigation List */}
      <div className="sidebar-content">
        <div className="sidebar-section">
          {!isCollapsed && <div className="section-label">NAVIGATION</div>}

          <nav className="nav-list" aria-label="Main Navigation">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = activePage === item.id;

              return (
                <button
                  key={item.id}
                  className={`nav-item ${isActive ? 'active' : ''}`}
                  onClick={() => onSelectPage(item.id)}
                  title={`${item.label} (${item.desc})`}
                >
                  <span className="nav-item-icon">
                    <Icon size={17} />
                  </span>

                  {!isCollapsed && (
                    <span className="nav-item-title">{item.label}</span>
                  )}

                  {!isCollapsed && (
                    <span className={`nav-tag ${isActive ? 'active-tag' : 'idle-tag'}`}>
                      {item.badge}
                    </span>
                  )}
                </button>
              );
            })}
          </nav>
        </div>

        {/* 3. Middle: Active Patient Glance Card */}
        {!isCollapsed && patientDetail && (
          <div className="sidebar-patient-card">
            <div className="card-header">
              <span className="card-label">CURRENT CASE</span>
              <span className={`card-status-pill ${isStroke ? 'status-stroke' : 'status-control'}`}>
                {isStroke ? (
                  <>
                    <ShieldAlert size={10} />
                    <span>Stroke</span>
                  </>
                ) : (
                  <>
                    <CheckCircle2 size={10} />
                    <span>Control</span>
                  </>
                )}
              </span>
            </div>

            <div className="card-body">
              <div className="card-id font-mono">{patientDetail.participant_id}</div>
              <div className="card-meta">
                <span>{patientDetail.age}y • {patientDetail.sex}</span>
                {strokeProb !== null && (
                  <span className="card-risk">
                    Risk: <strong style={{ color: strokeProb >= 50 ? '#dc2626' : '#16a34a' }}>{strokeProb}%</strong>
                  </span>
                )}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* 4. Footer / User Guide & Model Info */}
      <div className="sidebar-footer">
        <button
          className="sidebar-help-btn"
          onClick={onOpenTour}
          title="Clinical Decision Support System Walkthrough & SOP"
        >
          <HelpCircle size={15} color="#0284c7" />
          {!isCollapsed && <span>User Guide &amp; SOP</span>}
        </button>

        {!isCollapsed && (
          <div className="system-status-indicator">
            <span className="status-dot-pulse"></span>
            <span>FastAPI Online • E-ESN Active</span>
          </div>
        )}
      </div>
    </aside>
  );
}
