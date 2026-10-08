import React, { useState } from 'react';
import { Users, Search, ArrowRight, ShieldAlert, CheckCircle2, Eye } from 'lucide-react';
import PatientDetailModal from './PatientDetailModal';

export default function PatientCohortView({
  patients,
  selectedPatientId,
  onSelectPatient,
  onNavigateToDashboard
}) {
  const [searchTerm, setSearchTerm] = useState('');
  const [filterType, setFilterType] = useState('all'); // 'all', 'stroke', 'control'
  const [modalPatient, setModalPatient] = useState(null);

  const filteredPatients = patients.filter((p) => {
    const isStroke = p.diagnosis.toLowerCase().includes('stroke');
    if (filterType === 'stroke' && !isStroke) return false;
    if (filterType === 'control' && isStroke) return false;

    if (!searchTerm.trim()) return true;
    const term = searchTerm.toLowerCase();
    return (
      p.participant_id.toLowerCase().includes(term) ||
      p.diagnosis.toLowerCase().includes(term) ||
      String(p.age).includes(term) ||
      String(p.gender || '').toLowerCase().includes(term)
    );
  });

  const strokeCount = patients.filter((p) => p.diagnosis.toLowerCase().includes('stroke')).length;
  const controlCount = patients.length - strokeCount;

  return (
    <div className="patient-cohort-view">
      {/* Header */}
      <div className="page-view-header">
        <div>
          <div className="page-view-title">
            <Users size={20} color="#0284c7" />
            <span>Clinical Study Patient Cohort Database</span>
            <span className="badge badge-teal" style={{ fontSize: '0.68rem', padding: '1px 7px' }}>
              {patients.length} Cohort Records
            </span>
          </div>
          <p className="page-view-sub">
            {patients.length} clinical cases ({strokeCount} Acute Stroke / {controlCount} Healthy Control) from EPoC stroke study (Bouazizi &amp; Ltifi, DSS 2024)
          </p>
        </div>

        {/* Filter Badges */}
        <div style={{ display: 'flex', gap: '0.5rem', background: '#f1f5f9', padding: '4px', borderRadius: '10px', border: '1px solid #e2e8f0' }}>
          <button
            onClick={() => setFilterType('all')}
            style={{
              padding: '0.35rem 0.8rem',
              borderRadius: '6px',
              border: 'none',
              background: filterType === 'all' ? '#ffffff' : 'transparent',
              color: filterType === 'all' ? '#0f172a' : '#64748b',
              fontWeight: filterType === 'all' ? 700 : 500,
              fontSize: '0.78rem',
              cursor: 'pointer',
              boxShadow: filterType === 'all' ? '0 1px 3px rgba(0,0,0,0.06)' : 'none'
            }}
          >
            All Cohort ({patients.length})
          </button>
          <button
            onClick={() => setFilterType('stroke')}
            style={{
              padding: '0.35rem 0.8rem',
              borderRadius: '6px',
              border: 'none',
              background: filterType === 'stroke' ? '#fee2e2' : 'transparent',
              color: filterType === 'stroke' ? '#dc2626' : '#64748b',
              fontWeight: filterType === 'stroke' ? 700 : 500,
              fontSize: '0.78rem',
              cursor: 'pointer'
            }}
          >
            Acute Stroke ({strokeCount})
          </button>
          <button
            onClick={() => setFilterType('control')}
            style={{
              padding: '0.35rem 0.8rem',
              borderRadius: '6px',
              border: 'none',
              background: filterType === 'control' ? '#dcfce7' : 'transparent',
              color: filterType === 'control' ? '#16a34a' : '#64748b',
              fontWeight: filterType === 'control' ? 700 : 500,
              fontSize: '0.78rem',
              cursor: 'pointer'
            }}
          >
            Healthy Control ({controlCount})
          </button>
        </div>
      </div>

      {/* Search Input */}
      <div style={{ marginBottom: '1rem', position: 'relative', maxWidth: '360px' }}>
        <Search size={16} color="#94a3b8" style={{ position: 'absolute', left: '12px', top: '50%', transform: 'translateY(-50%)' }} />
        <input
          type="text"
          placeholder="Search by Patient ID, Diagnosis, Age..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          style={{
            width: '100%',
            padding: '0.5rem 0.75rem 0.5rem 2.25rem',
            borderRadius: '8px',
            border: '1px solid #cbd5e1',
            fontSize: '0.82rem',
            outline: 'none',
            background: '#ffffff'
          }}
        />
      </div>

      {/* Cohort Table */}
      <div style={{ background: '#ffffff', borderRadius: '12px', border: '1px solid #e2e8f0', boxShadow: 'var(--shadow-sm)', overflow: 'hidden' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.82rem' }}>
          <thead>
            <tr style={{ background: '#f8fafc', borderBottom: '1px solid #e2e8f0', color: '#475569', fontWeight: 700, fontSize: '0.74rem', textTransform: 'uppercase', letterSpacing: '0.4px' }}>
              <th style={{ padding: '0.75rem 1rem' }}>Patient ID</th>
              <th style={{ padding: '0.75rem 1rem' }}>Clinical Status</th>
              <th style={{ padding: '0.75rem 1rem' }}>Age</th>
              <th style={{ padding: '0.75rem 1rem' }}>Gender</th>
              <th style={{ padding: '0.75rem 1rem' }}>DAR (Delta/Alpha)</th>
              <th style={{ padding: '0.75rem 1rem' }}>DTR (Delta/Theta)</th>
              <th style={{ padding: '0.75rem 1rem' }}>Total Power</th>
              <th style={{ padding: '0.75rem 1rem', textAlign: 'right' }}>Actions</th>
            </tr>
          </thead>
          <tbody>
            {filteredPatients.map((p) => {
              const isSelected = p.id === selectedPatientId;
              const isStroke = p.diagnosis.toLowerCase().includes('stroke');
              const isDarElevated = p.dar > 3.7;

              return (
                <tr
                  key={p.id}
                  onClick={() => setModalPatient(p)}
                  title="Click to view full clinical details"
                  style={{
                    borderBottom: '1px solid #f1f5f9',
                    background: isSelected ? '#eff6ff' : '#ffffff',
                    cursor: 'pointer',
                    transition: 'background 0.15s ease'
                  }}
                  onMouseEnter={(e) => {
                    if (!isSelected) e.currentTarget.style.background = '#f8fafc';
                  }}
                  onMouseLeave={(e) => {
                    if (!isSelected) e.currentTarget.style.background = '#ffffff';
                  }}
                >
                  <td style={{ padding: '0.75rem 1rem', fontWeight: 700, fontFamily: 'var(--font-mono)', color: isSelected ? '#1d4ed8' : '#0f172a' }}>
                    {p.participant_id}
                    {isSelected && (
                      <span style={{ marginLeft: '8px', fontSize: '0.65rem', background: '#dbeafe', color: '#1d4ed8', padding: '1px 6px', borderRadius: '10px', fontWeight: 700 }}>
                        ACTIVE
                      </span>
                    )}
                  </td>

                  <td style={{ padding: '0.75rem 1rem' }}>
                    <span
                      style={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '0.35rem',
                        fontSize: '0.74rem',
                        fontWeight: 700,
                        padding: '2px 8px',
                        borderRadius: '12px',
                        background: isStroke ? '#fee2e2' : '#dcfce7',
                        color: isStroke ? '#dc2626' : '#16a34a'
                      }}
                    >
                      {isStroke ? <ShieldAlert size={13} /> : <CheckCircle2 size={13} />}
                      <span>{p.diagnosis}</span>
                    </span>
                  </td>

                  <td style={{ padding: '0.75rem 1rem', color: '#334155', fontWeight: 600 }}>{p.age}y</td>
                  <td style={{ padding: '0.75rem 1rem', color: '#334155' }}>{p.gender || 'Unknown'}</td>

                  <td style={{ padding: '0.75rem 1rem', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
                    <span style={{ color: isDarElevated ? '#dc2626' : '#0f172a' }}>
                      {typeof p.dar === 'number' ? p.dar.toFixed(3) : p.dar}
                    </span>
                    {isDarElevated && (
                      <span style={{ marginLeft: '6px', fontSize: '0.65rem', color: '#dc2626', background: '#fee2e2', padding: '1px 4px', borderRadius: '4px', fontWeight: 700 }}>
                        High
                      </span>
                    )}
                  </td>

                  <td style={{ padding: '0.75rem 1rem', fontFamily: 'var(--font-mono)', color: '#0f172a', fontWeight: 600 }}>
                    {typeof p.dtr === 'number' ? p.dtr.toFixed(3) : p.dtr}
                  </td>

                  <td style={{ padding: '0.75rem 1rem', fontFamily: 'var(--font-mono)', color: '#475569' }}>
                    {typeof p.total_power === 'number' ? p.total_power.toFixed(2) : p.total_power} µV²
                  </td>

                  <td style={{ padding: '0.75rem 1rem', textAlign: 'right' }}>
                    <div style={{ display: 'inline-flex', gap: '0.45rem', alignItems: 'center' }} onClick={(e) => e.stopPropagation()}>
                      <button
                        onClick={() => setModalPatient(p)}
                        title="View Full Clinical & EEG Details"
                        style={{
                          padding: '0.35rem 0.65rem',
                          borderRadius: '6px',
                          border: '1px solid #cbd5e1',
                          background: '#ffffff',
                          color: '#0f172a',
                          fontWeight: 600,
                          fontSize: '0.74rem',
                          cursor: 'pointer',
                          display: 'inline-flex',
                          alignItems: 'center',
                          gap: '0.35rem',
                          transition: 'all 0.15s ease'
                        }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.background = '#f1f5f9';
                          e.currentTarget.style.borderColor = '#94a3b8';
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.background = '#ffffff';
                          e.currentTarget.style.borderColor = '#cbd5e1';
                        }}
                      >
                        <Eye size={13} color="#0284c7" />
                        <span>View Details</span>
                      </button>

                      <button
                        onClick={() => {
                          onSelectPatient(p.id);
                          onNavigateToDashboard();
                        }}
                        title="Select patient and open Clinical Decision Dashboard"
                        style={{
                          padding: '0.35rem 0.75rem',
                          borderRadius: '6px',
                          border: '1px solid #cbd5e1',
                          background: isSelected ? '#0284c7' : '#f8fafc',
                          color: isSelected ? '#ffffff' : '#0284c7',
                          fontWeight: 700,
                          fontSize: '0.74rem',
                          cursor: 'pointer',
                          display: 'inline-flex',
                          alignItems: 'center',
                          gap: '0.35rem',
                          transition: 'all 0.15s ease'
                        }}
                        onMouseEnter={(e) => {
                          if (!isSelected) {
                            e.currentTarget.style.background = '#0284c7';
                            e.currentTarget.style.color = '#ffffff';
                          }
                        }}
                        onMouseLeave={(e) => {
                          if (!isSelected) {
                            e.currentTarget.style.background = '#f8fafc';
                            e.currentTarget.style.color = '#0284c7';
                          }
                        }}
                      >
                        <span>Analyze</span>
                        <ArrowRight size={13} />
                      </button>
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Patient Detail Modal */}
      {modalPatient && (
        <PatientDetailModal
          patient={modalPatient}
          onClose={() => setModalPatient(null)}
          onAnalyze={(patientId) => {
            onSelectPatient(patientId);
            onNavigateToDashboard();
          }}
        />
      )}
    </div>
  );
}
