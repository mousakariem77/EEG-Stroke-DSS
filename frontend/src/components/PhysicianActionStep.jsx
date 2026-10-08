import React, { useState } from 'react';
import { ClipboardCheck, CheckCircle2, XCircle, FileDown, ShieldCheck, Clock, AlertCircle, ArrowLeft } from 'lucide-react';

export default function PhysicianActionStep({
  recommendations = [],
  sessionState,
  onRecordAction,
  onDownloadPdf,
  onBack,
  downloadingPdf
}) {
  const [selectedTest, setSelectedTest] = useState(null);
  const [actionType, setActionType] = useState('CONFIRMED');
  const [notes, setNotes] = useState('');
  const [physicianId, setPhysicianId] = useState('Dr. Clinical Neurologist');
  const [showModal, setShowModal] = useState(false);

  const handleOpenActionModal = (test, action) => {
    if (test.status === action) return;
    setSelectedTest(test);
    setActionType(action);
    setNotes(action === 'CONFIRMED' ? 'Confirmed acute stroke diagnostic workup protocol.' : 'Patient has absolute contraindication or recent prior scan.');
    setShowModal(true);
  };

  const handleSubmitAction = () => {
    if (!selectedTest) return;
    onRecordAction({
      test_id: selectedTest.id,
      action: actionType,
      notes: notes,
      physician_id: physicianId
    });
    setShowModal(false);
  };

  return (
    <div>
      {/* Page Header */}
      <div className="page-view-header">
        <div>
          <div className="page-view-title">
            <ClipboardCheck size={20} color="#0284c7" />
            <span>Module 4: Clinical Decision Support &amp; Physician Orders</span>
            <span className="badge badge-teal" style={{ fontSize: '0.68rem', padding: '1px 7px' }}>Human-in-the-Loop</span>
          </div>
          <p className="page-view-sub">
            Attending physician verification and authorization of diagnostic protocol per Bouazizi &amp; Ltifi (DSS 2024)
          </p>
        </div>

        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          <span className="badge badge-low" style={{ fontSize: '0.72rem' }}>Confirmed: {sessionState?.confirmed_count ?? 0}</span>
          <span className="badge badge-high" style={{ background: '#f1f5f9', color: '#64748b', borderColor: '#cbd5e1', fontSize: '0.72rem' }}>
            Cancelled: {sessionState?.cancelled_count ?? 0}
          </span>
          <button
            id="download-pdf-btn-header"
            className="btn-primary"
            onClick={onDownloadPdf}
            disabled={downloadingPdf}
            style={{ fontSize: '0.78rem', padding: '0.38rem 0.85rem' }}
          >
            <FileDown size={14} />
            <span>{downloadingPdf ? 'Generating PDF...' : 'Download Clinical PDF'}</span>
          </button>
        </div>
      </div>

      {/* Human-in-the-loop Guidance Banner (Compact) */}
      <div className="clinical-card" style={{ background: '#f0fdf4', borderColor: '#bbf7d0', padding: '0.65rem 0.95rem', marginBottom: '0.85rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
          <div style={{ background: '#dcfce7', color: '#16a34a', padding: '5px', borderRadius: '6px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <ShieldCheck size={18} />
          </div>
          <p style={{ fontSize: '0.78rem', color: '#166534', margin: 0, lineHeight: 1.45 }}>
            <strong>Human-in-the-Loop Protocol:</strong> AI predictions do not automatically submit laboratory or imaging orders. The attending physician must explicitly <strong>Confirm</strong> or <strong>Cancel</strong> each suggested intervention before final PDF generation.
          </p>
        </div>
      </div>

      {/* Recommended Diagnostic Interventions List */}
      <div className="clinical-card" style={{ padding: '0.85rem 1rem' }}>
        <div className="card-header-flex" style={{ marginBottom: '0.65rem', paddingBottom: '0.45rem' }}>
          <div className="card-title-group">
            <div style={{ background: '#e0f2fe', color: '#0284c7', padding: '5px', borderRadius: '6px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <ClipboardCheck size={16} />
            </div>
            <div>
              <h2 style={{ fontSize: '0.98rem' }}>Actionable Clinical Recommendations</h2>
              <p style={{ fontSize: '0.74rem' }}>Tailored diagnostic pathway based on acute stroke electrophysiological stratification</p>
            </div>
          </div>
        </div>

        {/* Test Cards Grid (Compact) */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '0.75rem', marginBottom: '1rem' }}>
          {recommendations.map((t) => {
            const isConfirmed = t.status === 'CONFIRMED';
            const isCancelled = t.status === 'CANCELLED';

            return (
              <div
                key={t.id}
                style={{
                  background: isConfirmed ? '#f0fdf4' : isCancelled ? '#f8fafc' : '#ffffff',
                  border: `1.5px solid ${isConfirmed ? '#86efac' : isCancelled ? '#cbd5e1' : '#e2e8f0'}`,
                  borderRadius: '8px',
                  padding: '0.75rem 0.85rem',
                  display: 'flex',
                  flexDirection: 'column',
                  justifyContent: 'space-between',
                  boxShadow: '0 1px 2px rgba(0,0,0,0.04)',
                  opacity: isCancelled ? 0.75 : 1
                }}
              >
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.3rem' }}>
                    <h4 style={{ fontSize: '0.88rem', fontWeight: 700, color: '#0f172a' }}>{t.name}</h4>
                    <span
                      className={`badge ${t.urgency === 'CRITICAL' ? 'badge-high' : t.urgency === 'PRIORITY' ? 'badge-med' : 'badge-info'}`}
                      style={{ fontSize: '0.64rem', padding: '1px 6px' }}
                    >
                      {t.urgency}
                    </span>
                  </div>

                  <p style={{ fontSize: '0.76rem', color: '#475569', marginBottom: '0.4rem', lineHeight: 1.4 }}>
                    {t.rationale}
                  </p>

                  <div style={{ fontSize: '0.7rem', color: '#64748b', marginBottom: '0.65rem' }}>
                    <div>Window: <strong>{t.target_time}</strong> | Category: <strong>{t.category}</strong></div>
                  </div>
                </div>

                {/* Cancel (Left) / Confirm (Right) Buttons */}
                <div style={{ display: 'flex', gap: '0.4rem', borderTop: '1px solid #f1f5f9', paddingTop: '0.55rem' }}>
                  <button
                    className="btn-secondary"
                    onClick={() => handleOpenActionModal(t, 'CANCELLED')}
                    disabled={isCancelled}
                    title={isCancelled ? 'This test has already been cancelled' : 'Cancel this test'}
                    style={{
                      flex: 1,
                      justifyContent: 'center',
                      background: isCancelled ? '#f1f5f9' : '#ffffff',
                      color: isCancelled ? '#94a3b8' : '#dc2626',
                      borderColor: isCancelled ? '#e2e8f0' : '#fca5a5',
                      padding: '0.35rem 0.55rem',
                      fontSize: '0.74rem',
                      cursor: isCancelled ? 'not-allowed' : 'pointer',
                      opacity: isCancelled ? 0.65 : 1
                    }}
                  >
                    <XCircle size={13} />
                    <span>{isCancelled ? 'Cancelled ✕' : 'Cancel'}</span>
                  </button>

                  <button
                    className="btn-success"
                    onClick={() => handleOpenActionModal(t, 'CONFIRMED')}
                    disabled={isConfirmed}
                    title={isConfirmed ? 'This test has already been confirmed' : 'Confirm this test'}
                    style={{
                      flex: 1,
                      justifyContent: 'center',
                      background: isConfirmed ? '#059669' : '#10b981',
                      padding: '0.35rem 0.55rem',
                      fontSize: '0.74rem',
                      cursor: isConfirmed ? 'not-allowed' : 'pointer',
                      opacity: isConfirmed ? 0.85 : 1,
                      boxShadow: isConfirmed ? 'none' : undefined
                    }}
                  >
                    <CheckCircle2 size={13} />
                    <span>{isConfirmed ? 'Confirmed ✓' : 'Confirm'}</span>
                  </button>
                </div>
              </div>
            );
          })}
        </div>

        {/* Audit Trail Section */}
        <div style={{ borderTop: '1px solid #e2e8f0', paddingTop: '0.75rem' }}>
          <h4 style={{ fontSize: '0.88rem', fontWeight: 700, color: '#0f172a', marginBottom: '0.15rem' }}>
            Physician Action Log &amp; Audit Trail
          </h4>
          <p style={{ fontSize: '0.74rem', color: '#64748b' }}>
            Immutable chronological record of medical orders, timestamps, and physician notes.
          </p>

          {sessionState?.audit_log && sessionState.audit_log.length > 0 ? (
            <div style={{ overflowX: 'auto', marginTop: '0.5rem' }}>
              <table className="audit-table" style={{ fontSize: '0.76rem' }}>
                <thead>
                  <tr>
                    <th style={{ padding: '0.45rem 0.65rem' }}>Timestamp</th>
                    <th style={{ padding: '0.45rem 0.65rem' }}>Intervention</th>
                    <th style={{ padding: '0.45rem 0.65rem' }}>Action</th>
                    <th style={{ padding: '0.45rem 0.65rem' }}>Physician</th>
                    <th style={{ padding: '0.45rem 0.65rem' }}>Clinical Rationale / Notes</th>
                  </tr>
                </thead>
                <tbody>
                  {sessionState.audit_log.map((entry, idx) => (
                    <tr key={idx}>
                      <td style={{ padding: '0.4rem 0.65rem', fontFamily: 'monospace', fontSize: '0.72rem' }}>{entry.timestamp}</td>
                      <td style={{ padding: '0.4rem 0.65rem', fontWeight: 600 }}>{entry.test_name}</td>
                      <td style={{ padding: '0.4rem 0.65rem' }}>
                        <span className={`badge ${entry.action === 'CONFIRM' ? 'badge-low' : 'badge-high'}`} style={{ fontSize: '0.65rem', padding: '1px 5px' }}>
                          {entry.action}
                        </span>
                      </td>
                      <td style={{ padding: '0.4rem 0.65rem' }}>{entry.physician_id}</td>
                      <td style={{ padding: '0.4rem 0.65rem', color: '#475569' }}>{entry.notes || '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div style={{ padding: '1rem', textAlign: 'center', background: '#f8fafc', borderRadius: '7px', color: '#94a3b8', fontSize: '0.78rem', marginTop: '0.5rem' }}>
              No review actions recorded yet. Click <strong>Confirm</strong> or <strong>Cancel</strong> on recommended tests above.
            </div>
          )}
        </div>

        {/* Bottom Actions: Back + Download PDF */}
        <div style={{ marginTop: '1rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderTop: '1px solid #e2e8f0', paddingTop: '0.75rem' }}>
          <button className="btn-secondary" onClick={onBack} style={{ fontSize: '0.78rem', padding: '0.38rem 0.75rem' }}>
            <ArrowLeft size={14} />
            <span>Back to AI Diagnosis</span>
          </button>

          <button
            id="download-pdf-btn"
            className="btn-primary"
            onClick={onDownloadPdf}
            disabled={downloadingPdf}
            style={{ background: '#0284c7', padding: '0.45rem 1.1rem', fontSize: '0.82rem' }}
          >
            <FileDown size={15} />
            <span>{downloadingPdf ? 'Generating Signed PDF...' : 'Download Official Clinical Report (PDF)'}</span>
          </button>
        </div>
      </div>

      {/* Confirmation & Clinical Note Modal */}
      {showModal && selectedTest && (
        <div className="modal-backdrop">
          <div className="modal-card">
            <h3 style={{ fontSize: '1.15rem', fontWeight: 700, color: '#0f172a', marginBottom: '0.4rem' }}>
              {actionType === 'CONFIRMED' ? 'Confirm Clinical Test Order' : 'Cancel / Override Clinical Recommendation'}
            </h3>
            <p style={{ fontSize: '0.84rem', color: '#64748b', marginBottom: '1.2rem' }}>
              Intervention: <strong>{selectedTest.name}</strong>
            </p>

            <div style={{ marginBottom: '1rem' }}>
              <label style={{ display: 'block', fontSize: '0.8rem', fontWeight: 600, color: '#475569', marginBottom: '0.3rem' }}>
                Attending Physician Name / ID:
              </label>
              <input
                type="text"
                value={physicianId}
                onChange={(e) => setPhysicianId(e.target.value)}
                style={{ width: '100%', padding: '0.5rem 0.75rem', borderRadius: '6px', border: '1px solid #cbd5e1', fontSize: '0.85rem' }}
              />
            </div>

            <div style={{ marginBottom: '1.5rem' }}>
              <label style={{ display: 'block', fontSize: '0.8rem', fontWeight: 600, color: '#475569', marginBottom: '0.3rem' }}>
                Clinical Rationale & Notes:
              </label>
              <textarea
                rows={3}
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                placeholder="Enter justification or contraindications..."
                style={{ width: '100%', padding: '0.5rem 0.75rem', borderRadius: '6px', border: '1px solid #cbd5e1', fontSize: '0.85rem', fontFamily: 'inherit' }}
              />
            </div>

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.6rem' }}>
              <button className="btn-secondary" onClick={() => setShowModal(false)}>
                Cancel
              </button>
              <button
                className={actionType === 'CONFIRMED' ? 'btn-success' : 'btn-primary'}
                style={{ background: actionType === 'CONFIRMED' ? '#10b981' : '#dc2626' }}
                onClick={handleSubmitAction}
              >
                Save Action to Audit Log
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
