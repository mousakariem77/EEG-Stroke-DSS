import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import TopBar from './components/TopBar';
import Stepper from './components/Stepper';
import SignalInspectionStep from './components/SignalInspectionStep';
import DiagnosisStep from './components/DiagnosisStep';
import PhysicianActionStep from './components/PhysicianActionStep';
import PatientCohortView from './components/PatientCohortView';
import QuickTourModal from './components/QuickTourModal';
import { AlertCircle, RefreshCw } from 'lucide-react';

import DashboardFig8View from './components/DashboardFig8View';

export default function App() {
  const [patients, setPatients] = useState([]);
  const [selectedPatientId, setSelectedPatientId] = useState(() => {
    try {
      const saved = localStorage.getItem('eeg_dss_selected_patient_id');
      return saved !== null ? Number(saved) : 0;
    } catch {
      return 0;
    }
  });
  const [patientDetail, setPatientDetail] = useState(null);
  const [rawSignal, setRawSignal] = useState([]);
  const [filteredSignal, setFilteredSignal] = useState([]);
  const [processReport, setProcessReport] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [shapData, setShapData] = useState(null);
  const [limeData, setLimeData] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  const [sessionState, setSessionState] = useState(null);

  const getPatientSessionId = (pId) => `sess_patient_${pId ?? selectedPatientId ?? 0}`;

  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false); // Default: visible / expanded
  const [activePage, setActivePage] = useState(() => {
    try {
      return localStorage.getItem('eeg_dss_active_page') || 'dashboard';
    } catch {
      return 'dashboard';
    }
  });

  const handleSelectPage = (page) => {
    setActivePage(page);
    try {
      localStorage.setItem('eeg_dss_active_page', page);
    } catch (e) {}
  };

  const [activePreset, setActivePreset] = useState('stroke');
  const [showTourModal, setShowTourModal] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [downloadingPdf, setDownloadingPdf] = useState(false);

  // Initial load: Fetch patient list
  useEffect(() => {
    async function initData() {
      try {
        setLoading(true);
        const res = await fetch('/api/patients');
        if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to load patient cohort`);
        const data = await res.json();
        setPatients(data);
        if (data.length > 0) {
          let targetId = data[0].id;
          try {
            const saved = localStorage.getItem('eeg_dss_selected_patient_id');
            if (saved !== null && data.some((p) => p.id === Number(saved))) {
              targetId = Number(saved);
            }
          } catch (e) {}
          await loadPatient(targetId);
        }
      } catch (err) {
        console.error('API Init Error:', err);
        setError('Failed to connect to the EEG Stroke DSS backend. Please ensure the FastAPI server is running on port 8000.');
      } finally {
        setLoading(false);
      }
    }
    initData();
  }, []);

  // Load specific patient data & predictions
  const loadPatient = async (pId) => {
    try {
      setLoading(true);
      setError(null);
      setSelectedPatientId(pId);

      // 1. Patient Details
      const detailRes = await fetch(`/api/patients/${pId}`);
      if (!detailRes.ok) throw new Error('Failed to fetch patient details');
      const detail = await detailRes.json();
      setPatientDetail(detail);

      // 2. Simulate corresponding raw FP1 waveform for visual inspection
      const isStroke = detail.diagnosis.toLowerCase().includes('stroke');
      const simRes = await fetch('/api/eeg/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          pattern: isStroke ? 'stroke' : 'control',
          duration_sec: 8.0,
          sampling_rate: 128.0,
          seed: 42 + pId
        })
      });
      const simData = await simRes.json();
      setRawSignal(simData.signal);

      // 3. Process signal (Module 1 Preprocessing)
      const procRes = await fetch('/api/eeg/process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ signal: simData.signal, sampling_rate: 128.0 })
      });
      const procData = await procRes.json();
      setFilteredSignal(procData.filtered_signal);
      setProcessReport(procData);

      // 4. Model Prediction (Module 2 Ensemble ESN)
      const predRes = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ patient_id: pId })
      });
      const predData = await predRes.json();
      setPrediction(predData);

      // 5. Explainable AI: SHAP & LIME (Module 3)
      fetch('/api/explain/shap', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ patient_id: pId })
      })
        .then((r) => r.json())
        .then((data) => setShapData(data))
        .catch((e) => console.warn('SHAP computation delayed:', e));

      fetch('/api/explain/lime', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ patient_id: pId })
      })
        .then((r) => r.json())
        .then((data) => setLimeData(data))
        .catch((e) => console.warn('LIME computation delayed:', e));

      const currentSessionId = getPatientSessionId(pId);
      try {
        localStorage.setItem('eeg_dss_selected_patient_id', String(pId));
      } catch (e) {}

      // 6. Clinical Recommendations (Module 4)
      const recRes = await fetch(`/api/dss/recommendations?stroke_probability=${predData.stroke_probability}&session_id=${currentSessionId}&patient_id=${pId}`);
      const recData = await recRes.json();
      setRecommendations(recData);

      // 7. Session State
      const sessRes = await fetch(`/api/dss/session/${currentSessionId}`);
      const sessData = await sessRes.json();
      setSessionState(sessData);
    } catch (err) {
      console.error('Error loading patient data:', err);
      setError(err.message || 'Error occurred while loading clinical case.');
    } finally {
      setLoading(false);
    }
  };

  // Quick Preset Handler
  const handleSelectPreset = async (preset) => {
    setActivePreset(preset);
    if (preset === 'stroke') {
      const strokePatient = patients.find((p) => p.is_stroke);
      if (strokePatient) await loadPatient(strokePatient.id);
    } else if (preset === 'control') {
      const controlPatient = patients.find((p) => !p.is_stroke);
      if (controlPatient) await loadPatient(controlPatient.id);
    } else if (preset === 'sim') {
      // Direct raw simulation run
      try {
        setLoading(true);
        const simRes = await fetch('/api/eeg/simulate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ pattern: 'stroke', duration_sec: 12.0, sampling_rate: 128.0 })
        });
        const simData = await simRes.json();
        setRawSignal(simData.signal);

        const procRes = await fetch('/api/eeg/process', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ signal: simData.signal, sampling_rate: 128.0 })
        });
        const procData = await procRes.json();
        setFilteredSignal(procData.filtered_signal);
        setProcessReport(procData);

        const predRes = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ features: procData.features })
        });
        const predData = await predRes.json();
        setPrediction(predData);

        const recRes = await fetch(`/api/dss/recommendations?stroke_probability=${predData.stroke_probability}&session_id=${sessionId}`);
        const recData = await recRes.json();
        setRecommendations(recData);
      } catch (err) {
        setError('Simulation processing error: ' + err.message);
      } finally {
        setLoading(false);
      }
    }
  };

  // Physician Review Action (Confirm / Cancel)
  const handleRecordAction = async ({ test_id, action, notes, physician_id }) => {
    try {
      const currentSessionId = getPatientSessionId(selectedPatientId);
      const res = await fetch('/api/dss/action', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: currentSessionId,
          test_id: test_id,
          action: action,
          notes: notes,
          physician_id: physician_id
        })
      });
      const updatedSession = await res.json();
      setSessionState(updatedSession);

      // Refresh recommendations to reflect new status
      if (prediction) {
        const recRes = await fetch(`/api/dss/recommendations?stroke_probability=${prediction.stroke_probability}&session_id=${currentSessionId}&patient_id=${selectedPatientId}`);
        const recData = await recRes.json();
        setRecommendations(recData);
      }
    } catch (err) {
      console.error('Failed to record physician action:', err);
    }
  };

  // PDF Report Download
  const handleDownloadPdf = async () => {
    try {
      setDownloadingPdf(true);
      const currentSessionId = getPatientSessionId(selectedPatientId);
      const res = await fetch('/api/dss/pdf', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          patient_id: selectedPatientId,
          session_id: currentSessionId,
          stroke_probability: prediction?.stroke_probability,
          risk_level: prediction?.risk_level,
          prediction: prediction?.prediction,
          features: patientDetail?.features
        })
      });

      if (!res.ok) throw new Error('PDF generation failed on server');

      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `Stroke_DSS_Report_P${selectedPatientId + 1}.pdf`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      alert('Error downloading PDF: ' + err.message);
    } finally {
      setDownloadingPdf(false);
    }
  };

  return (
    <div className="app-layout">
      {/* Collapsible Left Clinical Navigation Sidebar (100% Page Navigation) */}
      <Sidebar
        isCollapsed={isSidebarCollapsed}
        onToggleCollapse={() => setIsSidebarCollapsed(!isSidebarCollapsed)}
        activePage={activePage}
        onSelectPage={handleSelectPage}
        onOpenTour={() => setShowTourModal(true)}
        patientDetail={patientDetail}
        prediction={prediction}
        patientCount={patients.length}
      />

      {/* Main Content Area */}
      <div className="main-wrapper">
        <TopBar
          isCollapsed={isSidebarCollapsed}
          onToggleSidebar={() => setIsSidebarCollapsed(false)}
          activePage={activePage}
          patients={patients}
          selectedPatientId={selectedPatientId}
          onSelectPatient={loadPatient}
          loading={loading}
        />

        <main className="main-content">
          {/* Error Notification Banner */}
          {error && (
            <div style={{ background: '#fef2f2', border: '1px solid #fecaca', borderRadius: '10px', padding: '1rem 1.25rem', marginBottom: '1.5rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', color: '#b91c1c', fontSize: '0.88rem' }}>
                <AlertCircle size={20} />
                <span>{error}</span>
              </div>
              <button
                className="btn-secondary"
                onClick={() => loadPatient(selectedPatientId)}
                style={{ fontSize: '0.78rem' }}
              >
                <RefreshCw size={14} />
                <span>Retry</span>
              </button>
            </div>
          )}

          {/* Clinical Page Routing */}
          {activePage === 'dashboard' && (
            <DashboardFig8View
              patientDetail={patientDetail}
              prediction={prediction}
              shapData={shapData}
              limeData={limeData}
              recommendations={recommendations}
              sessionState={sessionState}
              onRecordAction={handleRecordAction}
              onDownloadPdf={handleDownloadPdf}
              downloadingPdf={downloadingPdf}
              rawSignal={rawSignal}
              filteredSignal={filteredSignal}
            />
          )}

          {activePage === 'signal' && (
            <SignalInspectionStep
              patientDetail={patientDetail}
              rawSignal={rawSignal}
              filteredSignal={filteredSignal}
              processReport={processReport}
              onProceed={() => handleSelectPage('diagnosis')}
            />
          )}

          {activePage === 'diagnosis' && (
            <DiagnosisStep
              prediction={prediction}
              shapData={shapData}
              limeData={limeData}
              onProceed={() => handleSelectPage('orders')}
              onBack={() => handleSelectPage('signal')}
            />
          )}

          {activePage === 'explainability' && (
            <DiagnosisStep
              prediction={prediction}
              shapData={shapData}
              limeData={limeData}
              onProceed={() => handleSelectPage('orders')}
              onBack={() => handleSelectPage('dashboard')}
            />
          )}

          {activePage === 'orders' && (
            <PhysicianActionStep
              recommendations={recommendations}
              sessionState={sessionState}
              onRecordAction={handleRecordAction}
              onDownloadPdf={handleDownloadPdf}
              onBack={() => handleSelectPage('diagnosis')}
              downloadingPdf={downloadingPdf}
            />
          )}

          {activePage === 'cohort' && (
            <PatientCohortView
              patients={patients}
              selectedPatientId={selectedPatientId}
              onSelectPatient={loadPatient}
              onNavigateToDashboard={() => handleSelectPage('dashboard')}
            />
          )}
        </main>
      </div>

      {/* Onboarding Guide Modal */}
      {showTourModal && (
        <QuickTourModal onClose={() => setShowTourModal(false)} />
      )}
    </div>
  );
}
