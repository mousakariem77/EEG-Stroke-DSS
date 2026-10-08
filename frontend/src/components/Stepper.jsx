import React from 'react';
import { Activity, Brain, ClipboardCheck, ChevronRight } from 'lucide-react';

export default function Stepper({ currentStep, onSelectStep }) {
  const steps = [
    { id: 1, label: 'Patient & EEG Signals', sub: 'Module 1: Preprocessing', icon: Activity },
    { id: 2, label: 'AI Risk & Explainability', sub: 'Modules 2 & 3: ESN + XAI', icon: Brain },
    { id: 3, label: 'Clinical Review & PDF', sub: 'Module 4: Decision Support', icon: ClipboardCheck }
  ];

  return (
    <div className="stepper-container">
      {steps.map((step, idx) => {
        const Icon = step.icon;
        const isActive = currentStep === step.id;
        const isPassed = currentStep > step.id;

        return (
          <React.Fragment key={step.id}>
            <div
              id={`step-item-${step.id}`}
              className={`step-item ${isActive ? 'active' : ''}`}
              onClick={() => onSelectStep(step.id)}
            >
              <div
                className="step-number"
                style={{
                  background: isActive ? '#0284c7' : isPassed ? '#10b981' : '#f1f5f9',
                  color: isActive || isPassed ? '#ffffff' : '#64748b'
                }}
              >
                {isPassed ? '✓' : step.id}
              </div>
              <div>
                <div className="step-title">{step.label}</div>
                <div style={{ fontSize: '0.72rem', color: '#94a3b8' }}>{step.sub}</div>
              </div>
            </div>

            {idx < steps.length - 1 && (
              <ChevronRight className="step-arrow" size={20} />
            )}
          </React.Fragment>
        );
      })}
    </div>
  );
}
