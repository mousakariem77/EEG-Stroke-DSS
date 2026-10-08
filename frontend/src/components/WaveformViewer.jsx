import React, { useRef, useEffect, useState } from 'react';
import { ZoomIn, ZoomOut, RotateCcw, Eye, ShieldAlert } from 'lucide-react';

export default function WaveformViewer({
  rawSignal = [],
  filteredSignal = [],
  samplingRate = 128,
  durationSec = 8.0,
  title = "FP1 Pre-frontal Resting-State EEG Waveform"
}) {
  const canvasRef = useRef(null);
  const [showRaw, setShowRaw] = useState(false);
  const [zoom, setZoom] = useState(1);
  const [offset, setOffset] = useState(0);
  const [hoverInfo, setHoverInfo] = useState(null);

  const activeSignal = showRaw && rawSignal.length > 0 ? rawSignal : (filteredSignal.length > 0 ? filteredSignal : rawSignal);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    // Clear background
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, width, height);

    if (!activeSignal || activeSignal.length === 0) {
      ctx.fillStyle = '#94a3b8';
      ctx.font = '14px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('No signal loaded. Select a patient or run simulator.', width / 2, height / 2);
      return;
    }

    // Grid properties
    const centerY = height / 2;
    const maxAmplitude = 120; // uV range [-120, +120]
    const scaleY = (height / 2 - 20) / maxAmplitude;

    // Draw Subtle Grid Lines
    ctx.strokeStyle = '#f1f5f9';
    ctx.lineWidth = 1;

    // Horizontal voltage lines
    [-100, -50, 0, 50, 100].forEach((uv) => {
      const y = centerY - uv * scaleY;
      ctx.beginPath();
      ctx.moveTo(40, y);
      ctx.lineTo(width - 10, y);
      ctx.stroke();

      ctx.fillStyle = '#94a3b8';
      ctx.font = '10px JetBrains Mono, monospace';
      ctx.textAlign = 'right';
      ctx.fillText(`${uv} µV`, 36, y + 3);
    });

    // Draw Artifact Threshold Limits (+- 100 uV)
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = '#fca5a5'; // subtle red
    ctx.lineWidth = 1.2;

    const topThresholdY = centerY - 100 * scaleY;
    const bottomThresholdY = centerY - (-100) * scaleY;

    ctx.beginPath();
    ctx.moveTo(40, topThresholdY);
    ctx.lineTo(width - 10, topThresholdY);
    ctx.moveTo(40, bottomThresholdY);
    ctx.lineTo(width - 10, bottomThresholdY);
    ctx.stroke();
    ctx.setLineDash([]); // Reset dash

    // Draw 4.0-second Epoch Boundaries
    const samplesPerEpoch = Math.round(4.0 * samplingRate);
    const visibleSamples = Math.floor(activeSignal.length / zoom);
    const startIdx = Math.max(0, Math.min(offset, activeSignal.length - visibleSamples));
    const endIdx = Math.min(activeSignal.length, startIdx + visibleSamples);

    const stepX = (width - 50) / (visibleSamples - 1);

    ctx.strokeStyle = '#bfdbfe'; // Light blue epoch divider
    ctx.lineWidth = 1.5;
    for (let i = 0; i < activeSignal.length; i += samplesPerEpoch) {
      if (i >= startIdx && i <= endIdx) {
        const x = 45 + (i - startIdx) * stepX;
        ctx.beginPath();
        ctx.moveTo(x, 15);
        ctx.lineTo(x, height - 25);
        ctx.stroke();

        ctx.fillStyle = '#0284c7';
        ctx.font = '10px Inter, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText(`Epoch ${(i / samplesPerEpoch) + 1} (4s | 0.25 Hz)`, x + 4, 25);
      }
    }

    // Draw EEG Signal Waveform
    ctx.beginPath();
    ctx.lineWidth = 1.6;
    ctx.strokeStyle = showRaw ? '#f97316' : '#0284c7'; // Orange for raw, Sky blue for filtered

    for (let i = startIdx; i < endIdx; i++) {
      const x = 45 + (i - startIdx) * stepX;
      const val = activeSignal[i];
      const y = centerY - val * scaleY;

      if (i === startIdx) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();

    // Time Axis Labels
    ctx.fillStyle = '#64748b';
    ctx.font = '10px JetBrains Mono, monospace';
    ctx.textAlign = 'center';

    const timeStepSec = 1.0;
    const totalDuration = activeSignal.length / samplingRate;
    for (let sec = 0; sec <= totalDuration; sec += timeStepSec) {
      const sampleIdx = Math.round(sec * samplingRate);
      if (sampleIdx >= startIdx && sampleIdx <= endIdx) {
        const x = 45 + (sampleIdx - startIdx) * stepX;
        ctx.fillText(`${sec.toFixed(1)}s`, x, height - 8);
      }
    }
  }, [activeSignal, showRaw, zoom, offset, samplingRate]);

  return (
    <div style={{ background: '#ffffff', border: '1px solid #e2e8f0', borderRadius: '10px', padding: '0.85rem 1rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.65rem' }}>
        <div>
          <h3 style={{ fontSize: '0.94rem', fontWeight: 700, color: '#0f172a' }}>{title}</h3>
          <p style={{ fontSize: '0.74rem', color: '#64748b' }}>
            Sampling: {samplingRate} Hz | Epoch Length: 4.0s (FFT Resolution Δf = 0.25 Hz) | Artifact Threshold: &plusmn;100 &mu;V
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          {/* Toggle Raw vs Filtered */}
          {rawSignal.length > 0 && filteredSignal.length > 0 && (
            <button
              onClick={() => setShowRaw(!showRaw)}
              className="btn-secondary"
              style={{ fontSize: '0.75rem', padding: '0.35rem 0.75rem' }}
            >
              <Eye size={14} color={showRaw ? '#f97316' : '#0284c7'} />
              <span>{showRaw ? 'Showing: Raw EEG' : 'Showing: Filtered (0.5-30Hz)'}</span>
            </button>
          )}

          {/* Zoom controls */}
          <button
            onClick={() => setZoom(Math.min(4, zoom + 0.5))}
            className="btn-secondary"
            title="Zoom In"
            style={{ padding: '0.35rem 0.55rem' }}
          >
            <ZoomIn size={14} />
          </button>
          <button
            onClick={() => { setZoom(Math.max(1, zoom - 0.5)); setOffset(0); }}
            className="btn-secondary"
            title="Zoom Out"
            style={{ padding: '0.35rem 0.55rem' }}
          >
            <ZoomOut size={14} />
          </button>
          <button
            onClick={() => { setZoom(1); setOffset(0); }}
            className="btn-secondary"
            title="Reset View"
            style={{ padding: '0.35rem 0.55rem' }}
          >
            <RotateCcw size={14} />
          </button>
        </div>
      </div>

      <div style={{ position: 'relative', width: '100%', overflow: 'hidden', border: '1px solid #e2e8f0', borderRadius: '8px' }}>
        <canvas
          ref={canvasRef}
          width={1000}
          height={240}
          style={{ width: '100%', height: '240px', display: 'block', cursor: 'crosshair' }}
        />
      </div>

      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: '0.6rem', fontSize: '0.75rem', color: '#64748b' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem' }}>
            <span style={{ width: 12, height: 3, background: showRaw ? '#f97316' : '#0284c7', display: 'inline-block' }}></span>
            <span>{showRaw ? 'Raw EEG Signal' : 'Zero-Phase Butterworth (0.5-30 Hz) + 50Hz Notch'}</span>
          </span>
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem' }}>
            <span style={{ width: 12, height: 1, borderTop: '2px dashed #fca5a5', display: 'inline-block' }}></span>
            <span>&plusmn;100 &mu;V Artifact Threshold</span>
          </span>
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem' }}>
            <span style={{ width: 12, height: 2, background: '#bfdbfe', display: 'inline-block' }}></span>
            <span>4.0s Epoch Boundaries (0.25 Hz)</span>
          </span>
        </div>
      </div>
    </div>
  );
}
