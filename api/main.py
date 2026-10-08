"""
Main FastAPI Application Entrypoint.
Connects clinical REST API routers and serves compiled React frontend in production.
"""

import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.state import state
from api.routes import patients, eeg, predict, explain, dss

app = FastAPI(
    title="Explainable EEG Stroke Clinical Decision Support System API",
    description="RESTful API for acute stroke screening using 1-channel FP1 EEG, Ensemble ESN, XAI (SHAP/LIME), and clinical DSS.",
    version="2.0.0"
)

# Enable CORS for React frontend (development & production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register REST Routers
app.include_router(patients.router)
app.include_router(eeg.router)
app.include_router(predict.router)
app.include_router(explain.router)
app.include_router(dss.router)


@app.on_event("startup")
def on_startup():
    """Pre-initialize models, dataset, and explainers on service boot."""
    print("=" * 65)
    print("  EXPLAINABLE EEG STROKE DSS - FASTAPI BACKEND INITIALIZING")
    print("=" * 65)
    state.initialize()
    print("  FastAPI Backend Ready on http://localhost:8000")
    print("=" * 65)


@app.get("/api/health")
def health_check():
    """Health status endpoint."""
    return {
        "status": "healthy",
        "service": "EEG Stroke DSS Backend",
        "initialized": state.initialized,
        "n_samples": len(state.raw_df) if state.raw_df is not None else 0,
        "model": "Ensemble Echo State Network (7 estimators, 200 neurons)"
    }


# Mount compiled frontend static build if available
frontend_dist = PROJECT_ROOT / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")
