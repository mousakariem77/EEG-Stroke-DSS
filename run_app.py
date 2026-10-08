"""
Unified Single-Command Launcher for Explainable EEG Stroke DSS
Boots the FastAPI backend and serves the modern React web application on http://127.0.0.1:8000
"""

import sys
import os
import subprocess
import time
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

def main():
    print("=" * 72)
    print("      EXPLAINABLE EEG STROKE CLINICAL DECISION SUPPORT SYSTEM")
    print("       Bouazizi & Ltifi (Decision Support Systems 2024 Replication)")
    print("=" * 72)

    # 1. Verify frontend build
    dist_dir = PROJECT_ROOT / "frontend" / "dist"
    if not dist_dir.exists() or not (dist_dir / "index.html").exists():
        print("[BUILD] Compiling modern React frontend assets (one-time setup)...")
        subprocess.run(["npm", "run", "build"], cwd=str(PROJECT_ROOT / "frontend"), check=True, shell=True)
        print("[BUILD] Compilation complete.\n")

    # 2. Inform user
    print("[SERVER] Starting unified FastAPI server + React Frontend...")
    print("[SERVER] Access URL: http://127.0.0.1:8000")
    print("[SERVER] Press Ctrl+C at any time to stop the server.")
    print("=" * 72 + "\n")

    # Open browser automatically after a short delay
    def open_browser():
        time.sleep(1.2)
        try:
            webbrowser.open("http://127.0.0.1:8000")
        except Exception:
            pass

    import threading
    threading.Thread(target=open_browser, daemon=True).start()

    # 3. Launch Uvicorn
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="127.0.0.1",
        port=8000,
        log_level="info"
    )

if __name__ == "__main__":
    main()
