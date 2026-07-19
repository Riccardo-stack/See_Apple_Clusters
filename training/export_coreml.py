"""
CoreML Export Script

Exports a trained YOLO model to CoreML format. Uses a clean virtual
environment with compatible numpy versions to avoid conversion issues.
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path

def main():
    project_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description="Export YOLO model to CoreML format.")
    parser.add_argument("--weights", type=str, default=str(project_root / "models" / "best.pt"), help="Path to the model weights")
    args = parser.parse_args()

    WEIGHTS = args.weights
    VENV_DIR = project_root / ".export_venv"

    print("=" * 60)
    print("Clean CoreML Export (numpy < 2, no monkey-patches)")
    print("=" * 60)

    # ── Step 1: Create isolated venv with compatible numpy ──
    if not (VENV_DIR / "bin" / "python3").exists():
        print("\n📦 Creating isolated Python environment...")
        subprocess.check_call([sys.executable, "-m", "venv", str(VENV_DIR)])
        
        pip = str(VENV_DIR / "bin" / "pip")
        print("📦 Installing dependencies (this takes ~1 min)...")
        subprocess.check_call([
            pip, "install", "--quiet",
            "ultralytics>=8.4.90",
            "coremltools>=9.0",
            "numpy>=1.26,<2",
        ])
        print("✅ Environment ready")
    else:
        print("✅ Using existing export environment")

    # ── Step 2: Run export in the isolated environment ──
    python = str(VENV_DIR / "bin" / "python3")

    export_code = f"""
import os
from ultralytics import YOLO
import numpy as np
import coremltools as ct

print(f"   numpy: {{np.__version__}}")
print(f"   coremltools: {{ct.__version__}}")

model = YOLO("{WEIGHTS}")
result = model.export(format="coreml", imgsz=640)
print(f"\\n✅ Exported: {{result}}")
"""

    print(f"\n🔄 Exporting model...")
    subprocess.check_call([python, "-c", export_code])

if __name__ == '__main__':
    main()
