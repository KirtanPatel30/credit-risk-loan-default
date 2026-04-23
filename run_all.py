"""run_all.py — Full pipeline runner."""
import subprocess, sys
from pathlib import Path

def run(cmd, desc):
    print(f"\n{'='*60}\n  {desc}\n{'='*60}")
    r = subprocess.run(cmd, shell=True, cwd=Path(__file__).parent)
    if r.returncode != 0:
        print(f"ERROR: failed with code {r.returncode}")
        sys.exit(r.returncode)

if __name__ == "__main__":
    print("\n🏦 CREDIT RISK — LOAN DEFAULT PREDICTION PIPELINE")
    print("="*60)
    run("python data/generate.py",   "STEP 1/3: Generating loan dataset")
    run("python models/train.py",    "STEP 2/3: Training model + SHAP")
    run("python -m pytest tests/ -v","STEP 3/3: Running unit tests")
    print("\n✅ COMPLETE!")
    print("  streamlit run dashboard/app.py  → http://localhost:8501")
    print("  uvicorn api.main:app --reload   → http://localhost:8000/docs")
