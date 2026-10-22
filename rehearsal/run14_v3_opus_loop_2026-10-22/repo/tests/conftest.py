import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# predict.py loads the model at import time, so make sure it exists first.
if not (ROOT / "model.joblib").exists():
    subprocess.run([sys.executable, "train.py"], cwd=ROOT, check=True)
