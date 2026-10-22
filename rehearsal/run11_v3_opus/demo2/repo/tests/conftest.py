import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# predict.py loads the model at import time, so it must exist before test modules are collected.
if not (ROOT / "model.joblib").exists():
    import train

    train.main()
