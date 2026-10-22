"""End-to-end test of train.py, run in a temp dir so the real model.joblib is untouched."""
import re
import shutil
import subprocess
import sys

import joblib

from conftest import ROOT


def test_train_script_saves_model_and_reports_accuracy(tmp_path):
    for name in ["train.py", "data.csv", "eval.csv"]:
        shutil.copy(ROOT / name, tmp_path)

    out = subprocess.run(
        [sys.executable, "train.py"], cwd=tmp_path, capture_output=True, text=True,
        check=True, timeout=60,
    ).stdout

    match = re.search(r"Eval accuracy: (\d\.\d+) \((\d+) examples\)", out)
    assert match, out
    assert float(match.group(1)) >= 0.90
    assert int(match.group(2)) == 404

    model = joblib.load(tmp_path / "model.joblib")
    assert set(model.classes_) == {"positive", "negative"}


def test_training_is_reproducible(tmp_path):
    for name in ["train.py", "data.csv", "eval.csv"]:
        shutil.copy(ROOT / name, tmp_path)
    runs = [
        subprocess.run([sys.executable, "train.py"], cwd=tmp_path,
                       capture_output=True, text=True, check=True).stdout
        for _ in range(2)
    ]
    assert runs[0] == runs[1]
