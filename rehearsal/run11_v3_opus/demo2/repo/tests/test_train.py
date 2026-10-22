import re
import shutil
import subprocess
import sys

import joblib
import pandas as pd

from conftest import ROOT


def test_train_script_end_to_end(tmp_path):
    # Run a copy in a temp dir so the real model.joblib is not overwritten.
    for name in ("train.py", "data.csv", "eval.csv"):
        shutil.copy(ROOT / name, tmp_path)
    result = subprocess.run(
        [sys.executable, str(tmp_path / "train.py")],
        cwd=ROOT.parent, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr

    match = re.search(r"Eval accuracy: ([\d.]+) \((\d+) examples\)", result.stdout)
    assert match, result.stdout
    assert float(match.group(1)) >= 0.90
    assert int(match.group(2)) == len(pd.read_csv(ROOT / "eval.csv"))

    model = joblib.load(tmp_path / "model.joblib")
    assert set(model.classes_) == {"positive", "negative"}


def test_data_files_are_well_formed():
    for name in ("data.csv", "eval.csv"):
        df = pd.read_csv(ROOT / name)
        assert list(df.columns) == ["text", "label"]
        assert df["text"].notna().all()
        assert set(df["label"]) == {"positive", "negative"}


def test_no_train_eval_leakage():
    train = set(pd.read_csv(ROOT / "data.csv")["text"])
    test = set(pd.read_csv(ROOT / "eval.csv")["text"])
    assert not train & test
