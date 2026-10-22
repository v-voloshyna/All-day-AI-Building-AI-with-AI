"""python predict.py "<text>" as a real subprocess, from an unrelated working directory."""
import subprocess
import sys

import pytest


def run_cli(demo_dir, cwd, *args, stdin=None):
    return subprocess.run(
        [sys.executable, str(demo_dir / "predict.py"), *args],
        input=stdin, capture_output=True, text=True, cwd=cwd, timeout=120,
    )


def test_cli_with_argument(demo_dir, tmp_path):
    r = run_cli(demo_dir, tmp_path, "A wonderful, excellent film")
    assert r.returncode == 0
    assert r.stdout.strip() == "positive"
    assert r.stderr == ""


def test_cli_joins_multiple_arguments(demo_dir, tmp_path):
    r = run_cli(demo_dir, tmp_path, "boring", "awful", "worst")
    assert r.stdout.strip() == "negative"


def test_cli_reads_stdin_when_no_arguments(demo_dir, tmp_path):
    r = run_cli(demo_dir, tmp_path, stdin="what an excellent and wonderful movie\n")
    assert r.returncode == 0 and r.stdout.strip() == "positive"


def test_cli_empty_input_still_prints_a_valid_label(demo_dir, tmp_path):
    r = run_cli(demo_dir, tmp_path, stdin="")
    assert r.returncode == 0 and r.stdout.strip() in {"positive", "negative"}


@pytest.mark.parametrize("text", ["这部电影非常好", "café \U0001F600", "'; DROP TABLE x; --", "-5", "--help"])
def test_cli_handles_unusual_arguments(demo_dir, tmp_path, text):
    r = run_cli(demo_dir, tmp_path, text)
    assert r.returncode == 0
    assert r.stdout.strip() in {"positive", "negative"}


def test_cli_prints_exactly_one_line(demo_dir, tmp_path):
    r = run_cli(demo_dir, tmp_path, "great")
    assert r.stdout.count("\n") == 1
