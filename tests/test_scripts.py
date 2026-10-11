"""Script-level tests for ``train.py`` (spec sections 9, 13; plan Task 11).

Runs the trainer as a subprocess (the real Windows ``spawn`` path) and asserts
the CLI contract: ``--help`` exits 0, ``--dry-run`` prints the parameter count
and writes no checkpoint, the 4096-step smoke run produces a checkpoint, a
TensorBoard event file and a training CSV, and ``--resume`` continues from the
checkpoint (the step counter grows).

The only training executed is the short smoke run required by the plan; no long
run is ever started here.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from sb3_contrib import MaskablePPO

from domino.callbacks import enable_checkpoint_loading

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"
RUNS_DIR = PROJECT_ROOT / "runs"
RESULTS_DIR = PROJECT_ROOT / "results"

# Reloading the checkpoint to read num_timesteps needs the torch/SB3 workaround.
enable_checkpoint_loading()


def _run_train(*args: str, timeout: int = 900) -> subprocess.CompletedProcess:
    """Run ``train.py`` with ``args`` from the project root, capturing output."""
    return subprocess.run(
        [sys.executable, "train.py", *args],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_train_help_exits_zero():
    result = _run_train("--help", timeout=120)
    assert result.returncode == 0, result.stderr
    assert "usage" in result.stdout.lower()
    for flag in ("--config", "--run-name", "--steps", "--n-envs", "--seed", "--reward-mode",
                 "--opponent", "--snapshot-every", "--ckpt-every", "--device", "--resume",
                 "--dry-run", "--set"):
        assert flag in result.stdout


def test_train_dry_run_prints_param_count_and_writes_no_checkpoint():
    before = set(CHECKPOINT_DIR.glob("*.zip")) if CHECKPOINT_DIR.exists() else set()
    result = _run_train("--dry-run", "--n-envs", "1", "--run-name", "dry_run_test", timeout=300)
    assert result.returncode == 0, result.stderr
    lines = [line for line in result.stdout.splitlines() if "Parameter count:" in line]
    assert lines, result.stdout
    assert any(char.isdigit() for char in lines[0])
    after = set(CHECKPOINT_DIR.glob("*.zip")) if CHECKPOINT_DIR.exists() else set()
    assert after == before, "dry run must not write a checkpoint"


def test_train_smoke_run_writes_artifacts_and_resumes():
    result = _run_train(
        "--steps", "4096",
        "--n-envs", "2",
        "--run-name", "smoke_test",
        "--snapshot-every", "1024",
        "--ckpt-every", "2048",
        timeout=900,
    )
    assert result.returncode == 0, result.stderr
    assert "fps" in result.stdout, "SB3 should print the measured throughput"

    latest = CHECKPOINT_DIR / "latest.zip"
    assert latest.exists(), "no latest checkpoint"
    assert (CHECKPOINT_DIR / "ppo_4096_steps.zip").exists()

    events = list((RUNS_DIR / "smoke_test").rglob("events.out.tfevents.*"))
    assert events, "no TensorBoard event file under runs/smoke_test"

    csv_path = RESULTS_DIR / "training_smoke_test.csv"
    assert csv_path.exists(), "no training CSV"
    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "step,episode,total_reward,window_avg"
    assert len(lines) >= 2, "expected at least one episode row"

    first_steps = int(MaskablePPO.load(str(latest)).num_timesteps)
    assert first_steps >= 4096

    # Resume: continues from latest.zip and the step counter grows.
    resume = _run_train(
        "--resume", "--steps", "1024", "--n-envs", "2", "--run-name", "smoke_test", timeout=900
    )
    assert resume.returncode == 0, resume.stderr
    second_steps = int(MaskablePPO.load(str(latest)).num_timesteps)
    assert second_steps > first_steps, f"resume did not advance: {second_steps} <= {first_steps}"
