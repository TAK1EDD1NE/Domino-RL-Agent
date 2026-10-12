"""Script-level tests for ``train.py`` (spec sections 9, 13; plan Task 11).

Runs the trainer as a subprocess (the real Windows ``spawn`` path) and asserts
the CLI contract: ``--help`` exits 0, ``--dry-run`` prints the parameter count
and writes no checkpoint, the 4096-step smoke run produces a checkpoint, a
TensorBoard event file and a training CSV, and ``--resume`` continues from the
checkpoint.

Beyond the step-counter growth the resume path must actually **reload the saved
policy**: a resume that performs no further gradient steps has to leave the
policy weights bit-exact (a re-initialised policy that kept the step counter
would change them). The same bit-exact requirement is asserted for a plain
save->load round trip through the project's own loader
(``domino.agents.pool.load_policy``), and for a pool configured with
``{"snapshot": 1.0}`` loading a real on-disk checkpoint and returning legal
actions.

The only training executed is the short 4096-step smoke run required by the plan
plus two tiny resume runs (0 and 1024 further steps); no long run is ever
started here.
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch as th
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Make the root-level ``train`` module importable (it is not part of the
# installed ``domino`` package) so the pure CLI helpers can be unit-tested.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import train  # noqa: E402  (imported after the sys.path fix above)

from domino.agents import OpponentPool  # noqa: E402
from domino.agents.pool import load_policy  # noqa: E402
from domino.config import TrainConfig  # noqa: E402
from domino.engine import new_deal  # noqa: E402
from domino.env import DominoEnv  # noqa: E402
from domino.masks import assert_consistent  # noqa: E402
from domino.policy import DominoMaskablePolicy, policy_kwargs_for  # noqa: E402

CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"
RUNS_DIR = PROJECT_ROOT / "runs"
RESULTS_DIR = PROJECT_ROOT / "results"


def _run_train(*args: str, timeout: int = 900) -> subprocess.CompletedProcess:
    """Run ``train.py`` with ``args`` from the project root, capturing output."""
    return subprocess.run(
        [sys.executable, "train.py", *args],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# ---- helpers ---------------------------------------------------------------
def _rule_env(seed: int) -> DominoEnv:
    """A cheap env whose opponent is the deterministic rule bot (no snapshot IO)."""
    return DominoEnv(opponent=OpponentPool(weights={"rule": 1.0}, seed=seed), seed=seed)


def _tiny_model(seed: int = 0) -> tuple[MaskablePPO, DummyVecEnv]:
    """Build (but do not train) a small real ``MaskablePPO`` on a rule-only env."""
    venv = DummyVecEnv([partial(_rule_env, seed)])
    model = MaskablePPO(
        DominoMaskablePolicy,
        venv,
        policy_kwargs=policy_kwargs_for(TrainConfig()),
        n_steps=32,
        batch_size=32,
        n_epochs=1,
        seed=seed,
        device="cpu",
        verbose=0,
    )
    return model, venv


def _policy_state_hash(model: MaskablePPO) -> str:
    """SHA-256 over the policy state dict (sorted keys, CPU bytes)."""
    digest = hashlib.sha256()
    state = model.policy.state_dict()
    for key in sorted(state):
        value = state[key]
        if not isinstance(value, th.Tensor):
            continue
        digest.update(key.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.detach().to("cpu").contiguous().numpy().tobytes())
    return digest.hexdigest()


def _assert_state_dicts_bit_equal(a: MaskablePPO, b: MaskablePPO) -> None:
    """Assert two models' policy state dicts are bit-exact (not merely close)."""
    sa, sb = a.policy.state_dict(), b.policy.state_dict()
    assert set(sa) == set(sb), "state dict key sets differ"
    for key in sa:
        va, vb = sa[key], sb[key]
        if not isinstance(va, th.Tensor):
            assert va == vb, f"non-tensor entry {key} differs"
            continue
        assert th.equal(va.detach().to("cpu"), vb.detach().to("cpu")), f"weight {key} differs"


def _decision_states(n: int, seed: int) -> list[tuple[object, int]]:
    """Collect ``n`` real mid-deal decision points ``(state, acting_seat)``."""
    rng = np.random.default_rng(seed)
    states: list[tuple[object, int]] = []
    deal_seed = 0
    while len(states) < n:
        state = new_deal(deal_no=1, rng=np.random.default_rng(deal_seed))
        deal_seed += 1
        while not state.is_terminal() and len(states) < n:
            seat = state.turn
            if state.must_pass(seat):
                state.pass_turn(seat)
                continue
            states.append((state.clone(), seat))
            legal = state.legal_moves(seat)
            tid, end = legal[int(rng.integers(len(legal)))]
            state.play(seat, tid, end)
    return states


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


def test_opponent_self_maps_to_live_policy_when_available():
    # Minor 3: with a live learner available (the in-process --n-envs 1 path)
    # --opponent self is pure mirror self-play against the current policy; the
    # vectorised path keeps the spawn-safe frozen-snapshot fallback.
    assert train.opponent_weights("self", live_self=True) == {"self": 1.0}
    assert train.opponent_weights("self", live_self=False) == {"snapshot": 1.0}
    assert train.opponent_weights("league", live_self=True) == train.LIVE_LEAGUE_WEIGHTS
    assert train.opponent_weights("league", live_self=False) == train.FROZEN_LEAGUE_WEIGHTS


# ---- native checkpoint loading ---------------------------------------------
def test_save_load_roundtrip_via_pool_loader_is_bit_exact(tmp_path):
    # The project's own loading choke point (domino.agents.pool.load_policy) must
    # round-trip weights bit-exactly with no torch.load monkeypatch anywhere.
    model, venv = _tiny_model(seed=0)
    try:
        path = tmp_path / "policy_0.zip"
        model.save(str(path))
        assert path.exists()
        loaded = load_policy(str(path))
        _assert_state_dicts_bit_equal(model, loaded)
        assert _policy_state_hash(model) == _policy_state_hash(loaded)
    finally:
        venv.close()


def test_pool_snapshot_source_loads_real_checkpoint_and_returns_legal_actions(tmp_path):
    # A pool configured for pure snapshot self-play loads a real on-disk
    # checkpoint through the project's loader and produces legal actions.
    model, venv = _tiny_model(seed=0)
    try:
        path = tmp_path / "policy_0.zip"
        model.save(str(path))
    finally:
        venv.close()

    pool = OpponentPool(weights={"snapshot": 1.0}, snapshot_paths=[str(path)], seed=0)
    assert "snapshot" in pool._available_weights()
    agent = pool.sample()
    assert agent.name == "snapshot"

    for state, seat in _decision_states(30, seed=1):
        action = agent.act(state, seat)
        assert isinstance(action, int)
        assert 0 <= action < 56
        assert_consistent(state, seat, action)


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

    model_before = MaskablePPO.load(str(latest))
    first_steps = int(model_before.num_timesteps)
    assert first_steps >= 4096
    weights_before = _policy_state_hash(model_before)

    # Resume fidelity (Important 2): a resume that performs no further gradient
    # steps must reload the *saved* policy bit-exactly and keep the step counter.
    # A regression that re-initialised the policy but kept the counter would fail
    # here, where the step-growth check alone would not.
    noop = _run_train(
        "--resume", "--steps", "0", "--n-envs", "2", "--run-name", "smoke_test", timeout=900
    )
    assert noop.returncode == 0, noop.stderr
    model_noop = MaskablePPO.load(str(latest))
    assert int(model_noop.num_timesteps) == first_steps, "resume changed the step counter"
    assert _policy_state_hash(model_noop) == weights_before, (
        "resume did not preserve the saved policy weights (re-initialised?)"
    )
    _assert_state_dicts_bit_equal(model_before, model_noop)

    # Resume: continues from latest.zip and the step counter grows.
    resume = _run_train(
        "--resume", "--steps", "1024", "--n-envs", "2", "--run-name", "smoke_test", timeout=900
    )
    assert resume.returncode == 0, resume.stderr
    second_steps = int(MaskablePPO.load(str(latest)).num_timesteps)
    assert second_steps > first_steps, f"resume did not advance: {second_steps} <= {first_steps}"
