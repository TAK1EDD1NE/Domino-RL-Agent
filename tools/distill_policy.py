"""Fine-tune a checkpoint on PIMC targets (search distillation).

    uv run python tools/distill_policy.py --model models/checkpoints/ppo_16002774_steps.zip \\
        --targets results/pimc_targets_pimc_2026-10-14.npz --epochs 8 --lr 1e-4 --value-weight 0.2

Splits **by deal**, trains two losses with one optimiser (masked cross-entropy against the
search's action plus a KL against its soft ranking, and the public value head against the
mean-critic value), prints per-epoch held-out metrics, and saves the result as a checkpoint of
the *current* policy class so it can be evaluated like any other (``test.py``, the harness, the
judge, ``tools/leaf_ab.py``).

The fine-tuned checkpoint keeps the same observation layout, so the usual protocol applies:
evaluate it against the rule bot on the duplicate-deal harness before believing anything.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import pathlib

import numpy as np

from domino.agents.pool import load_policy
from domino.distill import distill, evaluate_agreement, load_targets, split_by_deal


def main(argv: list[str] | None = None) -> int:
    """Run the distillation and save the checkpoint."""
    parser = argparse.ArgumentParser(description="Distil PIMC decisions into the policy.")
    parser.add_argument("--model", required=True, help="starting checkpoint")
    parser.add_argument("--targets", required=True, help=".npz written by tools/gen_pimc_targets.py")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128, dest="batch_size")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="softens the expert's q into a distribution for the KL term")
    parser.add_argument("--soft-weight", type=float, default=0.5, dest="soft_weight")
    parser.add_argument("--value-weight", type=float, default=0.2, dest="value_weight")
    parser.add_argument("--val-fraction", type=float, default=0.25, dest="val_fraction")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    parser.add_argument("--history", default=None, help="CSV path for the per-epoch metrics")
    args = parser.parse_args(argv)

    targets = load_targets(args.targets)
    train, val = split_by_deal(targets, val_fraction=args.val_fraction,
                              rng=np.random.default_rng(args.seed))
    print(f"targets: {len(targets)} row(s) from {int(np.unique(targets.deal).size)} deal(s) -> "
          f"train {len(train)} / val {len(val)} (split by deal)")

    baseline = evaluate_agreement(load_policy(args.model), val, device=args.device)
    print("baseline on the validation split: " + " ".join(
        f"{key}={value:.4f}" for key, value in baseline.items()))

    model = load_policy(args.model)
    before = evaluate_agreement(model, val, device=args.device)
    tracker = distill(model, train, val=val, epochs=args.epochs, lr=args.lr,
                      batch_size=args.batch_size, temperature=args.temperature,
                      soft_weight=args.soft_weight, value_weight=args.value_weight,
                      device=args.device, log=print, rng=np.random.default_rng(args.seed + 1))
    after = evaluate_agreement(model, val, device=args.device)

    print("\nvalidation before -> after:")
    for key in sorted(after):
        print(f"  {key:<16}{before[key]:>10.4f} -> {after[key]:>10.4f}")
    agreement_up = after["val_agreement"] - before["val_agreement"]
    print(f"agreement {agreement_up:+.1%}, value MSE "
          f"{before['val_value_mse']:.1f} -> {after['val_value_mse']:.1f}")

    out = pathlib.Path(args.out) if args.out else pathlib.Path("models/checkpoints") / (
        f"distilled_{pathlib.Path(args.targets).stem}_{_dt.datetime.now().strftime('%Y%m%d-%H%M')}.zip")
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    history = pathlib.Path(args.history) if args.history else pathlib.Path("results") / (
        f"distill_{pathlib.Path(args.targets).stem}_{_dt.datetime.now().strftime('%Y%m%d-%H%M%S')}.csv")
    history.parent.mkdir(parents=True, exist_ok=True)
    with history.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(tracker.rows[0].keys()))
        writer.writeheader()
        writer.writerows(tracker.rows)
    print(f"\nwrote {out}\nwrote {history}")
    print("\nnext: uv run python tools/run_search_eval.py --matchups rl_vs_rule,rl_vs_random "
          f"--model {out} --deals 30")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
