"""Turn the CSV artefacts into the report's figures and tables.

Reads every ``results/eval_*.csv`` (the evaluation matrix) and every
``results/training_*.csv`` (the convergence mirror written during training) and
writes PNGs plus a Markdown metric table under ``docs/report/``. The design is
deliberately dumb: whatever rows exist get plotted, so the figures can be
regenerated after any training or evaluation run without editing code.

Usage
-----
    uv run python tools/make_report_figures.py
    uv run python tools/make_report_figures.py --results results --out docs/report
"""
from __future__ import annotations

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")  # headless: figures are files, never windows

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

WINDOW = 100


def _load_eval(results: pathlib.Path) -> pd.DataFrame:
    """Concatenate every evaluation CSV (ignoring the training mirrors)."""
    frames = []
    for path in sorted(results.glob("eval_*.csv")):
        frame = pd.read_csv(path)
        frame["source"] = path.name
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_training(results: pathlib.Path) -> pd.DataFrame:
    """Concatenate every training mirror CSV."""
    frames = []
    for path in sorted(results.glob("training_*.csv")):
        frame = pd.read_csv(path)
        frame["run"] = path.stem.replace("training_", "")
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def plot_convergence(training: pd.DataFrame, out_dir: pathlib.Path) -> pathlib.Path | None:
    """Total (undiscounted) reward per episode with its sliding-window average."""
    if training.empty:
        return None
    figure, axis = plt.subplots(figsize=(9, 4.5))
    for run, frame in training.groupby("run"):
        frame = frame.sort_values("episode")
        axis.plot(frame["episode"], frame["total_reward"], alpha=0.28, linewidth=0.8,
                  label=f"{run} (per episode)")
        axis.plot(frame["episode"], frame["window_avg"], linewidth=2.0,
                  label=f"{run} ({WINDOW}-episode average)")
    axis.axhline(0.0, color="black", linewidth=0.8, linestyle=":")
    axis.set_xlabel("episode (deal)")
    axis.set_ylabel("total undiscounted reward (points)")
    axis.set_title("Training convergence: total reward per episode")
    axis.legend(fontsize=8)
    figure.tight_layout()
    path = out_dir / "figures" / "convergence.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def plot_win_rates(evaluation: pd.DataFrame, out_dir: pathlib.Path) -> pathlib.Path | None:
    """Win rate per matchup with its Wilson 95 % interval."""
    if evaluation.empty or "win_rate_a" not in evaluation.columns:
        return None
    frame = evaluation.dropna(subset=["win_rate_a"]).copy()
    frame = frame.sort_values("win_rate_a")
    figure, axis = plt.subplots(figsize=(8, 4.5))
    errors = [
        frame["win_rate_a"] - frame["ci_low"],
        frame["ci_high"] - frame["win_rate_a"],
    ]
    axis.barh(frame["matchup"], frame["win_rate_a"], xerr=errors, capsize=4, color="#4a86c8")
    axis.axvline(0.5, color="black", linewidth=0.9, linestyle="--", label="coin flip")
    axis.set_xlim(0, 1)
    axis.set_xlabel("deals won by agent A (Wilson 95 % CI)")
    axis.set_title("Head-to-head win rates")
    axis.legend(fontsize=8)
    figure.tight_layout()
    path = out_dir / "figures" / "win_rates.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def plot_points(evaluation: pd.DataFrame, out_dir: pathlib.Path) -> pathlib.Path | None:
    """Points banked per deal by each side, plus the blocked/void rates."""
    if evaluation.empty or "points_per_deal_a" not in evaluation.columns:
        return None
    frame = evaluation.copy()
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    width = 0.38
    positions = range(len(frame))
    axes[0].bar([p - width / 2 for p in positions], frame["points_per_deal_a"], width,
                label="agent A", color="#4a86c8")
    axes[0].bar([p + width / 2 for p in positions], frame["points_per_deal_b"], width,
                label="agent B", color="#c86f4a")
    axes[0].set_xticks(list(positions))
    axes[0].set_xticklabels(frame["matchup"], rotation=20, ha="right", fontsize=8)
    axes[0].set_ylabel("points banked per deal")
    axes[0].set_title("Margin per deal")
    axes[0].legend(fontsize=8)

    axes[1].bar([p - width / 2 for p in positions], frame["block_rate"], width,
                label="blocked deals", color="#6aa84f")
    axes[1].bar([p + width / 2 for p in positions], frame["void_rate"], width,
                label="void deals", color="#b45f06")
    axes[1].set_xticks(list(positions))
    axes[1].set_xticklabels(frame["matchup"], rotation=20, ha="right", fontsize=8)
    axes[1].set_ylabel("fraction of rounds")
    axes[1].set_title("Blocked and void rates")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    path = out_dir / "figures" / "deal_margins.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table (no extra dependency)."""
    headers = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines) + "\n"


def write_table(evaluation: pd.DataFrame, out_dir: pathlib.Path) -> pathlib.Path | None:
    """Write the metric table as Markdown (for pasting into the slides)."""
    if evaluation.empty:
        return None
    columns = [
        "matchup", "deals", "wins_a", "wins_b", "win_rate_a", "ci_low", "ci_high",
        "points_per_deal_a", "points_per_deal_b", "match_win_rate_a", "matches_played",
        "block_rate", "void_rate", "mean_pips_left_a", "ms_per_action", "seed", "source",
    ]
    present = [column for column in columns if column in evaluation.columns]
    frame = evaluation[present].copy()
    for column in ("win_rate_a", "ci_low", "ci_high", "points_per_deal_a",
                   "points_per_deal_b", "match_win_rate_a", "block_rate", "void_rate",
                   "mean_pips_left_a", "ms_per_action"):
        if column in frame.columns:
            frame[column] = frame[column].astype(float).round(3)
    path = out_dir / "tables" / "metrics.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_markdown_table(frame), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    """Generate every figure and table that the available CSVs support."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", default="results", help="directory holding the CSVs")
    parser.add_argument("--out", default="docs/report", help="output root for figures/tables")
    args = parser.parse_args(argv)

    results = pathlib.Path(args.results)
    out_dir = pathlib.Path(args.out)
    evaluation = _load_eval(results)
    training = _load_training(results)

    written = [
        plot_convergence(training, out_dir),
        plot_win_rates(evaluation, out_dir),
        plot_points(evaluation, out_dir),
        write_table(evaluation, out_dir),
    ]
    for path in written:
        print(f"wrote {path}" if path else "skipped (no matching CSV)")
    if not any(written):
        print("no results/*.csv found — run train.py and test.py first")
    return 0


if __name__ == "__main__":  # pragma: no cover - artefact generation entry point
    raise SystemExit(main())
