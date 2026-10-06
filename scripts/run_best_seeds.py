"""Retrain the best configuration of a W&B sweep over several seeds (paper: 42-46).

The best run is selected by the sweep's own metric (valid/macro-f1 or valid/AUROC).
Its hyperparameters are replayed as Hydra overrides on the sweep's training script,
once per seed, and the test metrics are summarised as mean ± std.

Usage:
    uv run python scripts/run_best_seeds.py <entity>/<project>/<sweep_id>
    uv run python scripts/run_best_seeds.py <sweep_path> --seeds 42 43 44 45 46 --offline
    uv run python scripts/run_best_seeds.py <sweep_path> --dry-run

Options:
    --seeds      seeds to run (default: 42 43 44 45 46)
    --offline    do not log the reruns to W&B (adds the '~logger' override)
    --extra      additional Hydra overrides, e.g. --extra trainer=cpu model.args.device=cpu
    --dry-run    print the commands without running them
"""

import argparse
import re
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METRIC_LINE = re.compile(r"^\s*(test/[\w@.-]+)\s+([-+0-9.eE]+)\s*$")


def best_overrides(sweep_path: str) -> tuple[str, list[str], str]:
    """Return (training script, Hydra overrides of the best run, metric name)."""
    import wandb

    sweep = wandb.Api().sweep(sweep_path)
    metric = sweep.config["metric"]["name"]
    goal = sweep.config["metric"].get("goal", "maximize")
    best = sweep.best_run(order=metric if goal == "maximize" else f"-{metric}")
    if best is None:
        sys.exit(f"No finished runs with '{metric}' in sweep {sweep_path}.")

    overrides = []
    for name, spec in sweep.config["parameters"].items():
        value = spec["value"] if "value" in spec else best.config.get(name)
        if value is None:
            continue
        if isinstance(value, bool):
            value = str(value).lower()
        overrides.append(f"{name}={value}")

    print(f"Best run {best.id} ({metric} = {best.summary.get(metric)})")
    return sweep.config["program"], overrides, metric


def run_seed(program: str, overrides: list[str], seed: int, dry_run: bool) -> dict:
    cmd = [sys.executable, program, *overrides, f"seed={seed}"]
    print("\n$ " + " ".join(cmd), flush=True)
    if dry_run:
        return {}
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-2000:], proc.stderr[-4000:], sep="\n")
        sys.exit(f"Seed {seed} failed with exit code {proc.returncode}.")
    metrics = {}
    for line in proc.stdout.splitlines():
        match = METRIC_LINE.match(line)
        if match:
            metrics[match.group(1)] = float(match.group(2))
    print("  " + ", ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items())))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("sweep", help="W&B sweep path: <entity>/<project>/<sweep_id>")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--extra", nargs="*", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    program, overrides, _ = best_overrides(args.sweep)
    overrides += args.extra
    if args.offline:
        overrides.append("~logger")
    else:
        overrides.append(f"tags=[best_seeds,{args.sweep.split('/')[-1]}]")

    results = [run_seed(program, overrides, s, args.dry_run) for s in args.seeds]
    if args.dry_run:
        return

    print(f"\nTest metrics over seeds {args.seeds} (mean ± std, in %):")
    for name in sorted(results[0]):
        values = [100 * r[name] for r in results if name in r]
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        print(f"  {name:<20} {statistics.mean(values):6.2f} ± {std:.2f}")


if __name__ == "__main__":
    main()
