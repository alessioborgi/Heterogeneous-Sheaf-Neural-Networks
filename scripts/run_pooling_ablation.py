"""SheafPool ablation (paper, Table 1).

For each graph-classification sweep, take its best configuration and retrain it with
every pooling method (mean, max, add, attention, late_token = SheafPool) over seeds
42-46, keeping all other hyperparameters fixed. Prints Macro/Micro-F1 mean ± std.

Usage:
    uv run python scripts/run_pooling_ablation.py <entity>/<project>/<sweep_id> [<sweep_path> ...]
    uv run python scripts/run_pooling_ablation.py <sweep_path> --offline --dry-run

Pass one sweep per dataset (MUTAG, PROTEINS, ENZYMES, NCI1), e.g. the sweeps created from
sweeps/gc/<dataset>/<model>.yaml for the HetSheaf backbone you want to ablate.
"""

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_best_seeds import best_overrides, run_seed  # noqa: E402

POOLING = ["mean", "max", "add", "attention", "late_token"]
REPORTED = ["test/macro-f1", "test/micro-f1"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("sweeps", nargs="+", help="W&B sweep paths <entity>/<project>/<id>")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--offline", action="store_true", help="do not log to W&B")
    parser.add_argument("--extra", nargs="*", default=[], help="extra Hydra overrides")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    table = []
    for sweep in args.sweeps:
        program, overrides, _ = best_overrides(sweep)
        overrides = [o for o in overrides if not o.startswith("pooling=")] + args.extra
        dataset = next((o.split("=", 1)[1] for o in overrides if o.startswith("dataset=")), sweep)
        for pooling in POOLING:
            run_overrides = overrides + [f"pooling={pooling}"]
            run_overrides.append("~logger" if args.offline else f"tags=[pooling_ablation,{pooling}]")
            results = [run_seed(program, run_overrides, s, args.dry_run) for s in args.seeds]
            if args.dry_run:
                continue
            row = [dataset, pooling]
            for name in REPORTED:
                values = [100 * r[name] for r in results if name in r]
                std = statistics.stdev(values) if len(values) > 1 else 0.0
                row.append(f"{statistics.mean(values):.2f} ± {std:.2f}")
            table.append(row)

    if table:
        print(f"\n{'dataset':<12}{'pooling':<12}{'Macro-F1':<18}{'Micro-F1':<18}")
        for row in table:
            print(f"{row[0]:<12}{row[1]:<12}{row[2]:<18}{row[3]:<18}")


if __name__ == "__main__":
    main()
