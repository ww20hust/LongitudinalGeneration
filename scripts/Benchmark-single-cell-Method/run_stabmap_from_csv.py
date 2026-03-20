from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from clinical_static_baseline_benchmark.data import load_longitudinal_csv_benchmark, save_prepared_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run StabMap on a longitudinal two-visit CSV benchmark.")
    parser.add_argument("--csv", required=True, help="Longitudinal two-visit clinical CSV.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--id-column", type=str, default="eid")
    parser.add_argument("--visit-column", type=str, default="visit")
    parser.add_argument("--lab-columns", type=str, default=None)
    parser.add_argument("--metab-columns", type=str, default=None)
    parser.add_argument("--lab-prefix", type=str, default=None)
    parser.add_argument("--metab-prefix", type=str, default=None)
    parser.add_argument("--expected-lab-dim", type=int, default=61)
    parser.add_argument("--expected-metab-dim", type=int, default=251)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--rscript-binary", type=str, default="Rscript")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    prepared_dir = output_dir / "prepared"
    stabmap_dir = output_dir / "stabmap"
    raw_split, scaled_split, lab_scaler, metab_scaler = load_longitudinal_csv_benchmark(
        args.csv,
        id_column=args.id_column,
        visit_column=args.visit_column,
        lab_columns=args.lab_columns,
        metab_columns=args.metab_columns,
        lab_prefix=args.lab_prefix,
        metab_prefix=args.metab_prefix,
        expected_lab_dim=args.expected_lab_dim,
        expected_metab_dim=args.expected_metab_dim,
        train_ratio=args.train_ratio,
        split_seed=args.split_seed,
    )
    save_prepared_split(raw_split, scaled_split, lab_scaler, metab_scaler, prepared_dir)

    script_path = Path(__file__).resolve().parent / "r" / "run_stabmap_benchmark.R"
    stabmap_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [args.rscript_binary, str(script_path), str(prepared_dir), str(stabmap_dir)],
        check=True,
    )


if __name__ == "__main__":
    main()
