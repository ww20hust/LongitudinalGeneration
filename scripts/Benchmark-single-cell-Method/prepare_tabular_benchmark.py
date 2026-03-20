from __future__ import annotations

import argparse
from pathlib import Path

from clinical_static_baseline_benchmark.data import (
    fit_standardizers,
    load_benchmark_split,
    load_longitudinal_csv_benchmark,
    save_prepared_split,
    transform_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a standardized lab/metabolomics benchmark split.")
    parser.add_argument("--input", default=None, help="Directory with train/test CSVs or an .npz bundle.")
    parser.add_argument("--csv", default=None, help="Longitudinal two-visit clinical CSV.")
    parser.add_argument("--output-dir", required=True, help="Output directory for raw and scaled CSV files.")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.input is None) == (args.csv is None):
        raise ValueError("Specify exactly one of --input or --csv.")

    if args.csv is not None:
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
    else:
        raw_split = load_benchmark_split(args.input)
        lab_scaler, metab_scaler = fit_standardizers(raw_split)
        scaled_split = transform_split(raw_split, lab_scaler, metab_scaler)

    save_prepared_split(raw_split, scaled_split, lab_scaler, metab_scaler, Path(args.output_dir))
    print(f"Prepared benchmark written to {args.output_dir}")


if __name__ == "__main__":
    main()
