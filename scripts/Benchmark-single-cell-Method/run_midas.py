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
from clinical_static_baseline_benchmark.io_utils import save_prediction_csv
from clinical_static_baseline_benchmark.metrics import evaluate_reconstruction, save_metrics
from clinical_static_baseline_benchmark.midas_adapter import predict_midas, train_midas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MIDAS on lab-to-metabolomics imputation.")
    parser.add_argument("--input", default=None, help="Directory with train/test CSVs or an .npz bundle.")
    parser.add_argument("--csv", default=None, help="Longitudinal two-visit clinical CSV.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
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

    output_dir = Path(args.output_dir)
    prepared_dir = output_dir / "prepared"
    model_dir = output_dir / "midas_model"

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

    save_prepared_split(raw_split, scaled_split, lab_scaler, metab_scaler, prepared_dir)

    checkpoint_path = train_midas(
        scaled_split.train_lab,
        scaled_split.train_metab,
        model_dir,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
    )
    pred_scaled = predict_midas(
        checkpoint_path,
        scaled_split.test_lab,
        train_lab_dim=scaled_split.train_lab.shape[1],
        train_metab_dim=scaled_split.train_metab.shape[1],
        batch_size=args.batch_size,
    )
    pred_raw = metab_scaler.inverse_transform(pred_scaled)

    save_prediction_csv(pred_scaled, output_dir / "test_metab_pred_scaled.csv", prefix="metab")
    save_prediction_csv(pred_raw, output_dir / "test_metab_pred.csv", prefix="metab")
    save_metrics(
        evaluate_reconstruction(raw_split.test_metab, pred_raw),
        output_dir / "metrics.json",
    )
    print(f"MIDAS benchmark completed: {output_dir}")


if __name__ == "__main__":
    main()
