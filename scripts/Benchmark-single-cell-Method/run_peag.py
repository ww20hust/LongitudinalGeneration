from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from clinical_static_baseline_benchmark.data import save_prepared_split
from clinical_static_baseline_benchmark.io_utils import save_prediction_csv
from clinical_static_baseline_benchmark.metrics import save_metrics
from peag_adapter import build_bundle_from_csv, predict_followup_metab, train_peag


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PEAG on longitudinal lab-to-metabolomics imputation.")
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

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lambda-kl", type=float, default=1.0)
    parser.add_argument("--lambda-align", type=float, default=1.0)
    parser.add_argument("--lambda-adv", type=float, default=0.1)
    parser.add_argument("--train-mask-rate", type=float, default=0.6)
    parser.add_argument("--temporal-model", type=str, default="recurrent", choices=["recurrent", "transformer"])
    parser.add_argument("--temporal-num-heads", type=int, default=4)
    parser.add_argument("--temporal-num-layers", type=int, default=1)
    parser.add_argument("--temporal-dropout", type=float, default=0.1)
    parser.add_argument("--temporal-max-seq-len", type=int, default=128)
    parser.add_argument("--kl-anneal-epochs", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument(
        "--disable-history-fusion",
        action="store_true",
        help="Disable history contribution during inference (diagnostic only).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    prepared_dir = output_dir / "prepared"
    model_dir = output_dir / "peag_model"

    bundle = build_bundle_from_csv(
        csv_path=args.csv,
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

    save_prepared_split(
        bundle.raw_static_split,
        bundle.scaled_static_split,
        bundle.lab_scaler,
        bundle.metab_scaler,
        prepared_dir,
    )

    model, _ = train_peag(
        bundle,
        model_dir,
        device=args.device,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        val_ratio=args.val_ratio,
        patience=args.patience,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        lambda_kl=args.lambda_kl,
        lambda_align=args.lambda_align,
        lambda_adv=args.lambda_adv,
        train_mask_rate=args.train_mask_rate,
        temporal_model=args.temporal_model,
        temporal_num_heads=args.temporal_num_heads,
        temporal_num_layers=args.temporal_num_layers,
        temporal_dropout=args.temporal_dropout,
        temporal_max_seq_len=args.temporal_max_seq_len,
        kl_anneal_epochs=args.kl_anneal_epochs,
        save_every=args.save_every,
    )

    results = predict_followup_metab(
        model,
        bundle,
        device=args.device,
        use_history_in_fusion=not args.disable_history_fusion,
    )

    save_prediction_csv(results["pred_scaled"], output_dir / "test_metab_pred_scaled.csv", prefix="metab")
    save_prediction_csv(results["pred_raw"], output_dir / "test_metab_pred.csv", prefix="metab")

    pd.DataFrame(
        results["pred_scaled"],
        index=results["sample_ids"],
        columns=bundle.metab_columns,
    ).to_csv(output_dir / "test_metab_pred_scaled_by_id.csv")
    pd.DataFrame(
        results["pred_raw"],
        index=results["sample_ids"],
        columns=bundle.metab_columns,
    ).to_csv(output_dir / "test_metab_pred_by_id.csv")

    save_metrics(results["metrics"], output_dir / "metrics.json")
    print(f"PEAG benchmark completed: {output_dir}")


if __name__ == "__main__":
    main()
