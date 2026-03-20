from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    add_common_experiment_args,
    build_bundle_from_args,
    evaluate_followup_imputation,
    save_experiment_metadata,
    train_model,
)
from peag import format_mask_rate_tag, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan active-masking probabilities for PEAG metabolomics imputation.")
    add_common_experiment_args(parser)
    parser.add_argument(
        "--mask-rates",
        type=str,
        default="0.0,0.2,0.4,0.6,0.8,1.0",
        help="Comma-separated masking probabilities to scan.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = build_bundle_from_args(args)
    save_experiment_metadata(args, bundle, output_dir / "experiment_setup.json")

    mask_rates = [float(value.strip()) for value in args.mask_rates.split(",") if value.strip()]
    summary: dict[str, dict] = {}

    for mask_rate in mask_rates:
        args.train_mask_rate = mask_rate
        experiment_name = f"mask_rate_{format_mask_rate_tag(mask_rate)}"
        experiment_dir = output_dir / experiment_name
        model, _ = train_model(bundle, args, experiment_dir)
        evaluation = evaluate_followup_imputation(
            model,
            bundle,
            device=args.device,
            use_history_in_fusion=True,
            output_dir=experiment_dir / "evaluation",
        )
        summary[experiment_name] = {
            "mask_rate": mask_rate,
            "metrics": evaluation["metrics"],
        }

    save_json(summary, output_dir / "summary.json")


if __name__ == "__main__":
    main()
