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
from peag import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PEAG metabolomics-imputation ablations from a two-visit CSV.")
    add_common_experiment_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = build_bundle_from_args(args)
    save_experiment_metadata(args, bundle, output_dir / "experiment_setup.json")

    summary: dict[str, dict] = {}

    default_dir = output_dir / "default"
    model, _ = train_model(bundle, args, default_dir)
    default_eval = evaluate_followup_imputation(
        model,
        bundle,
        device=args.device,
        use_history_in_fusion=True,
        output_dir=default_dir / "evaluation",
    )
    summary["default"] = {
        "metrics": default_eval["metrics"],
        "train_mask_rate": args.train_mask_rate,
        "alignment_strategy": "jeffrey",
        "lambda_align": args.lambda_align,
        "lambda_adv": args.lambda_adv,
    }

    no_history_dir = output_dir / "historical_state_removed_at_inference"
    no_history_eval = evaluate_followup_imputation(
        model,
        bundle,
        device=args.device,
        use_history_in_fusion=False,
        output_dir=no_history_dir / "evaluation",
    )
    summary["historical_state_removed_at_inference"] = {
        "metrics": no_history_eval["metrics"],
        "uses_default_checkpoint": True,
    }

    experiment_settings = {
        "directional_stop_gradient": {"alignment_strategy": "directional_stop_gradient"},
        "pointwise_alignment": {"alignment_strategy": "pointwise"},
        "remove_l_align": {"lambda_align": 0.0},
        "remove_l_adv": {"lambda_adv": 0.0, "use_adversarial_loss": False},
    }

    for experiment_name, overrides in experiment_settings.items():
        experiment_dir = output_dir / experiment_name
        experiment_model, _ = train_model(bundle, args, experiment_dir, **overrides)
        experiment_eval = evaluate_followup_imputation(
            experiment_model,
            bundle,
            device=args.device,
            use_history_in_fusion=True,
            output_dir=experiment_dir / "evaluation",
        )
        summary[experiment_name] = {
            "metrics": experiment_eval["metrics"],
            **overrides,
        }

    save_json(summary, output_dir / "summary.json")


if __name__ == "__main__":
    main()
