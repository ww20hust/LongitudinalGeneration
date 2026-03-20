import argparse
from typing import List, Dict

import torch

from peag.model import PEAGModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference script for PEAG model.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a trained PEAG model checkpoint.",
    )
    parser.add_argument(
        "--temporal_model",
        type=str,
        default=None,
        choices=["recurrent", "transformer"],
        help="Optional override for checkpoints that do not store model_config.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device, for example cpu, cuda, or cuda:0.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = args.device

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_config = checkpoint.get(
        "model_config",
        {
            "modality_dims": {"lab": 61, "metab": 251},
            "latent_dim": 16,
            "hidden_dim": 128,
            "temporal_model": args.temporal_model or "recurrent",
        },
    )
    if args.temporal_model is not None:
        model_config["temporal_model"] = args.temporal_model

    modality_dims = model_config["modality_dims"]
    model = PEAGModel(**model_config)
    model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Example inference: one patient with three visits, missing metabolomics at visit 2.
    batch_size = 1
    visits_data: List[Dict[str, torch.Tensor]] = [
        {
            "lab": torch.randn(batch_size, modality_dims["lab"], device=device),
            "metab": torch.randn(batch_size, modality_dims["metab"], device=device),
        },
        {
            "lab": torch.randn(batch_size, modality_dims["lab"], device=device),
            "metab": None,
        },
        {
            "lab": torch.randn(batch_size, modality_dims["lab"], device=device),
            "metab": torch.randn(batch_size, modality_dims["metab"], device=device),
        },
    ]
    # Inference masks use 0 (available) and 2 (naturally missing); no active mask (1).
    missing_masks = [
        {"lab": 0, "metab": 0},
        {"lab": 0, "metab": 2},
        {"lab": 0, "metab": 0},
    ]

    imputed_visits = model.impute_missing(visits_data, missing_masks)

    # Simple sanity printout
    for i, visit in enumerate(imputed_visits):
        print(f"Visit {i + 1}:")
        for mod_name, data in visit.items():
            shape = tuple(data.shape) if isinstance(data, torch.Tensor) else None
            print(f"  {mod_name}: shape={shape}")


if __name__ == "__main__":
    main()

