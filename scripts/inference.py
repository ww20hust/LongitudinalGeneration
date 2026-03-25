import argparse
from typing import Dict, List, Optional

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
    parser.add_argument(
        "--fallback_modality_a_name",
        type=str,
        default="modality_a",
        help="Fallback first-modality name when the checkpoint does not store model_config.",
    )
    parser.add_argument(
        "--fallback_modality_a_dim",
        type=int,
        default=61,
        help="Fallback first-modality dimension when the checkpoint does not store model_config.",
    )
    parser.add_argument(
        "--fallback_modality_b_name",
        type=str,
        default="modality_b",
        help="Fallback second-modality name when the checkpoint does not store model_config.",
    )
    parser.add_argument(
        "--fallback_modality_b_dim",
        type=int,
        default=251,
        help="Fallback second-modality dimension when the checkpoint does not store model_config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = args.device

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_config = checkpoint.get(
        "model_config",
        {
            "modality_dims": {
                args.fallback_modality_a_name: args.fallback_modality_a_dim,
                args.fallback_modality_b_name: args.fallback_modality_b_dim,
            },
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

    # Example inference: one patient with three visits, where one modality is
    # naturally missing at visit 2.
    batch_size = 1
    modality_names = list(modality_dims.keys())
    missing_modality = modality_names[1] if len(modality_names) > 1 else modality_names[0]

    visits_data: List[Dict[str, Optional[torch.Tensor]]] = []
    for visit_idx in range(3):
        visit_data: Dict[str, Optional[torch.Tensor]] = {}
        for mod_name, mod_dim in modality_dims.items():
            if visit_idx == 1 and mod_name == missing_modality:
                visit_data[mod_name] = None
            else:
                visit_data[mod_name] = torch.randn(batch_size, mod_dim, device=device)
        visits_data.append(visit_data)

    # Inference masks use 0 (available) and 2 (naturally missing); no active mask (1).
    missing_masks = []
    for visit_idx in range(3):
        visit_mask = {}
        for mod_name in modality_dims.keys():
            visit_mask[mod_name] = 2 if visit_idx == 1 and mod_name == missing_modality else 0
        missing_masks.append(visit_mask)

    imputed_visits = model.impute_missing(visits_data, missing_masks)

    # Simple sanity printout
    for i, visit in enumerate(imputed_visits):
        print(f"Visit {i + 1}:")
        for mod_name, data in visit.items():
            shape = tuple(data.shape) if isinstance(data, torch.Tensor) else None
            print(f"  {mod_name}: shape={shape}")


if __name__ == "__main__":
    main()

