import argparse

import torch
from torch.utils.data import DataLoader

from peag.data.dataset import LongitudinalDataset, create_synthetic_data, collate_visits
from peag.model import PEAGModel
from peag.training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PEAG model with active masking.")
    parser.add_argument("--n_patients", type=int, default=100, help="Number of synthetic patients.")
    parser.add_argument("--n_visits", type=int, default=3, help="Number of visits per patient.")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument(
        "--min_completeness_ratio",
        type=float,
        default=0.8,
        help="Minimum completeness ratio per patient (0 to 1).",
    )
    parser.add_argument(
        "--train_mask_rate",
        type=float,
        default=0.2,
        help="Active mask rate for available (visit, modality) cells.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Optional directory to save checkpoints.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    modality_dims = {"lab": 61, "metab": 251}

    # Create synthetic data; natural missingness is controlled by missing_rate.
    # For training with active mask it is often useful to start from mostly
    # complete data (e.g., missing_rate close to 0).
    patient_ids, visits_data, missing_masks = create_synthetic_data(
        n_patients=args.n_patients,
        n_visits=args.n_visits,
        modality_dims=modality_dims,
        missing_rate=0.0,
        seed=42,
    )

    dataset = LongitudinalDataset(
        patient_ids=patient_ids,
        visits_data=visits_data,
        missing_masks=missing_masks,
        min_completeness_ratio=args.min_completeness_ratio,
        train_mask_rate=args.train_mask_rate,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_visits,
    )

    model = PEAGModel(
        modality_dims=modality_dims,
        latent_dim=16,
        hidden_dim=128,
        lambda_kl=1.0,
        lambda_align=1.0,
        lambda_adv=0.1,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    trainer = Trainer(model=model, optimizer=optimizer)
    trainer.train(dataloader=dataloader, n_epochs=args.epochs, save_dir=args.save_dir)


if __name__ == "__main__":
    main()

