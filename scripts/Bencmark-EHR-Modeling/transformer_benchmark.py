from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common import (
    EarlyStopper,
    LabDiscretizer,
    TransformerProteomicsDataset,
    as_serializable_config,
    build_token_vocab,
    collate_transformer_batch,
    compute_regression_metrics,
    infer_proteomics_dim,
    load_samples,
    save_json,
    set_seed,
)


class TransformerProteomicsRegressor(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        proteomics_dim: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        ff_dim: int,
        dropout: float,
        max_seq_len: int,
        max_age_years: int,
        num_type_ids: int = 3,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.type_embedding = nn.Embedding(num_type_ids, d_model)
        self.age_embedding = nn.Embedding(max_age_years + 2, d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, proteomics_dim),
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        type_ids: torch.Tensor,
        age_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len = token_ids.shape
        if seq_len > self.position_embedding.num_embeddings:
            raise ValueError(
                f"Sequence length {seq_len} exceeds configured max_seq_len {self.position_embedding.num_embeddings}."
            )

        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        hidden = self.token_embedding(token_ids)
        hidden = hidden + self.type_embedding(type_ids)
        hidden = hidden + self.age_embedding(age_ids)
        hidden = hidden + self.position_embedding(positions)
        hidden = self.layer_norm(hidden)
        hidden = self.dropout(hidden)
        hidden = hidden.transpose(0, 1)

        encoded = self.encoder(
            hidden,
            src_key_padding_mask=~attention_mask,
        ).transpose(0, 1)

        cls_representation = encoded[:, 0, :]
        return self.regressor(cls_representation)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TransformerEncoder benchmark for proteomics generation from age-aware ICD history and routine lab tests."
    )
    parser.add_argument("--train_path", type=str, required=True, help="Training samples (.json or .jsonl).")
    parser.add_argument("--valid_path", type=str, required=True, help="Validation samples (.json or .jsonl).")
    parser.add_argument("--test_path", type=str, required=True, help="Test samples (.json or .jsonl).")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to store checkpoints and metrics.")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--d_model", type=int, default=256, help="Transformer hidden size.")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of Transformer encoder layers.")
    parser.add_argument("--ff_dim", type=int, default=512, help="Feed-forward hidden dimension.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument("--max_history_events", type=int, default=512, help="Max number of ICD events retained before the current visit.")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Max token sequence length.")
    parser.add_argument("--max_age_years", type=int, default=120, help="Max age bucket for age embeddings.")
    parser.add_argument("--num_lab_bins", type=int, default=10, help="Number of discrete bins per lab feature.")
    parser.add_argument("--patience", type=int, default=10, help="Validation patience for early stopping.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Training device.",
    )
    return parser.parse_args()


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
) -> Dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    prediction_batches: List[np.ndarray] = []
    target_batches: List[np.ndarray] = []
    loss_fn = nn.MSELoss(reduction="sum")

    with torch.no_grad():
        for batch in dataloader:
            token_ids = batch["token_ids"].to(device)
            type_ids = batch["type_ids"].to(device)
            age_ids = batch["age_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)

            predictions = model(
                token_ids=token_ids,
                type_ids=type_ids,
                age_ids=age_ids,
                attention_mask=attention_mask,
            )
            batch_loss = loss_fn(predictions, targets)
            total_loss += float(batch_loss.item())
            total_examples += int(targets.shape[0])

            prediction_batches.append(predictions.cpu().numpy())
            target_batches.append(targets.cpu().numpy())

    predictions_np = np.concatenate(prediction_batches, axis=0)
    targets_np = np.concatenate(target_batches, axis=0)
    metrics = compute_regression_metrics(predictions_np, targets_np)
    metrics["loss_per_sample"] = total_loss / max(total_examples, 1)
    return {
        "metrics": metrics,
        "predictions": predictions_np,
        "targets": targets_np,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_samples = load_samples(args.train_path)
    valid_samples = load_samples(args.valid_path)
    test_samples = load_samples(args.test_path)

    proteomics_dim = infer_proteomics_dim(train_samples)
    if infer_proteomics_dim(valid_samples) != proteomics_dim:
        raise ValueError("Validation proteomics dimensionality does not match the training split.")
    if infer_proteomics_dim(test_samples) != proteomics_dim:
        raise ValueError("Test proteomics dimensionality does not match the training split.")

    discretizer = LabDiscretizer(num_bins=args.num_lab_bins).fit(train_samples)
    token_to_id = build_token_vocab(
        samples=train_samples,
        discretizer=discretizer,
    )

    train_dataset = TransformerProteomicsDataset(
        samples=train_samples,
        token_to_id=token_to_id,
        discretizer=discretizer,
        max_history_events=args.max_history_events,
        max_age_years=args.max_age_years,
    )
    valid_dataset = TransformerProteomicsDataset(
        samples=valid_samples,
        token_to_id=token_to_id,
        discretizer=discretizer,
        max_history_events=args.max_history_events,
        max_age_years=args.max_age_years,
    )
    test_dataset = TransformerProteomicsDataset(
        samples=test_samples,
        token_to_id=token_to_id,
        discretizer=discretizer,
        max_history_events=args.max_history_events,
        max_age_years=args.max_age_years,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_transformer_batch,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_transformer_batch,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_transformer_batch,
    )

    model = TransformerProteomicsRegressor(
        vocab_size=len(token_to_id),
        proteomics_dim=proteomics_dim,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
        max_age_years=args.max_age_years,
    ).to(args.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.MSELoss()
    early_stopper = EarlyStopper(patience=args.patience)
    best_checkpoint_path = save_dir / "best_model.pt"

    training_log: List[Dict[str, float]] = []
    best_valid_mse = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0
        total_train_examples = 0

        for batch in train_loader:
            token_ids = batch["token_ids"].to(args.device)
            type_ids = batch["type_ids"].to(args.device)
            age_ids = batch["age_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            targets = batch["targets"].to(args.device)

            predictions = model(
                token_ids=token_ids,
                type_ids=type_ids,
                age_ids=age_ids,
                attention_mask=attention_mask,
            )
            loss = loss_fn(predictions, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = int(targets.shape[0])
            total_train_loss += float(loss.item()) * batch_size
            total_train_examples += batch_size

        valid_result = evaluate(model=model, dataloader=valid_loader, device=args.device)
        train_loss = total_train_loss / max(total_train_examples, 1)
        valid_mse = float(valid_result["metrics"]["mse"])
        training_log.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "valid_mse": valid_mse,
                "valid_mean_feature_pearson": float(valid_result["metrics"]["mean_feature_pearson"]),
            }
        )
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | "
            f"valid_mse={valid_mse:.6f} | "
            f"valid_mean_feature_pearson={valid_result['metrics']['mean_feature_pearson']:.6f}"
        )

        if best_valid_mse is None or valid_mse < best_valid_mse:
            best_valid_mse = valid_mse
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "token_to_id": token_to_id,
                    "discretizer": discretizer.to_dict(),
                    "config": as_serializable_config(vars(args)),
                },
                best_checkpoint_path,
            )

        if early_stopper.step(valid_mse):
            print(f"Early stopping triggered after epoch {epoch}.")
            break

    checkpoint = torch.load(best_checkpoint_path, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])

    valid_result = evaluate(model=model, dataloader=valid_loader, device=args.device)
    test_result = evaluate(model=model, dataloader=test_loader, device=args.device)

    metrics_payload = {
        "benchmark": "transformer_encoder_age_aware",
        "config": as_serializable_config(vars(args)),
        "vocab_size": len(token_to_id),
        "proteomics_dim": proteomics_dim,
        "training_log": training_log,
        "validation_metrics": valid_result["metrics"],
        "test_metrics": test_result["metrics"],
    }
    save_json(save_dir / "metrics.json", metrics_payload)
    np.savez_compressed(
        save_dir / "test_predictions.npz",
        predictions=test_result["predictions"],
        targets=test_result["targets"],
    )

    print("Final test metrics:")
    print(
        f"  mean_feature_pearson={test_result['metrics']['mean_feature_pearson']:.6f}, "
        f"median_feature_pearson={test_result['metrics']['median_feature_pearson']:.6f}, "
        f"mean_feature_mae={test_result['metrics']['mean_feature_mae']:.6f}"
    )


if __name__ == "__main__":
    main()
