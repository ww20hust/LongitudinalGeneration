from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from common import (
    EarlyStopper,
    LabDiscretizer,
    as_serializable_config,
    compute_regression_metrics,
    infer_proteomics_dim,
    load_samples,
    render_sample_as_text,
    save_json,
    set_seed,
    targets_to_numpy,
)
from llama_embedding_backends import add_llama_embedding_args, build_text_embedder, default_torch_device


class EmbeddingRegressor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.model(embeddings)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Llama 3.1 text-embedding benchmark for proteomics generation from medical history and routine labs."
    )
    parser.add_argument("--train_path", type=str, required=True, help="Training samples (.json or .jsonl).")
    parser.add_argument("--valid_path", type=str, required=True, help="Validation samples (.json or .jsonl).")
    parser.add_argument("--test_path", type=str, required=True, help="Test samples (.json or .jsonl).")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to store metrics and checkpoints.")
    parser.add_argument(
        "--llama_model_name_or_path",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model name or local path for the Llama 3.1 encoder.",
    )
    parser.add_argument("--embedding_batch_size", type=int, default=4, help="Batch size for frozen Llama encoding.")
    parser.add_argument("--regression_batch_size", type=int, default=64, help="Batch size for decoder training.")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum tokenized prompt length.")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum decoder training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the decoder.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for the decoder.")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension of the decoder MLP.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout for the decoder MLP.")
    parser.add_argument("--num_lab_bins", type=int, default=10, help="Number of routine-lab bins used in text rendering.")
    parser.add_argument("--patience", type=int, default=10, help="Validation patience for early stopping.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        type=str,
        default=default_torch_device(),
        help="Execution device.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Optional directory for cached train/valid/test embeddings.",
    )
    add_llama_embedding_args(parser)
    return parser.parse_args()


def build_texts(samples, discretizer: LabDiscretizer) -> List[str]:
    return [render_sample_as_text(sample, discretizer) for sample in samples]


def load_or_compute_embeddings(
    split_name: str,
    texts: Sequence[str],
    args: argparse.Namespace,
    embedder=None,
) -> np.ndarray:
    if args.cache_dir is None:
        if embedder is None:
            raise ValueError("An initialized text embedder is required when cache_dir is not provided.")
        return embedder.encode_texts(
            texts=texts,
            batch_size=args.embedding_batch_size,
            max_length=args.max_length,
        )

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{split_name}_embeddings.npy"
    if cache_path.exists():
        return np.load(cache_path)

    if embedder is None:
        raise ValueError(f"Missing text embedder while computing uncached embeddings for split '{split_name}'.")
    embeddings = embedder.encode_texts(
        texts=texts,
        batch_size=args.embedding_batch_size,
        max_length=args.max_length,
    )
    np.save(cache_path, embeddings)
    return embeddings


def expected_embedding_cache_path(cache_dir: Optional[str], split_name: str) -> Optional[Path]:
    if cache_dir is None:
        return None
    return Path(cache_dir) / f"{split_name}_embeddings.npy"


def evaluate(
    model: nn.Module,
    embeddings: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    device: str,
) -> Dict[str, object]:
    model.eval()
    dataset = TensorDataset(
        torch.tensor(embeddings, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32),
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    prediction_batches: List[np.ndarray] = []
    target_batches: List[np.ndarray] = []
    total_loss = 0.0
    total_examples = 0
    loss_fn = nn.MSELoss(reduction="sum")

    with torch.no_grad():
        for batch_embeddings, batch_targets in dataloader:
            batch_embeddings = batch_embeddings.to(device)
            batch_targets = batch_targets.to(device)
            batch_predictions = model(batch_embeddings)
            total_loss += float(loss_fn(batch_predictions, batch_targets).item())
            total_examples += int(batch_targets.shape[0])
            prediction_batches.append(batch_predictions.cpu().numpy())
            target_batches.append(batch_targets.cpu().numpy())

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

    train_texts = build_texts(train_samples, discretizer)
    valid_texts = build_texts(valid_samples, discretizer)
    test_texts = build_texts(test_samples, discretizer)

    train_targets = targets_to_numpy(train_samples)
    valid_targets = targets_to_numpy(valid_samples)
    test_targets = targets_to_numpy(test_samples)

    split_names = ("train", "valid", "test")
    cache_paths = [expected_embedding_cache_path(args.cache_dir, split_name) for split_name in split_names]
    need_embedder = args.cache_dir is None or any(path is None or not path.exists() for path in cache_paths)
    embedder = build_text_embedder(args) if need_embedder else None

    train_embeddings = load_or_compute_embeddings("train", train_texts, args, embedder=embedder)
    valid_embeddings = load_or_compute_embeddings("valid", valid_texts, args, embedder=embedder)
    test_embeddings = load_or_compute_embeddings("test", test_texts, args, embedder=embedder)

    embedding_dim = int(train_embeddings.shape[1])
    model = EmbeddingRegressor(
        input_dim=embedding_dim,
        output_dim=proteomics_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(args.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.MSELoss()
    early_stopper = EarlyStopper(patience=args.patience)
    best_checkpoint_path = save_dir / "best_decoder.pt"

    train_dataset = TensorDataset(
        torch.tensor(train_embeddings, dtype=torch.float32),
        torch.tensor(train_targets, dtype=torch.float32),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.regression_batch_size,
        shuffle=True,
    )

    training_log: List[Dict[str, float]] = []
    best_valid_mse = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0
        total_train_examples = 0

        for batch_embeddings, batch_targets in train_loader:
            batch_embeddings = batch_embeddings.to(args.device)
            batch_targets = batch_targets.to(args.device)

            predictions = model(batch_embeddings)
            loss = loss_fn(predictions, batch_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = int(batch_targets.shape[0])
            total_train_loss += float(loss.item()) * batch_size
            total_train_examples += batch_size

        valid_result = evaluate(
            model=model,
            embeddings=valid_embeddings,
            targets=valid_targets,
            batch_size=args.regression_batch_size,
            device=args.device,
        )
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
                    "discretizer": discretizer.to_dict(),
                    "config": as_serializable_config(vars(args)),
                    "embedding_dim": embedding_dim,
                },
                best_checkpoint_path,
            )

        if early_stopper.step(valid_mse):
            print(f"Early stopping triggered after epoch {epoch}.")
            break

    checkpoint = torch.load(best_checkpoint_path, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])

    valid_result = evaluate(
        model=model,
        embeddings=valid_embeddings,
        targets=valid_targets,
        batch_size=args.regression_batch_size,
        device=args.device,
    )
    test_result = evaluate(
        model=model,
        embeddings=test_embeddings,
        targets=test_targets,
        batch_size=args.regression_batch_size,
        device=args.device,
    )

    metrics_payload = {
        "benchmark": "llama31_frozen_encoder",
        "config": as_serializable_config(vars(args)),
        "embedding_dim": embedding_dim,
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
