from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from peag.model import PEAGModel

from common import (
    EarlyStopper,
    RoutineLabVectorizer,
    as_serializable_config,
    compute_regression_metrics,
    infer_proteomics_dim,
    load_samples,
    render_history_as_text,
    save_json,
    set_seed,
)
from llama_embedding_backends import add_llama_embedding_args, build_text_embedder, default_torch_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PEAG benchmark for proteomics generation using Llama 3.1 history embeddings and structured routine lab tests."
    )
    parser.add_argument("--train_path", type=str, required=True, help="Training samples (.json or .jsonl).")
    parser.add_argument("--valid_path", type=str, required=True, help="Validation samples (.json or .jsonl).")
    parser.add_argument("--test_path", type=str, required=True, help="Test samples (.json or .jsonl).")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to store checkpoints and metrics.")
    parser.add_argument(
        "--llama_model_name_or_path",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model name or local path for the frozen Llama history encoder.",
    )
    parser.add_argument("--embedding_batch_size", type=int, default=4, help="Batch size for frozen Llama encoding.")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum tokenized history length.")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--latent_dim", type=int, default=16, help="PEAG latent dimension.")
    parser.add_argument("--hidden_dim", type=int, default=128, help="PEAG hidden dimension.")
    parser.add_argument("--lambda_kl", type=float, default=1.0, help="Weight of the KL term.")
    parser.add_argument("--lambda_align", type=float, default=1.0, help="Weight of the alignment term.")
    parser.add_argument("--lambda_adv", type=float, default=0.1, help="Weight of the adversarial term.")
    parser.add_argument("--train_mask_rate", type=float, default=0.6, help="Probability of actively masking one current modality.")
    parser.add_argument("--kl_anneal_epochs", type=int, default=50, help="Epochs used for linear KL annealing.")
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
        help="Optional directory for cached train/valid/test history embeddings.",
    )
    add_llama_embedding_args(parser)
    return parser.parse_args()


def build_history_texts(samples) -> List[str]:
    return [render_history_as_text(sample) for sample in samples]


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
    cache_path = cache_dir / f"{split_name}_history_embeddings.npy"
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
    return Path(cache_dir) / f"{split_name}_history_embeddings.npy"


def build_visit(
    history_embedding: np.ndarray,
    lab_vector: np.ndarray,
    proteomics_target: np.ndarray,
    device: str,
) -> Dict[str, torch.Tensor]:
    return {
        "proteomics": torch.tensor(proteomics_target, dtype=torch.float32, device=device).unsqueeze(0),
        "history": torch.tensor(history_embedding, dtype=torch.float32, device=device).unsqueeze(0),
        "lab": torch.tensor(lab_vector, dtype=torch.float32, device=device).unsqueeze(0),
    }


def sample_active_mask(train_mask_rate: float) -> Dict[str, int]:
    mask = {"proteomics": 0, "history": 0, "lab": 0}
    available_modalities = ["proteomics", "history", "lab"]
    if len(available_modalities) > 1 and np.random.rand() < float(train_mask_rate):
        masked_modality = str(np.random.choice(available_modalities))
        mask[masked_modality] = 1
    return mask


def apply_mask_to_visit(
    visit: Dict[str, torch.Tensor],
    mask: Dict[str, int],
) -> Dict[str, torch.Tensor | None]:
    masked_visit: Dict[str, torch.Tensor | None] = {}
    for modality_name, tensor in visit.items():
        mask_value = mask.get(modality_name, 2)
        if mask_value in (1, 2):
            masked_visit[modality_name] = None
        else:
            masked_visit[modality_name] = tensor
    return masked_visit


def compute_kl_weight(epoch_index: int, kl_anneal_epochs: int) -> float:
    if kl_anneal_epochs <= 0:
        return 1.0
    return min(1.0, float(epoch_index) / float(kl_anneal_epochs))


def train_epoch(
    model: PEAGModel,
    optimizer: torch.optim.Optimizer,
    samples,
    history_embeddings: np.ndarray,
    lab_vectors: np.ndarray,
    device: str,
    train_mask_rate: float,
    kl_weight: float,
) -> Dict[str, float]:
    model.train()
    indices = np.random.permutation(len(samples))
    totals: Dict[str, float] = {
        "total_loss": 0.0,
        "recon_loss": 0.0,
        "kl_loss": 0.0,
        "alignment_loss": 0.0,
        "adversarial_loss": 0.0,
    }

    for sample_index in indices:
        visit_full = build_visit(
            history_embedding=history_embeddings[sample_index],
            lab_vector=lab_vectors[sample_index],
            proteomics_target=samples[sample_index].proteomics,
            device=device,
        )
        visit_mask = sample_active_mask(train_mask_rate)
        visit_masked = apply_mask_to_visit(visit_full, visit_mask)

        output = model(
            visits_data=[visit_masked],
            missing_masks=[visit_mask],
            kl_annealing_weight=kl_weight,
            recon_targets=[visit_full],
        )
        loss = output["losses"]["total_loss"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for key in totals.keys():
            totals[key] += float(output["losses"][key].item())

    num_samples = max(len(samples), 1)
    for key in totals.keys():
        totals[key] /= num_samples
    return totals


def evaluate_generation(
    model: PEAGModel,
    samples,
    history_embeddings: np.ndarray,
    lab_vectors: np.ndarray,
    device: str,
) -> Dict[str, object]:
    model.eval()
    predictions: List[np.ndarray] = []
    targets: List[np.ndarray] = []

    with torch.no_grad():
        for sample_index, sample in enumerate(samples):
            visit_full = build_visit(
                history_embedding=history_embeddings[sample_index],
                lab_vector=lab_vectors[sample_index],
                proteomics_target=sample.proteomics,
                device=device,
            )
            visit_mask = {"proteomics": 2, "history": 0, "lab": 0}
            visit_masked = {
                "proteomics": None,
                "history": visit_full["history"],
                "lab": visit_full["lab"],
            }

            imputed = model.impute_missing(
                visits_data=[visit_masked],
                missing_masks=[visit_mask],
            )
            predictions.append(imputed[0]["proteomics"].squeeze(0).cpu().numpy())
            targets.append(sample.proteomics.astype(np.float32))

    predictions_np = np.stack(predictions, axis=0).astype(np.float32)
    targets_np = np.stack(targets, axis=0).astype(np.float32)
    metrics = compute_regression_metrics(predictions_np, targets_np)
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

    history_train_texts = build_history_texts(train_samples)
    history_valid_texts = build_history_texts(valid_samples)
    history_test_texts = build_history_texts(test_samples)

    split_names = ("train", "valid", "test")
    cache_paths = [expected_embedding_cache_path(args.cache_dir, split_name) for split_name in split_names]
    need_embedder = args.cache_dir is None or any(path is None or not path.exists() for path in cache_paths)
    embedder = build_text_embedder(args) if need_embedder else None

    train_history_embeddings = load_or_compute_embeddings("train", history_train_texts, args, embedder=embedder)
    valid_history_embeddings = load_or_compute_embeddings("valid", history_valid_texts, args, embedder=embedder)
    test_history_embeddings = load_or_compute_embeddings("test", history_test_texts, args, embedder=embedder)

    history_dim = int(train_history_embeddings.shape[1])
    lab_vectorizer = RoutineLabVectorizer().fit(train_samples)
    train_lab_vectors = lab_vectorizer.transform_samples(train_samples)
    valid_lab_vectors = lab_vectorizer.transform_samples(valid_samples)
    test_lab_vectors = lab_vectorizer.transform_samples(test_samples)

    model = PEAGModel(
        modality_dims={
            "proteomics": proteomics_dim,
            "history": history_dim,
            "lab": lab_vectorizer.dim,
        },
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        lambda_kl=args.lambda_kl,
        lambda_align=args.lambda_align,
        lambda_adv=args.lambda_adv,
        temporal_model="recurrent",
    ).to(args.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    early_stopper = EarlyStopper(patience=args.patience)
    best_checkpoint_path = save_dir / "best_model.pt"

    training_log: List[Dict[str, float]] = []
    best_valid_mse = None

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model=model,
            optimizer=optimizer,
            samples=train_samples,
            history_embeddings=train_history_embeddings,
            lab_vectors=train_lab_vectors,
            device=args.device,
            train_mask_rate=args.train_mask_rate,
            kl_weight=compute_kl_weight(epoch, args.kl_anneal_epochs),
        )
        valid_result = evaluate_generation(
            model=model,
            samples=valid_samples,
            history_embeddings=valid_history_embeddings,
            lab_vectors=valid_lab_vectors,
            device=args.device,
        )
        valid_mse = float(valid_result["metrics"]["mse"])
        training_log.append(
            {
                "epoch": float(epoch),
                "train_total_loss": train_metrics["total_loss"],
                "train_alignment_loss": train_metrics["alignment_loss"],
                "valid_mse": valid_mse,
                "valid_mean_feature_pearson": float(valid_result["metrics"]["mean_feature_pearson"]),
            }
        )
        print(
            f"Epoch {epoch:03d} | train_total_loss={train_metrics['total_loss']:.6f} | "
            f"valid_mse={valid_mse:.6f} | "
            f"valid_mean_feature_pearson={valid_result['metrics']['mean_feature_pearson']:.6f}"
        )

        if best_valid_mse is None or valid_mse < best_valid_mse:
            best_valid_mse = valid_mse
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "lab_vectorizer": lab_vectorizer.to_dict(),
                    "config": as_serializable_config(vars(args)),
                    "history_dim": history_dim,
                    "proteomics_dim": proteomics_dim,
                },
                best_checkpoint_path,
            )

        if early_stopper.step(valid_mse):
            print(f"Early stopping triggered after epoch {epoch}.")
            break

    checkpoint = torch.load(best_checkpoint_path, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])

    valid_result = evaluate_generation(
        model=model,
        samples=valid_samples,
        history_embeddings=valid_history_embeddings,
        lab_vectors=valid_lab_vectors,
        device=args.device,
    )
    test_result = evaluate_generation(
        model=model,
        samples=test_samples,
        history_embeddings=test_history_embeddings,
        lab_vectors=test_lab_vectors,
        device=args.device,
    )

    metrics_payload = {
        "benchmark": "peag_llama_history",
        "config": as_serializable_config(vars(args)),
        "history_dim": history_dim,
        "lab_dim": lab_vectorizer.dim,
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
