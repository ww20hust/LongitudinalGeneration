from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from peag import (  # noqa: E402
    PEAGModel,
    evaluate_reconstruction,
    parse_column_argument,
    prepare_two_visit_clinical_benchmark,
    save_json,
)
from peag.data.dataset import LongitudinalDataset, collate_visits  # noqa: E402
from peag.training.trainer import Trainer  # noqa: E402


def add_common_experiment_args(parser) -> None:
    parser.add_argument("--csv", required=True, help="Two-visit clinical CSV with eid/visit and feature columns.")
    parser.add_argument("--output-dir", required=True, help="Directory for experiment outputs.")
    parser.add_argument("--id-column", type=str, default="eid")
    parser.add_argument("--visit-column", type=str, default="visit")
    parser.add_argument("--lab-columns", type=str, default=None, help="Comma-separated lab columns. Optional.")
    parser.add_argument("--metab-columns", type=str, default=None, help="Comma-separated metabolomics columns. Optional.")
    parser.add_argument("--lab-prefix", type=str, default=None, help="Infer lab columns by prefix.")
    parser.add_argument("--metab-prefix", type=str, default=None, help="Infer metabolomics columns by prefix.")
    parser.add_argument("--expected-lab-dim", type=int, default=61)
    parser.add_argument("--expected-metab-dim", type=int, default=251)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
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


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_bundle_from_args(args):
    return prepare_two_visit_clinical_benchmark(
        csv_path=args.csv,
        id_column=args.id_column,
        visit_column=args.visit_column,
        lab_columns=parse_column_argument(args.lab_columns),
        metab_columns=parse_column_argument(args.metab_columns),
        lab_prefix=args.lab_prefix,
        metab_prefix=args.metab_prefix,
        expected_lab_dim=args.expected_lab_dim,
        expected_metab_dim=args.expected_metab_dim,
        train_ratio=args.train_ratio,
        seed=args.split_seed,
    )


def build_model_from_args(args, bundle, **overrides: Any) -> PEAGModel:
    lab_dim = len(bundle.lab_columns)
    metab_dim = len(bundle.metab_columns)
    model_kwargs = {
        "modality_dims": {"lab": lab_dim, "metab": metab_dim},
        "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim,
        "lambda_kl": args.lambda_kl,
        "lambda_align": args.lambda_align,
        "lambda_adv": args.lambda_adv,
        "temporal_model": args.temporal_model,
        "temporal_num_heads": args.temporal_num_heads,
        "temporal_num_layers": args.temporal_num_layers,
        "temporal_dropout": args.temporal_dropout,
        "temporal_max_seq_len": args.temporal_max_seq_len,
        "alignment_strategy": "jeffrey",
        "use_adversarial_loss": True,
        "adversarial_grl_lambda": 1.0,
    }
    model_kwargs.update(overrides)
    return PEAGModel(**model_kwargs).to(args.device)


def train_model(bundle, args, experiment_dir: Path, **model_overrides: Any):
    set_random_seed(args.seed)
    dataset = LongitudinalDataset(
        patient_ids=bundle.train_patient_ids,
        visits_data=bundle.train_visits,
        missing_masks=bundle.train_missing_masks,
        train_mask_rate=args.train_mask_rate,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_visits,
        pin_memory=str(args.device).startswith("cuda"),
    )

    model = build_model_from_args(args, bundle, **model_overrides)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=args.device,
        kl_anneal_epochs=args.kl_anneal_epochs,
    )
    checkpoints_dir = experiment_dir / "checkpoints"
    history = trainer.train(
        dataloader=dataloader,
        n_epochs=args.epochs,
        save_dir=str(checkpoints_dir),
        validate_every=args.save_every,
    )
    save_json({"history": history, "model_config": model.get_config()}, experiment_dir / "train_history.json")
    final_checkpoint = experiment_dir / "model_final.pt"
    trainer.save_checkpoint(str(final_checkpoint))
    return model, history


def _to_tensor_visit(visit: dict[str, np.ndarray | None], device: str) -> dict[str, torch.Tensor | None]:
    visit_tensors: dict[str, torch.Tensor | None] = {}
    for modality_name, values in visit.items():
        if values is None:
            visit_tensors[modality_name] = None
        else:
            visit_tensors[modality_name] = torch.as_tensor(values, dtype=torch.float32, device=device).unsqueeze(0)
    return visit_tensors


def evaluate_followup_imputation(
    model: PEAGModel,
    bundle,
    *,
    device: str,
    use_history_in_fusion: bool = True,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    model.eval()
    predicted_by_id: dict[str, np.ndarray] = {}
    eval_ids = list(bundle.raw_static_split.test_index)
    eval_id_set = set(eval_ids)

    with torch.no_grad():
        for patient_id, patient_visits, patient_masks in zip(
            bundle.test_patient_ids,
            bundle.test_visits,
            bundle.test_missing_masks,
        ):
            sample_id = f"{patient_id}_visit1"
            if sample_id not in eval_id_set:
                continue

            visits_input = []
            masks_input = []
            for visit_index, (visit, visit_mask) in enumerate(zip(patient_visits, patient_masks)):
                visit_tensor = _to_tensor_visit(visit, device)
                current_mask = dict(visit_mask)
                if visit_index == 1:
                    visit_tensor["metab"] = None
                    current_mask["metab"] = 2
                visits_input.append(visit_tensor)
                masks_input.append(current_mask)

            output = model(
                visits_data=visits_input,
                missing_masks=masks_input,
                kl_annealing_weight=1.0,
                use_history_in_fusion=use_history_in_fusion,
            )
            predicted_by_id[sample_id] = output["reconstructions"][-1]["metab"][0].detach().cpu().numpy()

    missing_predictions = [sample_id for sample_id in eval_ids if sample_id not in predicted_by_id]
    if missing_predictions:
        raise ValueError(f"Missing predictions for evaluation samples: {missing_predictions[:10]}")

    sample_ids = eval_ids
    predictions_scaled = np.asarray([predicted_by_id[sample_id] for sample_id in sample_ids], dtype=np.float32)
    predictions_raw = bundle.metab_scaler.inverse_transform(predictions_scaled)
    truth_raw = bundle.raw_static_split.test_metab
    metrics = evaluate_reconstruction(truth_raw, predictions_raw)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            predictions_scaled,
            index=sample_ids,
            columns=bundle.metab_columns,
        ).to_csv(output_dir / "pred_metab_scaled.csv")
        pd.DataFrame(
            predictions_raw,
            index=sample_ids,
            columns=bundle.metab_columns,
        ).to_csv(output_dir / "pred_metab.csv")
        save_json(metrics, output_dir / "metrics.json")

    return {
        "metrics": metrics,
        "predictions_scaled": predictions_scaled,
        "predictions_raw": predictions_raw,
        "sample_ids": sample_ids,
    }


def save_experiment_metadata(args, bundle, output_path: Path) -> None:
    payload = {
        "args": vars(args),
        "n_train_patients": len(bundle.train_patient_ids),
        "n_test_patients": len(bundle.test_patient_ids),
        "lab_columns": bundle.lab_columns,
        "metab_columns": bundle.metab_columns,
    }
    save_json(payload, output_path)
