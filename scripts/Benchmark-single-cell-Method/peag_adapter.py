from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from peag import PEAGModel, evaluate_reconstruction, parse_column_argument, prepare_two_visit_clinical_benchmark, save_json
from peag.data.dataset import LongitudinalDataset, collate_visits
from peag.training.trainer import Trainer


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EarlyStopper:
    def __init__(self, patience: int) -> None:
        self.patience = int(patience)
        self.best_value: float | None = None
        self.best_state: dict[str, torch.Tensor] | None = None
        self.num_bad_epochs = 0

    def step(self, value: float, model: torch.nn.Module) -> bool:
        improved = self.best_value is None or value < self.best_value
        if improved:
            self.best_value = float(value)
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            self.num_bad_epochs = 0
            return False
        self.num_bad_epochs += 1
        return self.num_bad_epochs >= self.patience

    def restore(self, model: torch.nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def _split_train_validation(
    patient_ids: Sequence[str],
    visits: Sequence[Sequence[dict[str, np.ndarray | None]]],
    masks: Sequence[Sequence[dict[str, int]]],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[list[str], list, list, list[str], list, list]:
    if val_ratio <= 0.0 or len(patient_ids) < 2:
        return list(patient_ids), list(visits), list(masks), [], [], []
    rng = np.random.RandomState(seed)
    indices = np.arange(len(patient_ids))
    rng.shuffle(indices)
    val_count = max(1, int(round(len(indices) * val_ratio)))
    val_indices = set(indices[:val_count].tolist())
    train_ids: list[str] = []
    train_visits: list = []
    train_masks: list = []
    val_ids: list[str] = []
    val_visits: list = []
    val_masks: list = []
    for idx, patient_id in enumerate(patient_ids):
        if idx in val_indices:
            val_ids.append(patient_id)
            val_visits.append(visits[idx])
            val_masks.append(masks[idx])
        else:
            train_ids.append(patient_id)
            train_visits.append(visits[idx])
            train_masks.append(masks[idx])
    return train_ids, train_visits, train_masks, val_ids, val_visits, val_masks


def _to_tensor_visit(visit: dict[str, np.ndarray | None], device: str) -> dict[str, torch.Tensor | None]:
    tensors: dict[str, torch.Tensor | None] = {}
    for modality_name, values in visit.items():
        if values is None:
            tensors[modality_name] = None
        else:
            tensors[modality_name] = torch.as_tensor(values, dtype=torch.float32, device=device).unsqueeze(0)
    return tensors


def _evaluate_followup(
    model: PEAGModel,
    *,
    visits: Sequence[Sequence[dict[str, np.ndarray | None]]],
    masks: Sequence[Sequence[dict[str, int]]],
    metab_scaler,
    device: str,
) -> dict[str, float]:
    model.eval()
    predictions: list[np.ndarray] = []
    truths_scaled: list[np.ndarray] = []
    with torch.no_grad():
        for patient_visits, patient_masks in zip(visits, masks):
            if len(patient_visits) < 2:
                continue
            followup_visit = patient_visits[1]
            followup_mask = patient_masks[1]
            if followup_mask.get("metab", 2) != 0 or followup_visit.get("metab") is None:
                continue
            if followup_mask.get("lab", 2) != 0 or followup_visit.get("lab") is None:
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
                use_history_in_fusion=True,
            )
            predictions.append(output["reconstructions"][-1]["metab"][0].detach().cpu().numpy())
            truths_scaled.append(followup_visit["metab"])

    if not predictions:
        raise ValueError("No validation samples available for follow-up evaluation.")

    pred_scaled = np.asarray(predictions, dtype=np.float32)
    truth_scaled = np.asarray(truths_scaled, dtype=np.float32)
    pred_raw = metab_scaler.inverse_transform(pred_scaled)
    truth_raw = metab_scaler.inverse_transform(truth_scaled)
    return evaluate_reconstruction(truth_raw, pred_raw)


def build_bundle_from_csv(
    csv_path: str | Path,
    *,
    id_column: str = "eid",
    visit_column: str = "visit",
    lab_columns: str | None = None,
    metab_columns: str | None = None,
    lab_prefix: str | None = None,
    metab_prefix: str | None = None,
    expected_lab_dim: int = 61,
    expected_metab_dim: int = 251,
    train_ratio: float = 0.7,
    split_seed: int = 0,
):
    return prepare_two_visit_clinical_benchmark(
        csv_path=csv_path,
        id_column=id_column,
        visit_column=visit_column,
        lab_columns=parse_column_argument(lab_columns),
        metab_columns=parse_column_argument(metab_columns),
        lab_prefix=lab_prefix,
        metab_prefix=metab_prefix,
        expected_lab_dim=expected_lab_dim,
        expected_metab_dim=expected_metab_dim,
        train_ratio=train_ratio,
        seed=split_seed,
    )


def train_peag(
    bundle,
    output_dir: str | Path,
    *,
    device: str = "cpu",
    seed: int = 0,
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    val_ratio: float = 0.2,
    patience: int = 10,
    latent_dim: int = 16,
    hidden_dim: int = 128,
    lambda_kl: float = 1.0,
    lambda_align: float = 1.0,
    lambda_adv: float = 0.1,
    train_mask_rate: float = 0.6,
    temporal_model: str = "recurrent",
    temporal_num_heads: int = 4,
    temporal_num_layers: int = 1,
    temporal_dropout: float = 0.1,
    temporal_max_seq_len: int = 128,
    kl_anneal_epochs: int = 50,
    save_every: int = 50,
) -> tuple[PEAGModel, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_path / "checkpoints"

    set_random_seed(seed)

    (
        train_ids,
        train_visits,
        train_masks,
        val_ids,
        val_visits,
        val_masks,
    ) = _split_train_validation(
        bundle.train_patient_ids,
        bundle.train_visits,
        bundle.train_missing_masks,
        val_ratio=val_ratio,
        seed=seed,
    )

    dataset = LongitudinalDataset(
        patient_ids=train_ids,
        visits_data=train_visits,
        missing_masks=train_masks,
        train_mask_rate=train_mask_rate,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_visits,
        pin_memory=str(device).startswith("cuda"),
    )

    model = PEAGModel(
        modality_dims={"lab": len(bundle.lab_columns), "metab": len(bundle.metab_columns)},
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        lambda_kl=lambda_kl,
        lambda_align=lambda_align,
        lambda_adv=lambda_adv,
        temporal_model=temporal_model,
        temporal_num_heads=temporal_num_heads,
        temporal_num_layers=temporal_num_layers,
        temporal_dropout=temporal_dropout,
        temporal_max_seq_len=temporal_max_seq_len,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        kl_anneal_epochs=kl_anneal_epochs,
    )
    history: list[dict[str, float]] = []
    best_epoch = None
    stopper = EarlyStopper(patience=patience) if val_ids else None

    for epoch in range(epochs):
        epoch_losses = trainer.train_epoch(dataloader)
        record: dict[str, float] = {key: float(value) for key, value in epoch_losses.items()}

        if val_ids:
            prior_best = stopper.best_value if stopper is not None else None
            val_metrics = _evaluate_followup(
                model,
                visits=val_visits,
                masks=val_masks,
                metab_scaler=bundle.metab_scaler,
                device=device,
            )
            record.update(
                {
                    "val_mse": float(val_metrics["mse"]),
                    "val_mae": float(val_metrics["mae"]),
                    "val_pearson_mean": float(val_metrics["pearson_mean"]),
                }
            )
            should_stop = stopper.step(float(val_metrics["mse"]), model) if stopper is not None else False
            improved = prior_best is None or float(val_metrics["mse"]) < prior_best
            if stopper is not None and improved:
                best_epoch = epoch + 1
                best_checkpoint = output_path / "model_best.pt"
                trainer.save_checkpoint(str(best_checkpoint))

        history.append(record)
        if save_every > 0 and (epoch + 1) % save_every == 0:
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(str(checkpoints_dir / f"checkpoint_epoch_{epoch + 1}.pt"))

        if val_ids and should_stop:
            break

    if stopper is not None:
        stopper.restore(model)
    save_json(
        {
            "history": history,
            "model_config": model.get_config(),
            "val_ratio": val_ratio,
            "patience": patience,
            "best_epoch": best_epoch,
        },
        output_path / "train_history.json",
    )

    final_checkpoint = output_path / "model_final.pt"
    trainer.save_checkpoint(str(final_checkpoint))
    return model, final_checkpoint


def predict_followup_metab(
    model: PEAGModel,
    bundle,
    *,
    device: str = "cpu",
    use_history_in_fusion: bool = True,
) -> dict[str, Any]:
    model.eval()
    eval_ids = list(bundle.raw_static_split.test_index)
    eval_id_set = set(eval_ids)
    predicted_by_id: dict[str, np.ndarray] = {}

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

    missing = [sample_id for sample_id in eval_ids if sample_id not in predicted_by_id]
    if missing:
        raise ValueError(f"Missing predictions for evaluation samples: {missing[:10]}")

    sample_ids = eval_ids
    pred_scaled = np.asarray([predicted_by_id[sample_id] for sample_id in sample_ids], dtype=np.float32)
    pred_raw = bundle.metab_scaler.inverse_transform(pred_scaled)
    truth_raw = bundle.raw_static_split.test_metab
    metrics = evaluate_reconstruction(truth_raw, pred_raw)
    return {
        "sample_ids": sample_ids,
        "pred_scaled": pred_scaled,
        "pred_raw": pred_raw,
        "truth_raw": truth_raw,
        "metrics": metrics,
    }
