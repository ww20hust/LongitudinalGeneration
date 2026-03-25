from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

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

    dataset = LongitudinalDataset(
        patient_ids=bundle.train_patient_ids,
        visits_data=bundle.train_visits,
        missing_masks=bundle.train_missing_masks,
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
    history = trainer.train(
        dataloader=dataloader,
        n_epochs=epochs,
        save_dir=str(checkpoints_dir),
        validate_every=save_every,
    )
    save_json({"history": history, "model_config": model.get_config()}, output_path / "train_history.json")

    final_checkpoint = output_path / "model_final.pt"
    trainer.save_checkpoint(str(final_checkpoint))
    return model, final_checkpoint


def _to_tensor_visit(visit: dict[str, np.ndarray | None], device: str) -> dict[str, torch.Tensor | None]:
    tensors: dict[str, torch.Tensor | None] = {}
    for modality_name, values in visit.items():
        if values is None:
            tensors[modality_name] = None
        else:
            tensors[modality_name] = torch.as_tensor(values, dtype=torch.float32, device=device).unsqueeze(0)
    return tensors


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
