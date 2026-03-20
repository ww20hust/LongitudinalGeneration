from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
SCVAEIT_REPO = BENCHMARK_ROOT / "baseline-model" / "scVAEIT"
if str(SCVAEIT_REPO) not in sys.path:
    sys.path.insert(0, str(SCVAEIT_REPO))

from scVAEIT.VAEIT import VAEIT  # noqa: E402


def default_scvaeit_config(lab_dim: int, metab_dim: int) -> dict[str, Any]:
    return {
        "dim_input_arr": np.array([lab_dim, metab_dim], dtype=np.int32),
        "dim_block": np.array([lab_dim, metab_dim], dtype=np.int32),
        "dim_block_enc": np.array([64, 64], dtype=np.int32),
        "dim_block_dec": np.array([64, 64], dtype=np.int32),
        "dim_block_embed": np.array([32, 32], dtype=np.int32),
        "dimensions": np.array([128], dtype=np.int32),
        "dist_block": np.array(["Gaussian", "Gaussian"]),
        "uni_block_names": np.array(["lab", "metab"]),
        "block_names": np.array(["lab", "metab"]),
        "beta_kl": 2.0,
        "beta_unobs": 0.5,
        "beta_reverse": 0.0,
        "beta_modal": np.array([1.0, 1.0], dtype=np.float32),
        "p_feat": 0.2,
        "p_modal": None,
        "skip_conn": False,
        "gamma": 0.0,
    }


def _to_serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.int32, np.int64)):
        return int(value)
    return value


def _build_train_mask(lab_dim: int, metab_dim: int, n_samples: int) -> np.ndarray:
    return np.zeros((n_samples, lab_dim + metab_dim), dtype=np.float32)


def _build_test_mask(lab_dim: int, metab_dim: int, n_samples: int) -> np.ndarray:
    masks = np.zeros((n_samples, lab_dim + metab_dim), dtype=np.float32)
    masks[:, lab_dim:] = -1.0
    return masks


def train_scvaeit(
    train_lab: np.ndarray,
    train_metab: np.ndarray,
    output_dir: str | Path,
    *,
    config_overrides: dict[str, Any] | None = None,
    random_seed: int = 0,
    learning_rate: float = 3e-4,
    batch_size: int = 128,
    batch_size_inference: int = 512,
    num_epoch: int = 200,
    early_stopping_patience: int = 20,
    verbose: bool = False,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    np.random.seed(random_seed)

    train_lab = np.asarray(train_lab, dtype=np.float32)
    train_metab = np.asarray(train_metab, dtype=np.float32)
    if train_lab.shape[0] != train_metab.shape[0]:
        raise ValueError("train_lab and train_metab must have the same number of rows.")

    config = default_scvaeit_config(train_lab.shape[1], train_metab.shape[1])
    if config_overrides:
        config.update(config_overrides)

    data = np.concatenate([train_lab, train_metab], axis=1)
    masks = _build_train_mask(train_lab.shape[1], train_metab.shape[1], data.shape[0])
    batches_cate = np.zeros((data.shape[0], 1), dtype=np.int32)

    model = VAEIT(
        config=SimpleNamespace(**config),
        data=data,
        masks=masks,
        batches_cate=batches_cate,
    )
    checkpoint_dir = output_path / "checkpoints"
    model.train(
        valid=False,
        learning_rate=learning_rate,
        batch_size=batch_size,
        batch_size_inference=batch_size_inference,
        num_epoch=num_epoch,
        early_stopping_patience=early_stopping_patience,
        checkpoint_dir=str(checkpoint_dir),
        verbose=verbose,
    )
    model.save_model(str(checkpoint_dir))

    with (output_path / "config.json").open("w", encoding="utf-8") as fh:
        json.dump({k: _to_serializable(v) for k, v in config.items()}, fh, indent=2)

    return checkpoint_dir


def predict_scvaeit(
    checkpoint_dir: str | Path,
    test_lab: np.ndarray,
    metab_dim: int,
    *,
    config_path: str | Path | None = None,
    batch_size_inference: int = 512,
) -> np.ndarray:
    checkpoint_path = Path(checkpoint_dir)
    if config_path is None:
        config_path = checkpoint_path.parent / "config.json"
    config_path = Path(config_path)

    with config_path.open("r", encoding="utf-8") as fh:
        config = json.load(fh)

    config["dim_input_arr"] = np.asarray(config["dim_input_arr"], dtype=np.int32)
    config["dim_block"] = np.asarray(config["dim_block"], dtype=np.int32)
    config["dim_block_enc"] = np.asarray(config["dim_block_enc"], dtype=np.int32)
    config["dim_block_dec"] = np.asarray(config["dim_block_dec"], dtype=np.int32)
    config["dim_block_embed"] = np.asarray(config["dim_block_embed"], dtype=np.int32)
    config["dimensions"] = np.asarray(config["dimensions"], dtype=np.int32)
    config["dist_block"] = np.asarray(config["dist_block"])
    config["uni_block_names"] = np.asarray(config["uni_block_names"])
    config["block_names"] = np.asarray(config["block_names"])
    config["beta_modal"] = np.asarray(config["beta_modal"], dtype=np.float32)

    test_lab = np.asarray(test_lab, dtype=np.float32)
    zeros_metab = np.zeros((test_lab.shape[0], metab_dim), dtype=np.float32)
    data = np.concatenate([test_lab, zeros_metab], axis=1)
    masks = _build_test_mask(test_lab.shape[1], metab_dim, data.shape[0])
    batches_cate = np.zeros((data.shape[0], 1), dtype=np.int32)

    model = VAEIT(
        config=SimpleNamespace(**config),
        data=data,
        masks=masks,
        batches_cate=batches_cate,
    )
    model.load_model(str(checkpoint_path))
    recon = model.get_denoised_data(
        masks=masks,
        zero_out=False,
        return_mean=True,
        batch_size_inference=batch_size_inference,
        training=False,
    )
    return np.asarray(recon[:, test_lab.shape[1]:], dtype=np.float32)
