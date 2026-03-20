from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from mudata import MuData


BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
MIDAS_SRC = BENCHMARK_ROOT / "baseline-model" / "midas" / "src"
if str(MIDAS_SRC) not in sys.path:
    sys.path.insert(0, str(MIDAS_SRC))

from scmidas.config import load_config  # noqa: E402
from scmidas.model import MIDAS  # noqa: E402
from scmidas.nn import distribution_registry  # noqa: E402


class GaussianMSELoss(torch.nn.Module):
    def forward(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mse_loss(input_tensor, target_tensor, reduction="none")


def _register_gaussian_distribution() -> None:
    if "GAUSSIAN" in distribution_registry.list_registered():
        return
    distribution_registry.register(
        "GAUSSIAN",
        GaussianMSELoss(),
        lambda data: data,
        torch.nn.Identity(),
    )


def default_midas_config() -> dict[str, Any]:
    config = dict(load_config())
    config.update(
        {
            "distribution_dec_lab": "GAUSSIAN",
            "distribution_dec_metab": "GAUSSIAN",
            "lam_recon_lab": 1.0,
            "lam_recon_metab": 1.0,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "n_iter_disc": 1,
        }
    )
    for key in ["trsf_before_enc_rna", "trsf_before_enc_adt", "trsf_before_enc_atac"]:
        config.pop(key, None)
    return config


def _make_anndata(matrix: np.ndarray, obs_names: list[str], var_prefix: str, batch_label: str) -> AnnData:
    obs = pd.DataFrame({"batch": batch_label}, index=obs_names)
    var = pd.DataFrame(index=[f"{var_prefix}_{i}" for i in range(matrix.shape[1])])
    return AnnData(X=np.asarray(matrix, dtype=np.float32), obs=obs, var=var)


def _make_paired_mudata(lab: np.ndarray, metab: np.ndarray, batch_label: str) -> MuData:
    obs_names = [f"{batch_label}_{i}" for i in range(lab.shape[0])]
    return MuData(
        {
            "lab": _make_anndata(lab, obs_names, "lab", batch_label),
            "metab": _make_anndata(metab, obs_names, "metab", batch_label),
        }
    )


def _make_query_mudata(test_lab: np.ndarray, batch_label: str) -> MuData:
    obs_names = [f"{batch_label}_{i}" for i in range(test_lab.shape[0])]
    return MuData(
        {
            "lab": _make_anndata(test_lab, obs_names, "lab", batch_label),
        }
    )


def train_midas(
    train_lab: np.ndarray,
    train_metab: np.ndarray,
    output_dir: str | Path,
    *,
    config_overrides: dict[str, Any] | None = None,
    max_epochs: int = 200,
    batch_size: int = 128,
) -> Path:
    _register_gaussian_distribution()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    configs = default_midas_config()
    if config_overrides:
        configs.update(config_overrides)

    mdata = _make_paired_mudata(train_lab, train_metab, batch_label="train")
    dims_x = {"lab": [train_lab.shape[1]], "metab": [train_metab.shape[1]]}
    model = MIDAS.configure_data_from_mdata(
        configs=configs,
        mdata=mdata,
        dims_x=dims_x,
        batch_key="batch",
        sampler_type="auto",
        save_model_path=str(output_path),
        batch_size=batch_size,
    )
    model.train(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )

    checkpoint_path = output_path / "midas_checkpoint.pt"
    model.save_checkpoint(str(checkpoint_path))
    with (output_path / "config.json").open("w", encoding="utf-8") as fh:
        json.dump(configs, fh, indent=2)
    return checkpoint_path


def predict_midas(
    checkpoint_path: str | Path,
    test_lab: np.ndarray,
    train_lab_dim: int,
    train_metab_dim: int,
    *,
    config_path: str | Path | None = None,
    batch_size: int = 128,
) -> np.ndarray:
    _register_gaussian_distribution()
    checkpoint_path = Path(checkpoint_path)
    if config_path is None:
        config_path = checkpoint_path.parent / "config.json"
    with Path(config_path).open("r", encoding="utf-8") as fh:
        configs = json.load(fh)

    mdata = _make_query_mudata(test_lab, batch_label="query")
    dims_x = {"lab": [train_lab_dim], "metab": [train_metab_dim]}
    model = MIDAS.configure_data_from_mdata(
        configs=configs,
        mdata=mdata,
        dims_x=dims_x,
        batch_key="batch",
        sampler_type="auto",
        save_model_path=str(checkpoint_path.parent),
        batch_size=batch_size,
    )
    model.load_checkpoint(str(checkpoint_path), map_location="cpu")
    pred = model.predict(
        return_in_memory=True,
        save_dir=None,
        joint_latent=False,
        mod_latent=False,
        impute=True,
        batch_correct=False,
        translate=False,
        input=False,
        verbose=False,
    )
    return np.asarray(pred["query"]["x_impt"]["metab"], dtype=np.float32)
