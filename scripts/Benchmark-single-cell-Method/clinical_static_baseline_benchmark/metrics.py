from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np


def _safe_featurewise_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    true_centered = y_true - y_true.mean(axis=0, keepdims=True)
    pred_centered = y_pred - y_pred.mean(axis=0, keepdims=True)
    denom = np.sqrt((true_centered ** 2).sum(axis=0) * (pred_centered ** 2).sum(axis=0))
    corr = np.divide(
        (true_centered * pred_centered).sum(axis=0),
        denom,
        out=np.zeros_like(denom, dtype=np.float64),
        where=denom > 0,
    )
    return corr


def evaluate_reconstruction(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: truth={y_true.shape}, pred={y_pred.shape}")

    pearson_by_feature = _safe_featurewise_pearson(y_true, y_pred)
    return {
        "n_samples": int(y_true.shape[0]),
        "n_features": int(y_true.shape[1]),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "mse": float(np.mean((y_true - y_pred) ** 2)),
        "pearson_mean": float(np.mean(pearson_by_feature)),
        "pearson_median": float(np.median(pearson_by_feature)),
        "pearson_min": float(np.min(pearson_by_feature)),
        "pearson_max": float(np.max(pearson_by_feature)),
    }


def save_metrics(metrics: Dict[str, float], path: str | Path) -> None:
    Path(path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
