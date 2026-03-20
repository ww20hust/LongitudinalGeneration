"""
Utilities for two-visit clinical benchmark experiments driven by a single CSV.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class TabularBenchmarkSplit:
    train_lab: np.ndarray
    train_metab: np.ndarray
    test_lab: np.ndarray
    test_metab: np.ndarray
    train_index: list[str]
    test_index: list[str]
    lab_columns: list[str]
    metab_columns: list[str]


@dataclass
class ClinicalBenchmarkBundle:
    id_column: str
    visit_column: str
    lab_columns: list[str]
    metab_columns: list[str]
    train_patient_ids: list[str]
    test_patient_ids: list[str]
    train_visits: list[list[dict[str, np.ndarray | None]]]
    test_visits: list[list[dict[str, np.ndarray | None]]]
    train_missing_masks: list[list[dict[str, int]]]
    test_missing_masks: list[list[dict[str, int]]]
    raw_static_split: TabularBenchmarkSplit
    scaled_static_split: TabularBenchmarkSplit
    lab_scaler: StandardScaler
    metab_scaler: StandardScaler


def parse_column_argument(value: str | None) -> list[str] | None:
    if value is None:
        return None
    normalized = [column.strip() for column in value.split(",") if column.strip()]
    return normalized or None


def _infer_feature_columns(
    frame: pd.DataFrame,
    id_column: str,
    visit_column: str,
    lab_columns: Sequence[str] | None,
    metab_columns: Sequence[str] | None,
    lab_prefix: str | None,
    metab_prefix: str | None,
    expected_lab_dim: int,
    expected_metab_dim: int,
) -> tuple[list[str], list[str]]:
    if lab_columns is not None and metab_columns is not None:
        return list(lab_columns), list(metab_columns)

    metadata_columns = {id_column, visit_column}
    candidate_columns = [column for column in frame.columns if column not in metadata_columns]

    if lab_columns is None and lab_prefix:
        lab_columns = [column for column in candidate_columns if str(column).startswith(lab_prefix)]
    if metab_columns is None and metab_prefix:
        metab_columns = [column for column in candidate_columns if str(column).startswith(metab_prefix)]

    if lab_columns is not None and metab_columns is not None:
        return list(lab_columns), list(metab_columns)

    if len(candidate_columns) != expected_lab_dim + expected_metab_dim:
        raise ValueError(
            "Unable to infer feature columns automatically. Provide explicit columns or prefixes. "
            f"Expected {expected_lab_dim + expected_metab_dim} feature columns but found "
            f"{len(candidate_columns)}."
        )

    inferred_lab = candidate_columns[:expected_lab_dim]
    inferred_metab = candidate_columns[expected_lab_dim : expected_lab_dim + expected_metab_dim]
    return list(inferred_lab), list(inferred_metab)


def load_two_visit_clinical_csv(
    csv_path: str | Path,
    *,
    id_column: str = "eid",
    visit_column: str = "visit",
    lab_columns: Sequence[str] | None = None,
    metab_columns: Sequence[str] | None = None,
    lab_prefix: str | None = None,
    metab_prefix: str | None = None,
    expected_lab_dim: int = 61,
    expected_metab_dim: int = 251,
    required_visits: Sequence[int] = (0, 1),
) -> tuple[pd.DataFrame, list[str], list[str]]:
    frame = pd.read_csv(csv_path)
    if id_column not in frame.columns:
        raise ValueError(f"Missing id column '{id_column}'.")
    if visit_column not in frame.columns:
        raise ValueError(f"Missing visit column '{visit_column}'.")

    lab_columns, metab_columns = _infer_feature_columns(
        frame=frame,
        id_column=id_column,
        visit_column=visit_column,
        lab_columns=lab_columns,
        metab_columns=metab_columns,
        lab_prefix=lab_prefix,
        metab_prefix=metab_prefix,
        expected_lab_dim=expected_lab_dim,
        expected_metab_dim=expected_metab_dim,
    )

    required_columns = [id_column, visit_column, *lab_columns, *metab_columns]
    frame = frame.loc[:, required_columns].copy()
    frame[id_column] = frame[id_column].astype(str)
    frame[visit_column] = frame[visit_column].astype(int)
    frame = frame.sort_values([id_column, visit_column]).reset_index(drop=True)

    if frame.duplicated(subset=[id_column, visit_column]).any():
        raise ValueError("Each patient/visit pair must appear exactly once.")

    expected_visits = list(required_visits)
    invalid_patients = []
    for patient_id, patient_rows in frame.groupby(id_column):
        visits = patient_rows[visit_column].tolist()
        if visits != expected_visits:
            invalid_patients.append(str(patient_id))
    if invalid_patients:
        raise ValueError(
            "Each patient must contain exactly the required visits in order. "
            f"Invalid examples: {invalid_patients[:10]}"
        )

    frame[lab_columns] = frame[lab_columns].astype(np.float32)
    frame[metab_columns] = frame[metab_columns].astype(np.float32)
    return frame, lab_columns, metab_columns


def split_patient_ids(
    patient_ids: Sequence[str],
    *,
    train_ratio: float = 0.7,
    seed: int = 0,
) -> tuple[list[str], list[str]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")

    patient_ids = np.asarray(list(patient_ids), dtype=object)
    rng = np.random.default_rng(seed)
    shuffled = patient_ids.copy()
    rng.shuffle(shuffled)
    n_train = int(round(len(shuffled) * train_ratio))
    n_train = min(max(n_train, 1), len(shuffled) - 1)
    train_ids = sorted(str(patient_id) for patient_id in shuffled[:n_train])
    test_ids = sorted(str(patient_id) for patient_id in shuffled[n_train:])
    return train_ids, test_ids


def _fit_scalers(
    frame: pd.DataFrame,
    train_patient_ids: Sequence[str],
    id_column: str,
    lab_columns: Sequence[str],
    metab_columns: Sequence[str],
) -> tuple[StandardScaler, StandardScaler]:
    train_frame = frame.loc[frame[id_column].isin(train_patient_ids)]
    lab_scaler = StandardScaler().fit(train_frame.loc[:, lab_columns].to_numpy(dtype=np.float32))
    metab_scaler = StandardScaler().fit(train_frame.loc[:, metab_columns].to_numpy(dtype=np.float32))
    return lab_scaler, metab_scaler


def _apply_scalers(
    frame: pd.DataFrame,
    lab_columns: Sequence[str],
    metab_columns: Sequence[str],
    lab_scaler: StandardScaler,
    metab_scaler: StandardScaler,
) -> pd.DataFrame:
    scaled = frame.copy()
    scaled.loc[:, lab_columns] = lab_scaler.transform(frame.loc[:, lab_columns].to_numpy(dtype=np.float32)).astype(np.float32)
    scaled.loc[:, metab_columns] = metab_scaler.transform(frame.loc[:, metab_columns].to_numpy(dtype=np.float32)).astype(np.float32)
    return scaled


def _row_to_visit(
    row: pd.Series,
    lab_columns: Sequence[str],
    metab_columns: Sequence[str],
) -> tuple[dict[str, np.ndarray | None], dict[str, int]]:
    visit_data: dict[str, np.ndarray | None] = {}
    visit_mask: dict[str, int] = {}
    for modality_name, columns in (("lab", lab_columns), ("metab", metab_columns)):
        values = row.loc[list(columns)].to_numpy(dtype=np.float32)
        if np.isnan(values).all():
            visit_data[modality_name] = None
            visit_mask[modality_name] = 2
        else:
            visit_data[modality_name] = values
            visit_mask[modality_name] = 0
    return visit_data, visit_mask


def build_patient_sequences(
    frame: pd.DataFrame,
    patient_ids: Sequence[str],
    *,
    id_column: str,
    visit_column: str,
    lab_columns: Sequence[str],
    metab_columns: Sequence[str],
) -> tuple[list[str], list[list[dict[str, np.ndarray | None]]], list[list[dict[str, int]]]]:
    order = {str(patient_id): index for index, patient_id in enumerate(patient_ids)}
    ordered_ids = sorted((str(patient_id) for patient_id in patient_ids), key=order.get)
    grouped = frame.loc[frame[id_column].isin(ordered_ids)].sort_values([id_column, visit_column]).groupby(id_column, sort=False)

    visits_data: list[list[dict[str, np.ndarray | None]]] = []
    missing_masks: list[list[dict[str, int]]] = []
    for patient_id in ordered_ids:
        patient_visits: list[dict[str, np.ndarray | None]] = []
        patient_masks: list[dict[str, int]] = []
        for _, row in grouped.get_group(patient_id).iterrows():
            visit_data, visit_mask = _row_to_visit(row, lab_columns, metab_columns)
            patient_visits.append(visit_data)
            patient_masks.append(visit_mask)
        visits_data.append(patient_visits)
        missing_masks.append(patient_masks)
    return ordered_ids, visits_data, missing_masks


def build_static_benchmark_split(
    frame: pd.DataFrame,
    train_patient_ids: Sequence[str],
    test_patient_ids: Sequence[str],
    *,
    id_column: str,
    visit_column: str,
    lab_columns: Sequence[str],
    metab_columns: Sequence[str],
    followup_visit: int = 1,
) -> TabularBenchmarkSplit:
    train_frame = frame.loc[frame[id_column].isin(train_patient_ids)].sort_values([id_column, visit_column])
    test_frame = frame.loc[(frame[id_column].isin(test_patient_ids)) & (frame[visit_column] == followup_visit)].sort_values([id_column, visit_column])

    train_index = [f"{row[id_column]}_visit{int(row[visit_column])}" for _, row in train_frame.iterrows()]
    test_index = [f"{row[id_column]}_visit{int(row[visit_column])}" for _, row in test_frame.iterrows()]

    return TabularBenchmarkSplit(
        train_lab=train_frame.loc[:, lab_columns].to_numpy(dtype=np.float32),
        train_metab=train_frame.loc[:, metab_columns].to_numpy(dtype=np.float32),
        test_lab=test_frame.loc[:, lab_columns].to_numpy(dtype=np.float32),
        test_metab=test_frame.loc[:, metab_columns].to_numpy(dtype=np.float32),
        train_index=train_index,
        test_index=test_index,
        lab_columns=list(lab_columns),
        metab_columns=list(metab_columns),
    )


def prepare_two_visit_clinical_benchmark(
    csv_path: str | Path,
    *,
    id_column: str = "eid",
    visit_column: str = "visit",
    lab_columns: Sequence[str] | None = None,
    metab_columns: Sequence[str] | None = None,
    lab_prefix: str | None = None,
    metab_prefix: str | None = None,
    expected_lab_dim: int = 61,
    expected_metab_dim: int = 251,
    train_ratio: float = 0.7,
    seed: int = 0,
) -> ClinicalBenchmarkBundle:
    frame, lab_columns, metab_columns = load_two_visit_clinical_csv(
        csv_path=csv_path,
        id_column=id_column,
        visit_column=visit_column,
        lab_columns=lab_columns,
        metab_columns=metab_columns,
        lab_prefix=lab_prefix,
        metab_prefix=metab_prefix,
        expected_lab_dim=expected_lab_dim,
        expected_metab_dim=expected_metab_dim,
    )
    patient_ids = sorted(frame[id_column].astype(str).unique().tolist())
    train_patient_ids, test_patient_ids = split_patient_ids(
        patient_ids,
        train_ratio=train_ratio,
        seed=seed,
    )

    lab_scaler, metab_scaler = _fit_scalers(
        frame,
        train_patient_ids,
        id_column,
        lab_columns,
        metab_columns,
    )
    scaled_frame = _apply_scalers(frame, lab_columns, metab_columns, lab_scaler, metab_scaler)

    train_patient_ids, train_visits, train_missing_masks = build_patient_sequences(
        scaled_frame,
        train_patient_ids,
        id_column=id_column,
        visit_column=visit_column,
        lab_columns=lab_columns,
        metab_columns=metab_columns,
    )
    test_patient_ids, test_visits, test_missing_masks = build_patient_sequences(
        scaled_frame,
        test_patient_ids,
        id_column=id_column,
        visit_column=visit_column,
        lab_columns=lab_columns,
        metab_columns=metab_columns,
    )

    raw_static_split = build_static_benchmark_split(
        frame,
        train_patient_ids,
        test_patient_ids,
        id_column=id_column,
        visit_column=visit_column,
        lab_columns=lab_columns,
        metab_columns=metab_columns,
    )
    scaled_static_split = build_static_benchmark_split(
        scaled_frame,
        train_patient_ids,
        test_patient_ids,
        id_column=id_column,
        visit_column=visit_column,
        lab_columns=lab_columns,
        metab_columns=metab_columns,
    )

    return ClinicalBenchmarkBundle(
        id_column=id_column,
        visit_column=visit_column,
        lab_columns=list(lab_columns),
        metab_columns=list(metab_columns),
        train_patient_ids=train_patient_ids,
        test_patient_ids=test_patient_ids,
        train_visits=train_visits,
        test_visits=test_visits,
        train_missing_masks=train_missing_masks,
        test_missing_masks=test_missing_masks,
        raw_static_split=raw_static_split,
        scaled_static_split=scaled_static_split,
        lab_scaler=lab_scaler,
        metab_scaler=metab_scaler,
    )


def save_tabular_split_csv(split: TabularBenchmarkSplit, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(split.train_lab, index=split.train_index, columns=split.lab_columns).to_csv(output_dir / "train_lab.csv")
    pd.DataFrame(split.train_metab, index=split.train_index, columns=split.metab_columns).to_csv(output_dir / "train_metab.csv")
    pd.DataFrame(split.test_lab, index=split.test_index, columns=split.lab_columns).to_csv(output_dir / "test_lab.csv")
    pd.DataFrame(split.test_metab, index=split.test_index, columns=split.metab_columns).to_csv(output_dir / "test_metab.csv")


def save_scaler_stats(
    lab_scaler: StandardScaler,
    metab_scaler: StandardScaler,
    lab_columns: Sequence[str],
    metab_columns: Sequence[str],
    output_path: str | Path,
) -> None:
    payload = {
        "lab": {
            "mean": lab_scaler.mean_.tolist(),
            "scale": lab_scaler.scale_.tolist(),
            "var": lab_scaler.var_.tolist(),
        },
        "metab": {
            "mean": metab_scaler.mean_.tolist(),
            "scale": metab_scaler.scale_.tolist(),
            "var": metab_scaler.var_.tolist(),
        },
        "lab_columns": list(lab_columns),
        "metab_columns": list(metab_columns),
    }
    Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def evaluate_reconstruction(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: truth={y_true.shape}, pred={y_pred.shape}")

    true_centered = y_true - y_true.mean(axis=0, keepdims=True)
    pred_centered = y_pred - y_pred.mean(axis=0, keepdims=True)
    denom = np.sqrt((true_centered ** 2).sum(axis=0) * (pred_centered ** 2).sum(axis=0))
    corr = np.divide(
        (true_centered * pred_centered).sum(axis=0),
        denom,
        out=np.zeros_like(denom, dtype=np.float64),
        where=denom > 0,
    )
    return {
        "n_samples": int(y_true.shape[0]),
        "n_features": int(y_true.shape[1]),
        "pearson_mean": float(np.mean(corr)),
        "pearson_median": float(np.median(corr)),
        "pearson_min": float(np.min(corr)),
        "pearson_max": float(np.max(corr)),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "mse": float(np.mean((y_true - y_pred) ** 2)),
    }


def save_json(payload: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def format_mask_rate_tag(mask_rate: float) -> str:
    return f"{mask_rate:.2f}".replace(".", "p")
