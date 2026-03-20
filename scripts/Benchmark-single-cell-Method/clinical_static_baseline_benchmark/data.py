from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from peag.clinical_benchmark import parse_column_argument, prepare_two_visit_clinical_benchmark  # noqa: E402


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


def _default_index(prefix: str, n: int) -> list[str]:
    return [f"{prefix}_{i:06d}" for i in range(n)]


def _default_columns(prefix: str, n: int) -> list[str]:
    return [f"{prefix}_{i:03d}" for i in range(n)]


def _read_csv_matrix(path: Path, prefix: str) -> tuple[np.ndarray, list[str], list[str]]:
    frame = pd.read_csv(path, index_col=0)
    index = frame.index.astype(str).tolist() or _default_index(prefix, len(frame))
    columns = frame.columns.astype(str).tolist() or _default_columns(prefix, frame.shape[1])
    return frame.to_numpy(dtype=np.float32), index, columns


def load_benchmark_split(input_path: str | Path) -> TabularBenchmarkSplit:
    path = Path(input_path)
    if path.is_dir():
        train_lab, train_index, lab_columns = _read_csv_matrix(path / "train_lab.csv", "train")
        train_metab, train_index_metab, metab_columns = _read_csv_matrix(path / "train_metab.csv", "train")
        test_lab, test_index, lab_columns_test = _read_csv_matrix(path / "test_lab.csv", "test")
        test_metab, test_index_metab, metab_columns_test = _read_csv_matrix(path / "test_metab.csv", "test")

        if train_index != train_index_metab:
            raise ValueError("train_lab.csv and train_metab.csv must share the same sample order.")
        if test_index != test_index_metab:
            raise ValueError("test_lab.csv and test_metab.csv must share the same sample order.")
        if lab_columns != lab_columns_test:
            raise ValueError("train_lab.csv and test_lab.csv must share the same lab feature order.")
        if metab_columns != metab_columns_test:
            raise ValueError("train_metab.csv and test_metab.csv must share the same metabolomics feature order.")

        return TabularBenchmarkSplit(
            train_lab=train_lab,
            train_metab=train_metab,
            test_lab=test_lab,
            test_metab=test_metab,
            train_index=train_index,
            test_index=test_index,
            lab_columns=lab_columns,
            metab_columns=metab_columns,
        )

    if path.suffix.lower() == ".npz":
        bundle = np.load(path, allow_pickle=True)
        train_lab = bundle["train_lab"].astype(np.float32)
        train_metab = bundle["train_metab"].astype(np.float32)
        test_lab = bundle["test_lab"].astype(np.float32)
        test_metab = bundle["test_metab"].astype(np.float32)

        train_index = bundle["train_index"].astype(str).tolist() if "train_index" in bundle else _default_index("train", train_lab.shape[0])
        test_index = bundle["test_index"].astype(str).tolist() if "test_index" in bundle else _default_index("test", test_lab.shape[0])
        lab_columns = bundle["lab_columns"].astype(str).tolist() if "lab_columns" in bundle else _default_columns("lab", train_lab.shape[1])
        metab_columns = bundle["metab_columns"].astype(str).tolist() if "metab_columns" in bundle else _default_columns("metab", train_metab.shape[1])

        return TabularBenchmarkSplit(
            train_lab=train_lab,
            train_metab=train_metab,
            test_lab=test_lab,
            test_metab=test_metab,
            train_index=train_index,
            test_index=test_index,
            lab_columns=lab_columns,
            metab_columns=metab_columns,
        )

    raise ValueError(f"Unsupported input path: {path}")


def load_longitudinal_csv_benchmark(
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
) -> tuple[TabularBenchmarkSplit, TabularBenchmarkSplit, StandardScaler, StandardScaler]:
    bundle = prepare_two_visit_clinical_benchmark(
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
    raw_split = TabularBenchmarkSplit(
        train_lab=bundle.raw_static_split.train_lab,
        train_metab=bundle.raw_static_split.train_metab,
        test_lab=bundle.raw_static_split.test_lab,
        test_metab=bundle.raw_static_split.test_metab,
        train_index=bundle.raw_static_split.train_index,
        test_index=bundle.raw_static_split.test_index,
        lab_columns=bundle.raw_static_split.lab_columns,
        metab_columns=bundle.raw_static_split.metab_columns,
    )
    scaled_split = TabularBenchmarkSplit(
        train_lab=bundle.scaled_static_split.train_lab,
        train_metab=bundle.scaled_static_split.train_metab,
        test_lab=bundle.scaled_static_split.test_lab,
        test_metab=bundle.scaled_static_split.test_metab,
        train_index=bundle.scaled_static_split.train_index,
        test_index=bundle.scaled_static_split.test_index,
        lab_columns=bundle.scaled_static_split.lab_columns,
        metab_columns=bundle.scaled_static_split.metab_columns,
    )
    return raw_split, scaled_split, bundle.lab_scaler, bundle.metab_scaler


def fit_standardizers(split: TabularBenchmarkSplit) -> tuple[StandardScaler, StandardScaler]:
    return StandardScaler().fit(split.train_lab), StandardScaler().fit(split.train_metab)


def transform_split(
    split: TabularBenchmarkSplit,
    lab_scaler: StandardScaler,
    metab_scaler: StandardScaler,
) -> TabularBenchmarkSplit:
    return TabularBenchmarkSplit(
        train_lab=lab_scaler.transform(split.train_lab).astype(np.float32),
        train_metab=metab_scaler.transform(split.train_metab).astype(np.float32),
        test_lab=lab_scaler.transform(split.test_lab).astype(np.float32),
        test_metab=metab_scaler.transform(split.test_metab).astype(np.float32),
        train_index=split.train_index,
        test_index=split.test_index,
        lab_columns=split.lab_columns,
        metab_columns=split.metab_columns,
    )


def _write_csv(matrix: np.ndarray, index: list[str], columns: list[str], path: Path) -> None:
    pd.DataFrame(matrix, index=index, columns=columns).to_csv(path)


def _scaler_to_dict(scaler: StandardScaler) -> Dict[str, Any]:
    return {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "var": scaler.var_.tolist(),
    }


def save_prepared_split(
    raw_split: TabularBenchmarkSplit,
    scaled_split: TabularBenchmarkSplit,
    lab_scaler: StandardScaler,
    metab_scaler: StandardScaler,
    output_dir: str | Path,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    _write_csv(raw_split.train_lab, raw_split.train_index, raw_split.lab_columns, output / "train_lab.csv")
    _write_csv(raw_split.train_metab, raw_split.train_index, raw_split.metab_columns, output / "train_metab.csv")
    _write_csv(raw_split.test_lab, raw_split.test_index, raw_split.lab_columns, output / "test_lab.csv")
    _write_csv(raw_split.test_metab, raw_split.test_index, raw_split.metab_columns, output / "test_metab.csv")

    _write_csv(scaled_split.train_lab, scaled_split.train_index, scaled_split.lab_columns, output / "train_lab_scaled.csv")
    _write_csv(scaled_split.train_metab, scaled_split.train_index, scaled_split.metab_columns, output / "train_metab_scaled.csv")
    _write_csv(scaled_split.test_lab, scaled_split.test_index, scaled_split.lab_columns, output / "test_lab_scaled.csv")
    _write_csv(scaled_split.test_metab, scaled_split.test_index, scaled_split.metab_columns, output / "test_metab_scaled.csv")

    scaler_stats = {
        "lab": _scaler_to_dict(lab_scaler),
        "metab": _scaler_to_dict(metab_scaler),
        "lab_columns": raw_split.lab_columns,
        "metab_columns": raw_split.metab_columns,
    }
    (output / "scaler_stats.json").write_text(json.dumps(scaler_stats, indent=2), encoding="utf-8")


def load_scaler_stats(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def inverse_transform_array(matrix: np.ndarray, scaler_stats: Dict[str, Any], key: str) -> np.ndarray:
    mean = np.asarray(scaler_stats[key]["mean"], dtype=np.float32)
    scale = np.asarray(scaler_stats[key]["scale"], dtype=np.float32)
    return matrix * scale + mean

