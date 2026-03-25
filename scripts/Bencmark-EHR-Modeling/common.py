from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"


@dataclass
class HistoryEvent:
    age: Optional[float]
    code: str


@dataclass
class ProteomicsSample:
    patient_id: str
    current_age: Optional[float]
    history_events: List[HistoryEvent]
    routine_labs: Dict[str, Optional[float]]
    proteomics: np.ndarray


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric_value):
        return None
    return numeric_value


def _normalize_history_events(raw_events: Any) -> List[HistoryEvent]:
    if raw_events is None:
        return []

    normalized: List[HistoryEvent] = []
    for raw_event in raw_events:
        if raw_event is None:
            continue

        if isinstance(raw_event, dict):
            code = raw_event.get("code", raw_event.get("icd_code", raw_event.get("icd")))
            age = _to_optional_float(raw_event.get("age", raw_event.get("event_age")))
        elif isinstance(raw_event, (list, tuple)) and len(raw_event) >= 2:
            age = _to_optional_float(raw_event[0])
            code = raw_event[1]
        else:
            age = None
            code = raw_event

        if code is None:
            continue
        code_str = str(code).strip()
        if not code_str:
            continue
        normalized.append(HistoryEvent(age=age, code=code_str))

    normalized.sort(key=lambda event: (float("inf") if event.age is None else event.age, event.code))
    return normalized


def _normalize_history_windows(raw_windows: Any) -> List[HistoryEvent]:
    if raw_windows is None:
        return []
    events: List[HistoryEvent] = []
    for window in raw_windows:
        if window is None:
            continue
        for code in window:
            if code is None:
                continue
            code_str = str(code).strip()
            if code_str:
                events.append(HistoryEvent(age=None, code=code_str))
    return events


def _normalize_routine_labs(raw_labs: Any) -> Dict[str, Optional[float]]:
    if raw_labs is None:
        return {}
    normalized: Dict[str, Optional[float]] = {}
    for lab_name, value in raw_labs.items():
        normalized[str(lab_name)] = _to_optional_float(value)
    return normalized


def _normalize_record(record: Dict[str, Any], index: int) -> ProteomicsSample:
    patient_id = str(record.get("patient_id", f"sample_{index:06d}"))
    current_age = _to_optional_float(record.get("current_age", record.get("visit_age")))

    history_events = _normalize_history_events(
        record.get("history_events", record.get("medical_history_events"))
    )
    if not history_events:
        history_events = _normalize_history_windows(
            record.get("history_windows", record.get("medical_history_windows", record.get("medical_history")))
        )

    routine_labs = _normalize_routine_labs(record.get("routine_labs", record.get("labs")))
    if "proteomics" not in record:
        raise ValueError(f"Record {patient_id} is missing 'proteomics'.")
    proteomics = np.asarray(record["proteomics"], dtype=np.float32)
    if proteomics.ndim != 1:
        raise ValueError(f"Record {patient_id} has non-vector proteomics target.")

    return ProteomicsSample(
        patient_id=patient_id,
        current_age=current_age,
        history_events=history_events,
        routine_labs=routine_labs,
        proteomics=proteomics,
    )


def load_samples(path: Union[str, Path]) -> List[ProteomicsSample]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    if path.suffix.lower() == ".jsonl":
        records: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    elif path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            records = list(payload.get("samples", []))
        elif isinstance(payload, list):
            records = list(payload)
        else:
            raise ValueError(f"Unsupported JSON payload in {path}.")
    else:
        raise ValueError(f"Unsupported data format for {path}. Use .json or .jsonl.")

    return [_normalize_record(record, index) for index, record in enumerate(records)]


def infer_proteomics_dim(samples: Sequence[ProteomicsSample]) -> int:
    if not samples:
        raise ValueError("No samples were provided.")
    dim = int(samples[0].proteomics.shape[0])
    for sample in samples[1:]:
        if int(sample.proteomics.shape[0]) != dim:
            raise ValueError("Inconsistent proteomics dimensionality across samples.")
    return dim


class LabDiscretizer:
    def __init__(self, num_bins: int = 10) -> None:
        if num_bins < 2:
            raise ValueError("num_bins must be at least 2.")
        self.num_bins = int(num_bins)
        self.bin_edges: Dict[str, List[float]] = {}

    def fit(self, samples: Sequence[ProteomicsSample]) -> "LabDiscretizer":
        grouped_values: Dict[str, List[float]] = {}
        for sample in samples:
            for lab_name, value in sample.routine_labs.items():
                if value is None:
                    continue
                grouped_values.setdefault(lab_name, []).append(float(value))

        self.bin_edges = {}
        quantiles = np.linspace(0.0, 1.0, self.num_bins + 1)[1:-1]
        for lab_name, values in grouped_values.items():
            values_array = np.asarray(values, dtype=np.float64)
            if values_array.size == 0:
                continue
            if np.allclose(values_array, values_array[0]):
                edges = np.repeat(values_array[0], self.num_bins - 1)
            else:
                edges = np.quantile(values_array, quantiles)
                edges = np.maximum.accumulate(edges)
            self.bin_edges[lab_name] = [float(edge) for edge in edges.tolist()]
        return self

    def transform(self, lab_name: str, value: Optional[float]) -> Optional[int]:
        if value is None or not np.isfinite(value):
            return None
        edges = self.bin_edges.get(lab_name)
        if not edges:
            return 0
        return int(np.digitize(float(value), np.asarray(edges, dtype=np.float64), right=False))

    @property
    def lab_names(self) -> List[str]:
        return sorted(self.bin_edges.keys())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_bins": self.num_bins,
            "bin_edges": self.bin_edges,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "LabDiscretizer":
        instance = cls(num_bins=int(payload["num_bins"]))
        instance.bin_edges = {
            str(lab_name): [float(edge) for edge in edges]
            for lab_name, edges in payload["bin_edges"].items()
        }
        return instance


class RoutineLabVectorizer:
    def __init__(self) -> None:
        self.lab_names: List[str] = []
        self.means: Dict[str, float] = {}
        self.stds: Dict[str, float] = {}

    def fit(self, samples: Sequence[ProteomicsSample]) -> "RoutineLabVectorizer":
        observed_values: Dict[str, List[float]] = {}
        for sample in samples:
            for lab_name, value in sample.routine_labs.items():
                if value is None:
                    continue
                observed_values.setdefault(lab_name, []).append(float(value))

        self.lab_names = sorted(observed_values.keys())
        self.means = {}
        self.stds = {}
        for lab_name in self.lab_names:
            values = np.asarray(observed_values[lab_name], dtype=np.float32)
            mean = float(values.mean()) if values.size > 0 else 0.0
            std = float(values.std()) if values.size > 0 else 1.0
            if std < 1e-6:
                std = 1.0
            self.means[lab_name] = mean
            self.stds[lab_name] = std
        return self

    @property
    def dim(self) -> int:
        return len(self.lab_names)

    def transform_sample(self, sample: ProteomicsSample) -> np.ndarray:
        values: List[float] = []
        for lab_name in self.lab_names:
            raw_value = sample.routine_labs.get(lab_name)
            if raw_value is None:
                raw_value = self.means[lab_name]
            standardized = (float(raw_value) - self.means[lab_name]) / self.stds[lab_name]
            values.append(float(standardized))
        return np.asarray(values, dtype=np.float32)

    def transform_samples(self, samples: Sequence[ProteomicsSample]) -> np.ndarray:
        return np.stack([self.transform_sample(sample) for sample in samples], axis=0).astype(np.float32)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lab_names": self.lab_names,
            "means": self.means,
            "stds": self.stds,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "RoutineLabVectorizer":
        instance = cls()
        instance.lab_names = [str(name) for name in payload["lab_names"]]
        instance.means = {str(key): float(value) for key, value in payload["means"].items()}
        instance.stds = {str(key): float(value) for key, value in payload["stds"].items()}
        return instance


def age_to_bucket(age: Optional[float], max_age_years: int) -> int:
    if age is None or not np.isfinite(age):
        return 0
    clipped_age = max(0, min(int(math.floor(float(age))), int(max_age_years)))
    return clipped_age + 1


def build_token_vocab(
    samples: Sequence[ProteomicsSample],
    discretizer: LabDiscretizer,
) -> Dict[str, int]:
    token_to_id: Dict[str, int] = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
        CLS_TOKEN: 2,
        SEP_TOKEN: 3,
    }

    def add_token(token: str) -> None:
        if token not in token_to_id:
            token_to_id[token] = len(token_to_id)

    for sample in samples:
        for event in sample.history_events:
            add_token(f"ICD::{event.code}")

    for lab_name in discretizer.lab_names:
        for bin_index in range(discretizer.num_bins):
            add_token(f"LAB::{lab_name}::BIN::{bin_index}")

    return token_to_id


def encode_transformer_sample(
    sample: ProteomicsSample,
    token_to_id: Dict[str, int],
    discretizer: LabDiscretizer,
    max_history_events: Optional[int],
    max_age_years: int,
    max_seq_len: Optional[int] = None,
) -> Tuple[List[int], List[int], List[int]]:
    token_ids: List[int] = [token_to_id[CLS_TOKEN]]
    type_ids: List[int] = [0]
    age_ids: List[int] = [0]
    unk_id = token_to_id[UNK_TOKEN]

    history_events = sample.history_events[-max_history_events:] if max_history_events is not None else sample.history_events
    for event in history_events:
        token_ids.append(token_to_id.get(f"ICD::{event.code}", unk_id))
        type_ids.append(1)
        age_ids.append(age_to_bucket(event.age, max_age_years))

    token_ids.append(token_to_id[SEP_TOKEN])
    type_ids.append(0)
    age_ids.append(0)

    current_age_bucket = age_to_bucket(sample.current_age, max_age_years)
    for lab_name in sorted(sample.routine_labs.keys()):
        bin_index = discretizer.transform(lab_name, sample.routine_labs[lab_name])
        if bin_index is None:
            continue
        token = f"LAB::{lab_name}::BIN::{bin_index}"
        token_ids.append(token_to_id.get(token, unk_id))
        type_ids.append(2)
        age_ids.append(current_age_bucket)

    token_ids.append(token_to_id[SEP_TOKEN])
    type_ids.append(0)
    age_ids.append(0)

    if max_seq_len is not None and len(token_ids) > max_seq_len:
        tail_length = max(1, int(max_seq_len) - 1)
        token_ids = [token_to_id[CLS_TOKEN]] + token_ids[-tail_length:]
        type_ids = [0] + type_ids[-tail_length:]
        age_ids = [0] + age_ids[-tail_length:]
    return token_ids, type_ids, age_ids


def _format_age(age: Optional[float]) -> Optional[str]:
    if age is None:
        return None
    if abs(age - round(age)) < 1e-6:
        return str(int(round(age)))
    return f"{age:.1f}"


def render_history_as_text(sample: ProteomicsSample) -> str:
    if not sample.history_events:
        return "No ICD-coded medical history was recorded before the current visit."

    grouped_codes: Dict[str, List[str]] = {}
    for event in sample.history_events:
        age_text = _format_age(event.age)
        key = age_text if age_text is not None else "unknown"
        grouped_codes.setdefault(key, []).append(event.code)

    ordered_keys = sorted(
        grouped_codes.keys(),
        key=lambda key: float("inf") if key == "unknown" else float(key),
    )

    parts: List[str] = []
    for key in ordered_keys:
        codes = ", ".join(grouped_codes[key])
        if key == "unknown":
            parts.append(f"Before the current visit, the patient had ICD codes {codes}.")
        else:
            parts.append(f"At age {key}, the patient had ICD codes {codes}.")
    return " ".join(parts)


def render_sample_as_text(sample: ProteomicsSample, discretizer: LabDiscretizer) -> str:
    segments: List[str] = []
    if sample.current_age is not None:
        segments.append(f"The current clinical visit occurred at age {_format_age(sample.current_age)}.")
    else:
        segments.append("The current clinical visit age is not available.")

    segments.append(render_history_as_text(sample))

    if sample.routine_labs:
        lab_entries: List[str] = []
        for lab_name in sorted(sample.routine_labs.keys()):
            value = sample.routine_labs[lab_name]
            if value is None:
                lab_entries.append(f"{lab_name} was not measured")
                continue
            bin_index = discretizer.transform(lab_name, value)
            lab_entries.append(
                f"{lab_name} equaled {value:.4f}, corresponding to discretized level {int(bin_index) + 1} of {discretizer.num_bins}"
            )
        segments.append("At the current visit, routine laboratory tests were as follows: " + "; ".join(lab_entries) + ".")
    else:
        segments.append("No routine laboratory tests were recorded at the current visit.")

    return " ".join(segments)


class TransformerProteomicsDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[ProteomicsSample],
        token_to_id: Dict[str, int],
        discretizer: LabDiscretizer,
        max_history_events: Optional[int],
        max_age_years: int,
        max_seq_len: Optional[int] = None,
    ) -> None:
        self.samples = list(samples)
        self.token_to_id = token_to_id
        self.discretizer = discretizer
        self.max_history_events = max_history_events
        self.max_age_years = max_age_years
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        token_ids, type_ids, age_ids = encode_transformer_sample(
            sample=sample,
            token_to_id=self.token_to_id,
            discretizer=self.discretizer,
            max_history_events=self.max_history_events,
            max_age_years=self.max_age_years,
            max_seq_len=self.max_seq_len,
        )
        return {
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "type_ids": torch.tensor(type_ids, dtype=torch.long),
            "age_ids": torch.tensor(age_ids, dtype=torch.long),
            "target": torch.tensor(sample.proteomics, dtype=torch.float32),
        }


def collate_transformer_batch(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    max_length = max(int(item["token_ids"].shape[0]) for item in batch)
    batch_size = len(batch)
    token_ids = torch.zeros(batch_size, max_length, dtype=torch.long)
    type_ids = torch.zeros(batch_size, max_length, dtype=torch.long)
    age_ids = torch.zeros(batch_size, max_length, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_length, dtype=torch.bool)
    targets = []

    for row_index, item in enumerate(batch):
        current_length = int(item["token_ids"].shape[0])
        token_ids[row_index, :current_length] = item["token_ids"]
        type_ids[row_index, :current_length] = item["type_ids"]
        age_ids[row_index, :current_length] = item["age_ids"]
        attention_mask[row_index, :current_length] = True
        targets.append(item["target"])

    return {
        "token_ids": token_ids,
        "type_ids": type_ids,
        "age_ids": age_ids,
        "attention_mask": attention_mask,
        "targets": torch.stack(targets, dim=0),
    }


def targets_to_numpy(samples: Sequence[ProteomicsSample]) -> np.ndarray:
    return np.stack([sample.proteomics for sample in samples], axis=0).astype(np.float32)


def compute_regression_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, Any]:
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have the same shape.")

    absolute_error = np.abs(predictions - targets)
    feature_mae = absolute_error.mean(axis=0)

    feature_pearson: List[float] = []
    for feature_index in range(targets.shape[1]):
        x = targets[:, feature_index]
        y = predictions[:, feature_index]
        x_std = float(x.std())
        y_std = float(y.std())
        if x_std < 1e-8 or y_std < 1e-8:
            feature_pearson.append(0.0)
            continue
        corr = float(np.corrcoef(x, y)[0, 1])
        if np.isnan(corr):
            corr = 0.0
        feature_pearson.append(corr)

    mse = float(np.mean((predictions - targets) ** 2))
    return {
        "mse": mse,
        "mean_feature_mae": float(feature_mae.mean()),
        "median_feature_mae": float(np.median(feature_mae)),
        "mean_feature_pearson": float(np.mean(feature_pearson)),
        "median_feature_pearson": float(np.median(feature_pearson)),
        "feature_mae": [float(value) for value in feature_mae.tolist()],
        "feature_pearson": [float(value) for value in feature_pearson],
    }


class EarlyStopper:
    def __init__(self, patience: int) -> None:
        self.patience = int(patience)
        self.best_value: Optional[float] = None
        self.num_bad_epochs = 0

    def step(self, value: float) -> bool:
        if self.best_value is None or value < self.best_value:
            self.best_value = float(value)
            self.num_bad_epochs = 0
            return False
        self.num_bad_epochs += 1
        return self.num_bad_epochs >= self.patience


def save_json(path: Union[str, Path], payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def as_serializable_config(config: Dict[str, Any]) -> Dict[str, Any]:
    serializable: Dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, Path):
            serializable[key] = str(value)
        else:
            serializable[key] = value
    return serializable
