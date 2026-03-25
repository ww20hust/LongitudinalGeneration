from __future__ import annotations

import hashlib
import json
import math
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset


@dataclass
class PreparedBinarySplit:
    labels: np.ndarray
    labs: np.ndarray
    lab_mask: np.ndarray
    peag_note_texts: List[List[str]]
    document_texts: List[str]
    patient_ids: List[str]
    subject_ids: List[int]
    hadm_ids: List[int]
    icustay_ids: List[int]
    meta: Dict[str, Any]


class TensorDictDataset(Dataset):
    def __init__(self, tensors: Dict[str, torch.Tensor], patient_ids: Sequence[str]) -> None:
        if not patient_ids:
            raise ValueError('patient_ids must not be empty.')
        self.tensors = tensors
        self.patient_ids = list(patient_ids)
        expected = len(self.patient_ids)
        for key, value in tensors.items():
            if len(value) != expected:
                raise ValueError(f'Tensor {key} has length {len(value)} but expected {expected}.')

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {key: value[index] for key, value in self.tensors.items()}


class EarlyStopper:
    def __init__(self, mode: str = 'max', patience: int = 10, min_delta: float = 0.0) -> None:
        if mode not in {'max', 'min'}:
            raise ValueError("mode must be 'max' or 'min'.")
        self.mode = mode
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best_value: Optional[float] = None
        self.best_state: Optional[Dict[str, torch.Tensor]] = None
        self.counter = 0

    def step(self, value: float, model: torch.nn.Module) -> bool:
        improved = False
        if self.best_value is None:
            improved = True
        elif self.mode == 'max' and value > self.best_value + self.min_delta:
            improved = True
        elif self.mode == 'min' and value < self.best_value - self.min_delta:
            improved = True

        if improved:
            self.best_value = float(value)
            self.best_state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience

    def restore(self, model: torch.nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512) -> None:
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)


def load_pickle(path: str | Path) -> Any:
    with Path(path).open('rb') as handle:
        return pickle.load(handle)


def save_pickle(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_prepared_split(path: str | Path) -> PreparedBinarySplit:
    payload = load_pickle(path)
    return PreparedBinarySplit(
        labels=np.asarray(payload['labels'], dtype=np.float32),
        labs=np.asarray(payload['labs'], dtype=np.float32),
        lab_mask=np.asarray(payload['lab_mask'], dtype=np.float32),
        peag_note_texts=[list(map(str, seq)) for seq in payload['peag_note_texts']],
        document_texts=[str(text) for text in payload['document_texts']],
        patient_ids=[str(x) for x in payload['patient_ids']],
        subject_ids=[int(x) for x in payload['subject_ids']],
        hadm_ids=[int(x) for x in payload['hadm_ids']],
        icustay_ids=[int(x) for x in payload['icustay_ids']],
        meta=dict(payload.get('meta', {})),
    )


def create_tensor_dataset(
    split: PreparedBinarySplit,
    *,
    document_embeddings: Optional[np.ndarray] = None,
    note_sequence_embeddings: Optional[np.ndarray] = None,
    note_sequence_mask: Optional[np.ndarray] = None,
) -> TensorDictDataset:
    tensors: Dict[str, torch.Tensor] = {
        'labels': torch.tensor(split.labels, dtype=torch.float32),
        'labs': torch.tensor(split.labs, dtype=torch.float32),
        'lab_mask': torch.tensor(split.lab_mask, dtype=torch.float32),
        'lab_timestep_mask': torch.tensor(split.lab_mask.any(axis=-1), dtype=torch.float32),
    }
    if document_embeddings is not None:
        tensors['document_embeddings'] = torch.tensor(document_embeddings, dtype=torch.float32)
    if note_sequence_embeddings is not None:
        tensors['note_embeddings'] = torch.tensor(note_sequence_embeddings, dtype=torch.float32)
    if note_sequence_mask is not None:
        tensors['note_mask'] = torch.tensor(note_sequence_mask, dtype=torch.float32)
    return TensorDictDataset(tensors=tensors, patient_ids=split.patient_ids)


def create_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int = 0) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def compute_binary_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    labels = np.asarray(labels, dtype=np.float32)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    predictions = (probabilities >= threshold).astype(np.int32)
    metrics = {
        'accuracy': float(accuracy_score(labels, predictions)),
        'f1': float(f1_score(labels, predictions, zero_division=0)),
        'precision': float(precision_score(labels, predictions, zero_division=0)),
        'recall': float(recall_score(labels, predictions, zero_division=0)),
        'positive_rate': float(labels.mean()),
    }
    try:
        metrics['auroc'] = float(roc_auc_score(labels, probabilities))
    except ValueError:
        metrics['auroc'] = float('nan')
    try:
        metrics['auprc'] = float(average_precision_score(labels, probabilities))
    except ValueError:
        metrics['auprc'] = float('nan')
    return metrics


def compute_pos_weight(labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.float32)
    positives = float(labels.sum())
    negatives = float(labels.shape[0] - positives)
    if positives <= 0:
        return 1.0
    return max(negatives / positives, 1.0)


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    mask = mask.to(dtype=x.dtype)
    while mask.ndim < x.ndim:
        mask = mask.unsqueeze(-1)
    numerator = (x * mask).sum(dim=dim)
    denominator = mask.sum(dim=dim).clamp_min(1e-6)
    return numerator / denominator


def batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def run_classifier_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    criterion: torch.nn.Module,
    device: torch.device,
) -> Dict[str, Any]:
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    total_examples = 0
    all_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for batch in loader:
        batch = batch_to_device(batch, device)
        labels = batch['labels']
        with torch.set_grad_enabled(training):
            logits = model(batch)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        probs = torch.sigmoid(logits.detach()).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.detach().cpu().numpy())
        batch_size = labels.shape[0]
        total_examples += batch_size
        total_loss += float(loss.item()) * batch_size

    probs_np = np.concatenate(all_probs, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    metrics = compute_binary_metrics(labels_np, probs_np)
    metrics['loss'] = total_loss / max(total_examples, 1)
    metrics['probabilities'] = probs_np
    metrics['labels'] = labels_np
    return metrics


def save_predictions(path: str | Path, probabilities: np.ndarray, labels: np.ndarray, patient_ids: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        probabilities=np.asarray(probabilities, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.float32),
        patient_ids=np.asarray(list(patient_ids)),
    )


def fit_classifier(
    model: torch.nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    device: torch.device,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    pos_weight: float,
    patience: int,
    selection_metric: str = 'auprc',
) -> Tuple[torch.nn.Module, Dict[str, List[float]], Dict[str, float]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device))
    stopper = EarlyStopper(mode='max', patience=patience)
    history: Dict[str, List[float]] = {
        'train_loss': [], 'train_auroc': [], 'train_auprc': [], 'train_f1': [],
        'valid_loss': [], 'valid_auroc': [], 'valid_auprc': [], 'valid_f1': [],
    }

    model.to(device)
    best_valid: Dict[str, float] = {}
    for epoch in range(1, epochs + 1):
        train_metrics = run_classifier_epoch(model, train_loader, optimizer, criterion, device)
        valid_metrics = run_classifier_epoch(model, valid_loader, None, criterion, device)
        history['train_loss'].append(float(train_metrics['loss']))
        history['train_auroc'].append(float(train_metrics['auroc']))
        history['train_auprc'].append(float(train_metrics['auprc']))
        history['train_f1'].append(float(train_metrics['f1']))
        history['valid_loss'].append(float(valid_metrics['loss']))
        history['valid_auroc'].append(float(valid_metrics['auroc']))
        history['valid_auprc'].append(float(valid_metrics['auprc']))
        history['valid_f1'].append(float(valid_metrics['f1']))

        score = float(valid_metrics[selection_metric])
        if stopper.best_value is None or score >= stopper.best_value:
            best_valid = {k: float(v) for k, v in valid_metrics.items() if isinstance(v, (int, float, np.floating))}
        should_stop = stopper.step(score, model)
        print(
            f"Epoch {epoch:03d} | train loss={train_metrics['loss']:.4f} auroc={train_metrics['auroc']:.4f} auprc={train_metrics['auprc']:.4f} "
            f"| valid loss={valid_metrics['loss']:.4f} auroc={valid_metrics['auroc']:.4f} auprc={valid_metrics['auprc']:.4f}"
        )
        if should_stop:
            print(f'Early stopping triggered at epoch {epoch}.')
            break

    stopper.restore(model)
    return model, history, best_valid


@torch.inference_mode()
def evaluate_classifier(model: torch.nn.Module, loader: DataLoader, device: torch.device, pos_weight: float) -> Dict[str, Any]:
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device))
    model.to(device)
    return run_classifier_epoch(model, loader, None, criterion, device)


def _cache_filename(
    split: PreparedBinarySplit,
    *,
    split_name: str,
    kind: str,
    embedder,
    max_length: int,
) -> str:
    model_name = getattr(embedder, 'model_name_or_path', None)
    if model_name is None:
        model = getattr(embedder, 'model', None)
        model_name = getattr(model, 'name_or_path', 'unknown_model')
    payload = {
        'split_name': split_name,
        'kind': kind,
        'task': split.meta.get('task', 'unknown'),
        'document_scope': split.meta.get('document_scope', ''),
        'peag_scope': split.meta.get('peag_scope', ''),
        'max_length': int(max_length),
        'num_documents': len(split.document_texts),
        'sequence_length': len(split.peag_note_texts[0]) if split.peag_note_texts else 0,
        'model_name_or_path': str(model_name),
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode('utf-8')).hexdigest()[:12]
    return f'{split_name}_{kind}_{digest}.npz'


def load_or_compute_document_embeddings(
    split: PreparedBinarySplit,
    *,
    split_name: str,
    cache_dir: Optional[str],
    embedder,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    cache_path: Optional[Path] = None
    if cache_dir is not None:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        cache_path = cache_path / _cache_filename(
            split,
            split_name=split_name,
            kind='document_embeddings',
            embedder=embedder,
            max_length=max_length,
        )
        if cache_path.exists():
            return np.load(cache_path, allow_pickle=False)['embeddings'].astype(np.float32)

    non_empty_indices = [idx for idx, text in enumerate(split.document_texts) if str(text).strip()]
    if non_empty_indices:
        encoded = embedder.encode_texts([split.document_texts[idx] for idx in non_empty_indices], batch_size=batch_size, max_length=max_length).astype(np.float32)
        dim = encoded.shape[1]
    else:
        dim = 4096
        encoded = np.zeros((0, dim), dtype=np.float32)
    embeddings = np.zeros((len(split.document_texts), dim), dtype=np.float32)
    for row_index, sample_index in enumerate(non_empty_indices):
        embeddings[sample_index] = encoded[row_index]
    if cache_path is not None:
        np.savez_compressed(cache_path, embeddings=embeddings)
    return embeddings


def load_or_compute_sequence_embeddings(
    split: PreparedBinarySplit,
    *,
    split_name: str,
    cache_dir: Optional[str],
    embedder,
    batch_size: int,
    max_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    cache_path: Optional[Path] = None
    if cache_dir is not None:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        cache_path = cache_path / _cache_filename(
            split,
            split_name=split_name,
            kind='sequence_embeddings',
            embedder=embedder,
            max_length=max_length,
        )
        if cache_path.exists():
            payload = np.load(cache_path, allow_pickle=False)
            return payload['embeddings'].astype(np.float32), payload['mask'].astype(np.float32)

    n_samples = len(split.peag_note_texts)
    seq_len = len(split.peag_note_texts[0]) if n_samples > 0 else 0
    mask = np.zeros((n_samples, seq_len), dtype=np.float32)
    flat_texts: List[str] = []
    flat_indices: List[Tuple[int, int]] = []
    for i, sequence in enumerate(split.peag_note_texts):
        if len(sequence) != seq_len:
            raise ValueError('All PEAG note sequences must have the same length.')
        for t, text in enumerate(sequence):
            cleaned = str(text).strip()
            if cleaned:
                flat_texts.append(cleaned)
                flat_indices.append((i, t))
                mask[i, t] = 1.0

    if flat_texts:
        flat_embeddings = embedder.encode_texts(flat_texts, batch_size=batch_size, max_length=max_length).astype(np.float32)
        dim = flat_embeddings.shape[1]
    else:
        dim = 4096
        flat_embeddings = np.zeros((0, dim), dtype=np.float32)

    embeddings = np.zeros((n_samples, seq_len, dim), dtype=np.float32)
    for idx, vector in enumerate(flat_embeddings):
        i, t = flat_indices[idx]
        embeddings[i, t] = vector

    if cache_path is not None:
        np.savez_compressed(cache_path, embeddings=embeddings, mask=mask)
    return embeddings, mask
