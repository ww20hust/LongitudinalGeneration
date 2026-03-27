from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))



EXPECTED_TASK_TAGS = {
    'mortality': 'mortality',
    'readmission': 'readmission',
}


def save_json(path: str | Path, payload: Dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)


def save_pickle(path: str | Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Prepare MIMIC-III binary benchmark splits for PEAG and baselines.')
    parser.add_argument('--extracted_dir', type=str, required=True, help='Directory produced by mimic_data_extract_preprocess.py.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory for train/valid/test prepared splits.')
    parser.add_argument('--task', type=str, choices=['mortality', 'readmission'], required=True)
    parser.add_argument('--train_size', type=float, default=0.7, help='Training fraction.')
    parser.add_argument('--valid_size', type=float, default=0.1, help='Validation fraction.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    return parser.parse_args()


def _load_pickle(path: Path):
    with path.open('rb') as handle:
        return pickle.load(handle)


def _load_json(path: Path) -> Dict[str, object]:
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def _validate_task(extracted_task: str, requested_task: str) -> None:
    expected_tag = EXPECTED_TASK_TAGS[requested_task]
    if expected_tag not in str(extracted_task).lower():
        raise ValueError(
            f'Extracted data task={extracted_task!r} does not match requested --task={requested_task!r}. '
            'Please point --extracted_dir to the matching extracted dataset.'
        )


def load_extracted_inputs(extracted_dir: Path, requested_task: str) -> Tuple[pd.DataFrame, Dict[int, np.ndarray], Dict[int, List[str]], Dict[str, object], pd.DataFrame | None]:
    cohort_path = extracted_dir / 'cohort.csv'
    structured_path = extracted_dir / 'structured_ts.pkl'
    notes_path = extracted_dir / 'notes_ts.pkl'
    meta_path = extracted_dir / 'meta.json'

    cohort = pd.read_csv(cohort_path)
    structured_ts = _load_pickle(structured_path)
    notes_ts = _load_pickle(notes_path)
    meta = _load_json(meta_path)
    _validate_task(str(meta.get('task', '')), requested_task)

    for column in ['INTIME', 'OUTTIME', 'DISCHTIME', 'ADMITTIME', 'DEATHTIME']:
        if column in cohort.columns:
            cohort[column] = pd.to_datetime(cohort[column], errors='coerce')

    labels = None
    labels_path = extracted_dir / 'labels.csv'
    if labels_path.exists():
        labels = pd.read_csv(labels_path)

    return cohort, structured_ts, notes_ts, meta, labels


def build_feature_names(meta: Dict[str, object], structured_example: np.ndarray) -> List[str]:
    vital_names = [str(name) for name in meta.get('vital_names', [])]
    lab_names = [str(name) for name in meta.get('lab_names', [])]
    stat_order = [str(name) for name in meta.get('stat_order', [])]

    if not vital_names and not lab_names:
        raise ValueError('meta.json must contain vital_names and lab_names produced by mimic_data_extract_preprocess.py.')
    if not stat_order:
        raise ValueError('meta.json must contain stat_order produced by mimic_data_extract_preprocess.py.')

    base_names = vital_names + lab_names
    feature_names = [f'{base_name}__{stat_name}' for base_name in base_names for stat_name in stat_order]
    if structured_example.shape[1] != len(feature_names):
        raise ValueError(
            f'Feature count mismatch: structured sequence has {structured_example.shape[1]} columns but meta defines {len(feature_names)}.'
        )
    return feature_names


def build_document_text(note_sequence: List[str]) -> str:
    return '\n'.join(text for text in note_sequence if str(text).strip())


def build_mortality_payload(
    cohort: pd.DataFrame,
    structured_ts: Dict[int, np.ndarray],
    notes_ts: Dict[int, List[str]],
    meta: Dict[str, object],
) -> Dict[str, object]:
    cohort = cohort.copy()
    cohort['LABEL'] = cohort['MORTALITY'].astype(np.float32)

    if not structured_ts:
        raise ValueError('structured_ts.pkl is empty.')
    first_sequence = np.asarray(next(iter(structured_ts.values())), dtype=np.float32)
    num_timesteps = int(meta['num_timesteps'])
    feature_names = build_feature_names(meta, first_sequence)
    n_features = len(feature_names)

    labs = np.zeros((len(cohort), num_timesteps, n_features), dtype=np.float32)
    lab_mask = np.zeros((len(cohort), num_timesteps, n_features), dtype=np.float32)
    peag_note_texts: List[List[str]] = []
    document_texts: List[str] = []
    patient_ids: List[str] = []
    subject_ids: List[int] = []
    hadm_ids: List[int] = []
    icustay_ids: List[int] = []

    for row_index, row in enumerate(cohort.itertuples(index=False)):
        icustay_id = int(getattr(row, 'ICUSTAY_ID'))
        seq = structured_ts.get(icustay_id)
        if seq is None:
            seq = np.full((num_timesteps, n_features), np.nan, dtype=np.float32)
        seq = np.asarray(seq, dtype=np.float32)
        if seq.shape != (num_timesteps, n_features):
            raise ValueError(f'Unexpected structured sequence shape for ICUSTAY_ID={icustay_id}: {seq.shape}')
        mask = np.isfinite(seq).astype(np.float32)
        labs[row_index] = np.where(np.isfinite(seq), seq, 0.0)
        lab_mask[row_index] = mask

        note_sequence = [str(text) for text in notes_ts.get(icustay_id, [''] * num_timesteps)]
        if len(note_sequence) != num_timesteps:
            raise ValueError(f'Unexpected note sequence length for ICUSTAY_ID={icustay_id}: {len(note_sequence)}')
        peag_note_texts.append(note_sequence)
        document_texts.append(build_document_text(note_sequence))

        patient_ids.append(f'icu_{icustay_id}')
        subject_ids.append(int(getattr(row, 'SUBJECT_ID')))
        hadm_ids.append(int(getattr(row, 'HADM_ID')))
        icustay_ids.append(icustay_id)

    return {
        'labels': cohort['LABEL'].to_numpy(dtype=np.float32),
        'labs': labs,
        'lab_mask': lab_mask,
        'peag_note_texts': peag_note_texts,
        'document_texts': document_texts,
        'patient_ids': patient_ids,
        'subject_ids': subject_ids,
        'hadm_ids': hadm_ids,
        'icustay_ids': icustay_ids,
        'feature_names': feature_names,
        'num_timesteps': num_timesteps,
        'document_scope': 'first_48h_notes',
        'peag_scope': 'first_48h_4h_bins',
    }


def build_readmission_payload(
    cohort: pd.DataFrame,
    structured_ts: Dict[int, np.ndarray],
    notes_ts: Dict[int, List[str]],
    meta: Dict[str, object],
    labels_df: pd.DataFrame | None,
) -> Dict[str, object]:
    if labels_df is None:
        raise ValueError('labels.csv is required for readmission benchmark preparation.')

    eligible = labels_df[labels_df['STATUS'].isin(['positive', 'negative'])][['HADM_ID', 'READMIT_30D']].copy()
    eligible['HADM_ID'] = eligible['HADM_ID'].astype(int)
    eligible['READMIT_30D'] = eligible['READMIT_30D'].astype(np.float32)

    cohort = cohort.copy()
    cohort['HADM_ID'] = cohort['HADM_ID'].astype(int)
    cohort = cohort.merge(eligible.rename(columns={'READMIT_30D': 'LABEL'}), on='HADM_ID', how='inner')
    if cohort.empty:
        raise ValueError('No eligible readmission samples remained after merging labels.csv with cohort.csv.')

    if not structured_ts:
        raise ValueError('structured_ts.pkl is empty.')
    first_sequence = np.asarray(next(iter(structured_ts.values())), dtype=np.float32)
    num_timesteps = int(meta['num_timesteps'])
    feature_names = build_feature_names(meta, first_sequence)
    n_features = len(feature_names)

    labs = np.zeros((len(cohort), num_timesteps, n_features), dtype=np.float32)
    lab_mask = np.zeros((len(cohort), num_timesteps, n_features), dtype=np.float32)
    peag_note_texts: List[List[str]] = []
    document_texts: List[str] = []
    patient_ids: List[str] = []
    subject_ids: List[int] = []
    hadm_ids: List[int] = []
    icustay_ids: List[int] = []

    for row_index, row in enumerate(cohort.itertuples(index=False)):
        icustay_id = int(getattr(row, 'ICUSTAY_ID'))
        seq = structured_ts.get(icustay_id)
        if seq is None:
            seq = np.full((num_timesteps, n_features), np.nan, dtype=np.float32)
        seq = np.asarray(seq, dtype=np.float32)
        if seq.shape != (num_timesteps, n_features):
            raise ValueError(f'Unexpected structured sequence shape for ICUSTAY_ID={icustay_id}: {seq.shape}')
        mask = np.isfinite(seq).astype(np.float32)
        labs[row_index] = np.where(np.isfinite(seq), seq, 0.0)
        lab_mask[row_index] = mask

        note_sequence = [str(text) for text in notes_ts.get(icustay_id, [''] * num_timesteps)]
        if len(note_sequence) != num_timesteps:
            raise ValueError(f'Unexpected note sequence length for ICUSTAY_ID={icustay_id}: {len(note_sequence)}')
        peag_note_texts.append(note_sequence)
        document_texts.append(build_document_text(note_sequence))

        patient_ids.append(f'icu_{icustay_id}')
        subject_ids.append(int(getattr(row, 'SUBJECT_ID')))
        hadm_ids.append(int(getattr(row, 'HADM_ID')))
        icustay_ids.append(icustay_id)

    return {
        'labels': cohort['LABEL'].to_numpy(dtype=np.float32),
        'labs': labs,
        'lab_mask': lab_mask,
        'peag_note_texts': peag_note_texts,
        'document_texts': document_texts,
        'patient_ids': patient_ids,
        'subject_ids': subject_ids,
        'hadm_ids': hadm_ids,
        'icustay_ids': icustay_ids,
        'feature_names': feature_names,
        'num_timesteps': num_timesteps,
        'document_scope': 'first_48h_notes',
        'peag_scope': 'first_48h_4h_bins',
    }


def train_valid_test_indices(labels: np.ndarray, train_size: float, valid_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(labels.shape[0])
    train_idx, temp_idx, _, y_temp = train_test_split(
        indices,
        labels,
        train_size=train_size,
        random_state=seed,
        stratify=labels,
    )
    relative_valid = valid_size / (1.0 - train_size)
    valid_idx, test_idx = train_test_split(
        temp_idx,
        train_size=relative_valid,
        random_state=seed,
        stratify=y_temp,
    )
    return np.sort(train_idx), np.sort(valid_idx), np.sort(test_idx)


def fit_scaler(labs: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n_features = labs.shape[-1]
    means = np.zeros(n_features, dtype=np.float32)
    stds = np.ones(n_features, dtype=np.float32)
    for feature_index in range(n_features):
        observed = labs[:, :, feature_index][mask[:, :, feature_index] > 0]
        if observed.size == 0:
            continue
        means[feature_index] = float(observed.mean())
        std = float(observed.std())
        stds[feature_index] = std if std >= 1e-6 else 1.0
    return means, stds


def apply_scaler(labs: np.ndarray, mask: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    scaled = (labs - means.reshape(1, 1, -1)) / stds.reshape(1, 1, -1)
    scaled[mask == 0] = 0.0
    return scaled.astype(np.float32)


def subset_payload(payload: Dict[str, object], indices: np.ndarray, meta: Dict[str, object]) -> Dict[str, object]:
    return {
        'labels': np.asarray(payload['labels'])[indices],
        'labs': np.asarray(payload['labs'])[indices],
        'lab_mask': np.asarray(payload['lab_mask'])[indices],
        'peag_note_texts': [payload['peag_note_texts'][int(i)] for i in indices],
        'document_texts': [payload['document_texts'][int(i)] for i in indices],
        'patient_ids': [payload['patient_ids'][int(i)] for i in indices],
        'subject_ids': [payload['subject_ids'][int(i)] for i in indices],
        'hadm_ids': [payload['hadm_ids'][int(i)] for i in indices],
        'icustay_ids': [payload['icustay_ids'][int(i)] for i in indices],
        'meta': meta,
    }


def main() -> None:
    args = parse_args()
    extracted_dir = Path(args.extracted_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cohort, structured_ts, notes_ts, extracted_meta, labels_df = load_extracted_inputs(extracted_dir, args.task)
    if args.task == 'mortality':
        payload = build_mortality_payload(cohort, structured_ts, notes_ts, extracted_meta)
    else:
        payload = build_readmission_payload(cohort, structured_ts, notes_ts, extracted_meta, labels_df)

    train_idx, valid_idx, test_idx = train_valid_test_indices(payload['labels'], args.train_size, args.valid_size, args.seed)
    train_labs = np.asarray(payload['labs'])[train_idx]
    train_mask = np.asarray(payload['lab_mask'])[train_idx]
    means, stds = fit_scaler(train_labs, train_mask)
    payload['labs'] = apply_scaler(np.asarray(payload['labs']), np.asarray(payload['lab_mask']), means, stds)

    split_meta = {
        'task': args.task,
        'feature_names': payload['feature_names'],
        'num_timesteps': int(payload['num_timesteps']),
        'num_features': len(payload['feature_names']),
        'document_scope': payload['document_scope'],
        'peag_scope': payload['peag_scope'],
        'train_size': args.train_size,
        'valid_size': args.valid_size,
        'seed': args.seed,
        'source_extracted_dir': str(extracted_dir),
        'source_task': extracted_meta.get('task', ''),
        'scaler': {'mean': means.tolist(), 'std': stds.tolist()},
    }

    train_payload = subset_payload(payload, train_idx, split_meta)
    valid_payload = subset_payload(payload, valid_idx, split_meta)
    test_payload = subset_payload(payload, test_idx, split_meta)

    save_pickle(output_dir / 'train.pkl', train_payload)
    save_pickle(output_dir / 'valid.pkl', valid_payload)
    save_pickle(output_dir / 'test.pkl', test_payload)
    save_json(output_dir / 'meta.json', {
        **split_meta,
        'class_balance': {
            'train_positive_rate': float(np.mean(train_payload['labels'])),
            'valid_positive_rate': float(np.mean(valid_payload['labels'])),
            'test_positive_rate': float(np.mean(test_payload['labels'])),
        },
        'num_samples': {
            'train': int(len(train_payload['labels'])),
            'valid': int(len(valid_payload['labels'])),
            'test': int(len(test_payload['labels'])),
        },
    })

    print(f'Prepared {args.task} benchmark in {output_dir}')
    print(f"  train={len(train_payload['labels'])} valid={len(valid_payload['labels'])} test={len(test_payload['labels'])}")
    print(f"  positive rates: train={np.mean(train_payload['labels']):.4f}, valid={np.mean(valid_payload['labels']):.4f}, test={np.mean(test_payload['labels']):.4f}")


if __name__ == '__main__':
    main()
