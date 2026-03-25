from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import save_json, save_pickle


LAB_TESTS = {
    'glucose': [50931, 50809],
    'potassium': [50971, 50822],
    'sodium': [50983, 50824],
    'chloride': [50902, 50806],
    'bicarbonate': [50882, 50803],
    'bun': [51006],
    'creatinine': [50912],
    'hemoglobin': [51222, 50811],
    'hematocrit': [51221, 50810],
    'wbc': [51301, 51300],
    'platelets': [51265],
    'lactate': [50813],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Prepare MIMIC-III binary benchmark splits for PEAG and baselines.')
    parser.add_argument('--extracted_dir', type=str, required=True, help='Directory produced by extract_with_notes.py.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory for train/valid/test prepared splits.')
    parser.add_argument('--task', type=str, choices=['mortality', 'readmission'], default='mortality')
    parser.add_argument('--mimic_path', type=str, default=None, help='Credentialed MIMIC-III root; required for readmission preparation.')
    parser.add_argument('--train_size', type=float, default=0.7, help='Training fraction.')
    parser.add_argument('--valid_size', type=float, default=0.1, help='Validation fraction.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--chunksize', type=int, default=200000, help='CSV chunk size for raw MIMIC extraction.')
    return parser.parse_args()


def _load_pickle(path: Path):
    with path.open('rb') as handle:
        return pickle.load(handle)


def _find_table_path(mimic_path: Path, table_name: str) -> Path:
    gz_path = mimic_path / f'{table_name}.csv.gz'
    csv_path = mimic_path / f'{table_name}.csv'
    if gz_path.exists():
        return gz_path
    if csv_path.exists():
        return csv_path
    raise FileNotFoundError(f'Table not found: {table_name} (.csv.gz or .csv)')


def _clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ''
    return ' '.join(str(value).split())


def _note_line(category: object, description: object, text: object) -> str:
    cat = _clean_text(category) or 'UNKNOWN'
    desc = _clean_text(description) or 'UNKNOWN'
    body = _clean_text(text)
    return f'CATEGORY: {cat}. DESCRIPTION: {desc}. TEXT: {body}'


def _build_predischarge_note_frame(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk = chunk.copy()
    chunk['CHARTTIME_TS'] = pd.to_datetime(chunk['CHARTTIME'], errors='coerce')
    chunk['CHARTDATE_TS'] = pd.to_datetime(chunk['CHARTDATE'], errors='coerce')
    chunk['HAS_CHARTTIME'] = chunk['CHARTTIME_TS'].notna()
    approx_ts = chunk['CHARTDATE_TS'] + pd.to_timedelta(12, unit='h')
    chunk['CHART_TS'] = chunk['CHARTTIME_TS'].where(chunk['HAS_CHARTTIME'], approx_ts)
    chunk = chunk.dropna(subset=['CHART_TS'])
    dischtime = pd.to_datetime(chunk['DISCHTIME'], errors='coerce')
    discharge_date = dischtime.dt.normalize()
    keep_exact = chunk['HAS_CHARTTIME'] & (chunk['CHART_TS'] <= dischtime)
    keep_date_only = (~chunk['HAS_CHARTTIME']) & chunk['CHARTDATE_TS'].notna() & (chunk['CHARTDATE_TS'] < discharge_date)
    return chunk[keep_exact | keep_date_only].copy()


def load_extracted_inputs(extracted_dir: Path) -> Tuple[pd.DataFrame, Dict[int, np.ndarray], Dict[int, List[str]], Dict[str, object]]:
    cohort = pd.read_csv(extracted_dir / 'cohort.csv')
    labs_ts = _load_pickle(extracted_dir / 'labs_ts.pkl')
    notes = _load_pickle(extracted_dir / 'notes.pkl')
    with (extracted_dir / 'meta.json').open('r', encoding='utf-8') as handle:
        meta = json.load(handle)
    cohort['INTIME'] = pd.to_datetime(cohort['INTIME'], errors='coerce')
    cohort['DISCHTIME'] = pd.to_datetime(cohort['DISCHTIME'], errors='coerce')
    return cohort, labs_ts, notes, meta


def compute_readmission_labels(cohort: pd.DataFrame, mimic_path: Path) -> pd.Series:
    admissions_path = _find_table_path(mimic_path, 'ADMISSIONS')
    admissions = pd.read_csv(
        admissions_path,
        usecols=['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME'],
        low_memory=False,
    )
    admissions['ADMITTIME'] = pd.to_datetime(admissions['ADMITTIME'], errors='coerce')
    admissions['DISCHTIME'] = pd.to_datetime(admissions['DISCHTIME'], errors='coerce')
    admissions = admissions.sort_values(['SUBJECT_ID', 'ADMITTIME'])
    grouped = {int(subject_id): frame.reset_index(drop=True) for subject_id, frame in admissions.groupby('SUBJECT_ID')}

    labels = []
    for row in cohort.itertuples(index=False):
        if int(getattr(row, 'MORTALITY')) == 1:
            labels.append(np.nan)
            continue
        dischtime = pd.to_datetime(getattr(row, 'DISCHTIME'), errors='coerce')
        if pd.isna(dischtime):
            labels.append(np.nan)
            continue
        subject_admissions = grouped.get(int(getattr(row, 'SUBJECT_ID')))
        if subject_admissions is None:
            labels.append(0.0)
            continue
        future = subject_admissions[
            (subject_admissions['HADM_ID'] != int(getattr(row, 'HADM_ID'))) &
            (subject_admissions['ADMITTIME'] > dischtime)
        ]
        if future.empty:
            labels.append(0.0)
            continue
        delta_days = (future.iloc[0]['ADMITTIME'] - dischtime).total_seconds() / 86400.0
        labels.append(1.0 if delta_days <= 30.0 else 0.0)
    return pd.Series(labels, index=cohort.index, dtype='float64')


def build_mortality_payload(cohort: pd.DataFrame, labs_ts: Dict[int, np.ndarray], notes: Dict[int, List[str]], meta: Dict[str, object]) -> Dict[str, object]:
    cohort = cohort.copy()
    cohort['LABEL'] = cohort['MORTALITY'].astype(np.float32)
    num_timesteps = int(meta['num_timesteps'])
    lab_names = [str(name) for name in meta.get('lab_names', [])]
    n_features = len(lab_names)

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
        seq = labs_ts.get(icustay_id)
        if seq is None:
            seq = np.full((num_timesteps, n_features), np.nan, dtype=np.float32)
        seq = np.asarray(seq, dtype=np.float32)
        if seq.shape != (num_timesteps, n_features):
            raise ValueError(f'Unexpected lab sequence shape for ICUSTAY_ID={icustay_id}: {seq.shape}')
        mask = np.isfinite(seq).astype(np.float32)
        labs[row_index] = np.where(np.isfinite(seq), seq, 0.0)
        lab_mask[row_index] = mask

        note_sequence = [str(text) for text in notes.get(icustay_id, [''] * num_timesteps)]
        if len(note_sequence) != num_timesteps:
            raise ValueError(f'Unexpected note sequence length for ICUSTAY_ID={icustay_id}: {len(note_sequence)}')
        peag_note_texts.append(note_sequence)
        document_texts.append('\n'.join(text for text in note_sequence if str(text).strip()))

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
        'feature_names': lab_names,
        'num_timesteps': num_timesteps,
        'document_scope': 'first_48h_notes',
        'peag_scope': 'first_48h_4h_bins',
    }


def extract_readmission_windows(cohort: pd.DataFrame, mimic_path: Path, lab_names: List[str], chunksize: int) -> Tuple[Dict[int, np.ndarray], Dict[int, List[str]], Dict[int, str]]:
    cohort_keys = cohort[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'DISCHTIME']].copy()
    hadm_ids = set(int(x) for x in cohort_keys['HADM_ID'].tolist())
    lab_itemid_to_name = {itemid: lab_name for lab_name, itemids in LAB_TESTS.items() for itemid in itemids}
    lab_name_to_index = {lab_name: idx for idx, lab_name in enumerate(lab_names)}

    n_days = 7
    lab_sums = defaultdict(lambda: np.zeros((n_days, len(lab_names)), dtype=np.float64))
    lab_counts = defaultdict(lambda: np.zeros((n_days, len(lab_names)), dtype=np.float64))

    labevents_path = _find_table_path(mimic_path, 'LABEVENTS')
    for chunk in pd.read_csv(
        labevents_path,
        usecols=['SUBJECT_ID', 'HADM_ID', 'ITEMID', 'CHARTTIME', 'VALUENUM'],
        chunksize=chunksize,
        low_memory=False,
    ):
        chunk = chunk[chunk['HADM_ID'].isin(hadm_ids) & chunk['ITEMID'].isin(lab_itemid_to_name.keys())]
        chunk = chunk.dropna(subset=['VALUENUM', 'CHARTTIME'])
        if chunk.empty:
            continue
        chunk['CHARTTIME'] = pd.to_datetime(chunk['CHARTTIME'], errors='coerce')
        chunk = chunk.dropna(subset=['CHARTTIME'])
        chunk['LAB_NAME'] = chunk['ITEMID'].map(lab_itemid_to_name)
        chunk = chunk.merge(cohort_keys, on=['SUBJECT_ID', 'HADM_ID'], how='left')
        chunk = chunk.dropna(subset=['ICUSTAY_ID', 'DISCHTIME', 'LAB_NAME'])
        delta_days = (pd.to_datetime(chunk['DISCHTIME']) - chunk['CHARTTIME']).dt.total_seconds() / 86400.0
        chunk['DELTA_DAYS'] = delta_days
        chunk = chunk[(chunk['DELTA_DAYS'] >= 0.0) & (chunk['DELTA_DAYS'] < 7.0)]
        if chunk.empty:
            continue
        chunk['SEQ_INDEX'] = 6 - np.floor(chunk['DELTA_DAYS']).astype(int)
        for row in chunk.itertuples(index=False):
            icustay_id = int(row.ICUSTAY_ID)
            feature_index = lab_name_to_index[str(row.LAB_NAME)]
            seq_index = int(row.SEQ_INDEX)
            value = float(row.VALUENUM)
            lab_sums[icustay_id][seq_index, feature_index] += value
            lab_counts[icustay_id][seq_index, feature_index] += 1.0

    lab_sequences: Dict[int, np.ndarray] = {}
    for icustay_id in cohort['ICUSTAY_ID'].tolist():
        icustay_id = int(icustay_id)
        sums = lab_sums[icustay_id]
        counts = lab_counts[icustay_id]
        seq = np.full((n_days, len(lab_names)), np.nan, dtype=np.float32)
        observed = counts > 0
        seq[observed] = (sums[observed] / counts[observed]).astype(np.float32)
        lab_sequences[icustay_id] = seq

    noteevents_path = _find_table_path(mimic_path, 'NOTEEVENTS')
    document_entries = defaultdict(list)
    day_entries = defaultdict(lambda: [list() for _ in range(n_days)])
    for chunk in pd.read_csv(
        noteevents_path,
        usecols=['SUBJECT_ID', 'HADM_ID', 'CHARTDATE', 'CHARTTIME', 'CATEGORY', 'DESCRIPTION', 'ISERROR', 'TEXT'],
        chunksize=chunksize,
        low_memory=False,
    ):
        chunk = chunk[chunk['HADM_ID'].isin(hadm_ids)]
        if 'ISERROR' in chunk.columns:
            chunk = chunk[chunk['ISERROR'].astype(str).str.strip() != '1']
        chunk = chunk.dropna(subset=['TEXT'])
        if chunk.empty:
            continue

        chunk = chunk.merge(cohort_keys, on=['SUBJECT_ID', 'HADM_ID'], how='left')
        chunk = chunk.dropna(subset=['ICUSTAY_ID', 'DISCHTIME'])
        chunk = _build_predischarge_note_frame(chunk)
        if chunk.empty:
            continue

        delta_days = (pd.to_datetime(chunk['DISCHTIME']) - chunk['CHART_TS']).dt.total_seconds() / 86400.0
        chunk['DELTA_DAYS'] = delta_days
        for row in chunk.itertuples(index=False):
            icustay_id = int(row.ICUSTAY_ID)
            line = _note_line(row.CATEGORY, row.DESCRIPTION, row.TEXT)
            document_entries[icustay_id].append((row.CHART_TS, line))
            if 0.0 <= float(row.DELTA_DAYS) < 7.0:
                seq_index = 6 - int(math.floor(float(row.DELTA_DAYS)))
                day_entries[icustay_id][seq_index].append((row.CHART_TS, line))

    peag_note_sequences: Dict[int, List[str]] = {}
    document_texts: Dict[int, str] = {}
    for icustay_id in cohort['ICUSTAY_ID'].tolist():
        icustay_id = int(icustay_id)
        doc_entries = sorted(document_entries[icustay_id], key=lambda item: item[0])
        document_texts[icustay_id] = '\n'.join(text for _, text in doc_entries)

        per_day_texts: List[str] = []
        for entries in day_entries[icustay_id]:
            entries = sorted(entries, key=lambda item: item[0])
            per_day_texts.append('\n'.join(text for _, text in entries))
        peag_note_sequences[icustay_id] = per_day_texts

    return lab_sequences, peag_note_sequences, document_texts


def build_readmission_payload(cohort: pd.DataFrame, mimic_path: Path, chunksize: int) -> Dict[str, object]:
    cohort = cohort.copy()
    cohort['LABEL'] = compute_readmission_labels(cohort, mimic_path)
    cohort = cohort.dropna(subset=['LABEL']).reset_index(drop=True)
    cohort['LABEL'] = cohort['LABEL'].astype(np.float32)

    lab_names = list(LAB_TESTS.keys())
    lab_sequences, peag_note_sequences, document_texts = extract_readmission_windows(cohort, mimic_path, lab_names, chunksize)
    n_days = 7
    n_features = len(lab_names)

    labs = np.zeros((len(cohort), n_days, n_features), dtype=np.float32)
    lab_mask = np.zeros((len(cohort), n_days, n_features), dtype=np.float32)
    peag_note_texts: List[List[str]] = []
    document_text_list: List[str] = []
    patient_ids: List[str] = []
    subject_ids: List[int] = []
    hadm_ids: List[int] = []
    icustay_ids: List[int] = []

    for row_index, row in enumerate(cohort.itertuples(index=False)):
        icustay_id = int(getattr(row, 'ICUSTAY_ID'))
        seq = np.asarray(lab_sequences[icustay_id], dtype=np.float32)
        mask = np.isfinite(seq).astype(np.float32)
        labs[row_index] = np.where(np.isfinite(seq), seq, 0.0)
        lab_mask[row_index] = mask
        peag_note_texts.append(list(peag_note_sequences[icustay_id]))
        document_text_list.append(str(document_texts[icustay_id]))
        patient_ids.append(f'icu_{icustay_id}')
        subject_ids.append(int(getattr(row, 'SUBJECT_ID')))
        hadm_ids.append(int(getattr(row, 'HADM_ID')))
        icustay_ids.append(icustay_id)

    return {
        'labels': cohort['LABEL'].to_numpy(dtype=np.float32),
        'labs': labs,
        'lab_mask': lab_mask,
        'peag_note_texts': peag_note_texts,
        'document_texts': document_text_list,
        'patient_ids': patient_ids,
        'subject_ids': subject_ids,
        'hadm_ids': hadm_ids,
        'icustay_ids': icustay_ids,
        'feature_names': lab_names,
        'num_timesteps': n_days,
        'document_scope': 'all_predischarge_notes',
        'peag_scope': 'last_week_daily_bins',
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

    cohort, labs_ts, notes, extracted_meta = load_extracted_inputs(extracted_dir)
    if args.task == 'mortality':
        payload = build_mortality_payload(cohort, labs_ts, notes, extracted_meta)
    else:
        if args.mimic_path is None:
            raise ValueError('--mimic_path is required for readmission preparation.')
        payload = build_readmission_payload(cohort, Path(args.mimic_path), args.chunksize)

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
