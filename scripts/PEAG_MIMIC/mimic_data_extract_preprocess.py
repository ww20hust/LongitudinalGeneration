
"""
Extract data for PEAG-style experiments from MIMIC-III.

Outputs:
  data/with_notes/mortality_48h_4h/
    - cohort.csv
    - structured_ts.pkl   (ICUSTAY_ID -> (T, D) numpy array)
    - notes_ts.pkl        (ICUSTAY_ID -> list[str], length T)
    - diagnoses.pkl       (ICUSTAY_ID -> list[str])
    - meta.json

  data/with_notes/readmission_48h_4h/
    - cohort.csv
    - structured_ts.pkl   (ICUSTAY_ID -> (T, D) numpy array)
    - notes_ts.pkl        (ICUSTAY_ID -> list[str], length T)
    - diagnoses.pkl       (ICUSTAY_ID -> list[str])
    - labels.csv          (HADM_ID -> status/label)
    - meta.json
"""

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data_extraction import MIMICDataExtractor


VITAL_NAMES = [
    "heart_rate",
    "sbp",
    "dbp",
    "mean_bp",
    "resp_rate",
    "temperature",
    "spo2",
    "gcs",
]

LAB_NAMES = [
    "glucose",
    "potassium",
    "sodium",
    "chloride",
    "bicarbonate",
    "bun",
    "creatinine",
    "hemoglobin",
    "hematocrit",
    "wbc",
    "platelets",
    "lactate",
]


def _find_table_path(mimic_path: Path, table_name: str) -> Path:
    gz_path = mimic_path / f"{table_name}.csv.gz"
    csv_path = mimic_path / f"{table_name}.csv"
    if gz_path.exists():
        return gz_path
    if csv_path.exists():
        return csv_path
    raise FileNotFoundError(f"Table not found: {table_name} (.csv.gz or .csv)")


def _save_pickle(path: Path, obj: object) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return " ".join(str(value).split())


def _note_line(category: object, description: object, text: object) -> str:
    cat = _clean_text(category) or "UNKNOWN"
    desc = _clean_text(description) or "UNKNOWN"
    body = _clean_text(text)
    return f"CATEGORY: {cat}. DESCRIPTION: {desc}. TEXT: {body}"


def _create_time_series_stats(
    data: pd.DataFrame,
    value_col: str,
    name_col: str,
    names: list,
    hours: int,
    interval: int,
) -> np.ndarray:
    """
    Create fixed-length time series with per-bin statistics.
    Stats order: mean, median, max, min, count, variance (ddof=0).
    """
    num_timesteps = hours // interval
    stats_per_var = 6
    num_features = len(names) * stats_per_var
    ts = np.full((num_timesteps, num_features), np.nan, dtype=np.float32)

    if data is None or len(data) == 0:
        # count columns should be 0
        for i in range(len(names)):
            ts[:, i * stats_per_var + 4] = 0.0
        return ts
    if name_col not in data.columns:
        for i in range(len(names)):
            ts[:, i * stats_per_var + 4] = 0.0
        return ts

    for j, name in enumerate(names):
        subset = data[data[name_col] == name][["HOURS", value_col]].copy()
        if len(subset) == 0:
            ts[:, j * stats_per_var + 4] = 0.0
            continue
        subset["BIN"] = (subset["HOURS"] // interval).astype(int)
        subset = subset[subset["BIN"] < num_timesteps]
        if len(subset) == 0:
            ts[:, j * stats_per_var + 4] = 0.0
            continue

        grouped = subset.groupby("BIN")[value_col]
        for bin_idx, values in grouped:
            if bin_idx < 0 or bin_idx >= num_timesteps:
                continue
            vals = values.dropna().values.astype(float)
            if len(vals) == 0:
                continue
            base = j * stats_per_var
            ts[int(bin_idx), base + 0] = float(np.mean(vals))
            ts[int(bin_idx), base + 1] = float(np.median(vals))
            ts[int(bin_idx), base + 2] = float(np.max(vals))
            ts[int(bin_idx), base + 3] = float(np.min(vals))
            ts[int(bin_idx), base + 4] = float(len(vals))
            ts[int(bin_idx), base + 5] = float(np.var(vals, ddof=0))

        # Ensure empty bins have count=0
        count_col = ts[:, j * stats_per_var + 4]
        count_col[np.isnan(count_col)] = 0.0
        ts[:, j * stats_per_var + 4] = count_col

    return ts


def aggregate_time_series_hourly(
    events_dict: dict,
    names: list,
    hours: int,
    interval: int,
) -> dict:
    ts_dict = {}
    for icustay_id, df in tqdm(events_dict.items(), desc="Aggregating hourly series"):
        ts = _create_time_series_stats(
            data=df,
            value_col="VALUENUM",
            name_col="NAME",
            names=names,
            hours=hours,
            interval=interval,
        )
        ts_dict[int(icustay_id)] = ts
    return ts_dict


def _empty_stats_ts(num_timesteps: int, num_vars: int) -> np.ndarray:
    ts = np.full((num_timesteps, num_vars * 6), np.nan, dtype=np.float32)
    for i in range(num_vars):
        ts[:, i * 6 + 4] = 0.0
    return ts


def extract_notes_mortality(
    mimic_path: Path,
    cohort: pd.DataFrame,
    hours: int,
    interval: int,
    chunksize: int,
) -> dict:
    print(f"\nExtracting ICU notes (first {hours} hours, interval {interval}h)...")

    num_timesteps = hours // interval
    hadm_ids = set(cohort["HADM_ID"].values)
    cohort_keys = cohort[["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "INTIME"]].copy()

    notes_path = _find_table_path(mimic_path, "NOTEEVENTS")
    usecols = [
        "SUBJECT_ID",
        "HADM_ID",
        "CHARTDATE",
        "CHARTTIME",
        "CATEGORY",
        "DESCRIPTION",
        "ISERROR",
        "TEXT",
    ]

    buckets = defaultdict(lambda: [list() for _ in range(num_timesteps)])

    with tqdm(unit=" lines") as pbar:
        for chunk in pd.read_csv(
            notes_path,
            usecols=usecols,
            chunksize=chunksize,
            low_memory=False,
        ):
            chunk = chunk[chunk["HADM_ID"].isin(hadm_ids)]

            if "ISERROR" in chunk.columns:
                is_error = chunk["ISERROR"].astype(str).str.strip() == "1"
                chunk = chunk[~is_error]

            chunk = chunk.dropna(subset=["TEXT"])
            if len(chunk) == 0:
                pbar.update(chunksize)
                continue

            chart_ts = chunk["CHARTTIME"].copy()
            missing_time = chart_ts.isna()
            if missing_time.any():
                chart_ts[missing_time] = chunk.loc[missing_time, "CHARTDATE"]
            chunk["CHART_TS"] = pd.to_datetime(chart_ts, errors="coerce")

            chunk = chunk.merge(cohort_keys, on=["SUBJECT_ID", "HADM_ID"], how="left")
            chunk = chunk.dropna(subset=["ICUSTAY_ID", "INTIME", "CHART_TS"])
            if len(chunk) == 0:
                pbar.update(chunksize)
                continue

            hours_since = (
                chunk["CHART_TS"] - pd.to_datetime(chunk["INTIME"], errors="coerce")
            ).dt.total_seconds() / 3600.0
            chunk["HOURS"] = hours_since
            chunk = chunk[(chunk["HOURS"] >= 0) & (chunk["HOURS"] <= hours)]
            if len(chunk) == 0:
                pbar.update(chunksize)
                continue

            chunk["BIN"] = (chunk["HOURS"] // interval).astype(int)
            chunk = chunk[chunk["BIN"] < num_timesteps]
            if len(chunk) == 0:
                pbar.update(chunksize)
                continue

            for row in chunk.itertuples(index=False):
                icustay_id = int(row.ICUSTAY_ID)
                bin_idx = int(row.BIN)
                line = _note_line(row.CATEGORY, row.DESCRIPTION, row.TEXT)
                buckets[icustay_id][bin_idx].append((row.CHART_TS, line))

            pbar.update(chunksize)

    notes_dict = {}
    for icustay_id, bin_lists in buckets.items():
        out_bins = []
        for entries in bin_lists:
            if not entries:
                out_bins.append("")
                continue
            entries.sort(key=lambda x: x[0])
            out_bins.append("\n".join([e[1] for e in entries]))
        notes_dict[icustay_id] = out_bins

    empty_bins = [""] * num_timesteps
    for icustay_id in cohort["ICUSTAY_ID"].values:
        icustay_id = int(icustay_id)
        if icustay_id not in notes_dict:
            notes_dict[icustay_id] = list(empty_bins)

    print(f"  ICU stays with notes: {len(notes_dict)}")
    return notes_dict


def extract_readmission_labels(mimic_path: Path) -> pd.DataFrame:
    """
    Compute readmission labels and mark excluded admissions.
    Exclusions: in-hospital death, newborn admission, no ICU stay.
    """
    print("\nExtracting readmission labels...")

    admissions = pd.read_csv(
        _find_table_path(mimic_path, "ADMISSIONS"),
        usecols=[
            "SUBJECT_ID",
            "HADM_ID",
            "ADMITTIME",
            "DISCHTIME",
            "DEATHTIME",
            "HOSPITAL_EXPIRE_FLAG",
            "ADMISSION_TYPE",
        ],
        low_memory=False,
    )
    admissions["ADMITTIME"] = pd.to_datetime(admissions["ADMITTIME"], errors="coerce")
    admissions["DISCHTIME"] = pd.to_datetime(admissions["DISCHTIME"], errors="coerce")
    admissions["DEATHTIME"] = pd.to_datetime(admissions["DEATHTIME"], errors="coerce")

    # Compute readmission label (based on next admission within 30 days)
    admissions = admissions.sort_values(["SUBJECT_ID", "ADMITTIME"])
    admissions["NEXT_ADMITTIME"] = admissions.groupby("SUBJECT_ID")["ADMITTIME"].shift(-1)
    delta = admissions["NEXT_ADMITTIME"] - admissions["DISCHTIME"]
    admissions["READMIT_30D"] = (delta <= pd.Timedelta(days=30)) & delta.notna()
    admissions["READMIT_30D"] = admissions["READMIT_30D"].astype(int)

    # Status assignment
    status = pd.Series(["candidate"] * len(admissions), index=admissions.index)
    deaths = (admissions["HOSPITAL_EXPIRE_FLAG"].fillna(0).astype(int) == 1) | (
        admissions["DEATHTIME"].notna()
    )
    status[deaths] = "excluded"

    is_newborn = admissions["ADMISSION_TYPE"].astype(str).str.upper() == "NEWBORN"
    status[is_newborn] = "excluded"

    admissions["STATUS"] = status
    print(f"  Total admissions: {len(admissions)}")
    return admissions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract PEAG-style MIMIC-III data")
    parser.add_argument(
        "--mimic_path",
        type=str,
        default="/data2404/ww/data/physionet.org/files/mimiciii/1.4",
        help="Path to MIMIC-III data directory",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="./data/with_notes",
        help="Output root directory",
    )
    parser.add_argument("--mortality_hours", type=int, default=48)
    parser.add_argument("--mortality_interval", type=int, default=4)
    parser.add_argument("--notes_chunksize", type=int, default=200_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    mimic_path = Path(args.mimic_path)
    if not mimic_path.exists():
        raise FileNotFoundError(f"MIMIC-III path not found: {mimic_path}")

    output_root = Path(args.output_root)
    mortality_dir = output_root / "mortality_48h_4h"
    readmit_dir = output_root / "readmission_48h_4h"
    mortality_dir.mkdir(parents=True, exist_ok=True)
    readmit_dir.mkdir(parents=True, exist_ok=True)

    # Task A: ICU mortality (48h, 4h bins)
    print("\n=== ICU Mortality Extraction ===")
    extractor = MIMICDataExtractor(str(mimic_path))
    cohort_mort, vitals, labs, diagnoses = extractor.extract_all(
        hours=args.mortality_hours,
        min_age=18,
        min_los_hours=24,
        max_los_days=30,
    )

    vitals_ts = aggregate_time_series_hourly(
        events_dict=vitals,
        names=VITAL_NAMES,
        hours=args.mortality_hours,
        interval=args.mortality_interval,
    )
    labs_ts = aggregate_time_series_hourly(
        events_dict=labs,
        names=LAB_NAMES,
        hours=args.mortality_hours,
        interval=args.mortality_interval,
    )
    structured_ts_mort = {}
    num_timesteps = args.mortality_hours // args.mortality_interval
    for icustay_id in cohort_mort["ICUSTAY_ID"].values:
        icustay_id = int(icustay_id)
        v = vitals_ts.get(icustay_id)
        l = labs_ts.get(icustay_id)
        if v is None:
            v = _empty_stats_ts(num_timesteps, len(VITAL_NAMES))
        if l is None:
            l = _empty_stats_ts(num_timesteps, len(LAB_NAMES))
        structured_ts_mort[icustay_id] = np.concatenate([v, l], axis=1)

    notes_ts_mort = extract_notes_mortality(
        mimic_path=mimic_path,
        cohort=cohort_mort,
        hours=args.mortality_hours,
        interval=args.mortality_interval,
        chunksize=args.notes_chunksize,
    )

    cohort_mort.to_csv(mortality_dir / "cohort.csv", index=False)
    _save_pickle(mortality_dir / "structured_ts.pkl", structured_ts_mort)
    _save_pickle(mortality_dir / "notes_ts.pkl", notes_ts_mort)
    _save_pickle(mortality_dir / "diagnoses.pkl", diagnoses)
    with open(mortality_dir / "meta.json", "w") as f:
        json.dump(
            {
                "task": "mortality",
                "hours": args.mortality_hours,
                "interval_hours": args.mortality_interval,
                "num_timesteps": args.mortality_hours // args.mortality_interval,
                "vital_names": VITAL_NAMES,
                "lab_names": LAB_NAMES,
                "stat_order": ["mean", "median", "max", "min", "count", "variance"],
                "variance_ddof": 0,
            },
            f,
            indent=2,
        )

    # Task B: 30-day readmission (ICU first 48h, 4h bins)
    print("\n=== 30-Day Readmission Extraction (ICU 48h) ===")
    readmit_labels = extract_readmission_labels(mimic_path)

    # Keep candidate admissions for ICU extraction
    candidates = readmit_labels[readmit_labels["STATUS"] == "candidate"].copy()

    icustays = pd.read_csv(
        _find_table_path(mimic_path, "ICUSTAYS"),
        usecols=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "INTIME", "OUTTIME", "LOS"],
        low_memory=False,
    )
    icustays["INTIME"] = pd.to_datetime(icustays["INTIME"], errors="coerce")
    icustays["OUTTIME"] = pd.to_datetime(icustays["OUTTIME"], errors="coerce")

    icustays = icustays[icustays["HADM_ID"].isin(candidates["HADM_ID"].values)]
    icustays = icustays.sort_values(["HADM_ID", "INTIME"])
    first_icu = icustays.groupby("HADM_ID").first().reset_index()

    # Mark admissions without ICU stay as excluded
    has_icu = set(first_icu["HADM_ID"].values)
    no_icu_mask = candidates["HADM_ID"].apply(lambda x: x not in has_icu)
    readmit_labels.loc[candidates[no_icu_mask].index, "STATUS"] = "excluded"

    # Final readmission cohort: first ICU stay per admission
    cohort_readmit = candidates[~no_icu_mask].merge(
        first_icu, on=["HADM_ID", "SUBJECT_ID"], how="left"
    )
    cohort_readmit["READMIT_30D"] = cohort_readmit["READMIT_30D"].astype(int)

    # Update labels status to positive/negative for included admissions
    included_mask = readmit_labels["HADM_ID"].isin(cohort_readmit["HADM_ID"].values)
    readmit_labels.loc[
        included_mask & (readmit_labels["READMIT_30D"] == 1), "STATUS"
    ] = "positive"
    readmit_labels.loc[
        included_mask & (readmit_labels["READMIT_30D"] == 0), "STATUS"
    ] = "negative"

    # Extract vitals/labs and aggregate with stats
    vitals_r, labs_r, diagnoses_r = None, None, None
    if len(cohort_readmit) > 0:
        vitals_r = extractor.extract_vitals(cohort_readmit, hours=args.mortality_hours)
        labs_r = extractor.extract_labs(cohort_readmit, hours=args.mortality_hours)
        diagnoses_r = extractor.extract_diagnoses(cohort_readmit)
    else:
        vitals_r, labs_r, diagnoses_r = {}, {}, {}

    vitals_ts_r = aggregate_time_series_hourly(
        events_dict=vitals_r,
        names=VITAL_NAMES,
        hours=args.mortality_hours,
        interval=args.mortality_interval,
    )
    labs_ts_r = aggregate_time_series_hourly(
        events_dict=labs_r,
        names=LAB_NAMES,
        hours=args.mortality_hours,
        interval=args.mortality_interval,
    )

    structured_ts_readmit = {}
    for icustay_id in cohort_readmit["ICUSTAY_ID"].values:
        icustay_id = int(icustay_id)
        v = vitals_ts_r.get(icustay_id)
        l = labs_ts_r.get(icustay_id)
        if v is None:
            v = _empty_stats_ts(num_timesteps, len(VITAL_NAMES))
        if l is None:
            l = _empty_stats_ts(num_timesteps, len(LAB_NAMES))
        structured_ts_readmit[icustay_id] = np.concatenate([v, l], axis=1)

    notes_ts_readmit = extract_notes_mortality(
        mimic_path=mimic_path,
        cohort=cohort_readmit,
        hours=args.mortality_hours,
        interval=args.mortality_interval,
        chunksize=args.notes_chunksize,
    )

    # Save outputs
    cohort_readmit.to_csv(readmit_dir / "cohort.csv", index=False)
    readmit_labels.to_csv(readmit_dir / "labels.csv", index=False)
    _save_pickle(readmit_dir / "structured_ts.pkl", structured_ts_readmit)
    _save_pickle(readmit_dir / "notes_ts.pkl", notes_ts_readmit)
    _save_pickle(readmit_dir / "diagnoses.pkl", diagnoses_r)
    with open(readmit_dir / "meta.json", "w") as f:
        json.dump(
            {
                "task": "readmission_30d",
                "hours": args.mortality_hours,
                "interval_hours": args.mortality_interval,
                "num_timesteps": args.mortality_hours // args.mortality_interval,
                "vital_names": VITAL_NAMES,
                "lab_names": LAB_NAMES,
                "stat_order": ["mean", "median", "max", "min", "count", "variance"],
                "variance_ddof": 0,
                "label_file": "labels.csv",
            },
            f,
            indent=2,
        )

    print("\nDone.")
    print(f"  Mortality output: {mortality_dir}")
    print(f"  Readmission output: {readmit_dir}")


if __name__ == "__main__":
    main()
