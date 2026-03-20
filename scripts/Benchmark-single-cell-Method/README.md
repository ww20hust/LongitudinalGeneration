# Static Single-Cell Baselines for Clinical Metabolomics Imputation

This folder adapts `MIDAS`, `scVAEIT`, and `StabMap` to the PEAG metabolomics-imputation benchmark using a single longitudinal CSV.

## Benchmark Protocol

All methods use the same patient-level train/test split.

- Training patients: every visit is treated as an independent paired sample `(lab, metabolomics)`.
- Test patients: only `visit=1` is evaluated.
- Test input: `visit=1` lab features only.
- Test target: the same patient's `visit=1` metabolomics profile.
- Static baselines never use baseline history or any historical-state variable.

This matches the manuscript framing: PEAG is the longitudinal model; MIDAS, scVAEIT, and StabMap are adapted static multimodal baselines.

## Expected CSV Format

The input CSV should contain:

- `eid`: patient identifier
- `visit`: visit index (`0` or `1`)
- `61` routine lab columns
- `251` metabolomics columns

Columns can be provided explicitly with `--lab-columns` and `--metab-columns`, inferred by prefix, or inferred automatically as the first `61` non-metadata columns followed by the next `251` columns.

## External Dependencies

`MIDAS`, `scVAEIT`, and `StabMap` are not vendored into this repository. The benchmark code expects them to exist under:

- `scripts/Benchmark-single-cell-Method/baseline-model/scVAEIT`
- `scripts/Benchmark-single-cell-Method/baseline-model/midas`
- `scripts/Benchmark-single-cell-Method/baseline-model/StabMap`

From the repository root, run:

```bash
cd scripts/Benchmark-single-cell-Method
mkdir -p baseline-model

git clone https://github.com/jaydu1/scVAEIT.git baseline-model/scVAEIT
git clone https://github.com/labomics/midas.git baseline-model/midas
git clone https://github.com/MarioniLab/StabMap.git baseline-model/StabMap
```

These clone locations are required because the Python adapters and the StabMap R wrapper resolve the external code through those exact relative paths.

## Scripts

- `prepare_tabular_benchmark.py`
  Converts the longitudinal CSV into the static benchmark split used by all baselines.
- `run_scvaeit.py`
  Runs scVAEIT on the metabolomics-imputation task.
- `run_midas.py`
  Runs MIDAS on the metabolomics-imputation task.
- `run_stabmap_from_csv.py`
  Prepares the split from the longitudinal CSV and then runs the StabMap R benchmark.
- `r/run_stabmap_benchmark.R`
  Runs StabMap when a prepared split directory already exists.

## Example Commands

```bash
python scripts/Benchmark-single-cell-Method/prepare_tabular_benchmark.py \
  --csv data/clinical_two_visit.csv \
  --output-dir outputs/static_benchmark_prepared
```

```bash
python scripts/Benchmark-single-cell-Method/run_scvaeit.py \
  --csv data/clinical_two_visit.csv \
  --output-dir outputs/scvaeit
```

```bash
python scripts/Benchmark-single-cell-Method/run_midas.py \
  --csv data/clinical_two_visit.csv \
  --output-dir outputs/midas
```

```bash
python scripts/Benchmark-single-cell-Method/run_stabmap_from_csv.py \
  --csv data/clinical_two_visit.csv \
  --output-dir outputs/stabmap
```

## Outputs

Each benchmark writes:

- prepared train/test matrices
- predicted metabolomics CSVs
- `metrics.json` with `Pearson r`, `MAE`, and `MSE`
