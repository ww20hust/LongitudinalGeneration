# Benchmarking Single-Cell Multimodal Frameworks and PEAG on Clinical Two-Visit Imputation

This folder benchmarks `MIDAS`, `scVAEIT`, `StabMap`, and `PEAG` on the same
clinical metabolomics-imputation task.

## Benchmark Goal

Target task: reconstruct follow-up Modality B (metabolomics) from follow-up
Modality A (routine labs), under patient-level train/test separation.

- Input to all methods at evaluation: held-out patient `visit=1` Modality A.
- Reconstruction target for all methods: the same held-out patient `visit=1`
  Modality B.
- Additional context used only by PEAG: the same patient's `visit=0` profile
  to form historical state.

This design keeps the reconstructed object identical across methods while
isolating PEAG's longitudinal conditioning capability.

## Fairness Protocol

All methods use the exact same patient-level split.

- No patient appears in both train and test.
- For static single-cell baselines (`MIDAS`, `scVAEIT`, `StabMap`): each visit
  from training patients is treated as an independent paired sample `(A, B)`.
- Static baselines do not use historical-state variables.
- For PEAG: training uses longitudinal sequences; evaluation uses `visit=0` as
  historical context and imputes `visit=1` Modality B.

## Input CSV Format

Expected columns:

- `eid`: patient identifier
- `visit`: visit index (`0` or `1`)
- Modality A columns (routine labs)
- Modality B columns (metabolomics)

Column selection options:

- explicit names: `--lab-columns`, `--metab-columns`
- prefix inference: `--lab-prefix`, `--metab-prefix`
- default fallback: first `61` non-metadata columns as Modality A and next
  `251` as Modality B

## Preprocessing Pipeline (shared)

For all methods, preprocessing follows the same steps:

1. Build patient-level split using `train_ratio` and `split_seed`.
2. Convert training-set visits into paired static samples for baseline methods.
3. Fit modality-specific standardization on training data only.
4. Apply the same transforms to train/test.
5. Evaluate in original scale using inverse transform.

Prepared matrices and scaler stats are exported under each method's
`<output-dir>/prepared` for reproducibility.

## Method Adaptations

### MIDAS (static baseline)

- Input structure: paired two-modality samples in MuData format.
- Architecture change: decoder distributions switched to Gaussian for
  continuous clinical variables.
- Inference: query uses follow-up Modality A only; model imputes follow-up
  Modality B.

### scVAEIT (static baseline)

- Input structure: concatenate `(A, B)` into a two-block Gaussian VAE input.
- Architecture change: two continuous blocks with Gaussian likelihood.
- Inference: mark Modality B block as unobserved; extract reconstructed
  Modality B block.

### StabMap (static baseline)

- Input structure: paired training visits as reference and follow-up Modality A
  from test patients as query.
- Adaptation: shared Modality A anchors transfer in the shared embedding.
- Inference: transfer/impute follow-up Modality B via reference-query mapping.

### PEAG (longitudinal model)

- Input structure: per-patient two-visit sequence.
- Training: active modality masking with longitudinal conditioning.
- Inference: use `visit=0` as historical context plus follow-up Modality A;
  reconstruct follow-up Modality B.

## External Dependencies for Static Baselines

`MIDAS`, `scVAEIT`, and `StabMap` are expected at:

- `scripts/Benchmark-single-cell-Method/baseline-model/scVAEIT`
- `scripts/Benchmark-single-cell-Method/baseline-model/midas`
- `scripts/Benchmark-single-cell-Method/baseline-model/StabMap`

From repository root:

```bash
cd scripts/Benchmark-single-cell-Method
mkdir -p baseline-model

git clone https://github.com/jaydu1/scVAEIT.git baseline-model/scVAEIT
git clone https://github.com/labomics/midas.git baseline-model/midas
git clone https://github.com/MarioniLab/StabMap.git baseline-model/StabMap
```

## Scripts

- `prepare_tabular_benchmark.py`
  Export shared prepared split from longitudinal CSV.
- `run_midas.py`
  Run MIDAS adaptation.
- `run_scvaeit.py`
  Run scVAEIT adaptation.
- `run_stabmap_from_csv.py`
  Run StabMap adaptation.
- `run_peag.py`
  Run PEAG with longitudinal historical-state conditioning.

## Example Commands

```bash
python scripts/Benchmark-single-cell-Method/run_midas.py \
  --csv data/clinical_two_visit.csv \
  --output-dir outputs/midas \
  --train-ratio 0.7 \
  --split-seed 0
```

```bash
python scripts/Benchmark-single-cell-Method/run_scvaeit.py \
  --csv data/clinical_two_visit.csv \
  --output-dir outputs/scvaeit \
  --train-ratio 0.7 \
  --split-seed 0
```

```bash
python scripts/Benchmark-single-cell-Method/run_stabmap_from_csv.py \
  --csv data/clinical_two_visit.csv \
  --output-dir outputs/stabmap \
  --train-ratio 0.7 \
  --split-seed 0
```

```bash
python scripts/Benchmark-single-cell-Method/run_peag.py \
  --csv data/clinical_two_visit.csv \
  --output-dir outputs/peag \
  --train-ratio 0.7 \
  --split-seed 0 \
  --train-mask-rate 0.6
```

## Outputs

Each method writes:

- prepared train/test matrices and scaler stats
- predicted follow-up Modality B CSVs
- `metrics.json` with `Pearson r`, `MAE`, and `MSE`
