# PEAG Ablation Scripts

This folder contains PEAG ablation code for the two-visit metabolomics-imputation task.

## Expected CSV Format

The input CSV should contain:

- `eid`: patient identifier
- `visit`: visit index (`0` or `1`)
- `61` routine lab columns
- `251` metabolomics columns

Columns can be passed explicitly or inferred by prefix. If no explicit mapping is given, the scripts assume the first `61` non-metadata columns are labs and the next `251` are metabolomics.

## Scripts

- `run_metabolomics_ablations.py`
  Runs the main PEAG ablations:
  - default model
  - historical-state ablation at inference
  - directional stop-gradient alignment
  - point-wise alignment
  - remove `L_align`
  - remove `L_adv`

- `run_mask_probability_sensitivity.py`
  Scans active-masking probabilities for the same metabolomics-imputation task.

## Example Commands

```bash
python scripts/Ablation/run_metabolomics_ablations.py \
  --csv data/clinical_two_visit.csv \
  --output-dir scripts/Ablation/outputs/ablation
```

```bash
python scripts/Ablation/run_mask_probability_sensitivity.py \
  --csv data/clinical_two_visit.csv \
  --output-dir scripts/Ablation/outputs/mask_sensitivity \
  --mask-rates 0.0,0.2,0.4,0.6,0.8,1.0
```

## Outputs

Each experiment writes:

- model checkpoints
- training history
- predicted follow-up metabolomics
- `metrics.json` with `Pearson r`, `MAE`, and `MSE`
- a folder-level `summary.json`
