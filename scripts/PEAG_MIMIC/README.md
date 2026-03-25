# PEAG MIMIC Benchmarks

This folder contains benchmark code for longitudinal MIMIC-III experiments
covering two tasks:

- `mortality`: first 48 hours, labs aggregated every 4 hours
- `readmission`: last 7 days before discharge, labs aggregated daily

The released benchmarks follow the task definitions you specified:

- `run_notes_only.py`
  Whole-note benchmark. For mortality, all notes from the first 48 hours are
  merged into one document. For readmission, all notes before discharge are
  merged into one document. The merged document is embedded with
  `Llama-3-8B-Instruct` and classified directly.
- `run_lab_transformer.py`
  Structured lab-sequence benchmark only.
- `run_simply_combined.py`
  Late-fusion benchmark combining the lab sequence with the whole-note Llama
  document embedding.
- `run_peag.py`
  PEAG benchmark. For mortality, note embeddings are built per 4-hour step. For
  readmission, note embeddings are built per day over the final 7 days before
  discharge.
- `prepare_binary_benchmark.py`
  Builds train/valid/test splits and derives the task-specific note/lab views.

## Inputs

`prepare_binary_benchmark.py` expects the outputs from
`MIMIC-III-ICU-Mortality-Prediction/extract_with_notes.py` for the mortality
cohort definition and first-48h data:

- `cohort.csv`
- `labs_ts.pkl`
- `notes.pkl`
- `meta.json`

For `readmission`, it also needs access to the raw MIMIC-III tables under
`--mimic_path` so it can reconstruct:

- labs from the final 7 days before discharge, aggregated daily
- PEAG note bins from the final 7 days before discharge, aggregated daily
- whole-note documents from all notes before discharge

## Example

```bash
python LongitudinalGeneration-main/scripts/PEAG_MIMIC/prepare_binary_benchmark.py   --extracted_dir data/with_notes   --task mortality   --output_dir data/mimic_mortality_benchmark

python LongitudinalGeneration-main/scripts/PEAG_MIMIC/run_notes_only.py   --train_path data/mimic_mortality_benchmark/train.pkl   --valid_path data/mimic_mortality_benchmark/valid.pkl   --test_path data/mimic_mortality_benchmark/test.pkl   --save_dir outputs/mimic_notes_only   --llama_model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct   --llama_cache_dir outputs/mimic_llama_cache

python LongitudinalGeneration-main/scripts/PEAG_MIMIC/run_peag.py   --train_path data/mimic_mortality_benchmark/train.pkl   --valid_path data/mimic_mortality_benchmark/valid.pkl   --test_path data/mimic_mortality_benchmark/test.pkl   --save_dir outputs/mimic_peag   --llama_model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct   --llama_cache_dir outputs/mimic_llama_cache
```
