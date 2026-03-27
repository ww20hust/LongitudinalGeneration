# PEAG MIMIC Benchmarks

This folder contains benchmark code for longitudinal MIMIC-III experiments
covering two tasks:

- `mortality`: first 48 hours, structured clinical measurements aggregated every 4 hours
- `readmission`: first 48 hours, structured clinical measurements aggregated every 4 hours

The released benchmarks follow the task definitions you specified:

- `run_notes_only.py`
  Whole-note benchmark. For both tasks, all notes from the first 48 hours are
  merged into one document. The merged document is embedded with
  `Llama-3.1-8B` and classified directly.
- `run_lab_transformer.py`
  Structured-clinical-measurements-only benchmark.
- `run_simply_combined.py`
  Late-fusion benchmark combining the structured clinical sequence with the
  whole-note Llama document embedding.
- `run_peag.py`
  PEAG benchmark. For both tasks, note embeddings are built per 4-hour step over
  the first 48 hours of the ICU stay.
- `prepare_binary_benchmark.py`
  Builds train/valid/test splits and derives the task-specific note/lab views.
- `precompute_vllm_embeddings.py`
  Optional precompute script that uses a vLLM OpenAI-compatible server to cache
  note embeddings (document and/or sequence).

## Inputs

`prepare_binary_benchmark.py` expects the outputs from
`mimic_data_extract_preprocess.py` for the matching task directory
(`mortality_48h_4h` or `readmission_48h_4h`):

- `cohort.csv`
- `structured_ts.pkl`
- `notes_ts.pkl`
- `meta.json`

For `readmission`, the extracted directory must also contain:

- `labels.csv`

## Example

```bash
python scripts/PEAG_MIMIC/prepare_binary_benchmark.py \
  --extracted_dir data/with_notes/mortality_48h_4h \
  --task mortality \
  --output_dir scripts/PEAG_MIMIC/data/mimic_mortality_benchmark

# Start a vLLM embeddings server (single GPU).
# vllm serve meta-llama/Llama-3.1-8B --task embedding --port 8000

# Multi-GPU (tensor-parallel) example:
# CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve meta-llama/Llama-3.1-8B \
#   --task embedding \
#   --tensor-parallel-size 4 \
#   --port 8000

python scripts/PEAG_MIMIC/precompute_vllm_embeddings.py \
  --train_path scripts/PEAG_MIMIC/data/mimic_mortality_benchmark/train.pkl \
  --valid_path scripts/PEAG_MIMIC/data/mimic_mortality_benchmark/valid.pkl \
  --test_path scripts/PEAG_MIMIC/data/mimic_mortality_benchmark/test.pkl \
  --cache_dir scripts/PEAG_MIMIC/outputs/mimic_vllm_cache \
  --llama_model_name_or_path meta-llama/Llama-3.1-8B \
  --vllm_base_url http://localhost:8000/v1

python scripts/PEAG_MIMIC/run_notes_only.py \
  --train_path scripts/PEAG_MIMIC/data/mimic_mortality_benchmark/train.pkl \
  --valid_path scripts/PEAG_MIMIC/data/mimic_mortality_benchmark/valid.pkl \
  --test_path scripts/PEAG_MIMIC/data/mimic_mortality_benchmark/test.pkl \
  --save_dir scripts/PEAG_MIMIC/outputs/mimic_notes_only \
  --llama_model_name_or_path meta-llama/Llama-3.1-8B \
  --llama_cache_dir scripts/PEAG_MIMIC/outputs/mimic_vllm_cache

python scripts/PEAG_MIMIC/run_peag.py \
  --train_path scripts/PEAG_MIMIC/data/mimic_mortality_benchmark/train.pkl \
  --valid_path scripts/PEAG_MIMIC/data/mimic_mortality_benchmark/valid.pkl \
  --test_path scripts/PEAG_MIMIC/data/mimic_mortality_benchmark/test.pkl \
  --save_dir scripts/PEAG_MIMIC/outputs/mimic_peag \
  --llama_model_name_or_path meta-llama/Llama-3.1-8B \
  --llama_cache_dir scripts/PEAG_MIMIC/outputs/mimic_vllm_cache
```
