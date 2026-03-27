# EHR Modeling Benchmarks

This folder contains the clinical patient-representation benchmarks used to
position PEAG against representative EHR / trajectory-modeling paradigms on the
proteomics generation task.

In this benchmark setting:

- input 1: longitudinal medical history before the current visit
- input 2: current routine laboratory tests
- target: current-visit plasma proteomics

The folder currently includes executable benchmarks for:

- `peag_benchmark.py`: PEAG with history representation + current labs
- `transformer_benchmark.py`: a standard Transformer encoder baseline
- `llama31_benchmark.py`: an LLM-embedding baseline based on Llama 3.1

The manuscript discussion also positions PEAG relative to trajectory-modeling
paradigms such as Delphi. If an implementation is not present in this folder,
it should be considered discussed in the paper rather than released here as a
reproducible script.

## Folder Contents

- `common.py`
  Shared utilities for loading JSON / JSONL benchmark data, preprocessing
  medical history and routine labs, training loops, and regression metrics.

- `transformer_benchmark.py`
  Encodes age-aware diagnosis history together with discretized current routine
  labs using `torch.nn.TransformerEncoder`, then predicts the proteomics vector
  with a regression head.

- `llama31_benchmark.py`
  Converts medical history and current routine labs into natural-language
  patient summaries, obtains frozen Llama 3.1 embeddings, and trains a small
  decoder to predict proteomics.

- `peag_benchmark.py`
  Uses the same clinical inputs, but models them with the PEAG
  alignment-generation framework to leverage longitudinal context and
  cross-modal structure.

## Data Format

Provide train / validation / test files in `.json` or `.jsonl` format. Each
record should follow this structure:

```json
{
  "patient_id": "P0001",
  "current_age": 63.4,
  "history_events": [
    {"age": 45.0, "code": "I10"},
    {"age": 58.0, "code": "E11.9"},
    {"age": 61.0, "code": "I25.1"}
  ],
  "routine_labs": {
    "ALT": 42.1,
    "AST": 28.3,
    "HbA1c": 6.4
  },
  "proteomics": [0.15, -0.33, 1.24]
}
```

Field conventions:

- `history_events` should contain all relevant events before the current visit
- each event includes both the age and the coded diagnosis / condition token
- `routine_labs` contains current-visit structured lab measurements
- `proteomics` is the current-visit regression target

## Usage

Run commands from the repository root.

### Transformer baseline

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/Bencmark-EHR-Modeling/transformer_benchmark.py \
  --device cuda:0 \
  --train_path data/proteomics_train.jsonl \
  --valid_path data/proteomics_valid.jsonl \
  --test_path data/proteomics_test.jsonl \
  --save_dir scripts/Bencmark-EHR-Modeling/outputs/transformer_benchmark
```

`transformer_benchmark.py` is configured to train end-to-end on a single 24 GB
GPU by default. The main memory-saving choices are:

- `batch_size=8`
- `max_history_events=256`
- `max_seq_len=512`
- `d_model=192`, `num_heads=6`, `num_layers=3`
- CUDA mixed precision enabled by default unless `--disable_amp` is passed

If your dataset still contains unusually long event histories, the encoded
sequence is truncated to `max_seq_len` while preserving the leading `[CLS]`
token and the most recent tokens.

### Llama 3.1 baseline

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/Bencmark-EHR-Modeling/llama31_benchmark.py \
  --device cuda:0 \
  --train_path data/proteomics_train.jsonl \
  --valid_path data/proteomics_valid.jsonl \
  --test_path data/proteomics_test.jsonl \
  --save_dir scripts/Bencmark-EHR-Modeling/outputs/llama31_benchmark \
  --llama_model_name_or_path meta-llama/Llama-3.1-8B \
  --cache_dir scripts/Bencmark-EHR-Modeling/outputs/llama31_cache
```

For a 24 GB GPU, prefer reduced-precision loading for the frozen Llama encoder.
This keeps the benchmark logic unchanged while making embedding extraction more
practical on a single RTX 3090 / 4090-class card:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/Bencmark-EHR-Modeling/llama31_benchmark.py \
  --device cuda:0 \
  --train_path data/proteomics_train.jsonl \
  --valid_path data/proteomics_valid.jsonl \
  --test_path data/proteomics_test.jsonl \
  --save_dir scripts/Bencmark-EHR-Modeling/outputs/llama31_benchmark \
  --llama_model_name_or_path meta-llama/Llama-3.1-8B \
  --llama_device cuda:0 \
  --llama_dtype float16 \
  --embedding_batch_size 1 \
  --max_length 512 \
  --cache_dir scripts/Bencmark-EHR-Modeling/outputs/llama31_cache
```

If 24 GB is still tight, add `--llama_load_in_8bit`. For the smallest memory
footprint on CUDA, add `--llama_load_in_4bit` instead.

### PEAG benchmark

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/Bencmark-EHR-Modeling/peag_benchmark.py \
  --device cuda:0 \
  --train_path data/proteomics_train.jsonl \
  --valid_path data/proteomics_valid.jsonl \
  --test_path data/proteomics_test.jsonl \
  --save_dir scripts/Bencmark-EHR-Modeling/outputs/peag_benchmark \
  --llama_model_name_or_path meta-llama/Llama-3.1-8B \
  --cache_dir scripts/Bencmark-EHR-Modeling/outputs/peag_cache
```

The same Llama loading flags are available in `peag_benchmark.py` because PEAG
also relies on frozen Llama history embeddings. A 24 GB GPU-friendly launch
looks like:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/Bencmark-EHR-Modeling/peag_benchmark.py \
  --device cuda:0 \
  --train_path data/proteomics_train.jsonl \
  --valid_path data/proteomics_valid.jsonl \
  --test_path data/proteomics_test.jsonl \
  --save_dir scripts/Bencmark-EHR-Modeling/outputs/peag_benchmark \
  --llama_model_name_or_path meta-llama/Llama-3.1-8B \
  --llama_device cuda:0 \
  --llama_dtype float16 \
  --embedding_batch_size 1 \
  --max_length 512 \
  --cache_dir scripts/Bencmark-EHR-Modeling/outputs/peag_cache
```

## Outputs

Each script writes benchmark outputs into its `--save_dir`, typically including:

- model checkpoints such as `best_model.pt` or `best_decoder.pt`
- `metrics.json` with validation / test metrics
- `test_predictions.npz` with predictions and targets for downstream analysis

## Metrics

These benchmarks are configured for proteomics generation and report regression
performance summaries such as:

- mean squared error (`MSE`)
- mean absolute error (`MAE`)
- protein-wise Pearson correlation summaries

## Dependencies

- `transformer_benchmark.py` depends on the repository's standard PyTorch / NumPy stack
- `llama31_benchmark.py` and `peag_benchmark.py` additionally require
  `transformers` and access to a Llama 3.1 checkpoint
- for CUDA 8-bit / 4-bit loading, install `bitsandbytes`

## Scope

This folder is for the paper's EHR / clinical representation benchmarks on the
proteomics generation task. It is separate from:

- `scripts/Ablation/` for PEAG ablations on two-visit metabolomics imputation
- `scripts/Benchmark-single-cell-Method/` for adapted static single-cell
  baselines on the same metabolomics imputation task
