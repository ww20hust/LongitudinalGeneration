# PEAG

Patient-context Enhanced Longitudinal Multimodal Alignment and Generation — a deep learning framework for longitudinal multimodal clinical data with multiple visits and missing modalities.

## Installation

```bash
pip install -e .
```

Requires Python ≥3.8, PyTorch ≥1.9, NumPy, and optionally `tqdm`.

## Quick Start

```python
import torch
from torch.utils.data import DataLoader
from peag.model import PEAGModel
from peag.data.dataset import create_synthetic_data, LongitudinalDataset, collate_visits
from peag.training.trainer import Trainer

modality_dims = {"lab": 61, "metab": 251}
model = PEAGModel(modality_dims=modality_dims, latent_dim=16, hidden_dim=128)

patient_ids, visits_data, missing_masks = create_synthetic_data(
    n_patients=100, n_visits=3, modality_dims=modality_dims, missing_rate=0.3
)
dataset = LongitudinalDataset(patient_ids, visits_data, missing_masks)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_visits)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
trainer = Trainer(model, optimizer)
history = trainer.train(dataloader, n_epochs=50)
```

**Imputation (inference):**

```python
imputed = model.impute_missing(visits_data, missing_masks)
# imputed[visit_idx][mod_name] is imputed when mask was 2 or data was None
```

## Scripts

**Train** (synthetic data, optional active masking and checkpoint saving):

```bash
python scripts/train.py --n_patients 100 --n_visits 3 --epochs 10 --save_dir ./ckpts
```

**Inference** (load checkpoint and run imputation):

```bash
python scripts/inference.py --checkpoint ./ckpts/checkpoint_epoch_10.pt
```

## Data Format

- **visits_data**: `List[Dict[str, Tensor]]` — one dict per visit; keys = modality names, value = `(batch, dim)` or `None` if missing.
- **missing_masks**: `List[Dict[str, int]]` — same length; `0` = available, `1` = actively masked, `2` = naturally missing.

Example: one patient, three visits, metabolomics missing at visit 2:

```python
visits_data = [
    {"lab": lab_t1, "metab": metab_t1},
    {"lab": lab_t2, "metab": None},
    {"lab": lab_t3, "metab": metab_t3},
]
missing_masks = [
    {"lab": 0, "metab": 0},
    {"lab": 0, "metab": 2},
    {"lab": 0, "metab": 0},
]
```

## Model Overview

For each visit:

1. Encode available modalities (skip where mask ≠ 0 or data is `None`).
2. Encode past state (zeros at first visit).
3. Align available modality distributions (pairwise Jeffrey divergence) and fuse with past state.
4. Visit state = mean of available modality means and past-state mean.
5. Decode all modalities from visit state; set past state = visit state for next visit.

**Losses:** reconstruction (on available/actively masked with targets), KL (on encoded modalities + past), alignment (on available only), optional adversarial on reconstructions.

## API Summary

| Component | Description |
|-----------|-------------|
| `PEAGModel(modality_dims, latent_dim=16, hidden_dim=128, lambda_kl, lambda_align, lambda_adv)` | Main model |
| `model.forward(visits_data, missing_masks, kl_annealing_weight=1.0, recon_targets=None)` | Returns `reconstructions`, `losses` |
| `model.impute_missing(visits_data, missing_masks)` | Returns list of per-visit dicts (imputed where missing) |
| `LongitudinalDataset(patient_ids, visits_data, missing_masks, min_completeness_ratio, train_mask_rate)` | Dataset with optional filtering and active masking |
| `create_synthetic_data(n_patients, n_visits, modality_dims, missing_rate, seed)` | Synthetic longitudinal data |
| `Trainer(model, optimizer).train(dataloader, n_epochs, save_dir, validate_every)` | Training loop with KL annealing |

## License

MIT
