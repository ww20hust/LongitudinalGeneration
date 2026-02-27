# PEAG Architecture

## Overview

PEAG processes longitudinal multimodal visits in a loop: encode available modalities, align their latent distributions with past state, fuse into a visit state, decode all modalities (for imputation), then pass the visit state to the next visit. Losses are computed only where data is available (or actively masked with a target).

**Mask convention:** `0` = available, `1` = actively masked (training), `2` = naturally missing.

## Data Flow

```
INPUT
  visits_data  = [visit_1, visit_2, ..., visit_T]
  missing_masks = [mask_1, mask_2, ..., mask_T]

  Per visit: {mod_name: tensor or None}
  Per mask:  {mod_name: 0 | 1 | 2}
                    │
                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  FOR EACH VISIT                                                            │
│                                                                            │
│  past_state = zeros(...) for t=1; else previous visit_state                │
│       │                                                                    │
│       ▼                                                                    │
│  ENCODE (skip where mask ≠ 0 or data is None)                             │
│    modality_dists[mod] = (mu, logvar)   only for available                │
│    past_mu, past_logvar = past_encoder(past_state)                        │
│       │                                                                    │
│       ▼                                                                    │
│  ALIGN (only available + past)                                            │
│    Pairwise Jeffrey divergence over (modality_dists + past), averaged    │
│       │                                                                    │
│       ▼                                                                    │
│  FUSE                                                                      │
│    visit_state = mean(available_mus + [past_mu])                          │
│       │                                                                    │
│       ▼                                                                    │
│  DECODE (all modalities)                                                  │
│    reconstructions[mod] = decoder[mod](visit_state)   for every mod      │
│       │                                                                    │
│       ▼                                                                    │
│  past_state = visit_state  for next visit                                 │
└───────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
LOSSES
  Recon:  only (visit, mod) with mask 0 or 1 and target not None
  KL:     only encoded modalities + past, then averaged
  Align:  per-visit alignment loss, averaged over visits
  Adv:    generator loss on (e.g. last visit) primary-modality reconstruction
                    │
                    ▼
OUTPUT
  reconstructions: List[Dict[str, Tensor]]  (one dict per visit)
  losses: { total_loss, recon_loss, kl_loss, alignment_loss, adversarial_loss, ... }
  visit_states: optional, if return_all_visit_states=True
```

## Design Choices

- **Exclusion, not substitution** — Missing modalities are not filled with the previous visit’s encoding; they are excluded from encoding, alignment, and fusion. Only available modalities and past state are used.
- **Dynamic alignment and fusion** — Alignment and fusion use whatever set of modalities is available at that visit (plus past state), so the number of terms varies by visit.
- **Supervised imputation** — Reconstruction loss is applied only where we have a target (available or actively masked with `recon_targets`). Missing modalities are still decoded (imputed) but not supervised at that visit.

## Example: 3 Visits, Metabolomics Missing at Visit 2

```
Visit 1: lab=✓, metab=✓
Visit 2: lab=✓, metab=✗
Visit 3: lab=✓, metab=✓

Visit 1: Encode lab, metab, past → align 3-way → fuse → decode all.
Visit 2: Encode lab, past (metab skipped) → align 2-way → fuse → decode all (metab = imputed).
Visit 3: Encode lab, metab, past → align 3-way → fuse → decode all.

Recon: V1(lab+metab), V2(lab only when no recon_targets; with recon_targets also metab), V3(lab+metab).
```

## Module Layout

```
peag/
├── model.py           # PEAGModel: forward, compute_losses, impute_missing
├── losses.py         # reconstruction_loss, kl_divergence_loss, compute_total_loss, adversarial
├── core/
│   ├── encoders.py    # LabTestsEncoder, MetabolomicsEncoder, PastStateEncoder, GenericModalityEncoder
│   ├── decoders.py    # LabTestsDecoder, MetabolomicsDecoder, GenericModalityDecoder
│   ├── alignment.py   # align_distributions_dynamic (Jeffrey divergence)
│   ├── fusion.py      # compute_visit_state_dynamic
│   └── adversarial.py # MissingnessDiscriminator
├── data/
│   └── dataset.py     # LongitudinalDataset, collate_visits, create_synthetic_data
├── training/
│   └── trainer.py     # Trainer: train_epoch, train, KL annealing
└── utils/
    └── distributions.py
```

## API (minimal)

```python
from peag.model import PEAGModel

model = PEAGModel(
    modality_dims={"lab": 61, "metab": 251},
    latent_dim=16,
    hidden_dim=128,
)

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

out = model.forward(visits_data, missing_masks)
imputed = model.impute_missing(visits_data, missing_masks)
```
