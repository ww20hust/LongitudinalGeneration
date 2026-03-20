# PEAG Architecture

## Overview

PEAG processes longitudinal multimodal visits in temporal order:

1. encode the currently available modalities at visit `N`
2. summarize previous visit states into a historical state `Z_P`
3. align current modality distributions with `Z_P`
4. fuse current modalities and `Z_P` into the visit state `Z^N`
5. decode every modality for reconstruction or imputation

Mask convention:

- `0`: currently available
- `1`: actively masked during training
- `2`: naturally missing

## Visit-State Fusion

The visit state uses equal-weight fusion:

```text
Z^N = (Z_P + sum(Z_M)) / (M_total + 1)
```

- `Z_P` is the historical latent state
- `Z_M` are the latent means of the currently observed modalities
- `M_total` counts only current modalities and does not include `Z_P`

This makes the implementation explicit that history is added once, and current
modalities are not double-counted.

## Active Masking During Training

The revised training strategy uses single-modality active masking:

- at each visit, PEAG samples whether to actively mask with probability `0.6`
- if masking is triggered, exactly one currently observed modality is selected
  and relabeled from `0` to `1`
- naturally missing modalities stay `2`
- a visit is never left without a current modality after active masking

This forces the model to reconstruct the masked modality from the remaining
current modality information together with the historical state.

## Temporal Autoregressive Module

The autoregressive component of PEAG is modular rather than fixed to an RNN.

### Recurrent option

- uses a GRU-style hidden-state update
- good default for shorter sequences

### Transformer option

- uses causal self-attention over previous visit states
- better suited to longer longitudinal visit sequences
- implemented as a lightweight temporal encoder, not a full redesign of PEAG

In both cases, the module outputs the historical state consumed by the
Past-State encoder at the next visit.

## Forward Pass

```text
for each visit N:
    historical_state = temporal_module(history_of_previous_visit_states)
    modality_dists = encode(currently_available_modalities_only)
    past_dist = past_encoder(historical_state)
    align(modality_dists, past_dist)
    visit_state = (past_mu + sum(current_modality_mus)) / (M_total + 1)
    decode_all_modalities(visit_state)
    append visit_state to temporal history
```

## Losses

- Reconstruction loss:
  - with active masking, supervised where mask is `0` or `1` and a target exists
  - naturally missing entries (`2`) are decoded but not supervised
- KL loss:
  - computed for encoded current modalities plus the historical latent branch
- Alignment loss:
  - computed only on available current modalities and history
- Adversarial loss:
  - optional gradient-reversal regularization on reconstructed modalities
  - each modality has a binary adversary that predicts whether that modality
    was missing at the current visit
  - missing includes both actively masked (`1`) and naturally missing (`2`)

## Module Layout

```text
peag/
  model.py              # main PEAG model
  losses.py             # reconstruction, KL, adversarial, total loss
  core/
    encoders.py         # modality encoders + past-state encoder
    decoders.py         # modality decoders
    alignment.py        # dynamic Jeffrey-divergence alignment
    fusion.py           # explicit equal-weight visit-state fusion
    temporal.py         # recurrent / transformer temporal modules
    adversarial.py      # discriminators
  data/
    dataset.py          # dataset and active masking
  training/
    trainer.py          # training loop and checkpointing
```
