# PEAG: Patient-context Enhanced Longitudinal Multimodal Alignment and Generation Framework

## Overview

PEAG is a deep learning framework for longitudinal multimodal clinical data imputation. It leverages patient historical state (Past-State) to enhance multimodal alignment and generation for the current visit, enabling accurate imputation of missing modalities from sparse measurements.

## Key Features

- **Longitudinal Context Integration**: Explicitly encodes personalized past state to enhance current visit multimodal alignment
- **Variational Autoencoder Architecture**: Based on VAE with recurrent neural network structure for sequential clinical visits
- **Multimodal Alignment**: Uses Jeffrey divergence to align modality-specific latent representations
- **Adversarial Training**: Prevents generated data from revealing missingness patterns
- **Flexible Ablation Modes**: Supports full, baseline-only, and follow-up-only modes

## Installation

```bash
pip install -r requirements.txt
```

Or install as a package:

```bash
pip install -e .
```

## Quick Start

### Basic Usage

```python
from peag.model import PEAGModel
from peag.data import LongitudinalDataset
from peag.training import Trainer

# Load data
dataset = LongitudinalDataset(baseline_data, followup_data)

# Initialize model
model = PEAGModel(
    lab_test_dim=61,
    metabolomics_dim=251,
    latent_dim=16
)

# Train model
trainer = Trainer(model, dataset)
trainer.train()

# Evaluate
results = trainer.evaluate(test_dataset)
```

## Architecture

PEAG processes longitudinal clinical visits as follows:

1. **Encoding**: Each modality (lab tests, metabolomics) and Past-State are encoded into 16-dimensional latent distributions
2. **Alignment**: Jeffrey divergence aligns the modality-specific latent representations
3. **Fusion**: Visit State is computed as the mean of aligned distributions
4. **Decoding**: All modalities are reconstructed from the fused Visit State
5. **Recurrence**: Visit State becomes the Past-State for the next visit

## Data Format

The framework expects paired Baseline and Follow-up visit data:
- **Baseline Visit**: Fully paired lab tests (61 features) and metabolomics (251 features)
- **Follow-up Visit**: Lab tests (61 features) available, metabolomics to be imputed

## Evaluation Metrics

- Pearson correlation coefficient (r)
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)

## Citation

If you use this framework, please cite the original paper.

## License

[Specify license]

