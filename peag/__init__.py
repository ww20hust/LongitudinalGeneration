"""
PEAG: Patient-context Enhanced Longitudinal Multimodal Alignment and Generation Framework

An enhanced framework for longitudinal multimodal clinical data imputation
with support for multiple visits and missing modalities.
"""

from peag.model import PEAGModel
from peag.losses import (
    reconstruction_loss,
    kl_divergence_loss,
    adversarial_loss_generator,
    adversarial_loss_discriminator,
    compute_total_loss
)

__version__ = "2.0.0"
__all__ = [
    "PEAGModel",
    "reconstruction_loss",
    "kl_divergence_loss",
    "adversarial_loss_generator",
    "adversarial_loss_discriminator",
    "compute_total_loss"
]
