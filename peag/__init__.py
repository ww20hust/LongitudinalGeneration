"""
PEAG: Patient-context Enhanced Longitudinal Multimodal Alignment and Generation Framework.
"""

from __future__ import annotations

from peag.clinical_benchmark import (
    ClinicalBenchmarkBundle,
    TabularBenchmarkSplit,
    evaluate_reconstruction,
    format_mask_rate_tag,
    parse_column_argument,
    prepare_two_visit_clinical_benchmark,
    save_json,
    save_scaler_stats,
    save_tabular_split_csv,
)

__version__ = "2.1.0"
__all__ = [
    "PEAGModel",
    "ClinicalBenchmarkBundle",
    "TabularBenchmarkSplit",
    "prepare_two_visit_clinical_benchmark",
    "parse_column_argument",
    "evaluate_reconstruction",
    "save_json",
    "save_scaler_stats",
    "save_tabular_split_csv",
    "format_mask_rate_tag",
    "reconstruction_loss",
    "kl_divergence_loss",
    "adversarial_loss_generator",
    "adversarial_loss_discriminator",
    "compute_total_loss"
]


def __getattr__(name: str):
    if name == "PEAGModel":
        from peag.model import PEAGModel

        return PEAGModel

    if name in {
        "reconstruction_loss",
        "kl_divergence_loss",
        "adversarial_loss_generator",
        "adversarial_loss_discriminator",
        "compute_total_loss",
    }:
        from peag import losses as _losses

        return getattr(_losses, name)

    raise AttributeError(f"module 'peag' has no attribute {name!r}")
