"""
Alignment mechanism using Jeffrey divergence.

This module implements the alignment penalty that minimizes divergence between
modality-specific latent distributions to create a unified latent space.

Reference: Methods section - Joint Latent Space and Recurrent Mechanism
"""

import torch
import torch.nn as nn
from peag.utils.distributions import kl_divergence_multivariate_normal


def jeffrey_divergence(
    mu1: torch.Tensor,
    logvar1: torch.Tensor,
    mu2: torch.Tensor,
    logvar2: torch.Tensor
) -> torch.Tensor:
    """
    Compute Jeffrey divergence between two multivariate normal distributions.
    
    Jeffrey divergence is the symmetric version of KL divergence:
    J(P||Q) = KL(P||Q) + KL(Q||P)
    
    Reference: Methods section - Alignment Penalty
    
    Args:
        mu1: Mean vector of first distribution (batch_size, latent_dim)
        logvar1: Log variance vector of first distribution (batch_size, latent_dim)
        mu2: Mean vector of second distribution (batch_size, latent_dim)
        logvar2: Log variance vector of second distribution (batch_size, latent_dim)
    
    Returns:
        Jeffrey divergence scalar value
    """
    kl_forward = kl_divergence_multivariate_normal(mu1, logvar1, mu2, logvar2)
    kl_reverse = kl_divergence_multivariate_normal(mu2, logvar2, mu1, logvar1)
    return kl_forward + kl_reverse


def align_distributions(
    z_lab_mu: torch.Tensor,
    z_lab_logvar: torch.Tensor,
    z_metab_mu: torch.Tensor,
    z_metab_logvar: torch.Tensor,
    z_past_mu: torch.Tensor,
    z_past_logvar: torch.Tensor
) -> torch.Tensor:
    """
    Compute alignment loss between lab tests, metabolomics, and Past-State encodings.
    
    Computes pairwise Jeffrey divergences between all three distributions and
    returns the sum as the alignment penalty.
    
    Reference: Methods section - Alignment Penalty
    
    Args:
        z_lab_mu: Lab tests encoder mean (batch_size, latent_dim)
        z_lab_logvar: Lab tests encoder log variance (batch_size, latent_dim)
        z_metab_mu: Metabolomics encoder mean (batch_size, latent_dim)
        z_metab_logvar: Metabolomics encoder log variance (batch_size, latent_dim)
        z_past_mu: Past-State encoder mean (batch_size, latent_dim)
        z_past_logvar: Past-State encoder log variance (batch_size, latent_dim)
    
    Returns:
        Total alignment loss (sum of pairwise Jeffrey divergences)
    """
    # Compute pairwise Jeffrey divergences
    j_lab_metab = jeffrey_divergence(
        z_lab_mu, z_lab_logvar,
        z_metab_mu, z_metab_logvar
    )
    
    j_lab_past = jeffrey_divergence(
        z_lab_mu, z_lab_logvar,
        z_past_mu, z_past_logvar
    )
    
    j_metab_past = jeffrey_divergence(
        z_metab_mu, z_metab_logvar,
        z_past_mu, z_past_logvar
    )
    
    # Sum all pairwise divergences
    alignment_loss = j_lab_metab + j_lab_past + j_metab_past
    
    return alignment_loss

