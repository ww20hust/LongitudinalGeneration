"""
Alignment mechanism using Jeffrey divergence with missing modality support.

This module implements the alignment penalty that minimizes divergence between
modality-specific latent distributions to create a unified latent space.
It supports dynamic alignment based on available (non-missing) modalities.

Reference: Methods section - Joint Latent Space and Recurrent Mechanism
"""
import torch
import torch.nn as nn
from typing import Dict, Optional


def kl_divergence_multivariate_normal(
    mu1: torch.Tensor,
    logvar1: torch.Tensor,
    mu2: torch.Tensor,
    logvar2: torch.Tensor
) -> torch.Tensor:
    """
    Compute KL divergence between two multivariate normal distributions.
    
    KL(N(mu1, sigma1) || N(mu2, sigma2)) = 
        0.5 * sum(sigma2^{-1} * sigma1 + (mu2-mu1)^2/sigma2 - 1 + log(sigma2/sigma1))
    
    Args:
        mu1: Mean vector of first distribution (batch_size, latent_dim)
        logvar1: Log variance vector of first distribution (batch_size, latent_dim)
        mu2: Mean vector of second distribution (batch_size, latent_dim)
        logvar2: Log variance vector of second distribution (batch_size, latent_dim)
    
    Returns:
        KL divergence scalar value
    """
    var1 = torch.exp(logvar1)
    var2 = torch.exp(logvar2)
    
    kl = 0.5 * torch.sum(
        logvar2 - logvar1 + (var1 + (mu1 - mu2) ** 2) / (var2 + 1e-8) - 1,
        dim=-1
    )
    return kl.mean()


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


def align_distributions_dynamic(
    modality_distributions: Dict[str, tuple[torch.Tensor, torch.Tensor]],
    z_past_mu: torch.Tensor,
    z_past_logvar: torch.Tensor,
    missing_mask: Dict[str, int]
) -> torch.Tensor:
    """
    Compute alignment loss between available modalities and Past-State.
    
    Computes pairwise Jeffrey divergences between all AVAILABLE distributions 
    (excluding missing modalities) and returns the sum as the alignment penalty.
    
    Reference: Methods section - Alignment Penalty
    
    Args:
        modality_distributions: Dictionary mapping modality names to their 
                                distribution tuples (mu, logvar).
        z_past_mu: Past-State encoder mean (batch_size, latent_dim)
        z_past_logvar: Past-State encoder log variance (batch_size, latent_dim)
        missing_mask: Dictionary indicating which modalities are missing.
                      Keys are modality names, values are integer codes:
                      0 = available, 1 = actively masked, 2 = naturally missing.
    
    Returns:
        Alignment loss scalar value. Returns 0 if fewer than 2 distributions available.
    """
    # Collect all available distributions
    available_dists = []
    available_names = []
    
    for mod_name, (mu, logvar) in modality_distributions.items():
        if missing_mask.get(mod_name, 2) == 0 and mu is not None:
            available_dists.append((mu, logvar))
            available_names.append(mod_name)
    
    # Always include past state
    available_dists.append((z_past_mu, z_past_logvar))
    available_names.append("past")
    
    # Need at least 2 distributions to align
    if len(available_dists) < 2:
        return torch.tensor(0.0, device=z_past_mu.device)
    
    # Compute pairwise Jeffrey divergences
    total_alignment_loss = 0.0
    num_pairs = 0
    
    for i in range(len(available_dists)):
        for j in range(i + 1, len(available_dists)):
            mu_i, logvar_i = available_dists[i]
            mu_j, logvar_j = available_dists[j]
            
            jeffrey_div = jeffrey_divergence(mu_i, logvar_i, mu_j, logvar_j)
            total_alignment_loss += jeffrey_div
            num_pairs += 1
    
    # Average over all pairs
    if num_pairs > 0:
        total_alignment_loss = total_alignment_loss / num_pairs
    
    return total_alignment_loss


def align_distributions_simple(
    z_lab_mu: Optional[torch.Tensor],
    z_lab_logvar: Optional[torch.Tensor],
    z_metab_mu: Optional[torch.Tensor],
    z_metab_logvar: Optional[torch.Tensor],
    z_past_mu: torch.Tensor,
    z_past_logvar: torch.Tensor,
    lab_available: bool = True,
    metab_available: bool = True
) -> torch.Tensor:
    """
    Simple alignment function for backward compatibility.
    
    Computes alignment between lab tests, metabolomics (if available), and Past-State.
    
    Args:
        z_lab_mu: Lab tests encoder mean or None if missing
        z_lab_logvar: Lab tests encoder log variance or None if missing
        z_metab_mu: Metabolomics encoder mean or None if missing
        z_metab_logvar: Metabolomics encoder log variance or None if missing
        z_past_mu: Past-State encoder mean
        z_past_logvar: Past-State encoder log variance
        lab_available: Whether lab tests are available
        metab_available: Whether metabolomics are available
    
    Returns:
        Alignment loss scalar value
    """
    distributions: Dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    missing_mask: Dict[str, int] = {}
    
    if lab_available and z_lab_mu is not None:
        distributions["lab"] = (z_lab_mu, z_lab_logvar)
        missing_mask["lab"] = 0
    else:
        missing_mask["lab"] = 2
    
    if metab_available and z_metab_mu is not None:
        distributions["metab"] = (z_metab_mu, z_metab_logvar)
        missing_mask["metab"] = 0
    else:
        missing_mask["metab"] = 2
    
    return align_distributions_dynamic(distributions, z_past_mu, z_past_logvar, missing_mask)
