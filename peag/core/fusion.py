"""
Visit State fusion mechanism.

This module implements the fusion of aligned modality-specific latent distributions
into a unified Visit State representation.

Reference: Methods section - Visit State Estimation
"""

import torch


def compute_visit_state(
    z_lab_mu: torch.Tensor,
    z_metab_mu: torch.Tensor,
    z_past_mu: torch.Tensor
) -> torch.Tensor:
    """
    Compute Visit State by averaging aligned distribution means.
    
    Formula: z_t = mean([z_lab, z_metab, z_past])
    
    The Visit State is the integrative latent state that combines information
    from all available modalities and the historical context.
    
    Reference: Methods section - Visit State Estimation
    
    Args:
        z_lab_mu: Lab tests encoder mean (batch_size, latent_dim)
        z_metab_mu: Metabolomics encoder mean (batch_size, latent_dim)
        z_past_mu: Past-State encoder mean (batch_size, latent_dim)
    
    Returns:
        Fused Visit State of shape (batch_size, latent_dim)
    """
    # Stack the means and compute average
    means = torch.stack([z_lab_mu, z_metab_mu, z_past_mu], dim=0)
    visit_state = torch.mean(means, dim=0)
    
    return visit_state

