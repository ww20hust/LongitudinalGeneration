"""
Utility functions for distribution operations.
"""
import torch
import torch.nn as nn


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Reparameterization trick for VAE.
    
    z = mu + sigma * epsilon
    where epsilon ~ N(0, 1)
    
    Args:
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
    
    Returns:
        Sampled latent vector
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def kl_divergence_multivariate_normal(
    mu1: torch.Tensor,
    logvar1: torch.Tensor,
    mu2: torch.Tensor,
    logvar2: torch.Tensor
) -> torch.Tensor:
    """
    Compute KL divergence between two multivariate normal distributions.
    
    Args:
        mu1: Mean of first distribution
        logvar1: Log variance of first distribution
        mu2: Mean of second distribution
        logvar2: Log variance of second distribution
    
    Returns:
        KL divergence
    """
    var1 = torch.exp(logvar1)
    var2 = torch.exp(logvar2)
    
    kl = 0.5 * torch.sum(
        logvar2 - logvar1 + (var1 + (mu1 - mu2) ** 2) / (var2 + 1e-8) - 1,
        dim=-1
    )
    return kl.mean()
