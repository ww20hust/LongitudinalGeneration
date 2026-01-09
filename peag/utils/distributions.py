"""
Distribution utilities for VAE reparameterization and sampling.

This module implements the reparameterization trick for variational autoencoders,
allowing backpropagation through stochastic sampling operations.

Reference: Methods section - Variational Inference and Objective Function
"""

import torch
import torch.nn as nn


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Reparameterization trick for VAE.
    
    Samples from a multivariate normal distribution using the reparameterization trick:
    z = mu + sigma * epsilon, where epsilon ~ N(0, I)
    
    This allows gradients to flow through the sampling operation.
    
    Args:
        mu: Mean vector of shape (batch_size, latent_dim)
        logvar: Log variance vector of shape (batch_size, latent_dim)
    
    Returns:
        Sampled latent vector of shape (batch_size, latent_dim)
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def sample_from_distribution(mu: torch.Tensor, logvar: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
    """
    Sample multiple times from a multivariate normal distribution.
    
    Args:
        mu: Mean vector of shape (batch_size, latent_dim)
        logvar: Log variance vector of shape (batch_size, latent_dim)
        n_samples: Number of samples to draw
    
    Returns:
        Sampled latent vectors of shape (n_samples, batch_size, latent_dim)
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn(n_samples, *mu.shape, device=mu.device, dtype=mu.dtype)
    return mu.unsqueeze(0) + std.unsqueeze(0) * eps


def kl_divergence_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence between a multivariate normal distribution and standard normal N(0, I).
    
    Formula: KL(q(z|x) || p(z)) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    
    Reference: Methods section - Variational Inference and Objective Function
    
    Args:
        mu: Mean vector of shape (batch_size, latent_dim)
        logvar: Log variance vector of shape (batch_size, latent_dim)
    
    Returns:
        KL divergence scalar value
    """
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()


def kl_divergence_multivariate_normal(
    mu1: torch.Tensor,
    logvar1: torch.Tensor,
    mu2: torch.Tensor,
    logvar2: torch.Tensor
) -> torch.Tensor:
    """
    Compute KL divergence between two multivariate normal distributions.
    
    Formula: KL(N(mu1, var1) || N(mu2, var2)) = 
        0.5 * (tr(var2^-1 * var1) + (mu2 - mu1)^T * var2^-1 * (mu2 - mu1) - k + ln(det(var2)/det(var1)))
    
    For isotropic distributions (diagonal covariance), this simplifies to:
        0.5 * sum(var1/var2 + (mu2 - mu1)^2/var2 - 1 + ln(var2/var1))
    
    Args:
        mu1: Mean vector of first distribution (batch_size, latent_dim)
        logvar1: Log variance vector of first distribution (batch_size, latent_dim)
        mu2: Mean vector of second distribution (batch_size, latent_dim)
        logvar2: Log variance vector of second distribution (batch_size, latent_dim)
    
    Returns:
        KL divergence scalar value
    """
    var1 = logvar1.exp()
    var2 = logvar2.exp()
    
    kl = 0.5 * torch.sum(
        var1 / var2 + (mu2 - mu1).pow(2) / var2 - 1 + logvar2 - logvar1,
        dim=1
    )
    return kl.mean()

