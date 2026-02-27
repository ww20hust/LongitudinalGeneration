"""
Loss functions for PEAG framework.

This module implements all loss components including reconstruction loss,
KL divergence, alignment penalty, and adversarial loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


def reconstruction_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    loss_type: str = "mse"
) -> torch.Tensor:
    """
    Compute reconstruction loss.
    
    Args:
        recon: Reconstructed data
        target: Target data
        mask: Optional mask for missing values (1 = valid, 0 = missing)
        loss_type: Type of loss ("mse" or "mae")
    
    Returns:
        Reconstruction loss
    """
    if loss_type == "mse":
        loss = F.mse_loss(recon, target, reduction='none')
    elif loss_type == "mae":
        loss = F.l1_loss(recon, target, reduction='none')
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    if mask is not None:
        # Apply mask and compute mean over valid values
        loss = loss * mask
        return loss.sum() / (mask.sum() + 1e-8)
    else:
        return loss.mean()


def kl_divergence_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence loss for VAE.
    
    KL(N(mu, sigma) || N(0, 1)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    
    Args:
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
    
    Returns:
        KL divergence loss
    """
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    return kl.mean()


def alignment_penalty_loss(alignment_loss: torch.Tensor) -> torch.Tensor:
    """
    Alignment penalty loss.
    
    Args:
        alignment_loss: Alignment loss from alignment module
    
    Returns:
        Alignment penalty
    """
    return alignment_loss


def adversarial_loss_generator(
    discriminator: nn.Module,
    fake_data: torch.Tensor
) -> torch.Tensor:
    """
    Adversarial loss for generator (to fool discriminator).
    
    Args:
        discriminator: Discriminator network
        fake_data: Generated data
    
    Returns:
        Adversarial loss
    """
    fake_pred = discriminator(fake_data)
    # Generator wants discriminator to predict 1 (real) for fake data
    loss = -torch.mean(torch.log(fake_pred + 1e-8))
    return loss


def adversarial_loss_discriminator(
    discriminator: nn.Module,
    real_data: torch.Tensor,
    fake_data: torch.Tensor
) -> torch.Tensor:
    """
    Adversarial loss for discriminator.
    
    Args:
        discriminator: Discriminator network
        real_data: Real data
        fake_data: Generated (fake) data
    
    Returns:
        Discriminator loss
    """
    real_pred = discriminator(real_data)
    fake_pred = discriminator(fake_data.detach())
    
    # Discriminator wants to predict 1 for real and 0 for fake
    real_loss = -torch.mean(torch.log(real_pred + 1e-8))
    fake_loss = -torch.mean(torch.log(1 - fake_pred + 1e-8))
    
    return real_loss + fake_loss


def compute_total_loss(
    recon_losses: Dict[str, torch.Tensor],
    kl_losses: Dict[str, torch.Tensor],
    alignment_loss: torch.Tensor,
    adversarial_loss: Optional[torch.Tensor] = None,
    lambda_kl: float = 1.0,
    lambda_align: float = 1.0,
    lambda_adv: float = 1.0,
    kl_annealing_weight: float = 1.0
) -> Dict[str, torch.Tensor]:
    """
    Compute total loss as weighted sum of all components.
    
    Args:
        recon_losses: Dictionary of reconstruction losses per modality
        kl_losses: Dictionary of KL divergence losses per encoder
        alignment_loss: Alignment penalty loss
        adversarial_loss: Adversarial loss (optional)
        lambda_kl: Weight for KL divergence
        lambda_align: Weight for alignment
        lambda_adv: Weight for adversarial loss
        kl_annealing_weight: Annealing weight for KL (0 to 1)
    
    Returns:
        Dictionary containing total loss and individual components
    """
    # Sum reconstruction losses
    total_recon_loss = sum(recon_losses.values())
    
    # Sum KL losses
    total_kl_loss = sum(kl_losses.values())
    
    # Total loss
    total_loss = total_recon_loss + \
                 lambda_kl * kl_annealing_weight * total_kl_loss + \
                 lambda_align * alignment_loss
    
    if adversarial_loss is not None:
        total_loss += lambda_adv * adversarial_loss
    
    return {
        "total_loss": total_loss,
        "recon_loss": total_recon_loss,
        "kl_loss": total_kl_loss,
        "alignment_loss": alignment_loss,
        "adversarial_loss": adversarial_loss if adversarial_loss is not None else torch.tensor(0.0),
        **{f"recon_{k}": v for k, v in recon_losses.items()},
        **{f"kl_{k}": v for k, v in kl_losses.items()}
    }
