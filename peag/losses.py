"""
Loss functions for PEAG framework.

This module implements all loss components: reconstruction loss, KL divergence,
alignment penalty, and adversarial loss.

Reference: Methods section - Variational Inference and Objective Function
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from peag.utils.distributions import kl_divergence_standard_normal
from peag.core.alignment import align_distributions


def reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor = None
) -> torch.Tensor:
    """
    Compute reconstruction loss (Mean Squared Error).
    
    Applied only to data points with ground-truth measurements.
    
    Reference: Methods section - Reconstruction Loss
    
    Args:
        pred: Predicted values of shape (batch_size, feature_dim)
        target: Ground-truth values of shape (batch_size, feature_dim)
        mask: Binary mask indicating available measurements (batch_size, feature_dim).
              If None, all points are considered available.
    
    Returns:
        Mean squared error loss
    """
    if mask is not None:
        # Only compute loss for available measurements
        mse = (pred - target).pow(2) * mask
        loss = mse.sum() / mask.sum()
    else:
        loss = F.mse_loss(pred, target)
    
    return loss


def kl_divergence_loss(
    mu: torch.Tensor,
    logvar: torch.Tensor
) -> torch.Tensor:
    """
    Compute KL divergence loss between latent distribution and standard normal prior.
    
    Reference: Methods section - Variational Inference and Objective Function
    
    Args:
        mu: Mean vector of shape (batch_size, latent_dim)
        logvar: Log variance vector of shape (batch_size, latent_dim)
    
    Returns:
        KL divergence loss
    """
    return kl_divergence_standard_normal(mu, logvar)


def alignment_penalty_loss(
    z_lab_mu: torch.Tensor,
    z_lab_logvar: torch.Tensor,
    z_metab_mu: torch.Tensor,
    z_metab_logvar: torch.Tensor,
    z_past_mu: torch.Tensor,
    z_past_logvar: torch.Tensor
) -> torch.Tensor:
    """
    Compute alignment penalty using Jeffrey divergence.
    
    Reference: Methods section - Alignment Penalty
    
    Args:
        z_lab_mu: Lab tests encoder mean (batch_size, latent_dim)
        z_lab_logvar: Lab tests encoder log variance (batch_size, latent_dim)
        z_metab_mu: Metabolomics encoder mean (batch_size, latent_dim)
        z_metab_logvar: Metabolomics encoder log variance (batch_size, latent_dim)
        z_past_mu: Past-State encoder mean (batch_size, latent_dim)
        z_past_logvar: Past-State encoder log variance (batch_size, latent_dim)
    
    Returns:
        Alignment penalty loss
    """
    return align_distributions(
        z_lab_mu, z_lab_logvar,
        z_metab_mu, z_metab_logvar,
        z_past_mu, z_past_logvar
    )


def adversarial_loss_discriminator(
    discriminator: nn.Module,
    real_data: torch.Tensor,
    fake_data: torch.Tensor
) -> torch.Tensor:
    """
    Compute discriminator loss for adversarial training.
    
    The discriminator is trained to distinguish real (fully measured) data
    from fake (imputed) data.
    
    Reference: Methods section - Missingness Adversarial Training
    
    Args:
        discriminator: MissingnessDiscriminator instance
        real_data: Real (fully measured) data of shape (batch_size, feature_dim)
        fake_data: Fake (imputed) data of shape (batch_size, feature_dim)
    
    Returns:
        Discriminator loss (binary cross-entropy)
    """
    # Real data should be classified as 1
    real_pred = discriminator(real_data)
    real_loss = F.binary_cross_entropy(real_pred, torch.ones_like(real_pred))
    
    # Fake data should be classified as 0
    fake_pred = discriminator(fake_data)
    fake_loss = F.binary_cross_entropy(fake_pred, torch.zeros_like(fake_pred))
    
    # Total discriminator loss
    disc_loss = (real_loss + fake_loss) / 2.0
    
    return disc_loss


def adversarial_loss_generator(
    discriminator: nn.Module,
    fake_data: torch.Tensor
) -> torch.Tensor:
    """
    Compute generator loss for adversarial training.
    
    The generator (decoder) is trained to confuse the discriminator by making
    imputed data indistinguishable from real data.
    
    Reference: Methods section - Missingness Adversarial Training
    
    Args:
        discriminator: MissingnessDiscriminator instance
        fake_data: Fake (imputed) data of shape (batch_size, feature_dim)
    
    Returns:
        Generator loss (binary cross-entropy, trying to fool discriminator)
    """
    # Generator wants discriminator to classify fake as real (1)
    fake_pred = discriminator(fake_data)
    gen_loss = F.binary_cross_entropy(fake_pred, torch.ones_like(fake_pred))
    
    return gen_loss


def compute_total_loss(
    # Reconstruction losses
    lab_recon_loss: torch.Tensor,
    metab_recon_loss: torch.Tensor,
    # KL divergences
    lab_kl_loss: torch.Tensor,
    metab_kl_loss: torch.Tensor,
    past_kl_loss: torch.Tensor,
    # Alignment penalty
    alignment_loss: torch.Tensor,
    # Adversarial loss
    adversarial_loss: torch.Tensor,
    # Loss weights
    lambda_kl: float = 1.0,
    lambda_align: float = 1.0,
    lambda_adv: float = 1.0,
    # KL annealing weight
    kl_annealing_weight: float = 1.0
) -> dict[str, torch.Tensor]:
    """
    Compute total loss from all components.
    
    Formula: L_total = L_recon + λ_KL * L_KL + λ_align * L_align + λ_adv * L_adv
    
    Reference: Methods section - Total Loss
    
    Args:
        lab_recon_loss: Lab tests reconstruction loss
        metab_recon_loss: Metabolomics reconstruction loss
        lab_kl_loss: Lab tests encoder KL divergence
        metab_kl_loss: Metabolomics encoder KL divergence
        past_kl_loss: Past-State encoder KL divergence
        alignment_loss: Alignment penalty loss
        adversarial_loss: Adversarial loss (generator)
        lambda_kl: Weight for KL divergence term
        lambda_align: Weight for alignment penalty term
        lambda_adv: Weight for adversarial loss term
        kl_annealing_weight: KL annealing weight (0 to 1) for gradual KL increase
    
    Returns:
        Dictionary containing total loss and individual components
    """
    # Total reconstruction loss
    recon_loss = lab_recon_loss + metab_recon_loss
    
    # Total KL divergence (with annealing)
    total_kl = (lab_kl_loss + metab_kl_loss + past_kl_loss) * kl_annealing_weight
    
    # Total loss
    total_loss = (
        recon_loss +
        lambda_kl * total_kl +
        lambda_align * alignment_loss +
        lambda_adv * adversarial_loss
    )
    
    return {
        "total_loss": total_loss,
        "reconstruction_loss": recon_loss,
        "kl_loss": total_kl,
        "alignment_loss": alignment_loss,
        "adversarial_loss": adversarial_loss,
        "lab_recon_loss": lab_recon_loss,
        "metab_recon_loss": metab_recon_loss,
        "lab_kl_loss": lab_kl_loss,
        "metab_kl_loss": metab_kl_loss,
        "past_kl_loss": past_kl_loss,
    }

