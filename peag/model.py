"""
Main PEAG model class.

This module implements the complete PEAG framework, integrating all encoders,
decoders, alignment mechanism, and adversarial components with recurrent
mechanism for longitudinal data processing.

Reference: Methods section - Patient-context Enhanced Longitudinal Multimodal Alignment and Generation Framework
"""

import torch
import torch.nn as nn
from peag.core.encoders import LabTestsEncoder, MetabolomicsEncoder, PastStateEncoder
from peag.core.decoders import LabTestsDecoder, MetabolomicsDecoder
from peag.core.alignment import align_distributions
from peag.core.fusion import compute_visit_state
from peag.core.adversarial import MissingnessDiscriminator
from peag.utils.distributions import reparameterize
from peag.losses import (
    reconstruction_loss,
    kl_divergence_loss,
    alignment_penalty_loss,
    adversarial_loss_generator,
    compute_total_loss
)


class PEAGModel(nn.Module):
    """
    Patient-context Enhanced Longitudinal Multimodal Alignment and Generation Model.
    
    The model processes longitudinal clinical visits by:
    1. Encoding current modalities and Past-State into latent distributions
    2. Aligning distributions using Jeffrey divergence
    3. Fusing into unified Visit State
    4. Decoding all modalities from Visit State
    5. Using Visit State as Past-State for next visit
    
    Reference: Methods section - The PEAG framework
    """
    
    def __init__(
        self,
        lab_test_dim: int = 61,
        metabolomics_dim: int = 251,
        latent_dim: int = 16,
        hidden_dim: int = 128,
        lambda_kl: float = 1.0,
        lambda_align: float = 1.0,
        lambda_adv: float = 1.0
    ):
        """
        Initialize PEAG model.
        
        Args:
            lab_test_dim: Dimension of lab test features (default: 61)
            metabolomics_dim: Dimension of metabolomics features (default: 251)
            latent_dim: Dimension of latent space (default: 16)
            hidden_dim: Dimension of hidden layers (default: 128)
            lambda_kl: Weight for KL divergence loss (default: 1.0)
            lambda_align: Weight for alignment penalty (default: 1.0)
            lambda_adv: Weight for adversarial loss (default: 1.0)
        """
        super(PEAGModel, self).__init__()
        
        self.lab_test_dim = lab_test_dim
        self.metabolomics_dim = metabolomics_dim
        self.latent_dim = latent_dim
        self.lambda_kl = lambda_kl
        self.lambda_align = lambda_align
        self.lambda_adv = lambda_adv
        
        # Encoders
        self.lab_encoder = LabTestsEncoder(latent_dim=latent_dim, hidden_dim=hidden_dim)
        self.metab_encoder = MetabolomicsEncoder(latent_dim=latent_dim, hidden_dim=hidden_dim)
        self.past_encoder = PastStateEncoder(latent_dim=latent_dim, hidden_dim=hidden_dim // 2)
        
        # Decoders
        self.lab_decoder = LabTestsDecoder(latent_dim=latent_dim, hidden_dim=hidden_dim)
        self.metab_decoder = MetabolomicsDecoder(latent_dim=latent_dim, hidden_dim=hidden_dim)
        
        # Adversarial discriminator
        self.discriminator = MissingnessDiscriminator(
            input_dim=metabolomics_dim,
            hidden_dim=hidden_dim
        )
    
    def encode(
        self,
        lab_tests: torch.Tensor = None,
        metabolomics: torch.Tensor = None,
        past_state: torch.Tensor = None
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode modalities and Past-State to latent distributions.
        
        Reference: Methods section - Modality-Specific Encoders and Past-State Integration
        
        Args:
            lab_tests: Lab test features of shape (batch_size, lab_test_dim) or None
            metabolomics: Metabolomics features of shape (batch_size, metabolomics_dim) or None
            past_state: Past-State vector of shape (batch_size, latent_dim) or None
        
        Returns:
            Dictionary with encoded distributions (mu, logvar) for each available modality
        """
        encodings = {}
        
        if lab_tests is not None:
            z_lab_mu, z_lab_logvar = self.lab_encoder(lab_tests)
            encodings["lab"] = (z_lab_mu, z_lab_logvar)
        
        if metabolomics is not None:
            z_metab_mu, z_metab_logvar = self.metab_encoder(metabolomics)
            encodings["metab"] = (z_metab_mu, z_metab_logvar)
        
        if past_state is not None:
            z_past_mu, z_past_logvar = self.past_encoder(past_state)
            encodings["past"] = (z_past_mu, z_past_logvar)
        
        return encodings
    
    def forward(
        self,
        lab_tests_baseline: torch.Tensor,
        metabolomics_baseline: torch.Tensor,
        lab_tests_followup: torch.Tensor,
        metabolomics_followup: torch.Tensor = None,
        mode: str = "full",
        kl_annealing_weight: float = 1.0,
        return_visit_state: bool = False
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Reference: Methods section - The PEAG framework
        
        Args:
            lab_tests_baseline: Baseline lab tests (batch_size, lab_test_dim)
            metabolomics_baseline: Baseline metabolomics (batch_size, metabolomics_dim)
            lab_tests_followup: Follow-up lab tests (batch_size, lab_test_dim)
            metabolomics_followup: Follow-up metabolomics (batch_size, metabolomics_dim) or None
            mode: Ablation mode - "full", "baseline_only", or "followup_only"
            kl_annealing_weight: KL annealing weight (0 to 1) for gradual KL increase
            return_visit_state: Whether to return Visit State for next visit
        
        Returns:
            Dictionary containing:
            - predictions: Decoded lab tests and metabolomics
            - losses: All loss components
            - visit_state: Visit State (if return_visit_state=True)
        """
        batch_size = lab_tests_baseline.shape[0]
        device = lab_tests_baseline.device
        
        # ===== BASELINE VISIT =====
        # Encode Baseline visit
        baseline_encodings = self.encode(
            lab_tests=lab_tests_baseline,
            metabolomics=metabolomics_baseline
        )
        
        z_lab_baseline_mu, z_lab_baseline_logvar = baseline_encodings["lab"]
        z_metab_baseline_mu, z_metab_baseline_logvar = baseline_encodings["metab"]
        
        # Initialize Past-State for Baseline (zeros for first visit)
        past_state_baseline = torch.zeros(batch_size, self.latent_dim, device=device)
        z_past_baseline_mu, z_past_baseline_logvar = self.past_encoder(past_state_baseline)
        
        # Align Baseline distributions
        alignment_loss_baseline = align_distributions(
            z_lab_baseline_mu, z_lab_baseline_logvar,
            z_metab_baseline_mu, z_metab_baseline_logvar,
            z_past_baseline_mu, z_past_baseline_logvar
        )
        
        # Compute Baseline Visit State
        visit_state_baseline = compute_visit_state(
            z_lab_baseline_mu,
            z_metab_baseline_mu,
            z_past_baseline_mu
        )
        
        # Decode Baseline visit
        lab_recon_baseline = self.lab_decoder(visit_state_baseline)
        metab_recon_baseline = self.metab_decoder(visit_state_baseline)
        
        # ===== FOLLOW-UP VISIT =====
        # Use Baseline Visit State as Past-State for Follow-up
        past_state_followup = visit_state_baseline
        
        # Encode Follow-up visit based on mode
        if mode == "full":
            # Full mode: use both Baseline full data and Follow-up lab tests
            followup_encodings = self.encode(
                lab_tests=lab_tests_followup,
                past_state=past_state_followup
            )
            z_lab_followup_mu, z_lab_followup_logvar = followup_encodings["lab"]
            z_past_followup_mu, z_past_followup_logvar = followup_encodings["past"]
            
            # Use Baseline metabolomics encoding for alignment
            z_metab_followup_mu = z_metab_baseline_mu
            z_metab_followup_logvar = z_metab_baseline_logvar
            
        elif mode == "baseline_only":
            # Baseline-only mode: use only Baseline full data
            followup_encodings = self.encode(
                past_state=past_state_followup
            )
            z_past_followup_mu, z_past_followup_logvar = followup_encodings["past"]
            
            # Use Baseline encodings
            z_lab_followup_mu = z_lab_baseline_mu
            z_lab_followup_logvar = z_lab_baseline_logvar
            z_metab_followup_mu = z_metab_baseline_mu
            z_metab_followup_logvar = z_metab_baseline_logvar
            
        elif mode == "followup_only":
            # Follow-up-only mode: use only Follow-up lab tests
            followup_encodings = self.encode(
                lab_tests=lab_tests_followup,
                past_state=past_state_followup
            )
            z_lab_followup_mu, z_lab_followup_logvar = followup_encodings["lab"]
            z_past_followup_mu, z_past_followup_logvar = followup_encodings["past"]
            
            # Use Baseline metabolomics encoding for alignment (but not for fusion)
            z_metab_followup_mu = z_metab_baseline_mu
            z_metab_followup_logvar = z_metab_baseline_logvar
            
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Align Follow-up distributions
        alignment_loss_followup = align_distributions(
            z_lab_followup_mu, z_lab_followup_logvar,
            z_metab_followup_mu, z_metab_followup_logvar,
            z_past_followup_mu, z_past_followup_logvar
        )
        
        # Compute Follow-up Visit State
        visit_state_followup = compute_visit_state(
            z_lab_followup_mu,
            z_metab_followup_mu,
            z_past_followup_mu
        )
        
        # Decode Follow-up visit
        lab_recon_followup = self.lab_decoder(visit_state_followup)
        metab_recon_followup = self.metab_decoder(visit_state_followup)
        
        # ===== COMPUTE LOSSES =====
        # Reconstruction losses (only for available data)
        lab_recon_loss = reconstruction_loss(
            lab_recon_baseline, lab_tests_baseline
        ) + reconstruction_loss(
            lab_recon_followup, lab_tests_followup
        )
        
        metab_recon_loss = reconstruction_loss(
            metab_recon_baseline, metabolomics_baseline
        )
        if metabolomics_followup is not None:
            metab_recon_loss += reconstruction_loss(
                metab_recon_followup, metabolomics_followup
            )
        
        # KL divergence losses
        lab_kl_loss = kl_divergence_loss(z_lab_baseline_mu, z_lab_baseline_logvar)
        if mode != "baseline_only":
            lab_kl_loss += kl_divergence_loss(z_lab_followup_mu, z_lab_followup_logvar)
        
        metab_kl_loss = kl_divergence_loss(z_metab_baseline_mu, z_metab_baseline_logvar)
        past_kl_loss = kl_divergence_loss(z_past_baseline_mu, z_past_baseline_logvar)
        past_kl_loss += kl_divergence_loss(z_past_followup_mu, z_past_followup_logvar)
        
        # Alignment loss
        alignment_loss = alignment_loss_baseline + alignment_loss_followup
        
        # Adversarial loss (only for metabolomics)
        if metabolomics_followup is not None:
            # Use real follow-up metabolomics
            real_metab = metabolomics_followup
        else:
            # Use baseline metabolomics as proxy for real data
            real_metab = metabolomics_baseline
        
        adversarial_loss = adversarial_loss_generator(
            self.discriminator,
            metab_recon_followup
        )
        
        # Total loss
        losses = compute_total_loss(
            lab_recon_loss=lab_recon_loss,
            metab_recon_loss=metab_recon_loss,
            lab_kl_loss=lab_kl_loss,
            metab_kl_loss=metab_kl_loss,
            past_kl_loss=past_kl_loss,
            alignment_loss=alignment_loss,
            adversarial_loss=adversarial_loss,
            lambda_kl=self.lambda_kl,
            lambda_align=self.lambda_align,
            lambda_adv=self.lambda_adv,
            kl_annealing_weight=kl_annealing_weight
        )
        
        # Prepare output
        output = {
            "lab_recon_baseline": lab_recon_baseline,
            "metab_recon_baseline": metab_recon_baseline,
            "lab_recon_followup": lab_recon_followup,
            "metab_recon_followup": metab_recon_followup,
            "losses": losses
        }
        
        if return_visit_state:
            output["visit_state_followup"] = visit_state_followup
        
        return output

