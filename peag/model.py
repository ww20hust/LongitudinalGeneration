"""
Main PEAG model class with multi-visit and missing modality support.

This module implements the complete PEAG framework with:
1. Loop-based processing for arbitrary number of visits
2. Missing modality handling with masks
3. Dynamic alignment and fusion based on available modalities
4. Supervised imputation using available samples

Reference: Methods section - Patient-context Enhanced Longitudinal 
           Multimodal Alignment and Generation Framework
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from peag.core.encoders import LabTestsEncoder, MetabolomicsEncoder, PastStateEncoder
from peag.core.decoders import LabTestsDecoder, MetabolomicsDecoder
from peag.core.alignment import align_distributions_dynamic
from peag.core.fusion import compute_visit_state_dynamic
from peag.core.adversarial import MissingnessDiscriminator
from peag.utils.distributions import reparameterize
from peag.losses import (
    reconstruction_loss,
    kl_divergence_loss,
    adversarial_loss_generator,
    compute_total_loss
)


class PEAGModel(nn.Module):
    """
    Patient-context Enhanced Longitudinal Multimodal Alignment and Generation Model.
    
    Enhanced version supporting:
    - Multiple visits (loop-based processing)
    - Missing modalities per visit (with masks)
    - Dynamic alignment and fusion
    - Supervised imputation training
    
    The model processes longitudinal clinical visits by:
    1. For each visit:
       a. Encode available modalities (skip missing ones)
       b. Align available distributions (exclude missing from alignment)
       c. Fuse into Visit State using only available modalities
       d. Decode all modalities from Visit State
       e. Use Visit State as Past-State for next visit
    2. Compute losses with missing value masking
    3. Train imputation using available samples as supervision
    
    Reference: Methods section - The PEAG framework
    """
    
    def __init__(
        self,
        modality_dims: Dict[str, int],
        latent_dim: int = 16,
        hidden_dim: int = 128,
        lambda_kl: float = 1.0,
        lambda_align: float = 1.0,
        lambda_adv: float = 1.0
    ):
        """
        Initialize PEAG model.
        
        Args:
            modality_dims: Dictionary mapping modality names to their dimensions.
                          e.g., {"lab": 61, "metab": 251}
            latent_dim: Dimension of latent space (default: 16)
            hidden_dim: Dimension of hidden layers (default: 128)
            lambda_kl: Weight for KL divergence loss (default: 1.0)
            lambda_align: Weight for alignment penalty (default: 1.0)
            lambda_adv: Weight for adversarial loss (default: 1.0)
        """
        super(PEAGModel, self).__init__()
        
        self.modality_dims = modality_dims
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.lambda_kl = lambda_kl
        self.lambda_align = lambda_align
        self.lambda_adv = lambda_adv
        
        # Create encoders for each modality
        self.encoders = nn.ModuleDict()
        for mod_name, mod_dim in modality_dims.items():
            if mod_name == "lab":
                self.encoders[mod_name] = LabTestsEncoder(
                    input_dim=mod_dim, latent_dim=latent_dim, hidden_dim=hidden_dim
                )
            elif mod_name == "metab":
                self.encoders[mod_name] = MetabolomicsEncoder(
                    input_dim=mod_dim, latent_dim=latent_dim, hidden_dim=hidden_dim
                )
            else:
                # Generic encoder for other modalities
                from peag.core.encoders import GenericModalityEncoder
                self.encoders[mod_name] = GenericModalityEncoder(
                    input_dim=mod_dim, latent_dim=latent_dim, hidden_dim=hidden_dim
                )
        
        # Past-State encoder
        self.past_encoder = PastStateEncoder(
            latent_dim=latent_dim, hidden_dim=hidden_dim // 2
        )
        
        # Create decoders for each modality
        self.decoders = nn.ModuleDict()
        for mod_name, mod_dim in modality_dims.items():
            if mod_name == "lab":
                self.decoders[mod_name] = LabTestsDecoder(
                    latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=mod_dim
                )
            elif mod_name == "metab":
                self.decoders[mod_name] = MetabolomicsDecoder(
                    latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=mod_dim
                )
            else:
                # Generic decoder for other modalities
                from peag.core.decoders import GenericModalityDecoder
                self.decoders[mod_name] = GenericModalityDecoder(
                    latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=mod_dim
                )
        
        # Adversarial discriminator (for metabolomics or primary modality)
        primary_modality = list(modality_dims.keys())[0] if modality_dims else "metab"
        self.discriminator = MissingnessDiscriminator(
            input_dim=modality_dims.get(primary_modality, 251),
            hidden_dim=hidden_dim
        )
    
    def encode_visit(
        self,
        visit_data: Dict[str, torch.Tensor],
        missing_mask: Dict[str, int],
        past_state: torch.Tensor
    ) -> Tuple[
        Dict[str, Tuple[torch.Tensor, torch.Tensor]],  # modality distributions
        Tuple[torch.Tensor, torch.Tensor],              # past distribution
        Dict[str, torch.Tensor]                        # available modality means
    ]:
        """
        Encode a single visit's data.
        
        Args:
            visit_data: Dictionary of modality data for this visit
            missing_mask: Dictionary indicating which modalities are missing
            past_state: Past-State from previous visit
        
        Returns:
            modality_dists: Distributions for each available modality
            past_dist: Distribution for Past-State
            available_mus: Means of available modalities for fusion
        """
        device = past_state.device
        
        # Encode available modalities
        modality_dists = {}
        available_mus = {}
        
        for mod_name, mod_data in visit_data.items():
            # Skip if modality is not available (mask != 0) or input is None
            if missing_mask.get(mod_name, 2) != 0 or mod_data is None:
                continue
            
            # Encode this modality
            mu, logvar = self.encoders[mod_name](mod_data)
            modality_dists[mod_name] = (mu, logvar)
            available_mus[mod_name] = mu
        
        # Encode Past-State
        past_mu, past_logvar = self.past_encoder(past_state)
        past_dist = (past_mu, past_logvar)
        
        return modality_dists, past_dist, available_mus
    
    def process_visit(
        self,
        visit_data: Dict[str, torch.Tensor],
        missing_mask: Dict[str, int],
        past_state: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, Dict[str, Any]]:
        """
        Process a single visit through encode-align-fuse-decode pipeline.
        
        Args:
            visit_data: Dictionary of modality data
            missing_mask: Dictionary of missing flags
            past_state: Past-State tensor
        
        Returns:
            visit_state: Computed Visit State
            reconstructions: Dictionary of reconstructed modalities
            alignment_loss: Alignment loss for this visit
            debug_info: Dictionary with intermediate values for debugging
        """
        # Encode
        modality_dists, (past_mu, past_logvar), available_mus = self.encode_visit(
            visit_data, missing_mask, past_state
        )
        
        # Align distributions (only available modalities)
        alignment_loss = align_distributions_dynamic(
            modality_distributions=modality_dists,
            z_past_mu=past_mu,
            z_past_logvar=past_logvar,
            missing_mask=missing_mask
        )
        
        # Compute Visit State (only using available modalities)
        visit_state = compute_visit_state_dynamic(available_mus, past_mu)
        
        # Decode all modalities from Visit State
        reconstructions = {}
        for mod_name in self.modality_dims.keys():
            reconstructions[mod_name] = self.decoders[mod_name](visit_state)
        
        debug_info = {
            "modality_dists": modality_dists,
            "past_dist": (past_mu, past_logvar),
            "available_mus": available_mus
        }
        
        return visit_state, reconstructions, alignment_loss, debug_info
    
    def forward(
        self,
        visits_data: List[Dict[str, torch.Tensor]],
        missing_masks: List[Dict[str, int]],
        kl_annealing_weight: float = 1.0,
        return_all_visit_states: bool = False,
        recon_targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass through the model with loop-based multi-visit processing.
        
        Args:
            visits_data: List of visit data dictionaries.
                        Each dictionary maps modality names to tensors.
                        Length = number of visits.
            missing_masks: List of missing mask dictionaries.
                          Each dictionary maps modality names to int codes:
                          0 = available, 1 = actively masked, 2 = naturally missing.
                          Length = number of visits.
            kl_annealing_weight: KL annealing weight (0 to 1)
            return_all_visit_states: Whether to return all visit states
        
        Returns:
            Dictionary containing:
            - reconstructions: List of reconstruction dicts per visit
            - losses: All loss components
            - visit_states: List of visit states (if return_all_visit_states=True)
        
        Example:
            >>> visits_data = [
            ...     {"lab": lab_t1, "metab": metab_t1},  # Visit 1
            ...     {"lab": lab_t2, "metab": None},       # Visit 2 (missing metab)
            ...     {"lab": lab_t3, "metab": metab_t3},   # Visit 3
            ... ]
            >>> missing_masks = [
            ...     {"lab": False, "metab": False},
            ...     {"lab": False, "metab": True},
            ...     {"lab": False, "metab": False},
            ... ]
        """
        if len(visits_data) == 0:
            raise ValueError("No visit data provided")
        
        if len(visits_data) != len(missing_masks):
            raise ValueError("visits_data and missing_masks must have same length")
        
        batch_size = list(visits_data[0].values())[0].shape[0]
        device = list(visits_data[0].values())[0].device
        
        # Initialize Past-State for first visit (zeros)
        past_state = torch.zeros(batch_size, self.latent_dim, device=device)
        
        # Storage for outputs
        all_reconstructions = []
        all_visit_states = []
        all_alignment_losses = []
        all_modality_dists = []
        all_past_dists = []
        
        # ===== LOOP THROUGH ALL VISITS =====
        for visit_idx, (visit_data, missing_mask) in enumerate(zip(visits_data, missing_masks)):
            # Process this visit
            visit_state, reconstructions, alignment_loss, debug_info = self.process_visit(
                visit_data, missing_mask, past_state
            )
            
            # Store outputs
            all_reconstructions.append(reconstructions)
            all_visit_states.append(visit_state)
            all_alignment_losses.append(alignment_loss)
            all_modality_dists.append(debug_info["modality_dists"])
            all_past_dists.append(debug_info["past_dist"])
            
            # Update Past-State for next visit
            past_state = visit_state
        
        # ===== COMPUTE LOSSES =====
        losses = self.compute_losses(
            visits_data=visits_data,
            missing_masks=missing_masks,
            all_reconstructions=all_reconstructions,
            all_modality_dists=all_modality_dists,
            all_past_dists=all_past_dists,
            all_alignment_losses=all_alignment_losses,
            kl_annealing_weight=kl_annealing_weight,
            recon_targets=recon_targets,
        )
        
        # Prepare output
        output = {
            "reconstructions": all_reconstructions,
            "losses": losses
        }
        
        if return_all_visit_states:
            output["visit_states"] = all_visit_states
        
        return output
    
    def compute_losses(
        self,
        visits_data: List[Dict[str, torch.Tensor]],
        missing_masks: List[Dict[str, int]],
        all_reconstructions: List[Dict[str, torch.Tensor]],
        all_modality_dists: List[Dict[str, Tuple[torch.Tensor, torch.Tensor]]],
        all_past_dists: List[Tuple[torch.Tensor, torch.Tensor]],
        all_alignment_losses: List[torch.Tensor],
        kl_annealing_weight: float,
        recon_targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses with proper handling of missing modalities.
        
        Reconstruction loss:
            - If recon_targets is provided (active-mask training), computed for
              modalities with mask 0 or 1 (available or actively masked) where
              recon_targets provide a target (mask 2 is naturally missing and
              does not contribute).
            - If recon_targets is None, computed only for modalities with mask 0
              and non-None inputs in visits_data (backward-compatible behavior).
        KL loss: Only computed for modalities that were encoded (not missing)
        Alignment loss: Already computed to exclude missing modalities
        Adversarial loss: Computed on all reconstructions
        """
        device = list(visits_data[0].values())[0].device
        
        # Reconstruction losses per modality
        recon_losses = {mod_name: 0.0 for mod_name in self.modality_dims.keys()}
        
        # KL losses per encoder
        kl_losses = {mod_name: 0.0 for mod_name in self.modality_dims.keys()}
        kl_losses["past"] = 0.0
        
        # Track counts for averaging
        recon_counts = {mod_name: 0 for mod_name in self.modality_dims.keys()}
        kl_counts = {mod_name: 0 for mod_name in self.modality_dims.keys()}
        kl_counts["past"] = 0
        
        # Process each visit
        for visit_idx, (visit_data, missing_mask, reconstructions, modality_dists, past_dist) in enumerate(
            zip(visits_data, missing_masks, all_reconstructions, all_modality_dists, all_past_dists)
        ):
            # Reconstruction losses
            for mod_name in self.modality_dims.keys():
                mask_value = missing_mask.get(mod_name, 2)
                if recon_targets is not None:
                    # Active-mask training: supervise for mask 0 and 1 when target exists
                    target_dict = recon_targets[visit_idx]
                    target = target_dict.get(mod_name)
                    if mask_value in (0, 1) and target is not None:
                        recon = reconstructions[mod_name]
                        loss = reconstruction_loss(recon, target)
                        recon_losses[mod_name] += loss
                        recon_counts[mod_name] += 1
                else:
                    # Legacy behavior: supervise only for mask 0 and non-None input
                    target = visit_data.get(mod_name)
                    if mask_value == 0 and target is not None:
                        recon = reconstructions[mod_name]
                        loss = reconstruction_loss(recon, target)
                        recon_losses[mod_name] += loss
                        recon_counts[mod_name] += 1
            
            # KL losses - only for modalities that were encoded
            for mod_name, (mu, logvar) in modality_dists.items():
                kl_losses[mod_name] += kl_divergence_loss(mu, logvar)
                kl_counts[mod_name] += 1
            
            # Past-State KL loss
            past_mu, past_logvar = past_dist
            kl_losses["past"] += kl_divergence_loss(past_mu, past_logvar)
            kl_counts["past"] += 1
        
        # Average losses over visits
        for mod_name in recon_losses.keys():
            if recon_counts[mod_name] > 0:
                recon_losses[mod_name] = recon_losses[mod_name] / recon_counts[mod_name]
            else:
                recon_losses[mod_name] = torch.tensor(0.0, device=device)
        
        for mod_name in kl_losses.keys():
            if kl_counts[mod_name] > 0:
                kl_losses[mod_name] = kl_losses[mod_name] / kl_counts[mod_name]
            else:
                kl_losses[mod_name] = torch.tensor(0.0, device=device)
        
        # Sum alignment losses
        alignment_loss = sum(all_alignment_losses) / len(all_alignment_losses)
        
        # Adversarial loss (use last visit's reconstructions)
        last_recon = all_reconstructions[-1]
        primary_modality = list(self.modality_dims.keys())[0]
        adversarial_loss = adversarial_loss_generator(
            self.discriminator, last_recon[primary_modality]
        )
        
        # Total loss
        total_loss_dict = compute_total_loss(
            recon_losses=recon_losses,
            kl_losses=kl_losses,
            alignment_loss=alignment_loss,
            adversarial_loss=adversarial_loss,
            lambda_kl=self.lambda_kl,
            lambda_align=self.lambda_align,
            lambda_adv=self.lambda_adv,
            kl_annealing_weight=kl_annealing_weight
        )
        
        return total_loss_dict
    
    def impute_missing(
        self,
        visits_data: List[Dict[str, torch.Tensor]],
        missing_masks: List[Dict[str, int]]
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Impute missing modalities for all visits.
        
        Args:
            visits_data: List of visit data dictionaries
            missing_masks: List of missing mask dictionaries
        
        Returns:
            List of dictionaries containing imputed data for all modalities
        """
        with torch.no_grad():
            output = self.forward(
                visits_data=visits_data,
                missing_masks=missing_masks,
                kl_annealing_weight=1.0,
                return_all_visit_states=False,
            )
        
        # Return reconstructions as imputed values
        imputed_data = []
        for visit_idx, (visit_data, missing_mask, reconstructions) in enumerate(
            zip(visits_data, missing_masks, output["reconstructions"])
        ):
            imputed_visit = {}
            for mod_name in self.modality_dims.keys():
                mask_value = missing_mask.get(mod_name, 0)
                if mask_value == 2 or visit_data.get(mod_name) is None:
                    # Naturally missing (or no original data) - use imputed value
                    imputed_visit[mod_name] = reconstructions[mod_name]
                else:
                    # Available modality - keep original value
                    imputed_visit[mod_name] = visit_data[mod_name]
            imputed_data.append(imputed_visit)
        
        return imputed_data
