"""
Visit State fusion mechanism with missing modality handling.

This module implements the fusion of aligned modality-specific latent distributions
into a unified Visit State representation, supporting dynamic number of modalities
and missing modality handling.

Reference: Methods section - Visit State Estimation
"""
import torch
import torch.nn as nn


def compute_visit_state_dynamic(
    modality_mus: dict[str, torch.Tensor],
    z_past_mu: torch.Tensor,
    include_history: bool = True,
) -> torch.Tensor:
    """
    Compute Visit State by equal-weight fusion of current modalities and history.

    Formula:
        z_t = (z_past + sum(z_mod for current available modalities)) / (M_total + 1)

    where M_total counts only the currently observed modalities and does not
    include the historical state.
    
    The Visit State is the integrative latent state that combines information
    from all AVAILABLE modalities and the historical context. Missing modalities
    are excluded from the fusion.
    
    Reference: Methods section - Visit State Estimation
    
    Args:
        modality_mus: Dictionary mapping modality names to their encoder means.
                      Only available (non-missing) modalities should be included.
                      Each tensor has shape (batch_size, latent_dim)
        z_past_mu: Past-State encoder mean (batch_size, latent_dim)
    
    Returns:
        Fused Visit State of shape (batch_size, latent_dim)
    """
    current_modality_means = []
    for mod_name, z_mu in modality_mus.items():
        if z_mu is not None:
            current_modality_means.append(z_mu)

    m_total = len(current_modality_means)
    if m_total == 0:
        if include_history:
            return z_past_mu
        return torch.zeros_like(z_past_mu)

    current_sum = torch.stack(current_modality_means, dim=0).sum(dim=0)
    if not include_history:
        return current_sum / float(m_total)
    return (z_past_mu + current_sum) / float(m_total + 1)


class AdaptiveVisitStateFusion(nn.Module):
    """
    Adaptive fusion module that learns to weight different modalities.
    
    This module learns importance weights for each modality based on
    their latent representations, allowing adaptive fusion even with
    missing modalities.
    """
    
    def __init__(self, latent_dim: int, num_modalities: int = 2):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_modalities = num_modalities
        
        # Attention mechanism for modality weighting
        self.attention = nn.Sequential(
            nn.Linear(latent_dim * (num_modalities + 1), hidden_dim := 64),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_modalities + 1),
            nn.Softmax(dim=-1)
        )
    
    def forward(
        self,
        modality_mus: dict[str, torch.Tensor],
        z_past_mu: torch.Tensor,
        missing_mask: dict[str, int]
    ) -> torch.Tensor:
        """
        Compute weighted Visit State using attention mechanism.
        
        Args:
            modality_mus: Dictionary of modality means
            z_past_mu: Past-State mean
            missing_mask: Dictionary indicating which modalities are missing
                          (0 = available, 1 = actively masked, 2 = naturally missing)
        
        Returns:
            Weighted Visit State
        """
        batch_size = z_past_mu.shape[0]
        
        # Collect means and create padding for missing modalities
        means_list = []
        valid_mask = []
        
        for mod_name in sorted(modality_mus.keys()):  # Ensure consistent ordering
            if missing_mask.get(mod_name, 2) == 0 and modality_mus[mod_name] is not None:
                means_list.append(modality_mus[mod_name])
                valid_mask.append(1.0)
            else:
                # Use zeros for missing or masked modalities
                means_list.append(torch.zeros_like(z_past_mu))
                valid_mask.append(0.0)
        
        # Add past state
        means_list.append(z_past_mu)
        valid_mask.append(1.0)
        
        # Stack means
        means_tensor = torch.stack(means_list, dim=1)  # (batch, n_mods+1, latent)
        valid_mask_tensor = torch.tensor(valid_mask, device=z_past_mu.device)
        
        # Compute attention weights
        concat_means = torch.cat(means_list, dim=-1)  # (batch, (n_mods+1)*latent)
        attn_weights = self.attention(concat_means)  # (batch, n_mods+1)
        
        # Mask out missing modalities and renormalize
        masked_weights = attn_weights * valid_mask_tensor.unsqueeze(0)
        normalized_weights = masked_weights / (masked_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        # Weighted sum
        visit_state = torch.sum(
            means_tensor * normalized_weights.unsqueeze(-1),
            dim=1
        )  # (batch, latent)
        
        return visit_state
