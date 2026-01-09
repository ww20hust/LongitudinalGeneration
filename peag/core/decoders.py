"""
Modality-specific decoders for PEAG framework.

Each decoder reconstructs modality features from the fused Visit State
using a two-layer neural network.

Reference: Methods section - Modality-Specific Decoders
"""

import torch
import torch.nn as nn


class ModalityDecoder(nn.Module):
    """
    Base class for modality decoders.
    
    Two-layer non-linear neural network that maps Visit State to
    reconstructed modality features.
    """
    
    def __init__(self, output_dim: int, latent_dim: int = 16, hidden_dim: int = 128):
        """
        Initialize modality decoder.
        
        Args:
            output_dim: Dimension of output features
            latent_dim: Dimension of latent space (default: 16)
            hidden_dim: Dimension of hidden layer (default: 128)
        """
        super(ModalityDecoder, self).__init__()
        
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        
        # Two-layer network with ReLU activation
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        
        # Output layer
        self.fc_out = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, visit_state: torch.Tensor) -> torch.Tensor:
        """
        Decode Visit State to modality features.
        
        Args:
            visit_state: Visit State vector of shape (batch_size, latent_dim)
        
        Returns:
            Reconstructed features of shape (batch_size, output_dim)
        """
        h = self.relu(self.fc1(visit_state))
        h = self.relu(self.fc2(h))
        output = self.fc_out(h)
        
        return output


class LabTestsDecoder(ModalityDecoder):
    """
    Decoder for laboratory tests (61 features).
    
    Reference: Methods section - Modality-Specific Decoders
    """
    
    def __init__(self, latent_dim: int = 16, hidden_dim: int = 128):
        """
        Initialize lab tests decoder.
        
        Args:
            latent_dim: Dimension of latent space (default: 16)
            hidden_dim: Dimension of hidden layer (default: 128)
        """
        super(LabTestsDecoder, self).__init__(
            output_dim=61,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim
        )


class MetabolomicsDecoder(ModalityDecoder):
    """
    Decoder for metabolomics data (251 features).
    
    Reference: Methods section - Modality-Specific Decoders
    """
    
    def __init__(self, latent_dim: int = 16, hidden_dim: int = 128):
        """
        Initialize metabolomics decoder.
        
        Args:
            latent_dim: Dimension of latent space (default: 16)
            hidden_dim: Dimension of hidden layer (default: 128)
        """
        super(MetabolomicsDecoder, self).__init__(
            output_dim=251,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim
        )

