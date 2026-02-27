"""
Modality-specific decoders for PEAG framework.

This module implements VAE decoders for different clinical modalities.
"""
import torch
import torch.nn as nn
from typing import Optional


class LabTestsDecoder(nn.Module):
    """VAE Decoder for lab test features."""
    
    def __init__(self, latent_dim: int = 16, hidden_dim: int = 128, output_dim: int = 61):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode Visit State to lab tests.
        
        Args:
            z: Visit State (batch_size, latent_dim)
        
        Returns:
            Reconstructed lab tests (batch_size, output_dim)
        """
        return self.decoder(z)


class MetabolomicsDecoder(nn.Module):
    """VAE Decoder for metabolomics features."""
    
    def __init__(self, latent_dim: int = 16, hidden_dim: int = 128, output_dim: int = 251):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode Visit State to metabolomics.
        
        Args:
            z: Visit State (batch_size, latent_dim)
        
        Returns:
            Reconstructed metabolomics (batch_size, output_dim)
        """
        return self.decoder(z)


class GenericModalityDecoder(nn.Module):
    """Generic decoder that can be used for any modality."""
    
    def __init__(self, latent_dim: int = 16, hidden_dim: int = 128, output_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
