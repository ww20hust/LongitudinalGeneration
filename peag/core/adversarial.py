"""
Adversarial components for PEAG framework.

This module implements the discriminator for adversarial training
to prevent generated data from revealing missingness patterns.
"""
import torch
import torch.nn as nn


class MissingnessDiscriminator(nn.Module):
    """
    Discriminator for adversarial training.
    
    Discriminates between real and generated (imputed) data
to ensure generated data is indistinguishable from real data.
    """
    
    def __init__(self, input_dim: int = 251, hidden_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Discriminate real vs generated data.
        
        Args:
            x: Input data (batch_size, input_dim)
        
        Returns:
            Probability of being real (batch_size, 1)
        """
        return self.discriminator(x)


class ModalityDiscriminator(nn.Module):
    """
    Multi-modal discriminator for different data types.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.discriminator(x)
