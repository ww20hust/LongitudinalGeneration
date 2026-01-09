"""
Modality-specific encoders for PEAG framework.

Each encoder maps input features to parameters of a 16-dimensional isotropic
multivariate normal distribution using a two-layer neural network.

Reference: Methods section - Modality-Specific Encoders and Past-State Integration
"""

import torch
import torch.nn as nn


class TabularEncoder(nn.Module):
    """
    Base class for tabular data encoders.
    
    Two-layer non-linear neural network that maps input features to
    16-dimensional latent distribution parameters.
    """
    
    def __init__(self, input_dim: int, latent_dim: int = 16, hidden_dim: int = 128):
        """
        Initialize tabular encoder.
        
        Args:
            input_dim: Dimension of input features
            latent_dim: Dimension of latent space (default: 16)
            hidden_dim: Dimension of hidden layer (default: 128)
        """
        super(TabularEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Two-layer network with ReLU activation
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        
        # Output layers for mean and log variance
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters.
        
        Args:
            x: Input features of shape (batch_size, input_dim)
        
        Returns:
            mu: Mean vector of shape (batch_size, latent_dim)
            logvar: Log variance vector of shape (batch_size, latent_dim)
        """
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar


class LabTestsEncoder(TabularEncoder):
    """
    Encoder for laboratory tests (61 features).
    
    Reference: Methods section - Modality-Specific Encoders
    """
    
    def __init__(self, latent_dim: int = 16, hidden_dim: int = 128):
        """
        Initialize lab tests encoder.
        
        Args:
            latent_dim: Dimension of latent space (default: 16)
            hidden_dim: Dimension of hidden layer (default: 128)
        """
        super(LabTestsEncoder, self).__init__(
            input_dim=61,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim
        )


class MetabolomicsEncoder(TabularEncoder):
    """
    Encoder for metabolomics data (251 features).
    
    Reference: Methods section - Modality-Specific Encoders
    """
    
    def __init__(self, latent_dim: int = 16, hidden_dim: int = 128):
        """
        Initialize metabolomics encoder.
        
        Args:
            latent_dim: Dimension of latent space (default: 16)
            hidden_dim: Dimension of hidden layer (default: 128)
        """
        super(MetabolomicsEncoder, self).__init__(
            input_dim=251,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim
        )


class PastStateEncoder(nn.Module):
    """
    Encoder for Past-State (16-dimensional input from previous visit).
    
    Processes the joint latent representation from the previous visit
    into a 16-dimensional distribution for alignment.
    
    Reference: Methods section - Modality-Specific Encoders and Past-State Integration
    """
    
    def __init__(self, latent_dim: int = 16, hidden_dim: int = 64):
        """
        Initialize Past-State encoder.
        
        Args:
            latent_dim: Dimension of latent space (default: 16)
            hidden_dim: Dimension of hidden layer (default: 64)
        """
        super(PastStateEncoder, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Two-layer network with ReLU activation
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        
        # Output layers for mean and log variance
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, past_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode Past-State to latent distribution parameters.
        
        Args:
            past_state: Past-State vector of shape (batch_size, latent_dim)
        
        Returns:
            mu: Mean vector of shape (batch_size, latent_dim)
            logvar: Log variance vector of shape (batch_size, latent_dim)
        """
        h = self.relu(self.fc1(past_state))
        h = self.relu(self.fc2(h))
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar

