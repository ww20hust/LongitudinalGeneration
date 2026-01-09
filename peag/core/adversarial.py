"""
Adversarial discriminator for missingness pattern detection.

The discriminator is trained to classify whether generated data was fully measured
or imputed, preventing the model from revealing missingness patterns.

Reference: Methods section - Missingness Adversarial Training
"""

import torch
import torch.nn as nn


class MissingnessDiscriminator(nn.Module):
    """
    Discriminator network that classifies generated data as fully measured or imputed.
    
    The discriminator takes generated metabolomics data and outputs a binary
    classification probability. During training, the generator (decoder) is
    trained to confuse this discriminator.
    """
    
    def __init__(self, input_dim: int = 251, hidden_dim: int = 128):
        """
        Initialize missingness discriminator.
        
        Args:
            input_dim: Dimension of input features (metabolomics: 251)
            hidden_dim: Dimension of hidden layers (default: 128)
        """
        super(MissingnessDiscriminator, self).__init__()
        
        self.input_dim = input_dim
        
        # Multi-layer network
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc_out = nn.Linear(hidden_dim // 2, 1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classify input as fully measured (1) or imputed (0).
        
        Args:
            x: Generated metabolomics data of shape (batch_size, input_dim)
        
        Returns:
            Binary classification probability of shape (batch_size, 1)
        """
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        h = self.relu(self.fc3(h))
        output = self.sigmoid(self.fc_out(h))
        
        return output

