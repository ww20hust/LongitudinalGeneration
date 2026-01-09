"""
Training configuration for PEAG framework.

This module defines hyperparameters and training settings as specified in the
Methods section.

Reference: Methods section - Training Implementation Details
"""

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """
    Training configuration class.
    
    All hyperparameters are set according to the Methods section.
    """
    # Optimizer settings
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 64
    
    # Training settings
    max_epochs: int = 500
    early_stopping_patience: int = 50
    kl_annealing_epochs: int = 50
    
    # Loss weights
    lambda_kl: float = 1.0
    lambda_align: float = 1.0
    lambda_adv: float = 1.0
    
    # Device
    device: str = "cuda"  # Will be set to "cpu" if CUDA not available
    
    # Checkpoint settings
    save_dir: str = "./checkpoints"
    save_best: bool = True
    
    # Validation settings
    val_split: float = 0.1  # 10% of training data for validation
    
    def __post_init__(self):
        """Set device based on availability."""
        import torch
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"

