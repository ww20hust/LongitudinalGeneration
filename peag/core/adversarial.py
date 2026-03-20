"""
Adversarial components for PEAG framework.

The adversarial regularizer operates on modality reconstructions. Each
adversary receives a reconstructed modality and tries to predict whether that
modality was missing at the current visit. Missing includes both actively
masked and naturally missing cases. A gradient reversal layer makes the
reconstruction pathway invariant to this cue.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.autograd import Function


class _GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, lambda_grl: float) -> torch.Tensor:
        ctx.lambda_grl = lambda_grl
        return input_tensor.view_as(input_tensor)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_grl * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_grl: float = 1.0):
        super().__init__()
        self.lambda_grl = float(lambda_grl)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return _GradientReversalFunction.apply(input_tensor, self.lambda_grl)


class ModalityMissingnessAdversary(nn.Module):
    """
    Predict whether a modality was missing from its reconstruction.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        lambda_grl: float = 1.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.grl = GradientReversalLayer(lambda_grl=lambda_grl)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        reversed_features = self.grl(features)
        return self.classifier(reversed_features)
