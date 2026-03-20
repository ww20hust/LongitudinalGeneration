"""
Temporal modules for longitudinal state propagation.

The autoregressive component of PEAG can be implemented with either a
recurrent update or a lightweight transformer-style attention block.
"""
from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn


def _build_sinusoidal_encoding(max_seq_len: int, latent_dim: int) -> torch.Tensor:
    positions = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, latent_dim, 2, dtype=torch.float32)
        * (-math.log(10000.0) / latent_dim)
    )

    encoding = torch.zeros(max_seq_len, latent_dim, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(positions * div_term)
    encoding[:, 1::2] = torch.cos(positions * div_term)
    return encoding


class RecurrentTemporalModule(nn.Module):
    """GRU-style temporal update for the historical latent state."""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.cell = nn.GRUCell(latent_dim, latent_dim)

    def init_context(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.latent_dim, device=device)

    def get_historical_state(
        self,
        context: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        del batch_size, device
        return context

    def update(self, context: torch.Tensor, visit_state: torch.Tensor) -> torch.Tensor:
        return self.cell(visit_state, context)


class TransformerTemporalModule(nn.Module):
    """Causal self-attention temporal update for longer visit histories."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
        max_seq_len: int = 128,
    ):
        super().__init__()
        if latent_dim % num_heads != 0:
            raise ValueError(
                f"latent_dim ({latent_dim}) must be divisible by num_heads ({num_heads}) "
                "for transformer temporal modeling."
            )

        self.latent_dim = latent_dim
        self.max_seq_len = max_seq_len
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=max(hidden_dim, latent_dim * 2),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.input_norm = nn.LayerNorm(latent_dim)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(latent_dim, latent_dim)
        self.register_buffer(
            "positional_encoding",
            _build_sinusoidal_encoding(max_seq_len=max_seq_len, latent_dim=latent_dim),
            persistent=False,
        )

    def init_context(self, batch_size: int, device: torch.device) -> List[torch.Tensor]:
        del batch_size, device
        return []

    def get_historical_state(
        self,
        context: List[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if not context:
            return torch.zeros(batch_size, self.latent_dim, device=device)

        history = context[-self.max_seq_len :]
        sequence = torch.stack(history, dim=1)
        seq_len = sequence.size(1)

        position = self.positional_encoding[:seq_len].to(sequence.device).unsqueeze(0)
        encoded_input = self.input_norm(sequence + position)

        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=sequence.device),
            diagonal=1,
        )
        encoded = self.encoder(encoded_input, mask=causal_mask)
        return self.output_proj(encoded[:, -1, :])

    def update(
        self,
        context: List[torch.Tensor],
        visit_state: torch.Tensor,
    ) -> List[torch.Tensor]:
        updated = list(context)
        updated.append(visit_state)
        if len(updated) > self.max_seq_len:
            updated = updated[-self.max_seq_len :]
        return updated


def build_temporal_module(
    temporal_model: str,
    latent_dim: int,
    hidden_dim: int,
    num_heads: int = 4,
    num_layers: int = 1,
    dropout: float = 0.1,
    max_seq_len: int = 128,
) -> nn.Module:
    """Factory for PEAG temporal modules."""
    temporal_model = temporal_model.lower()
    if temporal_model in {"recurrent", "gru", "rnn"}:
        return RecurrentTemporalModule(latent_dim=latent_dim)
    if temporal_model == "transformer":
        return TransformerTemporalModule(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )
    raise ValueError(
        f"Unsupported temporal_model '{temporal_model}'. "
        "Choose from: recurrent, transformer."
    )
