from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from common import PositionalEncoding, masked_mean


class LabSequenceEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 64,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim * 2, d_model)
        self.position = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_dim = d_model

    def forward(self, labs: torch.Tensor, lab_mask: torch.Tensor) -> torch.Tensor:
        timestep_mask = lab_mask.any(dim=-1)
        x = torch.cat([labs, lab_mask], dim=-1)
        x = self.input_projection(x)
        x = self.position(x)
        x = self.encoder(x, src_key_padding_mask=~timestep_mask.bool())
        return masked_mean(x, timestep_mask.float(), dim=1)


class DocumentEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.output_dim = hidden_dim
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, document_embeddings: torch.Tensor) -> torch.Tensor:
        return self.network(document_embeddings)


class NotesOnlyLlamaClassifier(nn.Module):
    def __init__(self, document_dim: int, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.document_encoder = DocumentEncoder(document_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        doc_repr = self.document_encoder(batch['document_embeddings'])
        return self.classifier(doc_repr).squeeze(-1)


class LabTransformerClassifier(nn.Module):
    def __init__(
        self,
        lab_dim: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 64,
    ) -> None:
        super().__init__()
        self.encoder = LabSequenceEncoder(
            input_dim=lab_dim,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max_len,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.encoder.output_dim),
            nn.Linear(self.encoder.output_dim, self.encoder.output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.encoder.output_dim // 2, 1),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        lab_repr = self.encoder(batch['labs'], batch['lab_mask'])
        return self.classifier(lab_repr).squeeze(-1)


class SimplyCombinedClassifier(nn.Module):
    def __init__(
        self,
        lab_dim: int,
        document_dim: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 64,
        document_hidden_dim: int = 256,
        fusion_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.lab_encoder = LabSequenceEncoder(
            input_dim=lab_dim,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max_len,
        )
        self.document_encoder = DocumentEncoder(document_dim, hidden_dim=document_hidden_dim, dropout=dropout)
        fused_dim = self.lab_encoder.output_dim + self.document_encoder.output_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim // 2, 1),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        lab_repr = self.lab_encoder(batch['labs'], batch['lab_mask'])
        doc_repr = self.document_encoder(batch['document_embeddings'])
        fused = torch.cat([lab_repr, doc_repr], dim=-1)
        return self.classifier(fused).squeeze(-1)
