"""
Main PEAG model class with multi-visit and missing modality support.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from peag.core.adversarial import ModalityMissingnessAdversary
from peag.core.alignment import align_distributions_dynamic
from peag.core.decoders import LabTestsDecoder, MetabolomicsDecoder
from peag.core.encoders import LabTestsEncoder, MetabolomicsEncoder, PastStateEncoder
from peag.core.fusion import compute_visit_state_dynamic
from peag.core.temporal import build_temporal_module
from peag.losses import (
    compute_total_loss,
    kl_divergence_loss,
    missingness_adversarial_loss,
    reconstruction_loss,
)


class PEAGModel(nn.Module):
    """
    Patient-context Enhanced Longitudinal Multimodal Alignment and Generation.

    The longitudinal autoregressive block is configurable: it can use a
    recurrent update or a lightweight transformer-style temporal module to
    summarize previous visit states into the historical state used at the next
    visit.
    """

    def __init__(
        self,
        modality_dims: Dict[str, int],
        latent_dim: int = 16,
        hidden_dim: int = 128,
        lambda_kl: float = 1.0,
        lambda_align: float = 1.0,
        lambda_adv: float = 1.0,
        temporal_model: str = "recurrent",
        temporal_num_heads: int = 4,
        temporal_num_layers: int = 1,
        temporal_dropout: float = 0.1,
        temporal_max_seq_len: int = 128,
        alignment_strategy: str = "jeffrey",
        use_adversarial_loss: bool = True,
        adversarial_grl_lambda: float = 1.0,
    ):
        super().__init__()

        self.modality_dims = modality_dims
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.lambda_kl = lambda_kl
        self.lambda_align = lambda_align
        self.lambda_adv = lambda_adv
        self.temporal_model = temporal_model
        self.temporal_num_heads = temporal_num_heads
        self.temporal_num_layers = temporal_num_layers
        self.temporal_dropout = temporal_dropout
        self.temporal_max_seq_len = temporal_max_seq_len
        self.alignment_strategy = alignment_strategy
        self.use_adversarial_loss = use_adversarial_loss
        self.adversarial_grl_lambda = adversarial_grl_lambda

        self.encoders = nn.ModuleDict()
        for mod_name, mod_dim in modality_dims.items():
            if mod_name == "lab":
                self.encoders[mod_name] = LabTestsEncoder(
                    input_dim=mod_dim,
                    latent_dim=latent_dim,
                    hidden_dim=hidden_dim,
                )
            elif mod_name == "metab":
                self.encoders[mod_name] = MetabolomicsEncoder(
                    input_dim=mod_dim,
                    latent_dim=latent_dim,
                    hidden_dim=hidden_dim,
                )
            else:
                from peag.core.encoders import GenericModalityEncoder

                self.encoders[mod_name] = GenericModalityEncoder(
                    input_dim=mod_dim,
                    latent_dim=latent_dim,
                    hidden_dim=hidden_dim,
                )

        self.past_encoder = PastStateEncoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim // 2,
        )

        self.decoders = nn.ModuleDict()
        for mod_name, mod_dim in modality_dims.items():
            if mod_name == "lab":
                self.decoders[mod_name] = LabTestsDecoder(
                    latent_dim=latent_dim,
                    hidden_dim=hidden_dim,
                    output_dim=mod_dim,
                )
            elif mod_name == "metab":
                self.decoders[mod_name] = MetabolomicsDecoder(
                    latent_dim=latent_dim,
                    hidden_dim=hidden_dim,
                    output_dim=mod_dim,
                )
            else:
                from peag.core.decoders import GenericModalityDecoder

                self.decoders[mod_name] = GenericModalityDecoder(
                    latent_dim=latent_dim,
                    hidden_dim=hidden_dim,
                    output_dim=mod_dim,
                )

        self.temporal_module = build_temporal_module(
            temporal_model=temporal_model,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_heads=temporal_num_heads,
            num_layers=temporal_num_layers,
            dropout=temporal_dropout,
            max_seq_len=temporal_max_seq_len,
        )

        self.adversaries = nn.ModuleDict(
            {
                mod_name: ModalityMissingnessAdversary(
                    input_dim=mod_dim,
                    hidden_dim=hidden_dim,
                    lambda_grl=adversarial_grl_lambda,
                )
                for mod_name, mod_dim in modality_dims.items()
            }
        )

    def get_config(self) -> Dict[str, Any]:
        """Return serializable model configuration for checkpointing."""
        return {
            "modality_dims": dict(self.modality_dims),
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "lambda_kl": self.lambda_kl,
            "lambda_align": self.lambda_align,
            "lambda_adv": self.lambda_adv,
            "temporal_model": self.temporal_model,
            "temporal_num_heads": self.temporal_num_heads,
            "temporal_num_layers": self.temporal_num_layers,
            "temporal_dropout": self.temporal_dropout,
            "temporal_max_seq_len": self.temporal_max_seq_len,
            "alignment_strategy": self.alignment_strategy,
            "use_adversarial_loss": self.use_adversarial_loss,
            "adversarial_grl_lambda": self.adversarial_grl_lambda,
        }

    @staticmethod
    def _infer_batch_metadata(
        visits_data: List[Dict[str, Optional[torch.Tensor]]],
        fallback_visits: Optional[List[Dict[str, Optional[torch.Tensor]]]] = None,
    ) -> Tuple[int, torch.device]:
        for visit_group in (visits_data, fallback_visits or []):
            for visit in visit_group:
                for tensor in visit.values():
                    if tensor is not None:
                        return tensor.shape[0], tensor.device
        raise ValueError("Unable to infer batch size/device because all modalities are missing.")

    def encode_visit(
        self,
        visit_data: Dict[str, Optional[torch.Tensor]],
        missing_mask: Dict[str, int],
        past_state: torch.Tensor,
    ) -> Tuple[
        Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        Tuple[torch.Tensor, torch.Tensor],
        Dict[str, torch.Tensor],
    ]:
        """Encode all currently available modalities plus the historical state."""
        modality_dists: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        available_mus: Dict[str, torch.Tensor] = {}

        for mod_name, mod_data in visit_data.items():
            if missing_mask.get(mod_name, 2) != 0 or mod_data is None:
                continue

            mu, logvar = self.encoders[mod_name](mod_data)
            modality_dists[mod_name] = (mu, logvar)
            available_mus[mod_name] = mu

        past_mu, past_logvar = self.past_encoder(past_state)
        return modality_dists, (past_mu, past_logvar), available_mus

    def process_visit(
        self,
        visit_data: Dict[str, Optional[torch.Tensor]],
        missing_mask: Dict[str, int],
        historical_state: torch.Tensor,
        use_history_in_fusion: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, Dict[str, Any]]:
        """Process one visit through encode-align-fuse-decode."""
        modality_dists, (past_mu, past_logvar), available_mus = self.encode_visit(
            visit_data=visit_data,
            missing_mask=missing_mask,
            past_state=historical_state,
        )

        alignment_loss = align_distributions_dynamic(
            modality_distributions=modality_dists,
            z_past_mu=past_mu,
            z_past_logvar=past_logvar,
            missing_mask=missing_mask,
            strategy=self.alignment_strategy,
        )

        visit_state = compute_visit_state_dynamic(
            modality_mus=available_mus,
            z_past_mu=past_mu,
            include_history=use_history_in_fusion,
        )

        reconstructions = {
            mod_name: self.decoders[mod_name](visit_state)
            for mod_name in self.modality_dims.keys()
        }

        debug_info = {
            "modality_dists": modality_dists,
            "past_dist": (past_mu, past_logvar),
            "available_mus": available_mus,
            "num_current_modalities": len(available_mus),
        }
        return visit_state, reconstructions, alignment_loss, debug_info

    def forward(
        self,
        visits_data: List[Dict[str, Optional[torch.Tensor]]],
        missing_masks: List[Dict[str, int]],
        kl_annealing_weight: float = 1.0,
        return_all_visit_states: bool = False,
        recon_targets: Optional[List[Dict[str, Optional[torch.Tensor]]]] = None,
        use_history_in_fusion: bool = True,
    ) -> Dict[str, Any]:
        """Forward pass through all visits in temporal order."""
        if len(visits_data) == 0:
            raise ValueError("No visit data provided")
        if len(visits_data) != len(missing_masks):
            raise ValueError("visits_data and missing_masks must have same length")

        batch_size, device = self._infer_batch_metadata(visits_data, recon_targets)
        temporal_context = self.temporal_module.init_context(batch_size, device)

        all_reconstructions: List[Dict[str, torch.Tensor]] = []
        all_visit_states: List[torch.Tensor] = []
        all_alignment_losses: List[torch.Tensor] = []
        all_modality_dists: List[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = []
        all_past_dists: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for visit_data, missing_mask in zip(visits_data, missing_masks):
            historical_state = self.temporal_module.get_historical_state(
                temporal_context,
                batch_size=batch_size,
                device=device,
            )
            visit_state, reconstructions, alignment_loss, debug_info = self.process_visit(
                visit_data=visit_data,
                missing_mask=missing_mask,
                historical_state=historical_state,
                use_history_in_fusion=use_history_in_fusion,
            )

            all_reconstructions.append(reconstructions)
            all_visit_states.append(visit_state)
            all_alignment_losses.append(alignment_loss)
            all_modality_dists.append(debug_info["modality_dists"])
            all_past_dists.append(debug_info["past_dist"])

            temporal_context = self.temporal_module.update(temporal_context, visit_state)

        losses = self.compute_losses(
            visits_data=visits_data,
            missing_masks=missing_masks,
            all_reconstructions=all_reconstructions,
            all_modality_dists=all_modality_dists,
            all_past_dists=all_past_dists,
            all_alignment_losses=all_alignment_losses,
            kl_annealing_weight=kl_annealing_weight,
            recon_targets=recon_targets,
        )

        output = {
            "reconstructions": all_reconstructions,
            "losses": losses,
        }
        if return_all_visit_states:
            output["visit_states"] = all_visit_states
        return output

    def compute_losses(
        self,
        visits_data: List[Dict[str, Optional[torch.Tensor]]],
        missing_masks: List[Dict[str, int]],
        all_reconstructions: List[Dict[str, torch.Tensor]],
        all_modality_dists: List[Dict[str, Tuple[torch.Tensor, torch.Tensor]]],
        all_past_dists: List[Tuple[torch.Tensor, torch.Tensor]],
        all_alignment_losses: List[torch.Tensor],
        kl_annealing_weight: float,
        recon_targets: Optional[List[Dict[str, Optional[torch.Tensor]]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute reconstruction, KL, alignment, and adversarial losses."""
        _, device = self._infer_batch_metadata(visits_data, recon_targets)

        recon_losses = {mod_name: 0.0 for mod_name in self.modality_dims.keys()}
        kl_losses = {mod_name: 0.0 for mod_name in self.modality_dims.keys()}
        kl_losses["past"] = 0.0

        recon_counts = {mod_name: 0 for mod_name in self.modality_dims.keys()}
        kl_counts = {mod_name: 0 for mod_name in self.modality_dims.keys()}
        kl_counts["past"] = 0

        for visit_idx, (visit_data, missing_mask, reconstructions, modality_dists, past_dist) in enumerate(
            zip(visits_data, missing_masks, all_reconstructions, all_modality_dists, all_past_dists)
        ):
            for mod_name in self.modality_dims.keys():
                mask_value = missing_mask.get(mod_name, 2)
                if recon_targets is not None:
                    target = recon_targets[visit_idx].get(mod_name)
                    if mask_value in (0, 1) and target is not None:
                        recon_losses[mod_name] += reconstruction_loss(
                            reconstructions[mod_name],
                            target,
                        )
                        recon_counts[mod_name] += 1
                else:
                    target = visit_data.get(mod_name)
                    if mask_value == 0 and target is not None:
                        recon_losses[mod_name] += reconstruction_loss(
                            reconstructions[mod_name],
                            target,
                        )
                        recon_counts[mod_name] += 1

            for mod_name, (mu, logvar) in modality_dists.items():
                kl_losses[mod_name] += kl_divergence_loss(mu, logvar)
                kl_counts[mod_name] += 1

            past_mu, past_logvar = past_dist
            kl_losses["past"] += kl_divergence_loss(past_mu, past_logvar)
            kl_counts["past"] += 1

        for mod_name in recon_losses.keys():
            if recon_counts[mod_name] > 0:
                recon_losses[mod_name] = recon_losses[mod_name] / recon_counts[mod_name]
            else:
                recon_losses[mod_name] = torch.tensor(0.0, device=device)

        for mod_name in kl_losses.keys():
            if kl_counts[mod_name] > 0:
                kl_losses[mod_name] = kl_losses[mod_name] / kl_counts[mod_name]
            else:
                kl_losses[mod_name] = torch.tensor(0.0, device=device)

        alignment_loss = sum(all_alignment_losses) / len(all_alignment_losses)

        if self.use_adversarial_loss and self.lambda_adv > 0.0:
            adversarial_terms: List[torch.Tensor] = []
            for missing_mask, reconstructions in zip(missing_masks, all_reconstructions):
                for mod_name, adversary in self.adversaries.items():
                    reconstructed_modality = reconstructions[mod_name]
                    if reconstructed_modality is None:
                        continue
                    missing_label_value = 0.0 if missing_mask.get(mod_name, 2) == 0 else 1.0
                    missing_labels = torch.full(
                        (reconstructed_modality.shape[0],),
                        missing_label_value,
                        dtype=torch.float32,
                        device=device,
                    )
                    adversarial_terms.append(
                        missingness_adversarial_loss(
                            adversary,
                            reconstructed_modality,
                            missing_labels,
                        )
                    )
            if adversarial_terms:
                adversarial_loss = torch.stack(adversarial_terms).mean()
            else:
                adversarial_loss = torch.tensor(0.0, device=device)
        else:
            adversarial_loss = torch.tensor(0.0, device=device)

        return compute_total_loss(
            recon_losses=recon_losses,
            kl_losses=kl_losses,
            alignment_loss=alignment_loss,
            adversarial_loss=adversarial_loss,
            lambda_kl=self.lambda_kl,
            lambda_align=self.lambda_align,
            lambda_adv=self.lambda_adv,
            kl_annealing_weight=kl_annealing_weight,
        )

    def impute_missing(
        self,
        visits_data: List[Dict[str, Optional[torch.Tensor]]],
        missing_masks: List[Dict[str, int]],
    ) -> List[Dict[str, torch.Tensor]]:
        """Impute naturally missing modalities for all visits."""
        with torch.no_grad():
            output = self.forward(
                visits_data=visits_data,
                missing_masks=missing_masks,
                kl_annealing_weight=1.0,
                return_all_visit_states=False,
            )

        imputed_data: List[Dict[str, torch.Tensor]] = []
        for visit_data, missing_mask, reconstructions in zip(
            visits_data,
            missing_masks,
            output["reconstructions"],
        ):
            imputed_visit = {}
            for mod_name in self.modality_dims.keys():
                mask_value = missing_mask.get(mod_name, 0)
                if mask_value == 2 or visit_data.get(mod_name) is None:
                    imputed_visit[mod_name] = reconstructions[mod_name]
                else:
                    imputed_visit[mod_name] = visit_data[mod_name]
            imputed_data.append(imputed_visit)
        return imputed_data
