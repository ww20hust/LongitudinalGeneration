from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from peag.model import PEAGModel

from common import EarlyStopper, batch_to_device, compute_binary_metrics, compute_pos_weight, create_loader, create_tensor_dataset, load_or_compute_sequence_embeddings, load_prepared_split, save_json, save_predictions, set_seed
from llama_embeddings import add_llama_embedding_args, build_text_embedder, default_torch_device


class PEAGBinaryClassifier(nn.Module):
    def __init__(self, lab_dim: int, note_dim: int, *, latent_dim: int = 16, hidden_dim: int = 128, lambda_kl: float = 1.0, lambda_align: float = 1.0, lambda_adv: float = 0.1, temporal_model: str = 'recurrent', temporal_num_heads: int = 4, temporal_num_layers: int = 1, temporal_dropout: float = 0.1, temporal_max_seq_len: int = 64, alignment_strategy: str = 'jeffrey', use_adversarial_loss: bool = True, train_mask_rate: float = 0.5, pooling: str = 'last') -> None:
        super().__init__()
        self.peag = PEAGModel(
            modality_dims={'lab': lab_dim, 'note': note_dim},
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            lambda_kl=lambda_kl,
            lambda_align=lambda_align,
            lambda_adv=lambda_adv,
            temporal_model=temporal_model,
            temporal_num_heads=temporal_num_heads,
            temporal_num_layers=temporal_num_layers,
            temporal_dropout=temporal_dropout,
            temporal_max_seq_len=temporal_max_seq_len,
            alignment_strategy=alignment_strategy,
            use_adversarial_loss=use_adversarial_loss,
        )
        self.pooling = pooling
        self.train_mask_rate = float(train_mask_rate)
        self.classifier = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _build_visit_payloads(self, batch: Dict[str, torch.Tensor], apply_active_mask: bool) -> Tuple[List[Dict[str, torch.Tensor | None]], List[Dict[str, int]], List[Dict[str, torch.Tensor | None]], torch.Tensor]:
        labs = batch['labs']
        lab_mask = batch['lab_mask']
        note_embeddings = batch['note_embeddings']
        note_mask = batch['note_mask']
        timestep_mask = torch.maximum(lab_mask.any(dim=-1).float(), note_mask)

        if labs.shape[0] != 1:
            raise ValueError('PEAGBinaryClassifier expects batch_size=1 because missing masks are sample-specific.')

        visits_data: List[Dict[str, torch.Tensor | None]] = []
        missing_masks: List[Dict[str, int]] = []
        recon_targets: List[Dict[str, torch.Tensor | None]] = []
        num_timesteps = labs.shape[1]
        for t in range(num_timesteps):
            lab_available = bool(lab_mask[0, t].any().item())
            note_available = bool(note_mask[0, t].item() > 0)
            visit_full = {'lab': labs[:, t, :], 'note': note_embeddings[:, t, :]}
            visit_mask = {'lab': 0 if lab_available else 2, 'note': 0 if note_available else 2}
            if apply_active_mask:
                available = [name for name, value in visit_mask.items() if value == 0]
                if len(available) > 1 and random.random() < self.train_mask_rate:
                    visit_mask[random.choice(available)] = 1
            visit_input = {
                'lab': None if visit_mask['lab'] != 0 else visit_full['lab'],
                'note': None if visit_mask['note'] != 0 else visit_full['note'],
            }
            recon_target = {
                'lab': visit_full['lab'] if lab_available else None,
                'note': visit_full['note'] if note_available else None,
            }
            visits_data.append(visit_input)
            missing_masks.append(visit_mask)
            recon_targets.append(recon_target)
        return visits_data, missing_masks, recon_targets, timestep_mask

    def forward_with_aux(self, batch: Dict[str, torch.Tensor], training: bool) -> Dict[str, torch.Tensor]:
        visits_data, missing_masks, recon_targets, timestep_mask = self._build_visit_payloads(batch, apply_active_mask=training)
        output = self.peag(visits_data=visits_data, missing_masks=missing_masks, recon_targets=recon_targets, return_all_visit_states=True)
        visit_states = torch.stack(output['visit_states'], dim=1)
        if self.pooling == 'mean':
            pooled = (visit_states * timestep_mask.unsqueeze(-1)).sum(dim=1) / timestep_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        else:
            pooled = visit_states[:, -1, :]
        logits = self.classifier(pooled).squeeze(-1)
        return {'logits': logits, 'peag_loss': output['losses']['total_loss']}

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.forward_with_aux(batch, training=self.training)['logits']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='PEAG benchmark on prepared MIMIC splits.')
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--valid_path', type=str, required=True)
    parser.add_argument('--test_path', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default=default_torch_device())
    parser.add_argument('--llama_model_name_or_path', type=str, default='meta-llama/Llama-3.1-8B')
    parser.add_argument('--embedding_batch_size', type=int, default=2)
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--llama_cache_dir', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--latent_dim', type=int, default=16)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--lambda_kl', type=float, default=1.0)
    parser.add_argument('--lambda_align', type=float, default=1.0)
    parser.add_argument('--lambda_adv', type=float, default=0.1)
    parser.add_argument('--peag_loss_weight', type=float, default=0.1)
    parser.add_argument('--train_mask_rate', type=float, default=0.5)
    parser.add_argument('--temporal_model', type=str, default='recurrent', choices=['recurrent', 'transformer'])
    parser.add_argument('--temporal_num_heads', type=int, default=4)
    parser.add_argument('--temporal_num_layers', type=int, default=1)
    parser.add_argument('--temporal_dropout', type=float, default=0.1)
    parser.add_argument('--alignment_strategy', type=str, default='jeffrey', choices=['jeffrey', 'stop_gradient'])
    parser.add_argument('--pooling', type=str, default='last', choices=['last', 'mean'])
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    add_llama_embedding_args(parser)
    return parser.parse_args()


def run_epoch(model: PEAGBinaryClassifier, loader, optimizer: torch.optim.Optimizer | None, criterion: nn.Module, device: torch.device, peag_loss_weight: float) -> Dict[str, np.ndarray | float]:
    training = optimizer is not None
    model.train() if training else model.eval()
    all_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    total_loss = 0.0
    total_examples = 0

    for batch in loader:
        batch = batch_to_device(batch, device)
        labels = batch['labels']
        with torch.set_grad_enabled(training):
            output = model.forward_with_aux(batch, training=training)
            cls_loss = criterion(output['logits'], labels)
            loss = cls_loss + peag_loss_weight * output['peag_loss']
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        probs = torch.sigmoid(output['logits'].detach()).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.detach().cpu().numpy())
        batch_size = labels.shape[0]
        total_examples += batch_size
        total_loss += float(loss.item()) * batch_size

    probs_np = np.concatenate(all_probs, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    metrics = compute_binary_metrics(labels_np, probs_np)
    metrics['loss'] = total_loss / max(total_examples, 1)
    metrics['probabilities'] = probs_np
    metrics['labels'] = labels_np
    return metrics


def fit_peag(model: PEAGBinaryClassifier, train_loader, valid_loader, *, device: torch.device, epochs: int, lr: float, weight_decay: float, pos_weight: float, patience: int, peag_loss_weight: float) -> tuple[PEAGBinaryClassifier, Dict[str, List[float]], Dict[str, float]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device))
    stopper = EarlyStopper(mode='max', patience=patience)
    history = {
        'train_loss': [], 'train_auroc': [], 'train_auprc': [], 'train_f1': [],
        'valid_loss': [], 'valid_auroc': [], 'valid_auprc': [], 'valid_f1': [],
    }
    best_valid: Dict[str, float] = {}
    model.to(device)
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, criterion, device, peag_loss_weight)
        valid_metrics = run_epoch(model, valid_loader, None, criterion, device, peag_loss_weight)
        history['train_loss'].append(float(train_metrics['loss']))
        history['train_auroc'].append(float(train_metrics['auroc']))
        history['train_auprc'].append(float(train_metrics['auprc']))
        history['train_f1'].append(float(train_metrics['f1']))
        history['valid_loss'].append(float(valid_metrics['loss']))
        history['valid_auroc'].append(float(valid_metrics['auroc']))
        history['valid_auprc'].append(float(valid_metrics['auprc']))
        history['valid_f1'].append(float(valid_metrics['f1']))
        if stopper.best_value is None or float(valid_metrics['auprc']) >= stopper.best_value:
            best_valid = {k: float(v) for k, v in valid_metrics.items() if isinstance(v, (int, float, np.floating))}
        should_stop = stopper.step(float(valid_metrics['auprc']), model)
        print(f"Epoch {epoch:03d} | train loss={train_metrics['loss']:.4f} auroc={train_metrics['auroc']:.4f} auprc={train_metrics['auprc']:.4f} | valid loss={valid_metrics['loss']:.4f} auroc={valid_metrics['auroc']:.4f} auprc={valid_metrics['auprc']:.4f}")
        if should_stop:
            print(f'Early stopping triggered at epoch {epoch}.')
            break
    stopper.restore(model)
    return model, history, best_valid


@torch.inference_mode()
def evaluate_peag(model: PEAGBinaryClassifier, loader, *, device: torch.device, pos_weight: float, peag_loss_weight: float) -> Dict[str, np.ndarray | float]:
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device))
    model.to(device)
    return run_epoch(model, loader, None, criterion, device, peag_loss_weight)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_split = load_prepared_split(args.train_path)
    valid_split = load_prepared_split(args.valid_path)
    test_split = load_prepared_split(args.test_path)
    embedder = build_text_embedder(args)

    train_note_embeddings, train_note_mask = load_or_compute_sequence_embeddings(train_split, split_name='train', cache_dir=args.llama_cache_dir, embedder=embedder, batch_size=args.embedding_batch_size, max_length=args.max_length)
    valid_note_embeddings, valid_note_mask = load_or_compute_sequence_embeddings(valid_split, split_name='valid', cache_dir=args.llama_cache_dir, embedder=embedder, batch_size=args.embedding_batch_size, max_length=args.max_length)
    test_note_embeddings, test_note_mask = load_or_compute_sequence_embeddings(test_split, split_name='test', cache_dir=args.llama_cache_dir, embedder=embedder, batch_size=args.embedding_batch_size, max_length=args.max_length)

    train_dataset = create_tensor_dataset(train_split, note_sequence_embeddings=train_note_embeddings, note_sequence_mask=train_note_mask)
    valid_dataset = create_tensor_dataset(valid_split, note_sequence_embeddings=valid_note_embeddings, note_sequence_mask=valid_note_mask)
    test_dataset = create_tensor_dataset(test_split, note_sequence_embeddings=test_note_embeddings, note_sequence_mask=test_note_mask)

    train_loader = create_loader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = create_loader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = create_loader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = PEAGBinaryClassifier(
        lab_dim=train_split.labs.shape[-1],
        note_dim=train_note_embeddings.shape[-1],
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        lambda_kl=args.lambda_kl,
        lambda_align=args.lambda_align,
        lambda_adv=args.lambda_adv,
        temporal_model=args.temporal_model,
        temporal_num_heads=args.temporal_num_heads,
        temporal_num_layers=args.temporal_num_layers,
        temporal_dropout=args.temporal_dropout,
        temporal_max_seq_len=train_split.labs.shape[1],
        alignment_strategy=args.alignment_strategy,
        train_mask_rate=args.train_mask_rate,
        pooling=args.pooling,
    )
    device = torch.device(args.device)
    pos_weight = compute_pos_weight(train_split.labels)
    model, history, best_valid = fit_peag(model, train_loader, valid_loader, device=device, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, pos_weight=pos_weight, patience=args.patience, peag_loss_weight=args.peag_loss_weight)
    test_metrics = evaluate_peag(model, test_loader, device=device, pos_weight=pos_weight, peag_loss_weight=args.peag_loss_weight)

    torch.save({'model_state_dict': model.state_dict(), 'note_dim': int(train_note_embeddings.shape[-1])}, save_dir / 'best_model.pt')
    save_predictions(save_dir / 'test_predictions.npz', test_metrics['probabilities'], test_metrics['labels'], test_split.patient_ids)
    save_json(save_dir / 'metrics.json', {
        'benchmark': 'peag_mimic_alignment_llama',
        'llama_model_name_or_path': args.llama_model_name_or_path,
        'best_valid': best_valid,
        'test': {k: float(v) for k, v in test_metrics.items() if isinstance(v, (int, float, np.floating))},
        'history': history,
        'config': {
            'latent_dim': args.latent_dim,
            'hidden_dim': args.hidden_dim,
            'lambda_kl': args.lambda_kl,
            'lambda_align': args.lambda_align,
            'lambda_adv': args.lambda_adv,
            'peag_loss_weight': args.peag_loss_weight,
            'train_mask_rate': args.train_mask_rate,
            'temporal_model': args.temporal_model,
            'pooling': args.pooling,
        },
    })
    print(f'PEAG benchmark complete: {save_dir}')
    print(f"Test AUROC={test_metrics['auroc']:.4f}, AUPRC={test_metrics['auprc']:.4f}, F1={test_metrics['f1']:.4f}")


if __name__ == '__main__':
    main()
