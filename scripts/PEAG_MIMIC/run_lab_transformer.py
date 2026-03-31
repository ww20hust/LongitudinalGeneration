from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common import compute_binary_metrics_ci, compute_pos_weight, create_loader, create_tensor_dataset, evaluate_classifier, fit_classifier, load_prepared_split, save_json, save_predictions, set_seed
from models import LabTransformerClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Structured-clinical-measurements-only Transformer benchmark on prepared MIMIC splits.')
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--valid_path', type=str, required=True)
    parser.add_argument('--test_path', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dim_feedforward', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ci_samples', type=int, default=1000)
    parser.add_argument('--ci_seed', type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_split = load_prepared_split(args.train_path)
    valid_split = load_prepared_split(args.valid_path)
    test_split = load_prepared_split(args.test_path)

    train_dataset = create_tensor_dataset(train_split)
    valid_dataset = create_tensor_dataset(valid_split)
    test_dataset = create_tensor_dataset(test_split)

    train_loader = create_loader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = create_loader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = create_loader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = LabTransformerClassifier(
        lab_dim=train_split.labs.shape[-1],
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_len=train_split.labs.shape[1],
    )

    device = torch.device(args.device)
    pos_weight = compute_pos_weight(train_split.labels)
    model, history, best_valid = fit_classifier(
        model,
        train_loader,
        valid_loader,
        device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        pos_weight=pos_weight,
        patience=args.patience,
    )
    test_metrics = evaluate_classifier(model, test_loader, device, pos_weight=pos_weight)
    test_ci = compute_binary_metrics_ci(
        test_metrics['labels'],
        test_metrics['probabilities'],
        n_boot=args.ci_samples,
        seed=args.ci_seed,
    )

    torch.save({'model_state_dict': model.state_dict()}, save_dir / 'best_model.pt')
    save_predictions(save_dir / 'test_predictions.npz', test_metrics['probabilities'], test_metrics['labels'], test_split.patient_ids)
    save_json(save_dir / 'metrics.json', {
        'benchmark': 'structured_measurements_only_transformer',
        'best_valid': best_valid,
        'test': {k: float(v) for k, v in test_metrics.items() if isinstance(v, (int, float))},
        'test_ci': test_ci,
        'history': history,
    })
    print(f'Structured-only benchmark complete: {save_dir}')
    print(f"Test AUROC={test_metrics['auroc']:.4f}, AUPRC={test_metrics['auprc']:.4f}, F1={test_metrics['f1']:.4f}")


if __name__ == '__main__':
    main()
