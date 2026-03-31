from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common import compute_binary_metrics_ci, compute_pos_weight, create_loader, create_tensor_dataset, evaluate_classifier, fit_classifier, load_or_compute_document_embeddings, load_prepared_split, save_json, save_predictions, set_seed
from llama_embeddings import add_llama_embedding_args, build_text_embedder, default_torch_device
from models import NotesOnlyLlamaClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Notes-only Llama embedding benchmark on prepared MIMIC splits.')
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
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ci_samples', type=int, default=1000)
    parser.add_argument('--ci_seed', type=int, default=0)
    add_llama_embedding_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_split = load_prepared_split(args.train_path)
    valid_split = load_prepared_split(args.valid_path)
    test_split = load_prepared_split(args.test_path)
    embedder = build_text_embedder(args)

    train_doc_embeddings = load_or_compute_document_embeddings(train_split, split_name='train', cache_dir=args.llama_cache_dir, embedder=embedder, batch_size=args.embedding_batch_size, max_length=args.max_length)
    valid_doc_embeddings = load_or_compute_document_embeddings(valid_split, split_name='valid', cache_dir=args.llama_cache_dir, embedder=embedder, batch_size=args.embedding_batch_size, max_length=args.max_length)
    test_doc_embeddings = load_or_compute_document_embeddings(test_split, split_name='test', cache_dir=args.llama_cache_dir, embedder=embedder, batch_size=args.embedding_batch_size, max_length=args.max_length)

    train_dataset = create_tensor_dataset(train_split, document_embeddings=train_doc_embeddings)
    valid_dataset = create_tensor_dataset(valid_split, document_embeddings=valid_doc_embeddings)
    test_dataset = create_tensor_dataset(test_split, document_embeddings=test_doc_embeddings)

    train_loader = create_loader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = create_loader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = create_loader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = NotesOnlyLlamaClassifier(document_dim=train_doc_embeddings.shape[-1], hidden_dim=args.hidden_dim, dropout=args.dropout)
    device = torch.device(args.device)
    pos_weight = compute_pos_weight(train_split.labels)
    model, history, best_valid = fit_classifier(model, train_loader, valid_loader, device, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, pos_weight=pos_weight, patience=args.patience)
    test_metrics = evaluate_classifier(model, test_loader, device, pos_weight=pos_weight)
    test_ci = compute_binary_metrics_ci(
        test_metrics['labels'],
        test_metrics['probabilities'],
        n_boot=args.ci_samples,
        seed=args.ci_seed,
    )

    torch.save({'model_state_dict': model.state_dict(), 'document_dim': int(train_doc_embeddings.shape[-1])}, save_dir / 'best_model.pt')
    save_predictions(save_dir / 'test_predictions.npz', test_metrics['probabilities'], test_metrics['labels'], test_split.patient_ids)
    save_json(save_dir / 'metrics.json', {
        'benchmark': 'notes_only_llama_document',
        'llama_model_name_or_path': args.llama_model_name_or_path,
        'best_valid': best_valid,
        'test': {k: float(v) for k, v in test_metrics.items() if isinstance(v, (int, float))},
        'test_ci': test_ci,
        'history': history,
    })
    print(f'Notes-only benchmark complete: {save_dir}')
    print(f"Test AUROC={test_metrics['auroc']:.4f}, AUPRC={test_metrics['auprc']:.4f}, F1={test_metrics['f1']:.4f}")


if __name__ == '__main__':
    main()
