from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

try:
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover
    AutoTokenizer = None

from common import PreparedBinarySplit, load_prepared_split, save_pickle
from common import _cache_filename as build_cache_filename


class _ModelRef:
    def __init__(self, model_name_or_path: str) -> None:
        self.model_name_or_path = model_name_or_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute MIMIC note embeddings via a vLLM OpenAI-compatible server."
    )
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--valid_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True, help="Directory to store cached .npz embeddings.")
    parser.add_argument(
        "--llama_model_name_or_path",
        type=str,
        default="meta-llama/Llama-3.1-8B",
        help="Model name used by the vLLM server and recorded in cache metadata.",
    )
    parser.add_argument(
        "--vllm_base_url",
        type=str,
        default="http://localhost:8000/v1",
        help="Base URL for the vLLM OpenAI-compatible server (e.g., http://host:8000/v1).",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument(
        "--embedding_kind",
        type=str,
        default="both",
        choices=["document", "sequence", "both"],
        help="Which embeddings to precompute.",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="Optional tokenizer name for length-aware truncation (defaults to model name).",
    )
    parser.add_argument("--request_timeout", type=float, default=120.0)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--retry_sleep", type=float, default=2.0)
    return parser.parse_args()


def _load_tokenizer(tokenizer_name_or_path: str | None, model_name_or_path: str):
    if AutoTokenizer is None:
        raise ImportError("transformers is required for tokenizer-based truncation.")
    name = tokenizer_name_or_path or model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _truncate_texts(tokenizer, texts: Sequence[str], max_length: int) -> List[str]:
    truncated: List[str] = []
    for text in texts:
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_attention_mask=False,
            return_tensors=None,
        )
        truncated.append(tokenizer.decode(encoded["input_ids"], skip_special_tokens=True))
    return truncated


def _post_embeddings(
    base_url: str,
    model_name_or_path: str,
    texts: Sequence[str],
    timeout: float,
) -> np.ndarray:
    url = base_url.rstrip("/") + "/embeddings"
    payload = {"model": model_name_or_path, "input": list(texts)}
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    parsed = json.loads(body)
    embeddings = [item["embedding"] for item in parsed["data"]]
    return np.asarray(embeddings, dtype=np.float32)


def _encode_texts(
    base_url: str,
    model_name_or_path: str,
    texts: Sequence[str],
    *,
    batch_size: int,
    timeout: float,
    max_retries: int,
    retry_sleep: float,
) -> np.ndarray:
    batches: List[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        for attempt in range(max_retries + 1):
            try:
                batches.append(_post_embeddings(base_url, model_name_or_path, batch, timeout))
                break
            except Exception:
                if attempt >= max_retries:
                    raise
                time.sleep(retry_sleep)
    if not batches:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(batches, axis=0)


def _save_document_embeddings(
    split: PreparedBinarySplit,
    *,
    split_name: str,
    cache_dir: Path,
    embedder_ref: _ModelRef,
    model_name_or_path: str,
    base_url: str,
    batch_size: int,
    max_length: int,
    tokenizer,
    timeout: float,
    max_retries: int,
    retry_sleep: float,
) -> None:
    non_empty_indices = [idx for idx, text in enumerate(split.document_texts) if str(text).strip()]
    if non_empty_indices:
        texts = [split.document_texts[idx] for idx in non_empty_indices]
        texts = _truncate_texts(tokenizer, texts, max_length)
        encoded = _encode_texts(
            base_url,
            model_name_or_path,
            texts,
            batch_size=batch_size,
            timeout=timeout,
            max_retries=max_retries,
            retry_sleep=retry_sleep,
        )
        dim = encoded.shape[1]
    else:
        dim = 0
        encoded = np.zeros((0, 0), dtype=np.float32)

    embeddings = np.zeros((len(split.document_texts), dim), dtype=np.float32)
    for row_index, sample_index in enumerate(non_empty_indices):
        embeddings[sample_index] = encoded[row_index]

    cache_path = cache_dir / build_cache_filename(
        split,
        split_name=split_name,
        kind="document_embeddings",
        embedder=embedder_ref,
        max_length=max_length,
    )
    np.savez_compressed(cache_path, embeddings=embeddings)


def _save_sequence_embeddings(
    split: PreparedBinarySplit,
    *,
    split_name: str,
    cache_dir: Path,
    embedder_ref: _ModelRef,
    model_name_or_path: str,
    base_url: str,
    batch_size: int,
    max_length: int,
    tokenizer,
    timeout: float,
    max_retries: int,
    retry_sleep: float,
) -> None:
    n_samples = len(split.peag_note_texts)
    seq_len = len(split.peag_note_texts[0]) if n_samples > 0 else 0
    mask = np.zeros((n_samples, seq_len), dtype=np.float32)
    flat_texts: List[str] = []
    flat_indices: List[Tuple[int, int]] = []
    for i, sequence in enumerate(split.peag_note_texts):
        if len(sequence) != seq_len:
            raise ValueError("All PEAG note sequences must have the same length.")
        for t, text in enumerate(sequence):
            cleaned = str(text).strip()
            if cleaned:
                flat_texts.append(cleaned)
                flat_indices.append((i, t))
                mask[i, t] = 1.0

    if flat_texts:
        flat_texts = _truncate_texts(tokenizer, flat_texts, max_length)
        flat_embeddings = _encode_texts(
            base_url,
            model_name_or_path,
            flat_texts,
            batch_size=batch_size,
            timeout=timeout,
            max_retries=max_retries,
            retry_sleep=retry_sleep,
        )
        dim = flat_embeddings.shape[1]
    else:
        dim = 0
        flat_embeddings = np.zeros((0, 0), dtype=np.float32)

    embeddings = np.zeros((n_samples, seq_len, dim), dtype=np.float32)
    for idx, vector in enumerate(flat_embeddings):
        i, t = flat_indices[idx]
        embeddings[i, t] = vector

    cache_path = cache_dir / build_cache_filename(
        split,
        split_name=split_name,
        kind="sequence_embeddings",
        embedder=embedder_ref,
        max_length=max_length,
    )
    np.savez_compressed(cache_path, embeddings=embeddings, mask=mask)


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = _load_tokenizer(args.tokenizer_name_or_path, args.llama_model_name_or_path)
    embedder_ref = _ModelRef(args.llama_model_name_or_path)

    splits = {
        "train": load_prepared_split(args.train_path),
        "valid": load_prepared_split(args.valid_path),
        "test": load_prepared_split(args.test_path),
    }

    for split_name, split in splits.items():
        if args.embedding_kind in {"document", "both"}:
            _save_document_embeddings(
                split,
                split_name=split_name,
                cache_dir=cache_dir,
                embedder_ref=embedder_ref,
                model_name_or_path=args.llama_model_name_or_path,
                base_url=args.vllm_base_url,
                batch_size=args.batch_size,
                max_length=args.max_length,
                tokenizer=tokenizer,
                timeout=args.request_timeout,
                max_retries=args.max_retries,
                retry_sleep=args.retry_sleep,
            )
        if args.embedding_kind in {"sequence", "both"}:
            _save_sequence_embeddings(
                split,
                split_name=split_name,
                cache_dir=cache_dir,
                embedder_ref=embedder_ref,
                model_name_or_path=args.llama_model_name_or_path,
                base_url=args.vllm_base_url,
                batch_size=args.batch_size,
                max_length=args.max_length,
                tokenizer=tokenizer,
                timeout=args.request_timeout,
                max_retries=args.max_retries,
                retry_sleep=args.retry_sleep,
            )

    meta = {
        "llama_model_name_or_path": args.llama_model_name_or_path,
        "vllm_base_url": args.vllm_base_url,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "embedding_kind": args.embedding_kind,
        "cache_dir": str(cache_dir),
    }
    save_pickle(cache_dir / "vllm_cache_meta.pkl", meta)
    print(f"vLLM embeddings cached at: {cache_dir}")


if __name__ == "__main__":
    main()
