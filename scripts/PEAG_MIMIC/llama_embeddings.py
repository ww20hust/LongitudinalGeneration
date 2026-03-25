from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:  # pragma: no cover
    AutoModel = None
    AutoTokenizer = None

try:
    from transformers import BitsAndBytesConfig
except ImportError:  # pragma: no cover
    BitsAndBytesConfig = None


def default_torch_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def add_llama_embedding_args(parser) -> None:
    parser.add_argument(
        "--llama_dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Precision used when loading the Llama encoder.",
    )
    parser.add_argument(
        "--llama_device",
        type=str,
        default="auto",
        help="Device for the Llama encoder. Examples: auto, cuda, cuda:0, mps, cpu.",
    )
    parser.add_argument(
        "--llama_device_map",
        type=str,
        default="none",
        help="Optional transformers device_map value such as auto, cpu, or cuda:0.",
    )
    parser.add_argument(
        "--llama_load_in_8bit",
        action="store_true",
        help="Load the transformers backend with 8-bit quantization.",
    )
    parser.add_argument(
        "--llama_load_in_4bit",
        action="store_true",
        help="Load the transformers backend with 4-bit quantization.",
    )
    parser.add_argument(
        "--llama_attn_implementation",
        type=str,
        default="auto",
        choices=["auto", "eager", "sdpa"],
        help="Attention kernel for the transformers backend.",
    )
    parser.add_argument(
        "--llama_disable_low_cpu_mem_usage",
        action="store_true",
        help="Disable transformers low_cpu_mem_usage when loading the Llama encoder.",
    )


def _mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    weights = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    summed = (hidden_states * weights).sum(dim=1)
    counts = weights.sum(dim=1).clamp(min=1.0)
    return summed / counts


def _resolve_device_name(device_name: str) -> str:
    if device_name == "auto":
        return default_torch_device()
    return device_name


def _resolve_torch_dtype(dtype_name: str, resolved_device: str):
    if dtype_name == "auto":
        if resolved_device.startswith("cuda"):
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        if resolved_device == "mps":
            return torch.float16
        return torch.float32
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[dtype_name]


def _infer_model_input_device(model: torch.nn.Module) -> torch.device:
    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict):
        for location in hf_device_map.values():
            if isinstance(location, int):
                return torch.device(f"cuda:{location}")
            if isinstance(location, str) and location not in {"cpu", "disk"}:
                return torch.device(location)
        return torch.device("cpu")

    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cpu")


class TransformersTextEmbedder:
    def __init__(
        self,
        model_name_or_path: str,
        device_name: str,
        dtype_name: str,
        device_map: str,
        load_in_4bit: bool,
        load_in_8bit: bool,
        attn_implementation: str,
        low_cpu_mem_usage: bool,
    ) -> None:
        if AutoModel is None or AutoTokenizer is None:
            raise ImportError(
                "transformers is required for the Llama embedding backend. Install it before running this benchmark."
            )
        if load_in_4bit and load_in_8bit:
            raise ValueError("Choose at most one of --llama_load_in_4bit and --llama_load_in_8bit.")

        self.model_name_or_path = model_name_or_path
        resolved_device = _resolve_device_name(device_name)
        resolved_dtype = _resolve_torch_dtype(dtype_name, resolved_device)
        model_kwargs = {"low_cpu_mem_usage": bool(low_cpu_mem_usage)}

        if attn_implementation != "auto":
            model_kwargs["attn_implementation"] = attn_implementation

        if load_in_4bit or load_in_8bit:
            if BitsAndBytesConfig is None:
                raise ImportError(
                    "bitsandbytes support requires a recent transformers install with BitsAndBytesConfig available."
                )
            if not resolved_device.startswith("cuda"):
                raise ValueError("bitsandbytes quantization is only enabled for CUDA devices in this benchmark.")
            quantization_kwargs = {}
            if load_in_4bit:
                quantization_kwargs["load_in_4bit"] = True
                quantization_kwargs["bnb_4bit_compute_dtype"] = resolved_dtype
            if load_in_8bit:
                quantization_kwargs["load_in_8bit"] = True
            model_kwargs["quantization_config"] = BitsAndBytesConfig(**quantization_kwargs)
            model_kwargs["device_map"] = "auto" if device_map == "none" else device_map
            model_kwargs["torch_dtype"] = resolved_dtype
        else:
            model_kwargs["torch_dtype"] = resolved_dtype
            if device_map != "none":
                model_kwargs["device_map"] = device_map

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModel.from_pretrained(model_name_or_path, **model_kwargs)
        if "device_map" not in model_kwargs:
            self.model.to(resolved_device)
        self.model.eval()
        self.input_device = _infer_model_input_device(self.model)

    def encode_texts(self, texts: Sequence[str], batch_size: int, max_length: int) -> np.ndarray:
        embedding_batches: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch_texts = list(texts[start : start + batch_size])
                tokenized = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                tokenized = {key: value.to(self.input_device) for key, value in tokenized.items()}
                outputs = self.model(**tokenized)
                pooled = _mean_pool(outputs.last_hidden_state, tokenized["attention_mask"])
                embedding_batches.append(pooled.cpu().numpy().astype(np.float32))
        return np.concatenate(embedding_batches, axis=0)


def build_text_embedder(args):
    return TransformersTextEmbedder(
        model_name_or_path=args.llama_model_name_or_path,
        device_name=args.llama_device,
        dtype_name=args.llama_dtype,
        device_map=args.llama_device_map,
        load_in_4bit=args.llama_load_in_4bit,
        load_in_8bit=args.llama_load_in_8bit,
        attn_implementation=args.llama_attn_implementation,
        low_cpu_mem_usage=not args.llama_disable_low_cpu_mem_usage,
    )
