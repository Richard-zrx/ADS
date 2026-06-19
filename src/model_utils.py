from __future__ import annotations

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = dtype_name.strip().lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[key]


def load_tokenizer(model_name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer has no pad_token and no eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(model_name_or_path: str, torch_dtype: torch.dtype, device: torch.device):
    model = AutoModel.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.eval()
    model.to(device)
    return model


def forward_last_hidden_state(model, model_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    outputs = model(**model_inputs, use_cache=False)
    if not hasattr(outputs, "last_hidden_state"):
        raise ValueError("Model output does not include last_hidden_state")
    return outputs.last_hidden_state


def load_causal_model(model_name_or_path: str, torch_dtype: torch.dtype, device: torch.device):
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.eval()
    model.to(device)
    return model
