from __future__ import annotations

import torch


def last_token_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    if last_hidden_state.ndim != 3:
        raise ValueError("last_hidden_state must be 3D [batch, seq, hidden]")
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must be 2D [batch, seq]")
    if last_hidden_state.shape[:2] != attention_mask.shape:
        raise ValueError("Shape mismatch between last_hidden_state and attention_mask")

    last_valid_index = attention_mask.long().sum(dim=1) - 1
    last_valid_index = torch.clamp(last_valid_index, min=0)
    batch_indices = torch.arange(last_hidden_state.size(0), device=last_hidden_state.device)
    return last_hidden_state[batch_indices, last_valid_index, :]


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    if last_hidden_state.ndim != 3:
        raise ValueError("last_hidden_state must be 3D [batch, seq, hidden]")
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must be 2D [batch, seq]")
    if last_hidden_state.shape[:2] != attention_mask.shape:
        raise ValueError("Shape mismatch between last_hidden_state and attention_mask")

    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts
