"""Phase 4: Distributed NLL-based difficulty score computation.

For each sample, computes:
  difficulty(x) = (1/T) * sum_t [ -log P(y_t | s, q, y_<t) ]

where q=question and y=golden answer tokens (no system prompt).

Runs distributed via accelerate (one model replica per GPU).
Output per rank: difficulty_ids_rank{N}.npy, difficulty_scores_rank{N}.npy
"""
from __future__ import annotations

import argparse
import math
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from tqdm import tqdm

from data_utils import _build_question_text, _extract_cot_answer, ensure_sample_id, load_raw_dataset
from io_utils import ensure_dir, save_npy
from logging_utils import setup_logger
from model_utils import load_causal_model, load_tokenizer, resolve_torch_dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distributed NLL difficulty score computation")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--input_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML object")
    return config


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    updated = dict(config)
    for key in ["model_name_or_path", "input_path", "output_dir", "batch_size", "max_length", "dtype"]:
        value = getattr(args, key)
        if value is not None:
            updated[key] = value
    if args.max_samples is not None:
        updated["max_samples"] = args.max_samples
    return updated


def _prepare_dataset(config: dict[str, Any]):
    cache_dir = config.get("datasets_cache_dir")
    if cache_dir is not None:
        cache_dir = str(cache_dir)
    dataset = load_raw_dataset(
        input_path=str(config["input_path"]),
        input_format=str(config.get("input_format", "parquet")),
        cache_dir=cache_dir,
    )
    id_field = str(config.get("id_field", "sample_id"))

    def _add_id(example, idx):
        return {"sample_id": ensure_sample_id(example, idx, id_field=id_field)}

    dataset = dataset.map(_add_id, with_indices=True)
    return dataset


def _compute_sample_nll(
    model,
    tokenizer,
    example: dict[str, Any],
    text_field: str,
    max_length: int,
    device: torch.device,
) -> float:
    """Compute average NLL of CoT answer tokens for one sample.

    Full sequence: question + full CoT answer (no system prompt).
    NLL is computed only on the CoT tokens via teacher-forcing.

    Returns float('inf') if answer is empty after truncation.
    """
    question_text = _build_question_text(
        example,
        text_field=text_field,
        prompt_text_mode="user_only",
    )
    golden_answer = _extract_cot_answer(example)

    # Build plain text format without system prompt.
    prompt_text = f"Question:\n{question_text}\n\nAnswer:\n"
    full_text = f"{prompt_text}{golden_answer}"

    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)

    if len(full_ids) > max_length:
        full_ids = full_ids[:max_length]

    prompt_len = len(prompt_ids)
    answer_len = len(full_ids) - prompt_len

    if answer_len <= 0:
        return float("inf")

    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)

    with torch.inference_mode():
        outputs = model(input_ids=input_ids, use_cache=False)
        logits = outputs.logits  # [1, seq_len, vocab_size]

    # For token at position t, logits[t-1] predicts full_ids[t].
    # Answer tokens are full_ids[prompt_len : prompt_len+answer_len].
    # So we need logits[prompt_len-1 : prompt_len+answer_len-1].
    answer_logits = logits[0, prompt_len - 1 : prompt_len + answer_len - 1, :]
    answer_targets = input_ids[0, prompt_len : prompt_len + answer_len]

    token_nll = F.cross_entropy(answer_logits, answer_targets, reduction="none")
    return float(token_nll.mean().item())


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)

    accelerator = Accelerator()
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    device = accelerator.device

    logger = setup_logger("compute_difficulty", rank=rank, world_size=world_size)

    output_dir = ensure_dir(str(config["output_dir"]))
    ids_path = output_dir / f"difficulty_ids_rank{rank}.npy"
    scores_path = output_dir / f"difficulty_scores_rank{rank}.npy"

    skip_if_exists = _parse_bool(config.get("skip_if_exists", True))
    if skip_if_exists and ids_path.exists() and scores_path.exists():
        logger.info("Shard already exists, skipping rank %d.", rank)
        return

    logger.info("Loading dataset from %s", config["input_path"])
    dataset = _prepare_dataset(config)

    max_samples = config.get("max_samples")
    if max_samples is not None:
        max_samples = int(max_samples)
        if max_samples > 0:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

    logger.info("Total samples before sharding: %d", len(dataset))
    shard = dataset.shard(num_shards=world_size, index=rank, contiguous=True)
    num_local = len(shard)
    logger.info("Rank %d shard size: %d", rank, num_local)

    model_name_or_path = str(config["model_name_or_path"])
    dtype = resolve_torch_dtype(str(config.get("dtype", "bfloat16")))

    logger.info("Loading causal model: %s", model_name_or_path)
    tokenizer = load_tokenizer(model_name_or_path)
    model = load_causal_model(model_name_or_path, torch_dtype=dtype, device=device)

    text_field = str(config.get("text_field", "prompt"))
    max_length = int(config.get("max_length", 8192))

    id_chunks: list[np.ndarray] = []
    score_chunks: list[np.ndarray] = []

    total = num_local
    wall_start = time.perf_counter()
    processed = 0

    progress = tqdm(
        range(num_local),
        disable=not accelerator.is_local_main_process,
        desc=f"rank{rank} NLL",
    )

    for i in progress:
        try:
            example = shard[i]
            sample_id = example["sample_id"]

            nll = _compute_sample_nll(
                model=model,
                tokenizer=tokenizer,
                example=example,
                text_field=text_field,
                max_length=max_length,
                device=device,
            )

            id_chunks.append(np.array([sample_id], dtype=np.int64))
            score_chunks.append(np.array([nll], dtype=np.float64))
            processed += 1

            if processed % 200 == 0 or processed == total:
                elapsed = time.perf_counter() - wall_start
                logger.info(
                    "rank%d: %d/%d  throughput=%.2f samples/s",
                    rank, processed, total, processed / elapsed,
                )
        except Exception as exc:
            logger.error("rank%d sample %d error: %s", rank, i, exc)
            logger.error(traceback.format_exc())
            raise

    all_ids = np.concatenate(id_chunks) if id_chunks else np.empty((0,), dtype=np.int64)
    all_scores = np.concatenate(score_chunks) if score_chunks else np.empty((0,), dtype=np.float64)

    save_npy(ids_path, all_ids)
    save_npy(scores_path, all_scores)

    elapsed_total = time.perf_counter() - wall_start
    logger.info(
        "rank%d done. samples=%d elapsed=%.1fs  score: min=%.4f max=%.4f mean=%.4f",
        rank, num_local, elapsed_total,
        float(np.nanmin(all_scores)) if len(all_scores) else float("nan"),
        float(np.nanmax(all_scores)) if len(all_scores) else float("nan"),
        float(np.nanmean(all_scores)) if len(all_scores) else float("nan"),
    )


if __name__ == "__main__":
    main()
