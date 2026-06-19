from __future__ import annotations

import argparse
import math
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from accelerate import Accelerator
from tqdm import tqdm

from data_utils import load_and_prepare_dataset
from io_utils import all_paths_exist, normalize_pooling_methods, save_npy, shard_output_paths
from logging_utils import setup_logger
from model_utils import forward_last_hidden_state, load_model, load_tokenizer, resolve_torch_dtype
from pooling import last_token_pool, mean_pool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distributed offline embedding extraction")
    parser.add_argument("--config", required=True, help="Path to YAML config")

    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--input_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--pooling_methods", default=None, help="Comma separated, e.g. last_token,mean")
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError("Config file must contain a YAML object")
    return config


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    updated = dict(config)
    for key in [
        "model_name_or_path",
        "input_path",
        "output_dir",
        "batch_size",
        "max_length",
        "dtype",
    ]:
        value = getattr(args, key)
        if value is not None:
            updated[key] = value

    if args.pooling_methods is not None:
        updated["pooling_methods"] = [p.strip() for p in args.pooling_methods.split(",") if p.strip()]

    if args.max_samples is not None:
        updated["max_samples"] = args.max_samples

    return updated


def _l2_normalize(emb: np.ndarray) -> np.ndarray:
    if emb.size == 0:
        return emb
    denom = np.linalg.norm(emb, ord=2, axis=1, keepdims=True)
    denom = np.clip(denom, a_min=1e-12, a_max=None)
    return emb / denom


def _to_numpy_ids(batch_ids: list[Any]) -> np.ndarray:
    try:
        return np.asarray(batch_ids, dtype=np.int64)
    except (TypeError, ValueError):
        return np.asarray([str(x) for x in batch_ids], dtype=object)


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)

    accelerator = Accelerator()
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    device = accelerator.device

    logger = setup_logger("extract_embeddings", rank=rank, world_size=world_size)

    pooling_methods = normalize_pooling_methods(list(config.get("pooling_methods", ["last_token", "mean"])))

    save_format = str(config.get("save_format", "npy")).lower()
    if save_format != "npy":
        raise ValueError("Phase-1 requires save_format=npy for primary artifacts.")

    output_dir = Path(str(config["output_dir"]))
    rank_paths = shard_output_paths(output_dir=output_dir, rank=rank, pooling_methods=pooling_methods)

    skip_if_exists = _parse_bool(config.get("skip_if_exists", True))
    overwrite = _parse_bool(config.get("overwrite", False))
    if skip_if_exists and (not overwrite) and all_paths_exist(rank_paths):
        logger.info("Output shard already exists. Skipping rank %s.", rank)
        return

    logger.info("Loading and preparing dataset from %s", config["input_path"])
    dataset = load_and_prepare_dataset(config)

    max_samples = config.get("max_samples")
    if max_samples is not None:
        max_samples = int(max_samples)
        if max_samples > 0:
            take_n = min(max_samples, len(dataset))
            dataset = dataset.select(range(take_n))

    logger.info("Prepared dataset size before sharding: %d", len(dataset))
    shard_dataset = dataset.shard(num_shards=world_size, index=rank, contiguous=True)
    num_samples_local = len(shard_dataset)
    logger.info("Rank local shard size: %d", num_samples_local)

    model_name_or_path = str(config["model_name_or_path"])
    dtype = resolve_torch_dtype(str(config.get("dtype", "bfloat16")))

    logger.info("Loading tokenizer and model: %s", model_name_or_path)
    tokenizer = load_tokenizer(model_name_or_path)
    model = load_model(model_name_or_path, torch_dtype=dtype, device=device)
    hidden_size = int(getattr(model.config, "hidden_size"))

    batch_size = int(config.get("batch_size", 8))
    max_length = int(config.get("max_length", 2048))
    normalize_embeddings = _parse_bool(config.get("normalize_embeddings", False))

    id_chunks: list[np.ndarray] = []
    last_chunks: list[np.ndarray] = []
    mean_chunks: list[np.ndarray] = []

    total_batches = math.ceil(num_samples_local / batch_size) if num_samples_local > 0 else 0
    wall_start = time.perf_counter()
    processed = 0

    progress = tqdm(
        range(0, num_samples_local, batch_size),
        disable=not accelerator.is_local_main_process,
        desc=f"rank{rank}",
    )

    for batch_idx, start_idx in enumerate(progress):
        try:
            batch = shard_dataset[start_idx : start_idx + batch_size]
            batch_ids = batch["sample_id"]
            batch_text = batch["input_text"]

            model_inputs = tokenizer(
                batch_text,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

            with torch.inference_mode():
                last_hidden = forward_last_hidden_state(model, model_inputs)

                if "last_token" in pooling_methods:
                    emb_last = last_token_pool(last_hidden, model_inputs["attention_mask"])
                    last_chunks.append(emb_last.detach().float().cpu().numpy())

                if "mean" in pooling_methods:
                    emb_mean = mean_pool(last_hidden, model_inputs["attention_mask"])
                    mean_chunks.append(emb_mean.detach().float().cpu().numpy())

            id_chunks.append(_to_numpy_ids(batch_ids))

            processed += len(batch_ids)
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                elapsed = time.perf_counter() - wall_start
                throughput = processed / elapsed if elapsed > 0 else 0.0
                logger.info(
                    "Processed batch %d/%d, samples=%d, throughput=%.2f samples/s",
                    batch_idx + 1,
                    total_batches,
                    processed,
                    throughput,
                )
        except Exception as exc:  # pragma: no cover
            logger.error(
                "Batch failure at rank=%d batch_idx=%d start_idx=%d error=%s",
                rank,
                batch_idx,
                start_idx,
                str(exc),
            )
            logger.error(traceback.format_exc())
            raise

    if id_chunks:
        ids_array = np.concatenate(id_chunks, axis=0)
    else:
        ids_array = np.empty((0,), dtype=np.int64)

    save_npy(rank_paths["ids"], ids_array)

    if "last_token" in pooling_methods:
        if last_chunks:
            emb_last_array = np.concatenate(last_chunks, axis=0).astype(np.float32, copy=False)
        else:
            emb_last_array = np.empty((0, hidden_size), dtype=np.float32)
        if normalize_embeddings:
            emb_last_array = _l2_normalize(emb_last_array)
        save_npy(rank_paths["last_token"], emb_last_array)

    if "mean" in pooling_methods:
        if mean_chunks:
            emb_mean_array = np.concatenate(mean_chunks, axis=0).astype(np.float32, copy=False)
        else:
            emb_mean_array = np.empty((0, hidden_size), dtype=np.float32)
        if normalize_embeddings:
            emb_mean_array = _l2_normalize(emb_mean_array)
        save_npy(rank_paths["mean"], emb_mean_array)

    elapsed_total = time.perf_counter() - wall_start
    logger.info("Rank %d completed. local_samples=%d elapsed=%.2fs", rank, num_samples_local, elapsed_total)


if __name__ == "__main__":
    main()
