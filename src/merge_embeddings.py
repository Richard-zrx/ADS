from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from io_utils import load_npy, merged_output_paths, normalize_pooling_methods, save_embeddings_parquet, save_npy, shard_output_paths
from logging_utils import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge rank shards from embedding extraction")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--num_shards", type=int, default=None)
    parser.add_argument("--pooling_methods", default=None, help="Comma separated, e.g. last_token,mean")
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config file must contain a YAML object")
    return cfg


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    updated = dict(config)
    if args.output_dir is not None:
        updated["output_dir"] = args.output_dir
    if args.num_shards is not None:
        updated["expected_num_processes"] = args.num_shards
    if args.pooling_methods is not None:
        updated["pooling_methods"] = [x.strip() for x in args.pooling_methods.split(",") if x.strip()]
    return updated


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _ensure_same_length(ids: np.ndarray, emb: np.ndarray, name: str) -> None:
    if len(ids) != len(emb):
        raise ValueError(f"Length mismatch for {name}: ids={len(ids)} emb={len(emb)}")


def _concat_or_empty(chunks: list[np.ndarray], ndim: int = 1) -> np.ndarray:
    if chunks:
        return np.concatenate(chunks, axis=0)
    if ndim == 1:
        return np.empty((0,), dtype=np.int64)
    return np.empty((0, 0), dtype=np.float32)


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)

    output_dir = Path(str(config["output_dir"]))
    pooling_methods = normalize_pooling_methods(list(config.get("pooling_methods", ["last_token", "mean"])))
    num_shards = int(config.get("expected_num_processes", 4))
    export_parquet_aux = _parse_bool(config.get("export_parquet_aux", False))

    logger = setup_logger("merge_embeddings", rank=0, world_size=1)
    logger.info("Merging from output_dir=%s, num_shards=%d", output_dir, num_shards)

    id_chunks: list[np.ndarray] = []
    last_chunks: list[np.ndarray] = []
    mean_chunks: list[np.ndarray] = []

    for rank in range(num_shards):
        shard_paths = shard_output_paths(output_dir=output_dir, rank=rank, pooling_methods=pooling_methods)
        for name, path in shard_paths.items():
            if not path.exists():
                raise FileNotFoundError(f"Missing shard file: {path} ({name})")

        ids = load_npy(shard_paths["ids"])
        id_chunks.append(ids)

        if "last_token" in pooling_methods:
            emb_last = load_npy(shard_paths["last_token"])
            _ensure_same_length(ids, emb_last, f"last_token_rank{rank}")
            last_chunks.append(emb_last)

        if "mean" in pooling_methods:
            emb_mean = load_npy(shard_paths["mean"])
            _ensure_same_length(ids, emb_mean, f"mean_rank{rank}")
            mean_chunks.append(emb_mean)

    all_ids = _concat_or_empty(id_chunks, ndim=1)
    all_last = _concat_or_empty(last_chunks, ndim=2) if "last_token" in pooling_methods else None
    all_mean = _concat_or_empty(mean_chunks, ndim=2) if "mean" in pooling_methods else None

    if all_ids.dtype == object:
        sort_idx = np.argsort(all_ids.astype(str))
    else:
        sort_idx = np.argsort(all_ids)

    sorted_ids = all_ids[sort_idx]
    if len(np.unique(sorted_ids)) != len(sorted_ids):
        raise ValueError("Duplicate sample_id detected after merge.")

    sorted_last = all_last[sort_idx] if all_last is not None else None
    sorted_mean = all_mean[sort_idx] if all_mean is not None else None

    merged_paths = merged_output_paths(output_dir, pooling_methods)
    save_npy(merged_paths["ids"], sorted_ids)
    if sorted_last is not None:
        save_npy(merged_paths["last_token"], sorted_last)
    if sorted_mean is not None:
        save_npy(merged_paths["mean"], sorted_mean)

    if export_parquet_aux:
        save_embeddings_parquet(
            path=merged_paths["parquet"],
            sample_ids=sorted_ids,
            embedding_last=sorted_last,
            embedding_mean=sorted_mean,
        )
        logger.info("Aux parquet exported: %s", merged_paths["parquet"])

    logger.info("Merge done. total_samples=%d", len(sorted_ids))


if __name__ == "__main__":
    main()
