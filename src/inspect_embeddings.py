from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from io_utils import ensure_dir, load_npy
from logging_utils import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect merged embeddings for sanity checks")
    parser.add_argument("--config", required=True, help="Path to clustering YAML config")
    parser.add_argument("--ids_path", default=None)
    parser.add_argument("--embedding_path", default=None)
    parser.add_argument("--embedding_name", default=None)
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config file must contain a YAML object")
    return cfg


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    updated = dict(config)
    for key in ["ids_path", "embedding_path", "embedding_name", "output_dir"]:
        value = getattr(args, key)
        if value is not None:
            updated[key] = value
    return updated


def _to_float(value: Any) -> float:
    return float(value) if value is not None else float("nan")


def _norm_stats(embeddings: np.ndarray) -> dict[str, float]:
    norms = np.linalg.norm(embeddings, ord=2, axis=1)
    return {
        "min": _to_float(np.min(norms)),
        "max": _to_float(np.max(norms)),
        "mean": _to_float(np.mean(norms)),
        "std": _to_float(np.std(norms)),
        "p50": _to_float(np.percentile(norms, 50)),
        "p95": _to_float(np.percentile(norms, 95)),
        "p99": _to_float(np.percentile(norms, 99)),
    }


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)

    logger = setup_logger("inspect_embeddings", rank=0, world_size=1)

    ids_path = Path(str(config["ids_path"]))
    embedding_path = Path(str(config["embedding_path"]))
    embedding_name = str(config.get("embedding_name", "embedding"))
    output_dir = ensure_dir(str(config["output_dir"]))

    ids = load_npy(ids_path)
    embeddings = load_npy(embedding_path)

    if embeddings.ndim != 2:
        raise ValueError(f"Embedding array must be 2D, got shape={embeddings.shape}")

    num_ids = int(len(ids))
    num_embeddings = int(embeddings.shape[0])
    if num_ids != num_embeddings:
        raise ValueError(f"Length mismatch: len(ids)={num_ids}, embeddings.shape[0]={num_embeddings}")

    unique_ids = int(len(np.unique(ids)))
    if unique_ids != num_ids:
        raise ValueError(f"sample_id must be unique. unique={unique_ids}, total={num_ids}")

    has_nan = bool(np.isnan(embeddings).any())
    has_inf = bool(np.isinf(embeddings).any())
    if has_nan or has_inf:
        raise ValueError(f"Invalid values detected: has_nan={has_nan}, has_inf={has_inf}")

    stats = {
        "embedding_name": embedding_name,
        "ids_path": str(ids_path),
        "embedding_path": str(embedding_path),
        "num_samples": num_ids,
        "embedding_shape": list(embeddings.shape),
        "embedding_dtype": str(embeddings.dtype),
        "ids_dtype": str(ids.dtype),
        "sample_id_unique": True,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "memory_bytes": int(embeddings.nbytes),
        "memory_mb": float(embeddings.nbytes / (1024 * 1024)),
        "norm_stats": _norm_stats(embeddings),
    }

    out_path = output_dir / f"embedding_stats_{embedding_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info("Embedding sanity check passed for %s", embedding_name)
    logger.info("Saved stats: %s", out_path)


if __name__ == "__main__":
    main()
