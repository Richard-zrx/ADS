"""Phase 4.5: Merge per-rank difficulty score shards into single arrays."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from io_utils import ensure_dir, load_npy, save_npy
from logging_utils import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge difficulty score shards")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--num_shards", type=int, default=None)
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML object")
    return cfg


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    output_dir = Path(str(args.output_dir or config["output_dir"]))
    num_shards = args.num_shards or int(config.get("expected_num_processes", 4))
    ensure_dir(output_dir)

    logger = setup_logger("merge_difficulty", rank=0, world_size=1)
    logger.info("Merging %d shards from %s", num_shards, output_dir)

    id_chunks: list[np.ndarray] = []
    score_chunks: list[np.ndarray] = []

    for rank in range(num_shards):
        ids_path = output_dir / f"difficulty_ids_rank{rank}.npy"
        scores_path = output_dir / f"difficulty_scores_rank{rank}.npy"
        if not ids_path.exists():
            raise FileNotFoundError(f"Missing shard: {ids_path}")
        if not scores_path.exists():
            raise FileNotFoundError(f"Missing shard: {scores_path}")

        ids = load_npy(ids_path)
        scores = load_npy(scores_path)
        if len(ids) != len(scores):
            raise ValueError(f"Rank {rank}: ids={len(ids)} scores={len(scores)} mismatch")

        id_chunks.append(ids)
        score_chunks.append(scores)
        logger.info("Loaded rank %d: %d samples", rank, len(ids))

    all_ids = np.concatenate(id_chunks, axis=0)
    all_scores = np.concatenate(score_chunks, axis=0)

    sort_idx = np.argsort(all_ids)
    all_ids = all_ids[sort_idx]
    all_scores = all_scores[sort_idx]

    if len(np.unique(all_ids)) != len(all_ids):
        raise ValueError("Duplicate sample_id detected after merge")

    save_npy(output_dir / "all_difficulty_ids.npy", all_ids)
    save_npy(output_dir / "all_difficulty_scores.npy", all_scores)

    finite = all_scores[np.isfinite(all_scores)]
    logger.info(
        "Merged %d samples. score: min=%.4f max=%.4f mean=%.4f  inf_count=%d",
        len(all_ids),
        float(np.min(finite)) if len(finite) else float("nan"),
        float(np.max(finite)) if len(finite) else float("nan"),
        float(np.mean(finite)) if len(finite) else float("nan"),
        int(np.sum(~np.isfinite(all_scores))),
    )


if __name__ == "__main__":
    main()
