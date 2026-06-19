from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SUPPORTED_POOLING = {"last_token", "mean"}


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def normalize_pooling_methods(methods: list[str]) -> list[str]:
    cleaned = []
    for method in methods:
        key = method.strip().lower()
        if key not in SUPPORTED_POOLING:
            raise ValueError(f"Unsupported pooling method: {method}")
        if key not in cleaned:
            cleaned.append(key)
    if not cleaned:
        raise ValueError("At least one pooling method is required.")
    return cleaned


def shard_output_paths(output_dir: str | Path, rank: int, pooling_methods: list[str]) -> dict[str, Path]:
    out_dir = ensure_dir(output_dir)
    paths: dict[str, Path] = {"ids": out_dir / f"ids_rank{rank}.npy"}
    if "last_token" in pooling_methods:
        paths["last_token"] = out_dir / f"emb_last_rank{rank}.npy"
    if "mean" in pooling_methods:
        paths["mean"] = out_dir / f"emb_mean_rank{rank}.npy"
    return paths


def merged_output_paths(output_dir: str | Path, pooling_methods: list[str]) -> dict[str, Path]:
    out_dir = ensure_dir(output_dir)
    paths: dict[str, Path] = {"ids": out_dir / "all_ids.npy"}
    if "last_token" in pooling_methods:
        paths["last_token"] = out_dir / "all_emb_last.npy"
    if "mean" in pooling_methods:
        paths["mean"] = out_dir / "all_emb_mean.npy"
    paths["parquet"] = out_dir / "embeddings_merged.parquet"
    return paths


def all_paths_exist(paths: dict[str, Path]) -> bool:
    return all(path.exists() for path in paths.values())


def save_npy(path: str | Path, array: np.ndarray) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    np.save(path, array)


def load_npy(path: str | Path) -> np.ndarray:
    return np.load(Path(path), allow_pickle=True)


def save_embeddings_parquet(
    path: str | Path,
    sample_ids: np.ndarray,
    embedding_last: np.ndarray | None,
    embedding_mean: np.ndarray | None,
) -> None:
    records: dict[str, Any] = {"sample_id": sample_ids.tolist()}
    if embedding_last is not None:
        records["embedding_last_token"] = [row.tolist() for row in embedding_last]
    if embedding_mean is not None:
        records["embedding_mean"] = [row.tolist() for row in embedding_mean]

    df = pd.DataFrame(records)
    out_path = Path(path)
    ensure_dir(out_path.parent)
    df.to_parquet(out_path, index=False)
