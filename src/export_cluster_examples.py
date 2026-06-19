from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from data_utils import build_id_to_text_map
from io_utils import ensure_dir
from logging_utils import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export representative examples per cluster")
    parser.add_argument("--config", required=True, help="Path to clustering YAML config")
    parser.add_argument("--assignments_path", default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--top_k_nearest", type=int, default=None)
    parser.add_argument("--top_k_random", type=int, default=None)
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config file must contain a YAML object")
    return cfg


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    updated = dict(config)
    if args.assignments_path is not None:
        updated["assignments_path"] = args.assignments_path
    if args.output_path is not None:
        updated["output_path"] = args.output_path
    if args.top_k_nearest is not None:
        updated["top_k_nearest"] = args.top_k_nearest
    if args.top_k_random is not None:
        updated["top_k_random"] = args.top_k_random
    return updated


def _sanitize_float_tag(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def _method_tag(config: dict[str, Any], method: str, num_clusters: int | None) -> str:
    if method == "kmeans":
        if num_clusters is None:
            raise ValueError("num_clusters is required for kmeans")
        return f"k{num_clusters}"
    if method == "spherical_kmeans":
        if num_clusters is None:
            raise ValueError("num_clusters is required for spherical_kmeans")
        return f"sk{num_clusters}"
    if method == "gmm":
        if num_clusters is None:
            raise ValueError("num_clusters is required for gmm")
        return f"gmm{num_clusters}"
    if method == "bisecting_kmeans":
        if num_clusters is None:
            raise ValueError("num_clusters is required for bisecting_kmeans")
        return f"bk{num_clusters}"
    if method == "knn_leiden":
        knn_k = int(config.get("knn_k", 30))
        resolution = _sanitize_float_tag(float(config.get("leiden_resolution", 1.0)))
        return f"leiden_k{knn_k}_r{resolution}"
    return method


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    try:
        return json.loads(json.dumps(value, ensure_ascii=False))
    except Exception:
        return str(value)


def _distance_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except Exception:
        return None
    if np.isnan(f) or np.isinf(f):
        return None
    return f


def _build_rows_from_subset(
    subset: pd.DataFrame,
    cluster_id: int,
    cluster_size: int,
    selection_type: str,
    embedding_name: str,
    method: str,
    id_to_record: dict[Any, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank_idx, (_, row) in enumerate(subset.iterrows(), start=1):
        sample_id = row["sample_id"]
        record = id_to_record[sample_id]
        rows.append(
            {
                "embedding_name": embedding_name,
                "clustering_method": method,
                "cluster_id": int(cluster_id),
                "cluster_size": int(cluster_size),
                "selection_type": selection_type,
                "rank_in_selection": int(rank_idx),
                "sample_id": int(sample_id) if isinstance(sample_id, (np.integer, int)) else sample_id,
                "input_text": record.get("input_text", ""),
                "distance_to_centroid": _distance_or_none(row.get("distance_to_centroid")),
                "cluster_confidence": _distance_or_none(row.get("cluster_confidence")),
                "source_split": record.get("source_split"),
                "target": _as_jsonable(record.get("target")),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)

    logger = setup_logger("export_cluster_examples", rank=0, world_size=1)

    output_dir = ensure_dir(str(config["output_dir"]))
    embedding_name = str(config.get("embedding_name", "mean"))
    method = str(config.get("clustering_method", "kmeans")).strip().lower()
    num_clusters = int(config.get("num_clusters")) if config.get("num_clusters") is not None else None
    method_tag = _method_tag(config, method, num_clusters)

    assignments_path = Path(
        str(
            config.get(
                "assignments_path",
                output_dir / f"cluster_assignments_{embedding_name}_{method_tag}.parquet",
            )
        )
    )

    output_path = Path(
        str(
            config.get(
                "output_path",
                output_dir / f"cluster_examples_{embedding_name}_{method_tag}.jsonl",
            )
        )
    )

    top_k_nearest = int(config.get("top_k_nearest", 10))
    top_k_random = int(config.get("top_k_random", 10))
    random_seed = int(config.get("random_seed", 42))

    logger.info("Loading assignments: %s", assignments_path)
    assignments = pd.read_parquet(assignments_path)

    if "sample_id" not in assignments.columns or "cluster_id" not in assignments.columns:
        raise ValueError("Assignments must contain sample_id and cluster_id columns")

    logger.info("Building sample_id -> text map from raw dataset")
    id_to_record = build_id_to_text_map(
        raw_data_path=str(config["raw_data_path"]),
        raw_data_format=str(config.get("raw_data_format", "parquet")),
        text_field=str(config.get("text_field", "prompt")),
        id_field=str(config.get("id_field", "sample_id")),
        prompt_text_mode=str(config.get("prompt_text_mode", "user_only")),
        input_text_mode=str(config.get("input_text_mode", "question_only")),
        num_proc=int(config.get("num_proc", 1)),
        cache_dir=str(config.get("datasets_cache_dir")) if config.get("datasets_cache_dir") else None,
    )

    missing_ids = [sample_id for sample_id in assignments["sample_id"].tolist() if sample_id not in id_to_record]
    if missing_ids:
        raise ValueError(f"Found {len(missing_ids)} sample_ids not present in raw dataset mapping. first_missing={missing_ids[:20]}")

    rows: list[dict[str, Any]] = []
    for cluster_id, group in assignments.groupby("cluster_id", sort=True):
        cluster_size = int(len(group))

        if "distance_to_centroid" in group.columns and group["distance_to_centroid"].notna().any() and top_k_nearest > 0:
            nearest = group.dropna(subset=["distance_to_centroid"]).sort_values("distance_to_centroid", ascending=True)
            nearest = nearest.head(min(top_k_nearest, len(nearest)))
            rows.extend(
                _build_rows_from_subset(
                    subset=nearest,
                    cluster_id=int(cluster_id),
                    cluster_size=cluster_size,
                    selection_type="nearest",
                    embedding_name=embedding_name,
                    method=method,
                    id_to_record=id_to_record,
                )
            )

        if top_k_random > 0:
            sample_n = min(top_k_random, cluster_size)
            random_state = int(random_seed + (int(cluster_id) & 0xFFFF))
            random_subset = group.sample(n=sample_n, random_state=random_state)
            rows.extend(
                _build_rows_from_subset(
                    subset=random_subset,
                    cluster_id=int(cluster_id),
                    cluster_size=cluster_size,
                    selection_type="random",
                    embedding_name=embedding_name,
                    method=method,
                    id_to_record=id_to_record,
                )
            )

    ensure_dir(output_path.parent)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info("Saved cluster examples: %s (rows=%d)", output_path, len(rows))

    if _parse_bool(config.get("export_examples_parquet", False)):
        parquet_path = output_path.with_suffix(".parquet")
        pd.DataFrame(rows).to_parquet(parquet_path, index=False)
        logger.info("Saved cluster examples parquet: %s", parquet_path)


if __name__ == "__main__":
    main()
