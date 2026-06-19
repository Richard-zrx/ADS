from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from data_utils import ensure_sample_id, load_raw_dataset
from io_utils import ensure_dir
from logging_utils import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Join cluster assignments back into the raw VERL training parquet")
    parser.add_argument(
        "--raw_data_path",
        default="dataset/train.parquet",
    )
    parser.add_argument("--raw_data_format", default="parquet")
    parser.add_argument(
        "--cluster_assignments_path",
        default="outputs/ads_run/phase2a/cluster_assignments_mean_k64.parquet",
    )
    parser.add_argument(
        "--output_path",
        default="outputs/ads_run/phase3/train_with_cluster_cot_k64.parquet",
    )
    parser.add_argument("--cluster_source", default="kmeans_k64")
    parser.add_argument("--id_field", default="sample_id")
    parser.add_argument("--num_proc", type=int, default=1)
    parser.add_argument("--datasets_cache_dir", default=None)
    parser.add_argument("--expected_num_rows", type=int, default=None)
    parser.add_argument("--mirror_into_extra_info", action="store_true", default=True)
    parser.add_argument("--no_mirror_into_extra_info", dest="mirror_into_extra_info", action="store_false")
    return parser.parse_args()


def _normalize_cluster_assignments(path: Path) -> tuple[pd.DataFrame, dict[Any, dict[str, Any]]]:
    assignments = pd.read_parquet(path)
    required_cols = {"sample_id", "cluster_id"}
    missing = required_cols - set(assignments.columns)
    if missing:
        raise ValueError(f"Cluster assignments missing required columns: {sorted(missing)}")

    if assignments["sample_id"].duplicated().any():
        dupes = assignments.loc[assignments["sample_id"].duplicated(), "sample_id"].head(20).tolist()
        raise ValueError(f"Duplicate sample_id values in cluster assignments: {dupes}")

    mapping: dict[Any, dict[str, Any]] = {}
    for row in assignments[["sample_id", "cluster_id"]].itertuples(index=False):
        mapping[row.sample_id] = {"cluster_id": int(row.cluster_id)}
    return assignments, mapping


def _join_cluster_fields(
    dataset,
    cluster_map: dict[Any, dict[str, Any]],
    id_field: str,
    cluster_source: str,
    mirror_into_extra_info: bool,
    num_proc: int,
):
    def map_fn(example: dict[str, Any], idx: int) -> dict[str, Any]:
        sample_id = ensure_sample_id(example, idx, id_field=id_field)
        if sample_id not in cluster_map:
            raise KeyError(f"sample_id missing from cluster assignments: {sample_id}")

        cluster_id = int(cluster_map[sample_id]["cluster_id"])
        updated: dict[str, Any] = {
            "sample_id": sample_id,
            "cluster_id": cluster_id,
            "cluster_source": cluster_source,
        }

        if mirror_into_extra_info:
            extra_info = example.get("extra_info")
            if extra_info is None:
                extra_info = {}
            elif hasattr(extra_info, "tolist") and not isinstance(extra_info, dict):
                extra_info = extra_info.tolist()
            if not isinstance(extra_info, dict):
                extra_info = {"raw_extra_info": extra_info}
            extra_info = dict(extra_info)
            extra_info["sample_id"] = sample_id
            extra_info["cluster_id"] = cluster_id
            extra_info["cluster_source"] = cluster_source
            updated["extra_info"] = extra_info
        return updated

    map_num_proc = num_proc if num_proc and num_proc > 1 else None
    try:
        return dataset.map(
            map_fn,
            with_indices=True,
            num_proc=map_num_proc,
            desc="Joining cluster assignments back to raw training data",
        )
    except Exception:
        if mirror_into_extra_info and map_num_proc is not None:
            raise
        raise


def _compute_manifest(
    raw_rows: int,
    assignments: pd.DataFrame,
    joined_dataset,
    cluster_source: str,
    mirror_into_extra_info: bool,
) -> dict[str, Any]:
    cluster_sizes = assignments["cluster_id"].value_counts().sort_index()
    sample_ids = joined_dataset["sample_id"]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Joined dataset contains duplicate sample_id values")

    cluster_ids = joined_dataset["cluster_id"]
    missing_cluster_count = sum(value is None for value in cluster_ids)
    if missing_cluster_count:
        raise ValueError(f"Joined dataset contains {missing_cluster_count} missing cluster_id values")

    return {
        "num_rows_raw": int(raw_rows),
        "num_rows_cluster": int(len(assignments)),
        "num_rows_joined": int(len(joined_dataset)),
        "num_unique_sample_id": int(len(set(sample_ids))),
        "num_unique_cluster_id": int(len(set(cluster_ids))),
        "cluster_size_min": int(cluster_sizes.min()),
        "cluster_size_max": int(cluster_sizes.max()),
        "cluster_size_mean": float(cluster_sizes.mean()),
        "cluster_source": cluster_source,
        "mirrored_into_extra_info": bool(mirror_into_extra_info),
    }


def main() -> None:
    args = parse_args()
    logger = setup_logger("prepare_verl_cluster_data", rank=0, world_size=1)

    raw_data_path = Path(args.raw_data_path)
    cluster_assignments_path = Path(args.cluster_assignments_path)
    output_path = Path(args.output_path)
    manifest_path = output_path.with_name(f"{output_path.stem}_manifest.json")
    datasets_cache_dir = args.datasets_cache_dir
    if not datasets_cache_dir:
        datasets_cache_dir = str(Path(__file__).resolve().parents[1] / ".cache" / "hf_datasets")

    logger.info("Loading cluster assignments: %s", cluster_assignments_path)
    assignments, cluster_map = _normalize_cluster_assignments(cluster_assignments_path)

    logger.info("Loading raw dataset: %s", raw_data_path)
    dataset = load_raw_dataset(
        input_path=str(raw_data_path),
        input_format=str(args.raw_data_format),
        cache_dir=datasets_cache_dir,
    )

    raw_rows = len(dataset)
    if raw_rows != len(assignments):
        raise ValueError(
            f"Raw dataset / cluster assignment row mismatch: raw={raw_rows} cluster={len(assignments)}"
        )

    logger.info("Joining cluster assignments back into raw dataset")
    try:
        joined_dataset = _join_cluster_fields(
            dataset=dataset,
            cluster_map=cluster_map,
            id_field=str(args.id_field),
            cluster_source=str(args.cluster_source),
            mirror_into_extra_info=bool(args.mirror_into_extra_info),
            num_proc=int(args.num_proc),
        )
        mirrored = bool(args.mirror_into_extra_info)
    except Exception as exc:
        if not bool(args.mirror_into_extra_info):
            raise
        logger.warning("extra_info mirroring failed (%s); retrying with top-level fields only", exc)
        joined_dataset = _join_cluster_fields(
            dataset=dataset,
            cluster_map=cluster_map,
            id_field=str(args.id_field),
            cluster_source=str(args.cluster_source),
            mirror_into_extra_info=False,
            num_proc=int(args.num_proc),
        )
        mirrored = False

    if args.expected_num_rows is not None and len(joined_dataset) != int(args.expected_num_rows):
        raise ValueError(
            f"Joined dataset row count mismatch: expected={int(args.expected_num_rows)} actual={len(joined_dataset)}"
        )

    manifest = _compute_manifest(
        raw_rows=raw_rows,
        assignments=assignments,
        joined_dataset=joined_dataset,
        cluster_source=str(args.cluster_source),
        mirror_into_extra_info=mirrored,
    )

    ensure_dir(output_path.parent)
    logger.info("Writing joined parquet: %s", output_path)
    joined_dataset.to_parquet(str(output_path))

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    logger.info("Wrote manifest: %s", manifest_path)
    logger.info(
        "Phase-3a data prep complete. rows=%d unique_clusters=%d",
        manifest["num_rows_joined"],
        manifest["num_unique_cluster_id"],
    )


if __name__ == "__main__":
    main()
