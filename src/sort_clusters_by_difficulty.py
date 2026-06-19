"""Phase 5: Sort samples within each cluster by difficulty score (ascending = easy first).

Reads:
  - Phase 3 parquet (with cluster_id)
  - Phase 4 merged difficulty scores

Adds columns:
  - difficulty_score   : average NLL under base model (lower = easier)
  - rank_in_cluster    : 0-based position within cluster after easy-to-hard sort
  - cluster_size       : total samples in that cluster

Mirrors all three into extra_info as well.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from io_utils import ensure_dir, load_npy
from logging_utils import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sort clusters by difficulty")
    parser.add_argument("--clustered_parquet", required=True)
    parser.add_argument("--difficulty_ids_path", required=True)
    parser.add_argument("--difficulty_scores_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--stats_output_path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger("sort_clusters_by_difficulty", rank=0, world_size=1)

    logger.info("Loading clustered parquet: %s", args.clustered_parquet)
    df = pd.read_parquet(args.clustered_parquet)

    logger.info("Loading difficulty scores")
    diff_ids = load_npy(args.difficulty_ids_path)
    diff_scores = load_npy(args.difficulty_scores_path)

    id_to_score: dict[int, float] = {int(k): float(v) for k, v in zip(diff_ids.tolist(), diff_scores.tolist())}

    missing = [sid for sid in df["sample_id"] if int(sid) not in id_to_score]
    if missing:
        raise ValueError(f"{len(missing)} sample_ids have no difficulty score. First 10: {missing[:10]}")

    df["difficulty_score"] = df["sample_id"].apply(lambda x: id_to_score[int(x)])

    # Replace inf (empty answer after truncation) with max_finite + 1 so they sort last
    inf_mask = ~np.isfinite(df["difficulty_score"])
    if inf_mask.any():
        finite_max = df.loc[~inf_mask, "difficulty_score"].max()
        df.loc[inf_mask, "difficulty_score"] = finite_max + 1.0
        logger.warning("Replaced %d inf scores with %.4f", int(inf_mask.sum()), finite_max + 1.0)

    # Sort: cluster first, then ascending difficulty (easy→hard)
    df = df.sort_values(["cluster_id", "difficulty_score"], ascending=[True, True]).reset_index(drop=True)

    df["rank_in_cluster"] = df.groupby("cluster_id").cumcount()
    df["cluster_size"] = df.groupby("cluster_id")["sample_id"].transform("count")

    # Mirror into extra_info
    if "extra_info" in df.columns:
        def _patch_extra_info(row):
            ei = row["extra_info"]
            if ei is None:
                ei = {}
            if isinstance(ei, np.ndarray):
                ei = ei.tolist()
            if not isinstance(ei, dict):
                ei = {"raw_extra_info": ei}
            ei = dict(ei)
            ei["difficulty_score"] = float(row["difficulty_score"])
            ei["rank_in_cluster"] = int(row["rank_in_cluster"])
            ei["cluster_size"] = int(row["cluster_size"])
            return ei

        df["extra_info"] = df.apply(_patch_extra_info, axis=1)

    output_path = Path(args.output_path)
    ensure_dir(output_path.parent)
    df.to_parquet(str(output_path), index=False)
    logger.info("Wrote final parquet: %s (%d rows)", output_path, len(df))

    # Per-cluster stats
    stats = {}
    for cid, grp in df.groupby("cluster_id"):
        stats[int(cid)] = {
            "cluster_id": int(cid),
            "size": int(len(grp)),
            "difficulty_min": float(grp["difficulty_score"].min()),
            "difficulty_max": float(grp["difficulty_score"].max()),
            "difficulty_mean": float(grp["difficulty_score"].mean()),
            "difficulty_std": float(grp["difficulty_score"].std()),
        }

    stats_path = args.stats_output_path or str(output_path.with_suffix("")) + "_cluster_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info("Wrote cluster stats: %s", stats_path)

    logger.info(
        "Done. clusters=%d  global difficulty: min=%.4f max=%.4f mean=%.4f",
        df["cluster_id"].nunique(),
        df["difficulty_score"].min(),
        df["difficulty_score"].max(),
        df["difficulty_score"].mean(),
    )


if __name__ == "__main__":
    main()
