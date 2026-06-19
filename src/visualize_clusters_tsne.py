from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from io_utils import ensure_dir, load_npy
from logging_utils import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="t-SNE visualization of phase-5 clusters")
    parser.add_argument("--config", required=True, help="Path to t-SNE YAML config")
    parser.add_argument("--ids_path", default=None)
    parser.add_argument("--embedding_path", default=None)
    parser.add_argument("--clusters_parquet", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--pca_dim", type=int, default=None)
    parser.add_argument("--tsne_perplexity", type=float, default=None)
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config file must contain a YAML object")
    return cfg


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    updated = dict(config)
    for key in ["ids_path", "embedding_path", "clusters_parquet", "output_dir"]:
        value = getattr(args, key)
        if value is not None:
            updated[key] = value
    if args.pca_dim is not None:
        updated["pca_dim"] = args.pca_dim
    if args.tsne_perplexity is not None:
        updated["tsne_perplexity"] = args.tsne_perplexity
    return updated


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, ord=2, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return x / norms


def _build_qualitative_palette(n: int) -> ListedColormap:
    base = (
        plt.colormaps["tab20"].colors
        + plt.colormaps["tab20b"].colors
        + plt.colormaps["tab20c"].colors
    )
    if n <= len(base):
        colors = list(base[:n])
    else:
        extra = plt.colormaps["hsv"](np.linspace(0, 1, n - len(base), endpoint=False))
        colors = list(base) + [tuple(c) for c in extra]
    return ListedColormap(colors, name=f"qual{n}")


def plot_by_cluster(
    coords: np.ndarray,
    cluster_ids: np.ndarray,
    out_path: Path,
    *,
    s: float,
    alpha: float,
    dpi: int,
) -> None:
    n_clusters = int(cluster_ids.max()) + 1
    cmap = _build_qualitative_palette(n_clusters)

    fig, ax = plt.subplots(figsize=(8.5, 8), constrained_layout=True)
    order = np.argsort(np.random.default_rng(0).random(coords.shape[0]))
    ax.scatter(
        coords[order, 0],
        coords[order, 1],
        c=cluster_ids[order],
        cmap=cmap,
        vmin=-0.5,
        vmax=n_clusters - 0.5,
        s=s,
        alpha=alpha,
        linewidths=0,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(f"t-SNE of mean embeddings, colored by cluster (K={n_clusters})")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_by_difficulty(
    coords: np.ndarray,
    difficulty: np.ndarray,
    out_path: Path,
    *,
    s: float,
    alpha: float,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    order = np.argsort(np.random.default_rng(1).random(coords.shape[0]))
    sc = ax.scatter(
        coords[order, 0],
        coords[order, 1],
        c=difficulty[order],
        cmap="viridis",
        s=s,
        alpha=alpha,
        linewidths=0,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("t-SNE of mean embeddings, colored by difficulty score")
    cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
    cbar.set_label("difficulty score (lower = easier)")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)

    logger = setup_logger("visualize_clusters_tsne", rank=0, world_size=1)

    ids_path = Path(str(config["ids_path"]))
    embedding_path = Path(str(config["embedding_path"]))
    clusters_parquet = Path(str(config["clusters_parquet"]))
    output_dir = ensure_dir(str(config["output_dir"]))

    random_seed = int(config.get("random_seed", 42))
    np.random.seed(random_seed)

    pca_dim = int(config.get("pca_dim", 50))
    perplexity = float(config.get("tsne_perplexity", 30))
    n_iter = int(config.get("tsne_n_iter", 1000))
    metric = str(config.get("tsne_metric", "cosine"))
    init = str(config.get("tsne_init", "pca"))
    learning_rate = config.get("tsne_learning_rate", "auto")
    if isinstance(learning_rate, str) and learning_rate.replace(".", "", 1).isdigit():
        learning_rate = float(learning_rate)

    figure_dpi = int(config.get("figure_dpi", 200))
    scatter_size = float(config.get("scatter_size", 4))
    scatter_alpha = float(config.get("scatter_alpha", 0.7))

    logger.info("Loading embeddings and ids")
    ids = load_npy(ids_path)
    embeddings = load_npy(embedding_path)
    if embeddings.ndim != 2 or embeddings.shape[0] != ids.shape[0]:
        raise ValueError(
            f"Shape mismatch: embeddings={embeddings.shape} ids={ids.shape}"
        )

    logger.info("Loading cluster assignments from parquet: %s", clusters_parquet)
    df = pd.read_parquet(clusters_parquet, columns=["sample_id", "cluster_id", "difficulty_score"])

    id_to_row = {int(sid): i for i, sid in enumerate(ids.tolist())}
    missing = [int(sid) for sid in df["sample_id"].tolist() if int(sid) not in id_to_row]
    if missing:
        raise ValueError(
            f"{len(missing)} sample_ids in parquet are absent from all_ids.npy. first={missing[:10]}"
        )
    row_idx = np.array([id_to_row[int(sid)] for sid in df["sample_id"].tolist()], dtype=np.int64)
    cluster_ids = df["cluster_id"].to_numpy().astype(np.int32)
    difficulty = df["difficulty_score"].to_numpy().astype(np.float32)

    work = embeddings[row_idx].astype(np.float32, copy=False)
    if bool(config.get("normalize_embeddings", True)):
        logger.info("L2-normalizing embeddings")
        work = _l2_normalize(work)

    pca_dim = min(pca_dim, work.shape[1], work.shape[0] - 1)
    logger.info("PCA: %d -> %d", work.shape[1], pca_dim)
    pca = PCA(n_components=pca_dim, random_state=random_seed, svd_solver="randomized")
    pca_features = pca.fit_transform(work).astype(np.float32, copy=False)
    explained = float(pca.explained_variance_ratio_.sum())
    logger.info("PCA cumulative explained variance: %.4f", explained)

    logger.info(
        "t-SNE: perplexity=%.1f n_iter=%d metric=%s init=%s lr=%s",
        perplexity,
        n_iter,
        metric,
        init,
        learning_rate,
    )
    t0 = time.time()
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=n_iter,
        metric=metric,
        init=init,
        learning_rate=learning_rate,
        random_state=random_seed,
        verbose=1,
    )
    coords = tsne.fit_transform(pca_features).astype(np.float32, copy=False)
    runtime_s = time.time() - t0
    logger.info("t-SNE done in %.1fs, kl_divergence=%.4f", runtime_s, float(tsne.kl_divergence_))

    coords_path = output_dir / "tsne_2d.npy"
    pca_path = output_dir / f"tsne_pca{pca_dim}.npy"
    meta_path = output_dir / "tsne_meta.json"
    cluster_png = output_dir / "tsne_by_cluster.png"
    difficulty_png = output_dir / "tsne_by_difficulty.png"

    np.save(coords_path, coords)
    np.save(pca_path, pca_features)
    coord_df = pd.DataFrame(
        {
            "sample_id": df["sample_id"].to_numpy().astype(np.int64),
            "cluster_id": cluster_ids,
            "difficulty_score": difficulty,
            "tsne_x": coords[:, 0],
            "tsne_y": coords[:, 1],
        }
    )
    coord_df.to_parquet(output_dir / "tsne_coords.parquet", index=False)

    meta = {
        "ids_path": str(ids_path),
        "embedding_path": str(embedding_path),
        "clusters_parquet": str(clusters_parquet),
        "num_samples": int(coords.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "normalize_embeddings": bool(config.get("normalize_embeddings", True)),
        "pca_dim": int(pca_dim),
        "pca_explained_variance_ratio": explained,
        "tsne_perplexity": perplexity,
        "tsne_n_iter": n_iter,
        "tsne_metric": metric,
        "tsne_init": init,
        "tsne_learning_rate": learning_rate if isinstance(learning_rate, str) else float(learning_rate),
        "tsne_kl_divergence": float(tsne.kl_divergence_),
        "random_seed": random_seed,
        "runtime_seconds": runtime_s,
        "num_clusters": int(cluster_ids.max()) + 1,
        "outputs": {
            "tsne_2d": str(coords_path),
            "tsne_pca": str(pca_path),
            "coords_parquet": str(output_dir / "tsne_coords.parquet"),
            "by_cluster_png": str(cluster_png),
            "by_difficulty_png": str(difficulty_png),
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.info("Rendering cluster-colored figure")
    plot_by_cluster(
        coords,
        cluster_ids,
        cluster_png,
        s=scatter_size,
        alpha=scatter_alpha,
        dpi=figure_dpi,
    )
    logger.info("Rendering difficulty-colored figure")
    plot_by_difficulty(
        coords,
        difficulty,
        difficulty_png,
        s=scatter_size,
        alpha=scatter_alpha,
        dpi=figure_dpi,
    )

    logger.info("Saved 2D coords: %s", coords_path)
    logger.info("Saved PCA features: %s", pca_path)
    logger.info("Saved meta: %s", meta_path)
    logger.info("Saved figures: %s, %s", cluster_png, difficulty_png)


if __name__ == "__main__":
    main()
