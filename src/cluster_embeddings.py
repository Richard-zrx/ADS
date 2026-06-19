from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

try:
    from sklearn.cluster import BisectingKMeans, DBSCAN, KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    from sklearn.mixture import GaussianMixture
    from sklearn.neighbors import NearestNeighbors
    _SKLEARN_IMPORT_ERROR: Exception | None = None
except ImportError as exc:  # pragma: no cover
    BisectingKMeans = None  # type: ignore[assignment]
    DBSCAN = None  # type: ignore[assignment]
    GaussianMixture = None  # type: ignore[assignment]
    KMeans = None  # type: ignore[assignment]
    NearestNeighbors = None  # type: ignore[assignment]
    PCA = None  # type: ignore[assignment]
    silhouette_score = None  # type: ignore[assignment]
    _SKLEARN_IMPORT_ERROR = exc

from data_utils import build_id_to_text_map
from io_utils import ensure_dir, load_npy
from logging_utils import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run clustering on merged embeddings")
    parser.add_argument("--config", required=True, help="Path to clustering YAML config")
    parser.add_argument("--ids_path", default=None)
    parser.add_argument("--embedding_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--embedding_name", default=None)
    parser.add_argument("--clustering_method", default=None)
    parser.add_argument("--num_clusters", type=int, default=None)
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config file must contain a YAML object")
    return cfg


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    updated = dict(config)
    for key in ["ids_path", "embedding_path", "output_dir", "embedding_name", "clustering_method"]:
        value = getattr(args, key)
        if value is not None:
            updated[key] = value
    if args.num_clusters is not None:
        updated["num_clusters"] = args.num_clusters
    return updated


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _require_sklearn() -> None:
    if _SKLEARN_IMPORT_ERROR is not None:
        raise ImportError(
            "scikit-learn is required for clustering. Please install `scikit-learn` in your Python environment."
        ) from _SKLEARN_IMPORT_ERROR


def _l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, ord=2, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return embeddings / norms


def _safe_json_or_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return str(value)


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
        resolution = float(config.get("leiden_resolution", 1.0))
        return f"leiden_k{knn_k}_r{_sanitize_float_tag(resolution)}"
    return method


def _silhouette_sampled(
    features: np.ndarray,
    labels: np.ndarray,
    method: str,
    random_seed: int,
    max_samples: int = 5000,
) -> float | None:
    if silhouette_score is None:  # pragma: no cover
        return None

    if method in {"dbscan", "hdbscan"}:
        valid = labels != -1
        if int(valid.sum()) < 2:
            return None
        work_x = features[valid]
        work_y = labels[valid]
    else:
        work_x = features
        work_y = labels

    unique_labels = np.unique(work_y)
    if unique_labels.shape[0] < 2:
        return None

    n = work_x.shape[0]
    if n > max_samples:
        rng = np.random.default_rng(random_seed)
        sel = rng.choice(n, size=max_samples, replace=False)
        work_x = work_x[sel]
        work_y = work_y[sel]

    return float(silhouette_score(work_x, work_y, metric="euclidean"))


def _ensure_dependencies(method: str) -> None:
    _require_sklearn()
    if method != "knn_leiden":
        return
    try:
        import igraph  # noqa: F401
        import leidenalg  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "igraph and leidenalg are required for clustering_method=knn_leiden."
        ) from exc


def _summarize_cluster_sizes(cluster_size_map: dict[int, int]) -> tuple[list[int], float]:
    non_noise_sizes = [size for label, size in cluster_size_map.items() if label != -1]
    if not non_noise_sizes:
        return [], 0.0
    top_10 = sorted(non_noise_sizes, reverse=True)[:10]
    singleton_ratio = float(sum(1 for size in non_noise_sizes if size == 1) / len(non_noise_sizes))
    return top_10, singleton_ratio


def _update_comparison_summary(output_dir: Path, row: dict[str, Any]) -> None:
    json_path = output_dir / "clustering_comparison_summary.json"
    csv_path = output_dir / "clustering_comparison_summary.csv"

    existing_rows: list[dict[str, Any]] = []
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, list):
            existing_rows = [item for item in loaded if isinstance(item, dict)]

    updated_rows = [
        item
        for item in existing_rows
        if not (
            item.get("embedding_name") == row.get("embedding_name")
            and item.get("method") == row.get("method")
            and item.get("method_tag") == row.get("method_tag")
        )
    ]
    updated_rows.append(row)
    updated_rows.sort(key=lambda item: (str(item.get("embedding_name")), str(item.get("method_tag"))))

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(updated_rows, f, ensure_ascii=False, indent=2)

    pd.DataFrame(updated_rows).to_csv(csv_path, index=False)


def _run_knn_leiden(features: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    import igraph as ig  # type: ignore
    import leidenalg  # type: ignore

    knn_k = int(config.get("knn_k", 30))
    if knn_k < 2:
        raise ValueError("knn_k must be >= 2 for knn_leiden")

    resolution = float(config.get("leiden_resolution", 1.0))
    iterations = int(config.get("leiden_iterations", -1))

    neighbors = NearestNeighbors(n_neighbors=knn_k, metric="euclidean")
    neighbors.fit(features)
    distances, indices = neighbors.kneighbors(features, return_distance=True)

    edges: set[tuple[int, int]] = set()
    weights: list[float] = []
    edge_pairs: list[tuple[int, int]] = []

    for src in range(features.shape[0]):
        for pos in range(1, indices.shape[1]):
            dst = int(indices[src, pos])
            if src == dst:
                continue
            edge = (src, dst) if src < dst else (dst, src)
            if edge in edges:
                continue
            edges.add(edge)
            edge_pairs.append(edge)
            weight = float(np.dot(features[edge[0]], features[edge[1]]))
            weights.append(max(0.0, weight))

    graph = ig.Graph(n=features.shape[0], edges=edge_pairs, directed=False)
    graph.es["weight"] = weights

    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights=graph.es["weight"],
        resolution_parameter=resolution,
        n_iterations=iterations,
    )
    return np.asarray(partition.membership, dtype=np.int32)


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)

    logger = setup_logger("cluster_embeddings", rank=0, world_size=1)

    ids_path = Path(str(config["ids_path"]))
    embedding_path = Path(str(config["embedding_path"]))
    output_dir = ensure_dir(str(config["output_dir"]))
    embedding_name = str(config.get("embedding_name", "mean"))

    method = str(config.get("clustering_method", "kmeans")).strip().lower()
    supported_methods = {
        "kmeans",
        "dbscan",
        "hdbscan",
        "spherical_kmeans",
        "knn_leiden",
        "gmm",
        "bisecting_kmeans",
    }
    if method not in supported_methods:
        raise ValueError(f"Unsupported clustering_method: {method}")

    _ensure_dependencies(method)

    random_seed = int(config.get("random_seed", 42))
    np.random.seed(random_seed)

    logger.info("Loading merged arrays")
    ids = load_npy(ids_path)
    embeddings = load_npy(embedding_path)

    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be 2D, got shape={embeddings.shape}")
    if len(ids) != embeddings.shape[0]:
        raise ValueError(f"Length mismatch: len(ids)={len(ids)} embeddings={embeddings.shape[0]}")
    if np.isnan(embeddings).any() or np.isinf(embeddings).any():
        raise ValueError("Embeddings contain NaN or Inf")

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

    missing: list[Any] = [sample_id for sample_id in ids.tolist() if sample_id not in id_to_record]
    if missing:
        raise ValueError(f"Found {len(missing)} ids that cannot map back to raw dataset. first_missing={missing[:20]}")

    normalize_embeddings = _parse_bool(config.get("normalize_embeddings", True))
    post_pca_normalize = _parse_bool(config.get("post_pca_normalize", False))
    if method == "spherical_kmeans":
        if not normalize_embeddings:
            raise ValueError("spherical_kmeans requires normalize_embeddings=true")
        post_pca_normalize = True

    work_embeddings = embeddings.astype(np.float32, copy=False)
    if normalize_embeddings:
        logger.info("Applying L2 normalization")
        work_embeddings = _l2_normalize(work_embeddings)

    requested_pca_dim = int(config.get("pca_dim", 128))
    pca_dim = min(requested_pca_dim, work_embeddings.shape[1], work_embeddings.shape[0] - 1)
    if pca_dim < 2:
        raise ValueError(f"Invalid PCA dimension after clamp: {pca_dim}")

    logger.info("Running PCA: requested=%d actual=%d", requested_pca_dim, pca_dim)
    pca = PCA(n_components=pca_dim, random_state=random_seed, svd_solver="randomized")
    pca_embeddings = pca.fit_transform(work_embeddings).astype(np.float32, copy=False)
    cluster_features = pca_embeddings
    if post_pca_normalize:
        logger.info("Applying post-PCA L2 normalization")
        cluster_features = _l2_normalize(cluster_features)

    num_clusters = int(config.get("num_clusters")) if config.get("num_clusters") is not None else None
    method_tag = _method_tag(config=config, method=method, num_clusters=num_clusters)

    logger.info("Running clustering: method=%s tag=%s", method, method_tag)
    inertia: float | None = None
    distance_to_centroid = np.full((cluster_features.shape[0],), np.nan, dtype=np.float32)
    cluster_confidence = np.full((cluster_features.shape[0],), np.nan, dtype=np.float32)
    cluster_probs: np.ndarray | None = None

    if method == "kmeans":
        if num_clusters is None:
            raise ValueError("num_clusters is required when clustering_method=kmeans")
        clusterer = KMeans(n_clusters=num_clusters, random_state=random_seed, n_init="auto")
        labels = clusterer.fit_predict(cluster_features)
        centers = clusterer.cluster_centers_
        distance_to_centroid = np.linalg.norm(cluster_features - centers[labels], axis=1).astype(np.float32)
        inertia = float(clusterer.inertia_)
    elif method == "spherical_kmeans":
        if num_clusters is None:
            raise ValueError("num_clusters is required when clustering_method=spherical_kmeans")
        clusterer = KMeans(n_clusters=num_clusters, random_state=random_seed, n_init="auto")
        labels = clusterer.fit_predict(cluster_features)
        centers = _l2_normalize(clusterer.cluster_centers_.astype(np.float32, copy=False))
        distance_to_centroid = np.linalg.norm(cluster_features - centers[labels], axis=1).astype(np.float32)
        inertia = float(clusterer.inertia_)
    elif method == "bisecting_kmeans":
        if num_clusters is None:
            raise ValueError("num_clusters is required when clustering_method=bisecting_kmeans")
        clusterer = BisectingKMeans(
            n_clusters=num_clusters,
            random_state=random_seed,
            n_init=int(config.get("n_init", 10)),
            bisecting_strategy=str(config.get("bisecting_strategy", "biggest_inertia")),
            init=str(config.get("init", "k-means++")),
        )
        labels = clusterer.fit_predict(cluster_features)
        centers = clusterer.cluster_centers_
        distance_to_centroid = np.linalg.norm(cluster_features - centers[labels], axis=1).astype(np.float32)
        inertia = float(clusterer.inertia_)
    elif method == "gmm":
        if num_clusters is None:
            raise ValueError("num_clusters is required when clustering_method=gmm")
        clusterer = GaussianMixture(
            n_components=num_clusters,
            covariance_type=str(config.get("gmm_covariance_type", "diag")),
            reg_covar=float(config.get("gmm_reg_covar", 1e-6)),
            max_iter=int(config.get("gmm_max_iter", 200)),
            n_init=int(config.get("gmm_n_init", 3)),
            random_state=random_seed,
        )
        clusterer.fit(cluster_features)
        cluster_probs = clusterer.predict_proba(cluster_features).astype(np.float32, copy=False)
        labels = cluster_probs.argmax(axis=1)
        cluster_confidence = cluster_probs.max(axis=1).astype(np.float32, copy=False)
    elif method == "dbscan":
        eps = float(config.get("dbscan_eps", 0.5))
        min_samples = int(config.get("dbscan_min_samples", 5))
        clusterer = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = clusterer.fit_predict(cluster_features)
    elif method == "hdbscan":
        try:
            import hdbscan  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError("hdbscan is not installed. Install hdbscan to use clustering_method=hdbscan") from exc
        min_cluster_size = int(config.get("hdbscan_min_cluster_size", 30))
        min_samples = config.get("hdbscan_min_samples")
        min_samples = int(min_samples) if min_samples is not None else None
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
        labels = clusterer.fit_predict(cluster_features)
    else:
        labels = _run_knn_leiden(cluster_features, config)

    labels = labels.astype(np.int32)

    unique_labels, label_counts = np.unique(labels, return_counts=True)
    cluster_size_map = {int(k): int(v) for k, v in zip(unique_labels.tolist(), label_counts.tolist())}
    non_noise_counts = [count for label, count in cluster_size_map.items() if label != -1]
    top_10_cluster_sizes, singleton_ratio = _summarize_cluster_sizes(cluster_size_map)

    silhouette = _silhouette_sampled(
        features=cluster_features,
        labels=labels,
        method=method,
        random_seed=random_seed,
        max_samples=int(config.get("silhouette_max_samples", 5000)),
    )

    sample_records = [id_to_record[sample_id] for sample_id in ids.tolist()]
    input_texts = [record.get("input_text", "") for record in sample_records]
    source_splits = [record.get("source_split") for record in sample_records]
    targets = [_safe_json_or_str(record.get("target")) for record in sample_records]

    assignments = pd.DataFrame(
        {
            "sample_id": ids,
            "cluster_id": labels,
            "pca_2d_x": pca_embeddings[:, 0].astype(np.float32),
            "pca_2d_y": pca_embeddings[:, 1].astype(np.float32),
            "distance_to_centroid": distance_to_centroid,
            "cluster_confidence": cluster_confidence,
            "embedding_name": embedding_name,
            "clustering_method": method,
            "input_text": input_texts,
            "source_split": source_splits,
            "target": targets,
        }
    )

    assignments_path = output_dir / f"cluster_assignments_{embedding_name}_{method_tag}.parquet"
    pca_path = output_dir / f"pca_{embedding_name}_{pca_dim}.npy"
    stats_path = output_dir / f"cluster_stats_{embedding_name}_{method_tag}.json"
    examples_path = output_dir / f"cluster_examples_{embedding_name}_{method_tag}.jsonl"

    assignments.to_parquet(assignments_path, index=False)
    np.save(pca_path, pca_embeddings.astype(np.float32, copy=False))
    if cluster_probs is not None and _parse_bool(config.get("export_cluster_probs", False)):
        probs_path = output_dir / f"cluster_probs_{embedding_name}_{method_tag}.npy"
        np.save(probs_path, cluster_probs.astype(np.float32, copy=False))

    stats: dict[str, Any] = {
        "embedding_name": embedding_name,
        "clustering_method": method,
        "method_tag": method_tag,
        "num_samples": int(len(ids)),
        "embedding_dim": int(embeddings.shape[1]),
        "pca_dim_requested": requested_pca_dim,
        "pca_dim_actual": int(pca_dim),
        "normalize_embeddings": normalize_embeddings,
        "post_pca_normalize": post_pca_normalize,
        "num_clusters_configured": int(num_clusters) if num_clusters is not None else None,
        "num_clusters_found": int(len(non_noise_counts)),
        "cluster_size_min": int(min(non_noise_counts)) if non_noise_counts else None,
        "cluster_size_max": int(max(non_noise_counts)) if non_noise_counts else None,
        "cluster_size_mean": float(np.mean(non_noise_counts)) if non_noise_counts else None,
        "cluster_size_std": float(np.std(non_noise_counts)) if non_noise_counts else None,
        "top_10_cluster_sizes": top_10_cluster_sizes,
        "singleton_ratio": singleton_ratio,
        "noise_ratio": float(cluster_size_map.get(-1, 0) / len(ids)),
        "inertia": inertia,
        "silhouette_sampled": silhouette,
        "cluster_size_map": cluster_size_map,
        "assignments_path": str(assignments_path),
        "pca_path": str(pca_path),
        "examples_path": str(examples_path),
    }
    if method == "gmm":
        stats["avg_max_posterior"] = float(np.mean(cluster_confidence))
        stats["bic"] = float(clusterer.bic(cluster_features))
        stats["aic"] = float(clusterer.aic(cluster_features))
    if method == "spherical_kmeans":
        stats["effective_similarity_metric"] = "cosine_via_l2_kmeans"

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    summary_row = {
        "embedding_name": embedding_name,
        "method": method,
        "method_tag": method_tag,
        "num_clusters_found": stats["num_clusters_found"],
        "singleton_ratio": singleton_ratio,
        "top_10_cluster_sizes": json.dumps(top_10_cluster_sizes),
        "cluster_size_min": stats["cluster_size_min"],
        "cluster_size_max": stats["cluster_size_max"],
        "cluster_size_mean": stats["cluster_size_mean"],
        "cluster_size_std": stats["cluster_size_std"],
        "noise_ratio": stats["noise_ratio"],
        "silhouette_sampled": stats["silhouette_sampled"],
        "examples_path": str(examples_path),
    }
    _update_comparison_summary(output_dir, summary_row)

    logger.info("Saved assignments: %s", assignments_path)
    logger.info("Saved pca output: %s", pca_path)
    logger.info("Saved cluster stats: %s", stats_path)


if __name__ == "__main__":
    main()
