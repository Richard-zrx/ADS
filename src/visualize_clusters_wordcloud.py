from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from wordcloud import WordCloud

from io_utils import ensure_dir
from logging_utils import setup_logger


_LATEX_CMD = re.compile(r"\\[a-zA-Z]+")
_MATH_DELIMS = re.compile(r"[$\{\}\\\[\]\(\)]")
_NON_ALPHA = re.compile(r"[^a-z\s]")
_PURE_NUM = re.compile(r"\b\d+(?:\.\d+)?\b")
_SHORT_TOKEN = re.compile(r"\b[a-z]{1}\b")
_WS = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="c-TF-IDF word clouds for phase-5 clusters")
    parser.add_argument("--config", required=True)
    parser.add_argument("--clusters_parquet", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--num_top_clusters", type=int, default=None)
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML object")
    return cfg


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    updated = dict(config)
    for key in ["clusters_parquet", "output_dir"]:
        v = getattr(args, key)
        if v is not None:
            updated[key] = v
    if args.num_top_clusters is not None:
        updated["num_top_clusters"] = args.num_top_clusters
    return updated


def extract_user_text(prompt_value: Any) -> str:
    """prompt is an array of {role, content} dicts. Concatenate the user-role contents."""
    if prompt_value is None:
        return ""
    if isinstance(prompt_value, str):
        return prompt_value
    parts: list[str] = []
    for msg in prompt_value:
        if isinstance(msg, dict):
            if msg.get("role") == "user":
                parts.append(str(msg.get("content", "")))
        else:
            try:
                role = msg["role"] if "role" in msg else None
                if role == "user":
                    parts.append(str(msg["content"]))
            except (TypeError, KeyError, IndexError):
                continue
    return "\n".join(parts)


def preprocess(text: str) -> str:
    if not text:
        return ""
    s = text.lower()
    s = _PURE_NUM.sub(" ", s)
    s = _LATEX_CMD.sub(" ", s)
    s = _MATH_DELIMS.sub(" ", s)
    s = _NON_ALPHA.sub(" ", s)
    s = _SHORT_TOKEN.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s


def build_stopwords(extra: list[str]) -> frozenset[str]:
    return frozenset(ENGLISH_STOP_WORDS) | frozenset(w.lower() for w in extra)


def render_grid_wordclouds(
    cluster_term_weights: dict[int, dict[str, float]],
    chosen_clusters: list[int],
    cluster_meta: dict[int, dict[str, float]],
    out_path: Path,
    *,
    width: int,
    height: int,
    figsize: tuple[float, float],
    dpi: int,
    seed: int,
    cols: int = 3,
) -> None:
    n = len(chosen_clusters)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize, constrained_layout=True)
    axes = np.atleast_2d(axes)

    for idx, cid in enumerate(chosen_clusters):
        ax = axes[idx // cols, idx % cols]
        weights = cluster_term_weights.get(cid, {})
        if not weights:
            ax.text(0.5, 0.5, "(empty)", ha="center", va="center")
            ax.axis("off")
            continue
        wc = WordCloud(
            width=width,
            height=height,
            background_color="white",
            colormap="tab10",
            random_state=seed,
            prefer_horizontal=0.92,
            relative_scaling=0.4,
            min_font_size=8,
        ).generate_from_frequencies(weights)
        ax.imshow(wc, interpolation="bilinear")
        meta = cluster_meta.get(cid, {})
        ax.set_title(f"cluster {idx}", fontsize=11)
        ax.axis("off")

    for idx in range(n, rows * cols):
        axes[idx // cols, idx % cols].axis("off")

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def render_grid_bars(
    cluster_top_terms_ranked: dict[int, list[tuple[str, float]]],
    chosen_clusters: list[int],
    cluster_meta: dict[int, dict[str, float]],
    out_path: Path,
    *,
    top_n: int,
    figsize: tuple[float, float],
    dpi: int,
    cols: int = 3,
) -> None:
    n = len(chosen_clusters)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize, constrained_layout=True)
    axes = np.atleast_2d(axes)

    for idx, cid in enumerate(chosen_clusters):
        ax = axes[idx // cols, idx % cols]
        ranked = cluster_top_terms_ranked.get(cid, [])[:top_n]
        if not ranked:
            ax.text(0.5, 0.5, "(empty)", ha="center", va="center")
            ax.axis("off")
            continue
        terms = [t for t, _ in ranked][::-1]
        weights = [w for _, w in ranked][::-1]
        ax.barh(terms, weights, color="steelblue")
        meta = cluster_meta.get(cid, {})
        ax.set_title(
            f"cluster {cid}  (n={int(meta.get('size', 0))}, diff={meta.get('mean_difficulty', float('nan')):.2f})",
            fontsize=11,
        )
        ax.tick_params(axis="y", labelsize=9)
        ax.tick_params(axis="x", labelsize=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    for idx in range(n, rows * cols):
        axes[idx // cols, idx % cols].axis("off")

    fig.suptitle(
        f"Top-{top_n} c-TF-IDF terms per chosen cluster", fontsize=14
    )
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)

    logger = setup_logger("visualize_clusters_wordcloud", rank=0, world_size=1)

    clusters_parquet = Path(str(config["clusters_parquet"]))
    output_dir = ensure_dir(str(config["output_dir"]))
    seed = int(config.get("random_seed", 42))
    np.random.seed(seed)

    num_top_clusters = int(config.get("num_top_clusters", 9))
    chosen_clusters_override = config.get("chosen_clusters", None)
    grid_cols = int(config.get("grid_cols", 3))
    distinctiveness_top_k = int(config.get("distinctiveness_top_k", 30))
    ngram_range = tuple(config.get("ngram_range", [1, 2]))
    max_df = float(config.get("max_df", 0.85))
    min_df = int(config.get("min_df", 3))
    sublinear_tf = bool(config.get("sublinear_tf", True))
    top_terms_per_cluster = int(config.get("top_terms_per_cluster", 20))
    top_terms_in_bar = int(config.get("top_terms_in_bar", 10))
    top_terms_in_cloud = int(config.get("top_terms_in_cloud", 60))
    figure_dpi = int(config.get("figure_dpi", 200))
    wc_w = int(config.get("wordcloud_width", 600))
    wc_h = int(config.get("wordcloud_height", 400))
    figsize = tuple(config.get("grid_figsize", [13, 13]))
    extra_stopwords = list(config.get("extra_stopwords", []))

    stopwords = build_stopwords(extra_stopwords)
    logger.info("Stopword count: %d (sklearn + %d extras)", len(stopwords), len(extra_stopwords))

    logger.info("Loading parquet: %s", clusters_parquet)
    df = pd.read_parquet(
        clusters_parquet,
        columns=["sample_id", "cluster_id", "difficulty_score", "cluster_size", "prompt"],
    )
    logger.info("Rows: %d, clusters: %d", len(df), df["cluster_id"].nunique())

    logger.info("Extracting user text and preprocessing")
    t0 = time.time()
    user_texts = df["prompt"].map(extract_user_text)
    cleaned = user_texts.map(preprocess)
    logger.info("Preprocess done in %.1fs", time.time() - t0)

    logger.info("Pooling per-cluster documents (c-TF-IDF style)")
    pooled = cleaned.groupby(df["cluster_id"]).agg(lambda x: " ".join(x)).sort_index()
    cluster_ids_sorted = pooled.index.to_numpy().astype(int)

    cluster_meta: dict[int, dict[str, float]] = {}
    for cid, sub in df.groupby("cluster_id"):
        cluster_meta[int(cid)] = {
            "size": int(sub["cluster_size"].iloc[0]),
            "mean_difficulty": float(sub["difficulty_score"].mean()),
        }

    logger.info(
        "Fitting TfidfVectorizer ngram=%s max_df=%.2f min_df=%d sublinear=%s",
        ngram_range,
        max_df,
        min_df,
        sublinear_tf,
    )
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_df=max_df,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
        stop_words=list(stopwords),
        lowercase=False,
        token_pattern=r"(?u)\b[a-z][a-z]+\b",
    )
    matrix = vectorizer.fit_transform(pooled.tolist())
    vocab = np.array(vectorizer.get_feature_names_out())
    logger.info("Vocabulary size: %d", len(vocab))

    cluster_term_weights: dict[int, dict[str, float]] = {}
    cluster_top_terms_ranked: dict[int, list[tuple[str, float]]] = {}
    distinctiveness: dict[int, float] = {}

    matrix_dense = matrix.toarray()
    for row_idx, cid in enumerate(cluster_ids_sorted):
        row = matrix_dense[row_idx]
        if row.sum() == 0:
            cluster_term_weights[int(cid)] = {}
            cluster_top_terms_ranked[int(cid)] = []
            distinctiveness[int(cid)] = 0.0
            continue
        order = np.argsort(-row)
        top_for_cloud = order[:top_terms_in_cloud]
        weights = {vocab[i]: float(row[i]) for i in top_for_cloud if row[i] > 0}
        cluster_term_weights[int(cid)] = weights

        top_for_table = order[:top_terms_per_cluster]
        ranked = [(vocab[i], float(row[i])) for i in top_for_table if row[i] > 0]
        cluster_top_terms_ranked[int(cid)] = ranked

        score_top_k = order[:distinctiveness_top_k]
        distinctiveness[int(cid)] = float(np.mean(row[score_top_k]))

    ranked_clusters = sorted(distinctiveness.items(), key=lambda kv: -kv[1])
    if chosen_clusters_override is not None:
        chosen = [int(c) for c in chosen_clusters_override]
        logger.info("Chosen clusters (from config override): %s", chosen)
    else:
        chosen = [cid for cid, _ in ranked_clusters[:num_top_clusters]]
        logger.info("Chosen clusters (by distinctiveness): %s", chosen)

    csv_rows = []
    for cid in cluster_ids_sorted:
        ranked = cluster_top_terms_ranked.get(int(cid), [])
        for rank, (term, weight) in enumerate(ranked, start=1):
            csv_rows.append(
                {
                    "cluster_id": int(cid),
                    "rank": rank,
                    "term": term,
                    "weight": weight,
                    "size": cluster_meta[int(cid)]["size"],
                    "mean_difficulty": cluster_meta[int(cid)]["mean_difficulty"],
                    "distinctiveness": distinctiveness[int(cid)],
                    "selected_for_main": int(cid) in set(chosen),
                }
            )
    csv_df = pd.DataFrame(csv_rows)
    csv_path = output_dir / "cluster_top_terms.csv"
    csv_df.to_csv(csv_path, index=False)
    logger.info("Wrote top terms CSV: %s", csv_path)

    distinctiveness_path = output_dir / "cluster_distinctiveness.json"
    with open(distinctiveness_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "top_k_for_score": distinctiveness_top_k,
                "selected_clusters": chosen,
                "ranking": [
                    {
                        "cluster_id": int(cid),
                        "score": float(score),
                        "size": cluster_meta[int(cid)]["size"],
                        "mean_difficulty": cluster_meta[int(cid)]["mean_difficulty"],
                        "top_terms": [t for t, _ in cluster_top_terms_ranked.get(int(cid), [])[:10]],
                    }
                    for cid, score in ranked_clusters
                ],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info("Wrote distinctiveness JSON: %s", distinctiveness_path)

    n_chosen = len(chosen)
    cloud_png = output_dir / f"wordcloud_{n_chosen}_grid.png"
    bars_png = output_dir / f"topterms_{n_chosen}_grid.png"

    logger.info("Rendering word cloud grid")
    render_grid_wordclouds(
        cluster_term_weights,
        chosen,
        cluster_meta,
        cloud_png,
        width=wc_w,
        height=wc_h,
        figsize=figsize,
        dpi=figure_dpi,
        seed=seed,
        cols=grid_cols,
    )

    logger.info("Rendering bar-chart grid")
    render_grid_bars(
        cluster_top_terms_ranked,
        chosen,
        cluster_meta,
        bars_png,
        top_n=top_terms_in_bar,
        figsize=figsize,
        dpi=figure_dpi,
        cols=grid_cols,
    )

    meta_path = output_dir / "wordcloud_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "clusters_parquet": str(clusters_parquet),
                "num_clusters": int(len(cluster_ids_sorted)),
                "selected_clusters": chosen,
                "vocab_size": int(len(vocab)),
                "ngram_range": list(ngram_range),
                "max_df": max_df,
                "min_df": min_df,
                "sublinear_tf": sublinear_tf,
                "stopword_count": int(len(stopwords)),
                "extra_stopwords_count": int(len(extra_stopwords)),
                "distinctiveness_top_k": distinctiveness_top_k,
                "outputs": {
                    "wordcloud_grid": str(cloud_png),
                    "bars_grid": str(bars_png),
                    "top_terms_csv": str(csv_path),
                    "distinctiveness_json": str(distinctiveness_path),
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info("Wrote meta: %s", meta_path)
    logger.info("Done")


if __name__ == "__main__":
    main()
