"""Generate 5 layout variants of the word cloud grid for visual comparison."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ── reuse data cached from main script ──────────────────────────────────────
import sys, yaml, pandas as pd, time
sys.path.insert(0, str(Path(__file__).parent))
from visualize_clusters_wordcloud import (
    extract_user_text, preprocess, build_stopwords
)
from sklearn.feature_extraction.text import TfidfVectorizer

CFG_PATH = Path(__file__).parent.parent / "configs/wordcloud_phase5_1_5b.yaml"
with open(CFG_PATH) as f:
    cfg = yaml.safe_load(f)

CHOSEN = [int(c) for c in cfg["chosen_clusters"]]
OUT_DIR = Path(cfg["output_dir"]) / "compact_rows"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = int(cfg.get("random_seed", 42))
np.random.seed(SEED)

# ── load & vectorize (same as main) ─────────────────────────────────────────
df = pd.read_parquet(cfg["clusters_parquet"],
    columns=["sample_id","cluster_id","difficulty_score","cluster_size","prompt"])
cleaned = df["prompt"].map(extract_user_text).map(preprocess)
pooled  = cleaned.groupby(df["cluster_id"]).agg(lambda x: " ".join(x)).sort_index()
cluster_ids = pooled.index.to_numpy().astype(int)

stopwords = build_stopwords(cfg.get("extra_stopwords", []))
vec = TfidfVectorizer(ngram_range=(1,1), max_df=0.85, min_df=3,
    sublinear_tf=True, stop_words=list(stopwords),
    lowercase=False, token_pattern=r"(?u)\b[a-z][a-z]+\b")
mat  = vec.fit_transform(pooled.tolist()).toarray()
vocab = np.array(vec.get_feature_names_out())

weights_map: dict[int, dict[str,float]] = {}
for ri, cid in enumerate(cluster_ids):
    row = mat[ri]
    order = np.argsort(-row)[:60]
    weights_map[int(cid)] = {vocab[i]: float(row[i]) for i in order if row[i]>0}

# ── 5 variants ───────────────────────────────────────────────────────────────
VARIANTS = [
    # (label,  figsize,    title_fs, hspace, wspace, wc_w, wc_h)  base: sp2_fs34, hspace=0.25, h=24
    ("cr1", (22, 21),   34,    0.18,  0.08,   600,  400),  # 轻微缩小
    ("cr2", (22, 20),   34,    0.14,  0.08,   600,  400),
    ("cr3", (22, 19),   34,    0.10,  0.08,   600,  400),
    ("cr4", (22, 18),   34,    0.06,  0.08,   600,  400),
    ("cr5", (22, 17),   34,    0.03,  0.08,   600,  400),  # 明显缩小
]

COLS = 4

for label, figsize, title_fs, hspace, wspace, wc_w, wc_h in VARIANTS:
    n = len(CHOSEN)
    rows = (n + COLS - 1) // COLS
    use_constrained = hspace is None
    fig, axes = plt.subplots(rows, COLS, figsize=figsize,
                             constrained_layout=use_constrained)
    axes = np.atleast_2d(axes)

    if not use_constrained:
        fig.subplots_adjust(
            left=0.02, right=0.98,
            top=0.97, bottom=0.02,
            hspace=hspace, wspace=wspace,
        )

    for idx, cid in enumerate(CHOSEN):
        ax = axes[idx // COLS, idx % COLS]
        w = weights_map.get(cid, {})
        if not w:
            ax.axis("off"); continue
        wc = WordCloud(
            width=wc_w, height=wc_h,
            background_color="white", colormap="tab10",
            random_state=SEED, prefer_horizontal=0.92,
            relative_scaling=0.4, min_font_size=8,
        ).generate_from_frequencies(w)
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(f"cluster {idx}", fontsize=title_fs, pad=6, fontweight="bold")
        ax.axis("off")

    for idx in range(n, rows * COLS):
        axes[idx // COLS, idx % COLS].axis("off")

    out = OUT_DIR / f"wordcloud_16_grid_{label}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

print("Done.")
