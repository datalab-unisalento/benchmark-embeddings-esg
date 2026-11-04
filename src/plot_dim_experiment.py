import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR.parent / "figures"

TARGET_DIMS = [256, 128, 64, 32, 16]
MODEL_STYLES = {
    "all-minilm":             {"color": "#e41a1c", "marker": "o", "ls": "-"},
    "nomic-embed-text":       {"color": "#377eb8", "marker": "s", "ls": "-"},
    "mxbai-embed-large":      {"color": "#4daf4a", "marker": "D", "ls": "-"},
    "text-embedding-3-small": {"color": "#984ea3", "marker": "^", "ls": "-"},
    "qwen3-embedding-4b":     {"color": "#ff7f00", "marker": "v", "ls": "-"},
}

FIGURES_DIR.mkdir(exist_ok=True)

def load_results(language):
    path = RESULTS_DIR / f"dim_experiment_{language.lower()}.json"
    with open(path) as f:
        return json.load(f)

def make_figure(results, language):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    all_dims = []
    for model_name, data in results.items():
        full_dim = data["native_dim"]
        ndcg_full = data["ndcg_full"]
        style = MODEL_STYLES.get(model_name, {"color": "gray", "marker": "x", "ls": "-"})

        reduced_dims = [d for d in TARGET_DIMS if f"ndcg_{d}" in data]
        reduced_ndcgs = [data[f"ndcg_{d}"] for d in reduced_dims]

        xs = reduced_dims + [full_dim]
        ys = reduced_ndcgs + [ndcg_full]
        order = np.argsort(xs)
        xs = np.array(xs)[order]
        ys = np.array(ys)[order]

        all_dims.extend(xs)

        # horizontal dashed line at full-dim NDCG
        ax.axhline(y=ndcg_full, color=style["color"], linestyle=":", alpha=0.3)

        ax.plot(xs, ys, label=model_name, color=style["color"],
                marker=style["marker"], linestyle=style["ls"], linewidth=1.5,
                markersize=6)

    ax.set_xscale("log")
    ax.set_xticks([16, 32, 64, 128, 256, 512, 1024, 2560])
    ax.set_xticklabels(["16", "32", "64", "128", "256", "512", "1024", "2560"])
    ax.set_xlabel("Embedding Dimension", fontsize=12)
    ax.set_ylabel("NDCG", fontsize=12)
    ax.set_title(f"{language} — NDCG vs. Embedding Dimension", fontsize=13, fontweight="bold")
    ax.set_xlim(10, 4000)
    ax.set_ylim(0.89, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower left")
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    return fig

for lang in ["English", "Italian"]:
    results = load_results(lang)
    fig = make_figure(results, lang)
    outpath = FIGURES_DIR / f"dim_vs_ndcg_{lang.lower()}.pdf"
    fig.savefig(outpath, dpi=300)
    fig.savefig(outpath.with_suffix(".png"), dpi=300)
    print(f"Saved: {outpath}")
    plt.close(fig)
