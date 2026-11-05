"""Bootstrap confidence intervals and paired comparisons of NDCG across models.

Reads the per-model embedding files written by genvec.py (results/<Language>-<model>.json),
computes per-query NDCG for each model, and reports:
  - mean NDCG with a 95% bootstrap CI (queries resampled with replacement)
  - for the top model, paired differences against every other model with a 95% bootstrap CI
    and a two-sided bootstrap p-value

Usage (from code-data/):
    python3 src/significance.py results English
    python3 src/significance.py results Italian
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

MODELS = [
    "all-minilm",
    "bge-m3",
    "embeddinggemma-300m",
    "granite-embedding-278m",
    "mxbai-embed-large",
    "nomic-embed-text",
    "nomic-embed-text-v2-moe",
    "qwen3-embedding-0.6b",
    "qwen3-embedding-4b",
    "text-embedding-3-large",
    "text-embedding-3-small",
    "text-embedding-ada-002",
]

N_BOOT = 10_000
SEED = 0


def cosine(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def ndcg_one_query(relevance, similarities):
    """NDCG for one query, using graded relevance as gains (same as sklearn.ndcg_score here)."""
    order = sorted(range(len(similarities)), key=lambda i: -similarities[i])
    dcg = sum(relevance[i] / math.log2(rank + 2) for rank, i in enumerate(order))
    ideal = sorted(relevance, reverse=True)
    idcg = sum(g / math.log2(rank + 2) for rank, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def per_query_ndcg(path):
    """Return {row_index: ndcg}; rows with missing embeddings are skipped."""
    rows = json.loads(path.read_text(encoding="utf-8"))
    scores = {}
    for i, row in enumerate(rows):
        q_emb, d_embs = row["query_embedding"], row["documents_embeddings"]
        if q_emb is None or d_embs is None:
            continue
        relevance = [doc["relevance"] for doc in row["documents"]]
        sims = [cosine(q_emb, d) for d in d_embs]
        scores[i] = ndcg_one_query(relevance, sims)
    return scores


def main():
    parser = argparse.ArgumentParser(description="Bootstrap NDCG comparisons between models")
    parser.add_argument("results_dir", type=Path, help="directory with <Language>-<model>.json files")
    parser.add_argument("language", choices=["English", "Italian"])
    args = parser.parse_args()

    per_model = {}
    for model in MODELS:
        path = args.results_dir / f"{args.language}-{model}.json"
        if not path.exists():
            raise FileNotFoundError(f"missing {path}")
        per_model[model] = per_query_ndcg(path)

    # Compare on queries that are valid for every model.
    common = sorted(set.intersection(*[set(s) for s in per_model.values()]))
    scores = {m: np.array([per_model[m][q] for q in common]) for m in MODELS}

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(common), (N_BOOT, len(common)))
    boots = {m: scores[m][idx].mean(axis=1) for m in MODELS}

    order = sorted(MODELS, key=lambda m: -scores[m].mean())
    top = order[0]

    print(f"{args.language}: {len(common)} queries, {N_BOOT} bootstrap resamples, seed {SEED}")
    print(f"{'model':28s} {'NDCG':>7s}  {'95% CI':>17s}")
    for m in order:
        lo, hi = np.percentile(boots[m], [2.5, 97.5])
        print(f"{m:28s} {scores[m].mean():7.4f}  [{lo:.4f}, {hi:.4f}]")

    print(f"\nPaired difference: {top} minus each model")
    for m in order[1:]:
        diff = scores[top] - scores[m]
        bd = boots[top] - boots[m]
        lo, hi = np.percentile(bd, [2.5, 97.5])
        p = 2 * min((bd <= 0).mean(), (bd >= 0).mean())
        flag = "significant" if (lo > 0 or hi < 0) else "n.s."
        print(f"{m:28s} diff={diff.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] p={p:.4f} {flag}")


if __name__ == "__main__":
    main()
