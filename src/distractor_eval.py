"""Harder candidate pool: rank each query against its 3 candidates plus N extra negatives.

The extra negatives are documents of other queries in the same ESG category, labelled relevance 0.
They are sampled from the dataset alone (same seed, same pool for every model), so models are compared
on identical candidate sets. Vectors are reused from the per-model embedding files written by
genvec.py, so no new embedding calls are needed.

Reports, per model, averaged over queries:
  - NDCG@10 on the 3-candidate baseline and on the enlarged pool
  - MRR on the enlarged pool (first document with relevance > 1)
  - Precision@1 on the enlarged pool (top-ranked document has relevance > 1)
with 95% bootstrap CIs (queries resampled) for NDCG@10 and MRR, and writes everything to JSON.

Caveat: a sampled negative can still be relevant to the query by chance. Sampling within the same
category reduces this; the output reports the pool size so the effect can be discussed.

Usage (from code-data/, after the per-model files exist in results/):
    python3 src/distractor_eval.py results English --negatives 17
    python3 src/distractor_eval.py results Italian --negatives 17

The per-model files are committed as <Language>-<model>.json.gz and read directly.
"""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from significance import MODELS  # the 12 models reported in the paper

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
N_BOOT = 10_000
SEED = 0


def normalize_domain(label):
    """The raw labels contain case variants and one misspelling; map them to three categories."""
    label = label.strip().lower()
    if label.startswith("enviro"):
        return "environmental"
    return label


def cosine_to_each(query, docs):
    q = np.asarray(query, dtype=float)
    d = np.asarray(docs, dtype=float)
    return (d @ q) / (np.linalg.norm(d, axis=1) * np.linalg.norm(q))


def ndcg_at_k(relevance, sims, k=10):
    relevance = np.asarray(relevance, dtype=float)
    order = np.argsort(-np.asarray(sims))[:k]
    discounts = 1.0 / np.log2(np.arange(2, len(order) + 2))
    dcg = float(np.sum(relevance[order] * discounts))
    ideal = np.sort(relevance)[::-1][:k]
    idcg = float(np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2))))
    return dcg / idcg if idcg > 0 else 0.0


def mrr_and_p1(relevance, sims):
    binary = np.asarray(relevance) > 1
    ranked = binary[np.argsort(-np.asarray(sims))]
    p1 = float(ranked[0])
    hits = np.flatnonzero(ranked)
    mrr = 1.0 / (hits[0] + 1) if hits.size else 0.0
    return mrr, p1


def build_negatives(rows, n_negatives, seed):
    """For each query, sample N (row, doc) pairs from other queries in the same category."""
    rng = np.random.default_rng(seed)
    domains = [normalize_domain(r["domain"]) for r in rows]
    pool = {}
    for i, dom in enumerate(domains):
        pool.setdefault(dom, []).extend((i, j) for j in range(len(rows[i]["documents"])))
    negatives = []
    for i, dom in enumerate(domains):
        candidates = [pair for pair in pool[dom] if pair[0] != i]
        if len(candidates) < n_negatives:
            raise ValueError(f"category {dom!r} has {len(candidates)} candidate documents, fewer than {n_negatives}")
        pick = rng.choice(len(candidates), size=n_negatives, replace=False)
        negatives.append([candidates[k] for k in pick])
    return negatives


def evaluate_model(model_rows, negatives):
    """Per-query (ndcg_baseline, ndcg_pool, mrr_pool, p1_pool) for one model.

    Queries with missing embeddings, or with a negative whose document embedding is missing, are skipped.
    """
    per_query = []
    for i, row in enumerate(model_rows):
        q_emb, d_embs = row["query_embedding"], row["documents_embeddings"]
        if q_emb is None or d_embs is None:
            continue
        rel = [doc["relevance"] for doc in row["documents"]]
        base_sims = cosine_to_each(q_emb, d_embs)
        ndcg_base = ndcg_at_k(rel, base_sims, k=10)

        neg_vecs = []
        for (r, j) in negatives[i]:
            other = model_rows[r]["documents_embeddings"]
            if other is None or other[j] is None:
                neg_vecs = None
                break
            neg_vecs.append(other[j])
        if neg_vecs is None:
            continue
        neg_sims = cosine_to_each(q_emb, neg_vecs) if neg_vecs else np.empty(0)
        pool_rel = list(rel) + [0] * len(neg_vecs)
        pool_sims = np.concatenate([base_sims, neg_sims])
        mrr, p1 = mrr_and_p1(pool_rel, pool_sims)
        per_query.append((ndcg_base, ndcg_at_k(pool_rel, pool_sims, k=10), mrr, p1))
    return np.asarray(per_query)


def bootstrap_ci(values, rng):
    idx = rng.integers(0, len(values), (N_BOOT, len(values)))
    lo, hi = np.percentile(values[idx].mean(axis=1), [2.5, 97.5])
    return float(lo), float(hi)


def load_model_rows(results_dir, language, model):
    """Read a per-model file; the committed copies are gzipped (.json.gz), plain .json also works."""
    gz = results_dir / f"{language}-{model}.json.gz"
    plain = results_dir / f"{language}-{model}.json"
    if gz.exists():
        with gzip.open(gz, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    if plain.exists():
        return json.loads(plain.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"missing {gz} (or {plain})")


def main():
    parser = argparse.ArgumentParser(description="Evaluate embedding models on a harder candidate pool")
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("language", choices=["English", "Italian"])
    parser.add_argument("--negatives", type=int, default=17, help="extra negatives per query")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, default=None,
                        help="output JSON (default: <results_dir>/distractor_<Language>.json)")
    args = parser.parse_args()

    data_path = DATA_DIR / f"{args.language}-cleaned_data_embeddings_benchmark.json"
    rows_data = json.loads(data_path.read_text(encoding="utf-8"))
    negatives = build_negatives(rows_data, args.negatives, args.seed)
    rng = np.random.default_rng(args.seed)

    summary = {}
    for model in MODELS:
        model_rows = load_model_rows(args.results_dir, args.language, model)
        per_query = evaluate_model(model_rows, negatives)
        ndcg_base, ndcg_pool, mrr, p1 = per_query.T
        lo, hi = bootstrap_ci(ndcg_pool, rng)
        mlo, mhi = bootstrap_ci(mrr, rng)
        summary[model] = {
            "queries": int(per_query.shape[0]),
            "pool_size": 3 + args.negatives,
            "ndcg10_baseline": float(ndcg_base.mean()),
            "ndcg10_pool": float(ndcg_pool.mean()),
            "ndcg10_pool_ci": [lo, hi],
            "mrr_pool": float(mrr.mean()),
            "mrr_pool_ci": [mlo, mhi],
            "p1_pool": float(p1.mean()),
        }
        print(f"{model:28s} NDCG@10 base={ndcg_base.mean():.4f} pool={ndcg_pool.mean():.4f} "
              f"[{lo:.4f}, {hi:.4f}]  MRR={mrr.mean():.4f}  P@1={p1.mean():.4f}", flush=True)

    out = args.out or (args.results_dir / f"distractor_{args.language}.json")
    out.write_text(json.dumps({"language": args.language, "negatives": args.negatives, "seed": args.seed,
                               "models": summary}, indent=2), encoding="utf-8")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
