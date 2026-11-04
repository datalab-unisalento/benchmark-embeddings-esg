import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import ndcg_score, precision_score
import json

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

parser = argparse.ArgumentParser(description="Dimension-vs-quality experiment using PCA")
parser.add_argument("language", choices=["English", "Italian"])
args = parser.parse_args()

SELECTED_MODELS = [
    "all-minilm",
    "nomic-embed-text",
    "mxbai-embed-large",
    "text-embedding-3-small",
    "qwen3-embedding-4b",
]

TARGET_DIMS = [256, 128, 64, 32, 16]

def cosine_similarity(vec1, vec2):
    v1, v2 = np.array(vec1), np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def compute_metrics(query_emb, docs_embs, documents):
    relevance_scores = [doc["relevance"] for doc in documents]
    similarities = [cosine_similarity(query_emb, d) for d in docs_embs]
    return ndcg_score([relevance_scores], [similarities])

def evaluate_df(df):
    scores = []
    for _, row in df.iterrows():
        qe = row["query_embedding"]
        de = row["documents_embeddings"]
        if qe is None or de is None:
            continue
        scores.append(compute_metrics(qe, de, row["documents"]))
    return np.mean(scores)

def safe_fn(m):
    return m.replace("models/", "").replace(":", "-").replace("/", "-")

def gather_all_embeddings(df):
    all_vecs = []
    for _, row in df.iterrows():
        qe = row["query_embedding"]
        de = row["documents_embeddings"]
        if qe is not None:
            all_vecs.append(qe)
        if de is not None:
            for d in de:
                if d is not None:
                    all_vecs.append(d)
    return np.array(all_vecs)

results = {}

for model_name in SELECTED_MODELS:
    fpath = RESULTS_DIR / f"{args.language}-{safe_fn(model_name)}.json"
    if not fpath.exists():
        print(f"Skipping {model_name}: file not found")
        continue

    df = pd.read_json(fpath)
    native_dim = len(df.iloc[0]["query_embedding"])

    ndcg_full = evaluate_df(df)
    entry = {"native_dim": native_dim, "ndcg_full": round(ndcg_full, 4)}

    all_embs = gather_all_embeddings(df)
    max_components = min(all_embs.shape[0], all_embs.shape[1])
    valid_dims = [d for d in TARGET_DIMS if d <= max_components]

    for target_dim in valid_dims:
        pca = PCA(n_components=target_dim, random_state=42)
        pca.fit(all_embs)
        transformed = pca.transform(all_embs)

        idx = 0
        ndcg_scores = []
        for _, row_data in df.iterrows():
            qe = row_data["query_embedding"]
            de = row_data["documents_embeddings"]
            if qe is None or de is None:
                continue
            q_t = transformed[idx]; idx += 1
            d_t = [transformed[idx + i] for i in range(len(de))]; idx += len(de)
            ndcg_scores.append(compute_metrics(q_t, d_t, row_data["documents"]))

        entry[f"ndcg_{target_dim}"] = round(np.mean(ndcg_scores), 4)

    results[model_name] = entry

    print(f"{model_name:30s} native={native_dim:4d}  full={entry['ndcg_full']:.4f}", end="")
    for d in TARGET_DIMS:
        key = f"ndcg_{d}"
        if key in entry:
            print(f"  {d:>3d}={entry[key]:.4f}", end="")
    print()

print("\n=== Dimension vs NDCG (print-ready) ===")
print(f"{'Model':30s} {'Full':>7s}", end="")
for d in TARGET_DIMS:
    print(f" {d:>6d}", end="")
print()
for model_name in SELECTED_MODELS:
    if model_name not in results:
        continue
    r = results[model_name]
    print(f"{model_name:30s} {r['ndcg_full']:>7.4f}", end="")
    for d in TARGET_DIMS:
        key = f"ndcg_{d}"
        if key in r:
            print(f" {r[key]:>6.4f}", end="")
        else:
            print(f" {'n/a':>6}", end="")
    print()

output = RESULTS_DIR / f"dim_experiment_{args.language.lower()}.json"
with open(output, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {output}")
