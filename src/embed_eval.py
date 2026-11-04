import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score, precision_score

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

parser = argparse.ArgumentParser(description="Evaluate embedding models for ESG benchmark")
parser.add_argument("language", choices=["English", "Italian"], help="Language of the dataset")
args = parser.parse_args()

###############################################################

def cosine_similarity(vec1, vec2):
    v1, v2 = np.array(vec1), np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def recall_at_k(binary_relevance, similarities, k=None):
    k = k or len(similarities)
    total_relevant = sum(binary_relevance)
    if total_relevant == 0:
        return 0.0
    ranked = np.argsort(similarities)[::-1][:k]
    return sum(binary_relevance[i] for i in ranked) / total_relevant

def average_precision_at_k(binary_relevance, similarities, k=None):
    k = k or len(similarities)
    total_relevant = sum(binary_relevance)
    if total_relevant == 0:
        return 0.0
    ranked = np.argsort(similarities)[::-1][:k]
    hits, sum_prec = 0, 0.0
    for rank, idx in enumerate(ranked, 1):
        if binary_relevance[idx]:
            hits += 1
            sum_prec += hits / rank
    return sum_prec / total_relevant

def reciprocal_rank(binary_relevance, similarities, k=None):
    k = k or len(similarities)
    ranked = np.argsort(similarities)[::-1][:k]
    for rank, idx in enumerate(ranked, 1):
        if binary_relevance[idx]:
            return 1.0 / rank
    return 0.0

###############################################################

# Discover per-model result files
exclude = {"vectors", "metrics", "average"}
model_files = sorted([
    f for f in RESULTS_DIR.glob(f"{args.language}-*.json")
    if not any(x in f.stem for x in exclude)
])

if not model_files:
    raise FileNotFoundError(
        f"No model result files found in {RESULTS_DIR} for language '{args.language}'. "
        "Run genvec.py first."
    )

print(f"Found {len(model_files)} model files:")
for f in model_files:
    print(f"  {f.name}")

###############################################################

all_metrics = []

for model_file in model_files:
    model_name = model_file.stem[len(args.language) + 1:]
    df = pd.read_json(model_file)

    ndcg_scores, prec_scores, recall_scores, map_scores, mrr_scores = [], [], [], [], []

    for i, row in df.iterrows():
        try:
            query_embedding = row["query_embedding"]
            docs_embeddings = row["documents_embeddings"]
            if query_embedding is None or docs_embeddings is None:
                continue

            documents = row["documents"]
            relevance_scores = [doc["relevance"] for doc in documents]
            binary_relevance = [1 if s > 1 else 0 for s in relevance_scores]
            similarities = [cosine_similarity(query_embedding, d) for d in docs_embeddings]

            ndcg      = ndcg_score([relevance_scores], [similarities])
            precision = precision_score(
                binary_relevance,
                [1 if s > 0.5 else 0 for s in similarities],
                zero_division=0,
            )
            recall    = recall_at_k(binary_relevance, similarities)
            ap        = average_precision_at_k(binary_relevance, similarities)
            rr        = reciprocal_rank(binary_relevance, similarities)

            ndcg_scores.append(ndcg)
            prec_scores.append(precision)
            recall_scores.append(recall)
            map_scores.append(ap)
            mrr_scores.append(rr)
        except Exception as e:
            print(f"Error in {model_name} row {i}: {e}")

    avg = lambda lst: np.mean(lst) if lst else None
    avg_ndcg = avg(ndcg_scores)
    avg_prec = avg(prec_scores)
    avg_rec  = avg(recall_scores)
    avg_map  = avg(map_scores)
    avg_mrr  = avg(mrr_scores)

    all_metrics.append({
        "model_name": model_name,
        "ndcg":      avg_ndcg,
        "precision": avg_prec,
        "recall":    avg_rec,
        "map":       avg_map,
        "mrr":       avg_mrr,
    })

    if avg_ndcg is not None:
        print(f"{model_name}: NDCG={avg_ndcg:.4f}  Precision={avg_prec:.4f}  "
              f"Recall={avg_rec:.4f}  MAP={avg_map:.4f}  MRR={avg_mrr:.4f}")
    else:
        print(f"{model_name}: no valid rows — check embeddings file")

###############################################################

average_metrics_df = pd.DataFrame(all_metrics).set_index("model_name")
print("\n", average_metrics_df)

lang_code = "eng" if args.language == "English" else "ita"
RESULTS_DIR.mkdir(exist_ok=True)
output_metrics = RESULTS_DIR / f"average_metrics_df_{lang_code}.json"
average_metrics_df.to_json(output_metrics, indent=2)
print(f"Saved: {output_metrics}")
print("done")
