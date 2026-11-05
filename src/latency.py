"""Local embedding latency via the Ollama HTTP API.

Times each local model on the English benchmark texts (queries and documents), one text per request,
after a warm-up request that loads the model. Requires a running Ollama server with the models pulled
(see pull_models.sh). Prints one JSON line per model and writes all results to the output file.

Usage (from code-data/):
    python3 src/latency.py results/latency.json
"""

import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/embed"
DATA = Path(__file__).resolve().parent.parent / "data" / "English-cleaned_data_embeddings_benchmark.json"
MODELS = [
    "all-minilm",
    "granite-embedding:278m",
    "nomic-embed-text",
    "embeddinggemma:300m",
    "mxbai-embed-large",
    "bge-m3",
    "nomic-embed-text-v2-moe",
    "qwen3-embedding:0.6b",
    "qwen3-embedding:4b",
]


def embed(model, text):
    """Return the embedding dimension for one text (the vector itself is discarded)."""
    body = json.dumps({"model": model, "input": text}).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        out = json.loads(resp.read())
    return len(out["embeddings"][0])


def main():
    parser = argparse.ArgumentParser(description="Local embedding latency via Ollama")
    parser.add_argument("output", type=Path, help="JSON file to write results to")
    args = parser.parse_args()

    rows = json.loads(DATA.read_text(encoding="utf-8"))
    texts = []
    for row in rows:
        texts.append(row["query"])
        texts += [doc["text"] for doc in row["documents"]]

    results = {}
    for model in MODELS:
        dim = embed(model, "warm-up")  # loads the model; not timed
        latencies = []
        start = time.perf_counter()
        for text in texts:
            t0 = time.perf_counter()
            embed(model, text)
            latencies.append(time.perf_counter() - t0)
        total = time.perf_counter() - start
        results[model] = {
            "dim": dim,
            "n_texts": len(texts),
            "total_s": total,
            "per_text_ms_mean": 1000 * statistics.mean(latencies),
            "per_text_ms_median": 1000 * statistics.median(latencies),
            "texts_per_s": len(texts) / total,
        }
        print(json.dumps({model: results[model]}), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
