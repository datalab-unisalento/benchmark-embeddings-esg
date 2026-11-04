# Benchmarking Embedding Models for ESG Data

Code and data for benchmarking text embedding models on Environmental, Social, and Governance (ESG) content in **English and Italian**, for information retrieval and Retrieval-Augmented Generation (RAG).

The benchmark accompanies the paper *Benchmarking Embedding Models for ESG Data* (Saad, Cretì, Gentile, Kazemi, Longo; ITADATA 2026, CEUR-WS). If you use the dataset or code, please cite it (see [Citation](#citation)).

## Contents

- `data/`: the benchmark dataset (one JSON file per language)
- `src/`: embedding generation, evaluation, and dimensionality experiment scripts
- `results/`: aggregate metrics and the dimension-vs-quality experiment output
- `old-results/`: earlier aggregate metrics, kept for reference

## Dataset

| | English | Italian |
|---|---|---|
| File | `data/English-cleaned_data_embeddings_benchmark.json` | `data/Italian-cleaned_data_embeddings_benchmark.json` |
| Records | 312 | 312 |

- **Creation**: queries and candidate documents were generated with GPT-4.1 using structured prompts, balanced across the three ESG pillars (Environmental, Social, Governance). Every record was then reviewed by an ESG domain expert on semantic accuracy, relevance grades, factual correctness, and retrieval realism. 63 records per language were excluded in review.
- **Relevance grades**: 3 = highly relevant, 2 = moderately relevant, 0 = irrelevant. For Precision, a document counts as relevant when its grade is greater than 1.

Each file is a JSON array of records:

```json
{
  "query": "What are the company's carbon emission targets?",
  "domain": "Environmental",
  "documents": [
    {"text": "The company aims for net-zero carbon emissions by 2030.",
     "relevance": 3, "label": "highly relevant"},
    {"text": "Investing in renewable energy is a company goal.",
     "relevance": 2, "label": "moderately relevant"},
    {"text": "The company reported record profits this year.",
     "relevance": 0, "label": "irrelevant"}
  ]
}
```

The Italian file uses the same schema with Italian text. The English and Italian files are not translations of each other; they are separately generated and reviewed sets.

## Models Benchmarked

14 models: 9 open-source run locally through Ollama, and 5 closed-source models accessed through APIs.

| Provider | Models |
|----------|--------|
| **Ollama** (local) | `nomic-embed-text`, `nomic-embed-text-v2-moe`, `mxbai-embed-large`, `bge-m3`, `all-minilm`, `embeddinggemma:300m`, `qwen3-embedding:0.6b`, `qwen3-embedding:4b`, `granite-embedding:278m` |
| **OpenAI** | `text-embedding-3-small`, `text-embedding-3-large`, `text-embedding-ada-002` |
| **Google Gemini** | `gemini-embedding-001`, `gemini-embedding-2` |

OpenAI and Gemini models are skipped automatically when the corresponding API key is not set.

## Metrics

Each query is scored against its three documents by cosine similarity between the query embedding and each document embedding. Scores are then averaged over all queries.

| Metric | Definition |
|--------|-----------|
| **NDCG** | Normalized Discounted Cumulative Gain over graded relevance (`sklearn.metrics.ndcg_score`). Primary metric. |
| **Precision** | Fraction of documents with similarity > 0.5 that are relevant (grade > 1). |
| **MAP** | Mean Average Precision over the ranked list. |
| **MRR** | Mean Reciprocal Rank of the first relevant document. |

`embed_eval.py` also computes Recall, but it is always 1.0 because the candidate list is the full set of documents for each query, so it is not reported.

## Results

NDCG on the full benchmark (from the paper, Tables 3 and 4):

| Rank | English | NDCG | Italian | NDCG |
|---|---|---|---|---|
| 1 | qwen3-embedding-4b | 0.9830 | text-embedding-3-large | 0.9745 |
| 2 | qwen3-embedding-0.6b | 0.9784 | qwen3-embedding-0.6b | 0.9717 |
| 3 | gemini-embedding-2 | 0.9771 | qwen3-embedding-4b | 0.9704 |

Full per-model results for both languages are in the paper. The paper's tables use the names `qwen3-embedding-0.6b` and `qwen3-embedding-4b`, which correspond to the Ollama tags `qwen3-embedding:0.6b` and `qwen3-embedding:4b`.

## Setup

Requirements: Python 3.9+, [Ollama](https://ollama.com/) (for the open-source models), and API keys for OpenAI and/or Google if you want the closed-source models.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the repository root with the keys you have. Keys you leave out just skip those models:

```env
OPENAI_API_KEY="your_openai_key_here"
GOOGLE_API_KEY="your_google_key_here"
```

Start Ollama and pull the open-source models:

```bash
ollama serve &
bash pull_models.sh
```

## Running the Benchmark

Run both languages end to end (creates `.venv` if missing, installs requirements, pulls Ollama models):

```bash
bash run_experiments.sh
```

Or step by step, from the repository root:

```bash
python3 src/genvec.py English     # generate embeddings -> results/English-<model>.json
python3 src/embed_eval.py English # compute metrics   -> results/average_metrics_df_eng.json
```

The language argument accepts `English` or `Italian`.

`genvec.py` checkpoints the Gemini runs every 25 records and resumes from the last checkpoint. It skips any model whose output file already exists.

### Dimensionality experiment

This experiment reduces the embeddings of five models with PCA (256, 128, 64, 32, 16 dimensions) and re-evaluates NDCG:

```bash
bash run_dim_experiments.sh        # writes results/dim_experiment_{english,italian}.json
python3 src/plot_dim_experiment.py # plots both languages (no arguments)
```

## Outputs and Reproducibility

- **Per-model embeddings** (`results/<Language>-<model>.json`) are generated by `genvec.py`. They are excluded from this repository because of their size, so run `genvec.py` to recreate them.
- **Aggregate metrics** (`results/average_metrics_df_eng.json`, `results/average_metrics_df_ita.json`) are written by `embed_eval.py`. The copies in this repository were last written when only a subset of the models had results, so they may not list all 14 models. Running the full pipeline regenerates them.
- **Figures**: `plot_dim_experiment.py` writes PDF and PNG files to `figures/` one level above `src/`. In this repository that is the parent of the repository root, so the figures land outside the repository. Adjust `FIGURES_DIR` in `src/plot_dim_experiment.py` if you want them inside.

## Repository Layout

```
.
├── data/                              # Benchmark datasets (English, Italian)
├── src/
│   ├── genvec.py                      # Generate embeddings for all models
│   ├── embed_eval.py                  # Compute NDCG, Precision, MAP, MRR
│   ├── dim_experiment.py              # PCA dimensionality experiment
│   └── plot_dim_experiment.py         # Plot dimensionality results
├── results/                           # Aggregate metrics and dimensionality results
├── old-results/                       # Earlier aggregate metrics
├── pull_models.sh                     # Download Ollama models
├── run_experiments.sh                 # Full benchmark, both languages
├── run_dim_experiments.sh             # Dimensionality experiment
├── requirements.txt                   # Python dependencies
└── LICENSE                            # Apache License 2.0
```

## Citation

If you use this dataset or code, please cite:

```bibtex
@misc{saad2026esgbenchmark,
  title={Benchmarking Embedding Models for ESG Data: Code and Data},
  author={Saad, Motaz and Cret{\`i}, Veronica and Gentile, Ivan and Kazemi, Kianna and Longo, Antonella},
  howpublished={GitHub repository, \url{https://github.com/datalab-unisalento/benchmark-embeddings-esg}},
  year={2026},
  license={Apache-2.0}
}
```

## License

Code and data are released under the Apache License 2.0. See [LICENSE](LICENSE).
