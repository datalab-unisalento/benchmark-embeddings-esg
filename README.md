# ESG Embedding Models Benchmarking

This directory contains the code, data, and scripts for benchmarking embedding models on Environmental, Social, and Governance (ESG) data in English and Italian.

## Dataset

- **Source**: Generated using Generative AI (GPT-4.1) and validated by ESG domain experts.
- **Languages**: English and Italian.
- **Size**: 312 records per language (624 total).
- **Structure**: Each record contains a query, 3 documents, and graded relevance scores (0, 2, 3).

## Models Benchmarked

| Category | Models |
|----------|--------|
| **Ollama** | `nomic-embed-text`, `mxbai-embed-large`, `bge-m3`, `all-minilm`, `nomic-embed-text-v2-moe`, `gemma-embedding`, `qwen3-embed-4b`, `granite-embedding` |
| **OpenAI** | `text-embedding-3-small`, `text-embedding-3-large`, `text-embedding-ada-002` |
| **Gemini** | `models/text-embedding-004` |

## Setup

### 1. Prerequisites

- Python 3.9+
- Ollama installed and running
- Valid API keys for OpenAI and Google (Gemini)

### 2. Environment Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. API Keys

Edit the `.env` file and add your API keys:

```env
OPENAI_API_KEY="your_openai_key_here"
ANTHROPIC_API_KEY="your_anthropic_key_here"
GOOGLE_API_KEY="your_google_key_here"
```

### 4. Download Models

```bash
# Make sure Ollama is running
ollama serve &

# Pull all required models
bash pull_models.sh
```

## Running the Benchmark

Run both languages:

```bash
bash run_experiments.sh
```

The runner creates and reuses `.venv` automatically when it is missing.

Or run individually:

```bash
python3 src/genvec.py English
python3 src/embed_eval.py English
```

The language argument accepts `English` or `Italian`.

## File Structure

```
code-data/
├── data/                          # Raw dataset JSON files
│   ├── English-cleaned_data_embeddings_benchmark.json
│   └── Italian-cleaned_data_embeddings_benchmark.json
├── src/                           # Source code
│   ├── genvec.py                  # Embedding generation script
│   └── embed_eval.py              # Evaluation and metrics script
├── results/                       # Generated vectors and metrics
├── old-results/                   # Previous benchmark results
├── pull_models.sh                 # Ollama model download script
├── run_experiments.sh             # Run full benchmark
├── requirements.txt               # Python dependencies
├── .env                           # API keys (not committed)
└── .venv/                         # Python virtual environment
```

## Metrics

| Metric | Description |
|--------|-------------|
| **NDCG@K** | Normalized Discounted Cumulative Gain. Measures ranking quality with graded relevance. |
| **Precision@K** | Proportion of retrieved items in top K that are relevant. Threshold: similarity > 0.5. |

## Notes

- The evaluation uses cosine similarity to rank documents.
- Results are averaged across all queries in the dataset.
- Models that fail to load are skipped with a warning message.

## License

See `LICENSE` file for details.
