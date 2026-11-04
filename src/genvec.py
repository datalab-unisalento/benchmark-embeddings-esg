import os
import argparse
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

load_dotenv(BASE_DIR / ".env")

###############################################################

parser = argparse.ArgumentParser(description="Generate embeddings for ESG benchmark")
parser.add_argument("language", choices=["English", "Italian"], help="Language of the dataset")
args = parser.parse_args()

###############################################################

# Ollama Models
ollama_models = [
    "nomic-embed-text",
    "mxbai-embed-large",
    "bge-m3",
    "all-minilm",
    "nomic-embed-text-v2-moe",
    "embeddinggemma:300m",
    "qwen3-embedding:0.6b",
    "qwen3-embedding:4b",
    "granite-embedding:278m",
]

# OpenAI Models (skipped if OPENAI_API_KEY not set)
openAI_embedding_models = [
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
] if os.environ.get("OPENAI_API_KEY") else []

# Gemini Models (skipped if GOOGLE_API_KEY not set)
gemini_models = [
    "gemini-embedding-001",
    "gemini-embedding-2",
] if os.environ.get("GOOGLE_API_KEY") else []

###############################################################

from openai import OpenAI
_openai_client = OpenAI()
def generate_embedding_openai(text, model_name):
    response = _openai_client.embeddings.create(input=text, model=model_name)
    return response.data[0].embedding

try:
    import time
    from google import genai as google_genai
    _gemini_client = google_genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    def generate_embedding_gemini(text, model_name, retries=5):
        delay = 20
        for attempt in range(retries):
            try:
                result = _gemini_client.models.embed_content(model=model_name, contents=text)
                return list(result.embeddings[0].values)
            except Exception as e:
                if "429" in str(e) and attempt < retries - 1:
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise
except ImportError:
    def generate_embedding_gemini(text, model_name, retries=5):
        raise NotImplementedError("Please install google-genai: pip install google-genai")

from llama_index.embeddings.ollama import OllamaEmbedding
ollama_embedding_models = []
for model_name in ollama_models:
    try:
        ollama_embedding_models.append((model_name, OllamaEmbedding(model_name=model_name)))
    except Exception as e:
        print(f"Warning: Could not load Ollama model {model_name}. Error: {e}")

###############################################################

import pandas as pd
from tqdm.auto import tqdm

input_file = DATA_DIR / f"{args.language}-cleaned_data_embeddings_benchmark.json"
if not input_file.exists():
    raise FileNotFoundError(f"Dataset not found: {input_file}")

df = pd.read_json(input_file)
RESULTS_DIR.mkdir(exist_ok=True)

###############################################################

def safe_filename(model_name):
    return model_name.replace("models/", "").replace(":", "-").replace("/", "-")

def model_output_path(model_name):
    return RESULTS_DIR / f"{args.language}-{safe_filename(model_name)}.json"

def checkpoint_path(model_name):
    return RESULTS_DIR / f"{args.language}-{safe_filename(model_name)}.ckpt.json"

def already_done(model_name):
    path = model_output_path(model_name)
    if path.exists():
        print(f"Skipping [{model_name}] — already exists: {path}")
        return True
    return False

def load_checkpoint(model_name):
    ckpt = checkpoint_path(model_name)
    if ckpt.exists():
        import json
        data = json.loads(ckpt.read_text())
        print(f"Resuming [{model_name}] from row {data['row']}")
        return data["row"], data["query_embeddings"], data["docs_embeddings_list"]
    return 0, [], []

def save_checkpoint(model_name, row_idx, query_embeddings, docs_embeddings_list):
    import json
    ckpt = checkpoint_path(model_name)
    data = {"row": row_idx, "query_embeddings": query_embeddings, "docs_embeddings_list": docs_embeddings_list}
    ckpt.write_text(json.dumps(data))
    print(f"Checkpoint saved at row {row_idx}: {ckpt}")

def save_model_results(model_name, query_embeddings, docs_embeddings_list):
    model_df = df[["query", "documents"]].copy()
    model_df["query_embedding"] = pd.Series(query_embeddings, dtype=object)
    model_df["documents_embeddings"] = pd.Series(docs_embeddings_list, dtype=object)
    out = model_output_path(model_name)
    model_df.to_json(out, orient="records", indent=4)
    print(f"Saved: {out}")
    ckpt = checkpoint_path(model_name)
    if ckpt.exists():
        ckpt.unlink()

###############################################################

print("=== Open-Source Models (Ollama) ===")
for model_name, embedding_model in ollama_embedding_models:
    if already_done(model_name):
        continue
    query_embeddings, docs_embeddings_list = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"[{model_name}]"):
        query = row["query"]
        documents = [doc["text"] for doc in row["documents"]]
        try:
            q_emb = embedding_model.get_query_embedding(query)
            d_emb = embedding_model.get_text_embedding_batch(documents, show_progress=False)
        except Exception as e:
            print(f"Error with {model_name}: {e}")
            q_emb, d_emb = None, None
        query_embeddings.append(q_emb)
        docs_embeddings_list.append(d_emb)
    save_model_results(model_name, query_embeddings, docs_embeddings_list)

print("=== Closed-Source Models (OpenAI) ===")
for model_name in openAI_embedding_models:
    if already_done(model_name):
        continue
    query_embeddings, docs_embeddings_list = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"[{model_name}]"):
        query = row["query"]
        documents = [doc["text"] for doc in row["documents"]]
        try:
            q_emb = generate_embedding_openai(query, model_name)
            d_emb = [generate_embedding_openai(doc, model_name) for doc in documents]
        except Exception as e:
            print(f"Error with {model_name}: {e}")
            q_emb, d_emb = None, None
        query_embeddings.append(q_emb)
        docs_embeddings_list.append(d_emb)
    save_model_results(model_name, query_embeddings, docs_embeddings_list)

print("=== Closed-Source Models (Gemini) ===")
for model_name in gemini_models:
    if already_done(model_name):
        continue
    start_row, query_embeddings, docs_embeddings_list = load_checkpoint(model_name)
    for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc=f"[{model_name}]")):
        if idx < start_row:
            continue
        query = row["query"]
        documents = [doc["text"] for doc in row["documents"]]
        try:
            q_emb = generate_embedding_gemini(query, model_name)
            d_emb = [generate_embedding_gemini(doc, model_name) for doc in documents]
        except Exception as e:
            print(f"Error with {model_name}: {e}")
            q_emb, d_emb = None, None
        query_embeddings.append(q_emb)
        docs_embeddings_list.append(d_emb)
        if (idx + 1) % 25 == 0:
            save_checkpoint(model_name, idx + 1, query_embeddings, docs_embeddings_list)
    save_model_results(model_name, query_embeddings, docs_embeddings_list)

print("done")
