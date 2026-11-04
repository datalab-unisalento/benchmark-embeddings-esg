#!/bin/bash

# Script to pull/download all embedding models required for the benchmark

echo "---------------------------------"
echo "🤖 Starting Ollama Model Downloads"
echo "---------------------------------"

# List of models to pull
MODELS=(
    "nomic-embed-text"
    "mxbai-embed-large"
    "bge-m3"
    "all-minilm"
    "nomic-embed-text-v2-moe"
    "embeddinggemma:300m"
    "qwen3-embedding:0.6b"
    "qwen3-embedding:4b"
    "granite-embedding:278m"
)

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama is not installed. Please install Ollama first."
    exit 1
fi

# Pull models
for model in "${MODELS[@]}"; do
    echo "📥 Pulling model: $model"
    ollama pull "$model"
    echo "✅ Finished pulling $model"
    echo ""
done

echo "---------------------------------"
echo "✨ All models have been downloaded!"
echo "---------------------------------"
