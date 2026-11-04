#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
RESULTS_DIR="$SCRIPT_DIR/results"

echo "=== ESG Embedding Benchmark ==="
echo ""

# Run from code-data/ so the Python scripts resolve data/ and results/ correctly.
cd "$SCRIPT_DIR"

# Create and activate virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

mkdir -p "$RESULTS_DIR"

echo "Activating virtual environment..."
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Install requirements
echo "Installing requirements..."
python3 -m pip install -r requirements.txt

# Pull models
echo "Pulling models..."
bash "$SCRIPT_DIR/pull_models.sh"

echo ""

# Run English experiments
echo "--- English ---"
python3 "$SCRIPT_DIR/src/genvec.py" English
python3 "$SCRIPT_DIR/src/embed_eval.py" English

# Run Italian experiments
echo ""
echo "--- Italian ---"
python3 "$SCRIPT_DIR/src/genvec.py" Italian
python3 "$SCRIPT_DIR/src/embed_eval.py" Italian

echo ""
echo "=== Done ==="
