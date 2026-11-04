#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

cd "$SCRIPT_DIR"

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python3 -m pip install -q -r requirements.txt

echo "=== Dimension-vs-Quality Experiment ==="
echo "--- English ---"
python3 "$SCRIPT_DIR/src/dim_experiment.py" English
echo ""
echo "--- Italian ---"
python3 "$SCRIPT_DIR/src/dim_experiment.py" Italian
echo ""
echo "=== Done ==="
