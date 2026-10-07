#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="AIDC-AI/Ovis2.5-2B"
TARGET_DIR="./model"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "$TARGET_DIR"

echo "[1/2] Checking huggingface_hub..."
if ! "$PYTHON_BIN" -c "import huggingface_hub" >/dev/null 2>&1; then
  echo "huggingface_hub not found. Installing..."
  "$PYTHON_BIN" -m pip install -U huggingface_hub
fi

echo "[2/2] Downloading $MODEL_ID -> $TARGET_DIR"
MODEL_ID="$MODEL_ID" TARGET_DIR="$TARGET_DIR" "$PYTHON_BIN" - <<'PY'
import os
from huggingface_hub import snapshot_download

model_id = os.environ["MODEL_ID"]
target_dir = os.environ["TARGET_DIR"]

snapshot_download(
    repo_id=model_id,
    local_dir=target_dir,
)
print(f"Download complete: {target_dir}")
PY
