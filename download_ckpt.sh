#!/usr/bin/env bash
set -euo pipefail

REPO_ID="Jchavan010/LaddFinal"
LOCAL_DIR="/root/Grace/checkpoints"

echo "Installing huggingface_hub..."
python -m pip install -U "huggingface_hub[cli]"

mkdir -p "$LOCAL_DIR"

echo "Downloading ckpt into $LOCAL_DIR ..."

hf download "$REPO_ID" \
  --repo-type model \
  --include "checkpoints/checkpoint-5300/*" \
  --local-dir "$LOCAL_DIR"

echo "Download complete. Listing files:"
find "$LOCAL_DIR" -maxdepth 4 -type f | sort
echo "Done."