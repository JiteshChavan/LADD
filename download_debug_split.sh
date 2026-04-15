#!/usr/bin/env bash
set -euo pipefail

REPO_ID="Jchavan010/zimage-debug-latents"
LOCAL_DIR="/root/Grace/VideoX-Fun/datasets/debug/precomputed_pt"

echo "Installing huggingface_hub..."
python -m pip install -U "huggingface_hub[cli]"

mkdir -p "$LOCAL_DIR"

echo "Downloading dataset into $LOCAL_DIR ..."

hf download "$REPO_ID" \
  --repo-type dataset \
  --local-dir "$LOCAL_DIR"

echo "Download complete. Listing files:"
find "$LOCAL_DIR" -maxdepth 2 -type f | sort
echo "Done."