#!/usr/bin/env bash
set -euo pipefail

REPO_ID="Jchavan010/data"
LOCAL_DIR="/root/Grace/VideoX-Fun/datasets/8k/precomputed_pt"

echo "Installing huggingface_hub CLI..."
python -m pip install -U "huggingface_hub[cli]"

mkdir -p "$LOCAL_DIR"

echo "Downloading dataset repo ${REPO_ID} into ${LOCAL_DIR} ..."

# Public repo path: no token needed
hf download "$REPO_ID" \
  --repo-type dataset \
  --local-dir "$LOCAL_DIR"

echo "Download complete. Listing files:"
find "$LOCAL_DIR" -maxdepth 2 -type f | sort
echo "Done."