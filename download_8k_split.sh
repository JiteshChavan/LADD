  #!/usr/bin/env bash
set -euo pipefail


export HF_TOKEN="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"
REPO_ID="Jchavan010/data"
LOCAL_DIR="/root/Grace/VideoX-Fun/datasets/8k/precomputed_pt"


echo "Installing huggingface_hub..."
python -m pip install -U huggingface_hub

echo "Downloading dataset into $LOCAL_DIR ..."

hf download $REPO_ID \
  --repo-type dataset \
  --local-dir $LOCAL_DIR \
  --token $HF_TOKEN


echo "Download complete. Listing files:"
find $LOCAL_DIR -maxdepth 2 -type f | sort
echo "Done."