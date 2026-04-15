REPO_ID="Jchavan010/LaddFinal"
LOCAL_DIR="/root/Grace"

echo "Installing huggingface_hub..."
python -m pip install -U "huggingface_hub[cli]"

mkdir -p "$LOCAL_DIR"

echo "Downloading ckpt into $LOCAL_DIR ..."

hf download "$REPO_ID" \
  --repo-type model \
  --include "checkpoints/checkpoint-20300/**" \
  --local-dir "$LOCAL_DIR"

echo "Download complete. Listing files:"
find "$LOCAL_DIR/checkpoints/checkpoint-20300" -type f | sort