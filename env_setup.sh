apt-get update
apt-get install -y python3-venv python3-pip
apt install tmux -y

python3 -m venv /root/Grace/.venv
source /root/Grace/.venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo Installing Project dependencies
cd /root/Grace/VideoX-Fun

pip install -e .
pip install -U huggingface_hub
pip install bitsandbytes
pip install wandb
hf download Tongyi-MAI/Z-Image --include "text_encoder/*"  --local-dir /root/Grace/VideoX-Fun/models/Z-Image
hf download Tongyi-MAI/Z-Image --include "vae/*" --local-dir /root/Grace/VideoX-Fun/models/Z-Image
hf download Tongyi-MAI/Z-Image --include "transformer/*" --local-dir /root/Grace/VideoX-Fun/models/Z-Image
hf download Tongyi-MAI/Z-Image --include "scheduler/*"  --local-dir /root/Grace/VideoX-Fun/models/Z-Image
hf download Tongyi-MAI/Z-Image --include "model_index.json" --local-dir /root/Grace/VideoX-Fun/models/Z-Image
hf download Tongyi-MAI/Z-Image --include "tokenizer/*" --local-dir /root/Grace/VideoX-Fun/models/Z-Image

