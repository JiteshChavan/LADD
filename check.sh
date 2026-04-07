python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch cuda version:", torch.version.cuda)
print("cudnn available:", torch.backends.cudnn.is_available())
print("cudnn version:", torch.backends.cudnn.version())
if torch.cuda.is_available():
    print("gpu 0:", torch.cuda.get_device_name(0))
PY