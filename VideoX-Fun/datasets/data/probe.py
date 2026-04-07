import sys
sys.path.append("/mnt/e/purple/VideoX-Fun")

import json
import traceback
from pathlib import Path
from typing import List, Optional, Union

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer, Qwen3ForCausalLM
from diffusers import AutoencoderKL


# =========================
# User-configurable paths
# =========================
PRETRAINED_MODEL_NAME_OR_PATH = "/mnt/e/purple/VideoX-Fun/models/Z-Image-turbo"
CAPTIONS_JSONL = Path("/mnt/e/purple/datasets/prepare/jdb/raw/train/train_anno_realease_repath.jsonl")
IMAGES_DIR = Path("/mnt/e/purple/datasets/prepare/jdb/raw/train/imgs")
OUTPUT_DIR = Path("/mnt/e/purple/datasets/prepare/jdb/probe_outputs")

# Probe only one image
RESOLUTION = 512

# Safer defaults for debugging
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32


def vprint(msg: str):
    print(msg, flush=True)


def encode_prompt(
    prompt: Union[str, List[str]],
    device: Optional[torch.device] = None,
    text_encoder=None,
    tokenizer=None,
    max_sequence_length: int = 512,
):
    """
    Copied from train.py behavior so probe matches training as closely as possible.
    Returns a list of variable-length tensors, one per prompt.
    """
    if isinstance(prompt, str):
        prompt = [prompt]

    prompt = list(prompt)

    vprint("[encode_prompt] original prompt list length: {}".format(len(prompt)))

    for i, prompt_item in enumerate(prompt):
        messages = [
            {"role": "user", "content": prompt_item},
        ]
        prompt_item = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        prompt[i] = prompt_item

    vprint("[encode_prompt] after chat template, first prompt preview:")
    vprint(prompt[0][:500] if len(prompt) > 0 else "<empty>")

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    )

    vprint(f"[encode_prompt] tokenized input_ids shape: {tuple(text_inputs.input_ids.shape)}")
    vprint(f"[encode_prompt] tokenized attention_mask shape: {tuple(text_inputs.attention_mask.shape)}")

    text_input_ids = text_inputs.input_ids.to(device)
    prompt_masks = text_inputs.attention_mask.to(device).bool()

    vprint(f"[encode_prompt] moved input_ids dtype/device: {text_input_ids.dtype} / {text_input_ids.device}")
    vprint(f"[encode_prompt] moved prompt_masks dtype/device: {prompt_masks.dtype} / {prompt_masks.device}")

    output = text_encoder(
        input_ids=text_input_ids,
        attention_mask=prompt_masks,
        output_hidden_states=True,
    )

    vprint("[encode_prompt] text encoder forward succeeded")
    vprint(f"[encode_prompt] num hidden states: {len(output.hidden_states)}")

    prompt_embeds = output.hidden_states[-2]
    vprint(f"[encode_prompt] hidden_states[-2] shape: {tuple(prompt_embeds.shape)} dtype={prompt_embeds.dtype}")

    embeddings_list = []
    for i in range(len(prompt_embeds)):
        kept = prompt_embeds[i][prompt_masks[i]]
        embeddings_list.append(kept)
        vprint(f"[encode_prompt] sample {i} masked embedding shape: {tuple(kept.shape)} dtype={kept.dtype}")

    return embeddings_list


def image_transform(size: int):
    return transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def debug_list_dir(path: Path, max_items: int = 20):
    if not path.exists():
        vprint(f"[debug_list_dir] MISSING: {path}")
        return
    vprint(f"[debug_list_dir] Listing {path}")
    items = sorted(list(path.iterdir()))
    for i, item in enumerate(items[:max_items]):
        vprint(f"  {i}: {item}")
    if len(items) > max_items:
        vprint(f"  ... and {len(items) - max_items} more")


def resolve_image_path(rel: str) -> Optional[Path]:
    rel = rel.strip()
    rel = rel.strip("./")

    candidates = [
        IMAGES_DIR / rel,
        CAPTIONS_JSONL.parent / rel,
        CAPTIONS_JSONL.parent.parent / rel,
        Path("/mnt/e/purple/datasets/prepare/jdb/raw") / rel,
    ]

    for c in candidates:
        if c.exists():
            return c

    return None


def load_first_valid_sample():
    vprint(f"[sample] Reading captions file: {CAPTIONS_JSONL}")
    if not CAPTIONS_JSONL.exists():
        raise FileNotFoundError(f"Captions JSONL not found: {CAPTIONS_JSONL}")

    with open(CAPTIONS_JSONL, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            try:
                row = json.loads(line)
            except Exception as e:
                vprint(f"[sample] Failed to parse json line {line_idx}: {e}")
                continue

            text = row.get("text") or row.get("prompt") or row.get("caption") or ""
            rel = row.get("file_path") or row.get("img_path") or row.get("image") or ""

            if line_idx < 5:
                vprint(f"[sample] line {line_idx} keys: {list(row.keys())}")
                vprint(f"[sample] line {line_idx} rel raw: {rel}")
                vprint(f"[sample] line {line_idx} text preview: {text[:120]}")

            if not text.strip():
                continue
            if not rel:
                continue

            img_path = resolve_image_path(rel)
            if img_path is not None:
                vprint(f"[sample] Found valid sample at line {line_idx}")
                return img_path, text, row

    raise RuntimeError("Could not find a valid sample with existing image path")


def save_tensor(tensor: torch.Tensor, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor.detach().cpu(), path)
    vprint(f"[save] Saved tensor: {path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    vprint("========== PROBE START ==========")
    vprint(f"[env] torch version: {torch.__version__}")
    vprint(f"[env] cuda available: {torch.cuda.is_available()}")
    vprint(f"[env] device: {DEVICE}")
    vprint(f"[env] dtype: {DTYPE}")
    if torch.cuda.is_available():
        vprint(f"[env] gpu: {torch.cuda.get_device_name(0)}")

    vprint(f"[paths] model root: {PRETRAINED_MODEL_NAME_OR_PATH}")
    vprint(f"[paths] captions jsonl: {CAPTIONS_JSONL}")
    vprint(f"[paths] images dir: {IMAGES_DIR}")
    vprint(f"[paths] output dir: {OUTPUT_DIR}")

    model_root = Path(PRETRAINED_MODEL_NAME_OR_PATH)
    if not model_root.exists():
        raise FileNotFoundError(f"Model root does not exist: {model_root}")

    debug_list_dir(model_root, max_items=50)
    debug_list_dir(model_root / "tokenizer", max_items=50)
    debug_list_dir(model_root / "text_encoder", max_items=50)
    debug_list_dir(model_root / "vae", max_items=50)

    # -------------------------
    # Load tokenizer
    # -------------------------
    try:
        vprint("[load] About to load tokenizer from model root + subfolder=tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(
            PRETRAINED_MODEL_NAME_OR_PATH,
            subfolder="tokenizer",
            use_fast=False,
            trust_remote_code=True,
        )
        vprint(f"[load] Loaded tokenizer type: {type(tokenizer)}")
    except Exception as e:
        vprint("[error] Failed tokenizer load from model root + subfolder")
        vprint(str(e))
        vprint(traceback.format_exc())

        vprint("[load] Trying fallback: direct tokenizer folder path")
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_root / "tokenizer"),
            use_fast=False,
            trust_remote_code=True,
        )
        vprint(f"[load] Loaded tokenizer via fallback type: {type(tokenizer)}")

    # -------------------------
    # Load text encoder
    # -------------------------
    try:
        vprint("[load] About to load text encoder")
        text_encoder = Qwen3ForCausalLM.from_pretrained(
            PRETRAINED_MODEL_NAME_OR_PATH,
            subfolder="text_encoder",
            torch_dtype=DTYPE,
            trust_remote_code=True,
        )
        vprint("[load] Text encoder object created")
        text_encoder = text_encoder.to(DEVICE).eval()
        vprint("[load] Text encoder moved to device and set to eval")
        vprint(f"[load] Text encoder class: {type(text_encoder)}")
    except Exception as e:
        vprint("[error] Failed to load text encoder")
        vprint(str(e))
        vprint(traceback.format_exc())
        raise

    # -------------------------
    # Load VAE
    # -------------------------
    try:
        vprint("[load] About to load VAE")
        vae = AutoencoderKL.from_pretrained(
            PRETRAINED_MODEL_NAME_OR_PATH,
            subfolder="vae",
        )
        vprint("[load] VAE object created")
        vae = vae.to(device=DEVICE, dtype=DTYPE).eval()
        vprint("[load] VAE moved to device and set to eval")
        vprint(f"[load] VAE class: {type(vae)}")
        vprint(f"[load] VAE scaling_factor: {getattr(vae.config, 'scaling_factor', 'N/A')}")
        vprint(f"[load] VAE shift_factor: {getattr(vae.config, 'shift_factor', 'N/A')}")
    except Exception as e:
        vprint("[error] Failed to load VAE")
        vprint(str(e))
        vprint(traceback.format_exc())
        raise

    # -------------------------
    # Load one sample
    # -------------------------
    try:
        img_path, text, row = load_first_valid_sample()
        vprint(f"[sample] Image path: {img_path}")
        vprint(f"[sample] Caption length: {len(text)}")
        vprint(f"[sample] Caption preview: {text[:300]}")
        vprint(f"[sample] Raw row keys: {list(row.keys())}")
    except Exception as e:
        vprint("[error] Failed to find/load sample")
        vprint(str(e))
        vprint(traceback.format_exc())
        raise

    # -------------------------
    # Image preprocessing
    # -------------------------
    try:
        vprint("[image] Opening image")
        image = Image.open(img_path).convert("RGB")
        vprint(f"[image] PIL size before transform: {image.size}")

        tfm = image_transform(RESOLUTION)
        pixel_values = tfm(image).unsqueeze(0)
        vprint(f"[image] pixel_values CPU shape: {tuple(pixel_values.shape)} dtype={pixel_values.dtype}")

        pixel_values = pixel_values.to(device=DEVICE, dtype=DTYPE)
        vprint(f"[image] pixel_values moved to device: shape={tuple(pixel_values.shape)} dtype={pixel_values.dtype} device={pixel_values.device}")
    except Exception as e:
        vprint("[error] Failed image preprocessing")
        vprint(str(e))
        vprint(traceback.format_exc())
        raise

    # -------------------------
    # VAE encode/decode
    # -------------------------
    try:
        with torch.no_grad():
            vprint("[vae] About to encode image")
            latent_out = vae.encode(pixel_values)
            vprint("[vae] VAE encode succeeded")

            # diffusers AutoencoderKLOutput usually has .latent_dist
            latent_dist = latent_out.latent_dist
            vprint(f"[vae] latent_dist type: {type(latent_dist)}")

            latents_raw = latent_dist.sample()
            vprint(f"[vae] latents_raw shape: {tuple(latents_raw.shape)} dtype={latents_raw.dtype} device={latents_raw.device}")

            scaling_factor = getattr(vae.config, "scaling_factor", 1.0)
            shift_factor = getattr(vae.config, "shift_factor", 0.0)
            scaled_latents = (latents_raw - shift_factor) * scaling_factor
            vprint(f"[vae] scaled_latents shape: {tuple(scaled_latents.shape)} dtype={scaled_latents.dtype}")

            # Decode raw latent sample for sanity
            vprint("[vae] About to decode raw latents")
            decoded = vae.decode(latents_raw).sample
            vprint(f"[vae] decoded shape: {tuple(decoded.shape)} dtype={decoded.dtype} device={decoded.device}")
    except Exception as e:
        vprint("[error] Failed VAE encode/decode")
        vprint(str(e))
        vprint(traceback.format_exc())
        raise

    # -------------------------
    # Text encode
    # -------------------------
    try:
        with torch.no_grad():
            vprint("[text] About to encode prompt")
            prompt_embeds = encode_prompt(
                [text],
                device=DEVICE,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
            )
            vprint("[text] Prompt encoding succeeded")
    except Exception as e:
        vprint("[error] Failed prompt encoding")
        vprint(str(e))
        vprint(traceback.format_exc())
        raise

    # -------------------------
    # Print shapes
    # -------------------------
    try:
        vprint("========== FINAL SHAPES ==========")
        vprint(f"pixel_values: {tuple(pixel_values.shape)} {pixel_values.dtype}")
        vprint(f"latents_raw: {tuple(latents_raw.shape)} {latents_raw.dtype}")
        vprint(f"scaled_latents: {tuple(scaled_latents.shape)} {scaled_latents.dtype}")
        vprint(f"decoded: {tuple(decoded.shape)} {decoded.dtype}")

        vprint(f"prompt_embeds type: {type(prompt_embeds)}")
        vprint(f"num prompts: {len(prompt_embeds)}")

        if len(prompt_embeds) > 0:
            vprint(f"prompt_embeds[0]: {tuple(prompt_embeds[0].shape)} {prompt_embeds[0].dtype}")
            if prompt_embeds[0].ndim == 2:
                vprint(f"num_tokens: {prompt_embeds[0].shape[0]}")
                vprint(f"embed_dim: {prompt_embeds[0].shape[1]}")
    except Exception as e:
        vprint("[error] Failed during final shape reporting")
        vprint(str(e))
        vprint(traceback.format_exc())
        raise

    # -------------------------
    # Save debug outputs
    # -------------------------
    try:
        vprint("[save] Saving tensors and recon image")
        save_tensor(latents_raw, OUTPUT_DIR / "latents_raw.pt")
        save_tensor(scaled_latents, OUTPUT_DIR / "scaled_latents.pt")
        if len(prompt_embeds) > 0:
            save_tensor(prompt_embeds[0], OUTPUT_DIR / "prompt_embeds_0.pt")

        recon = decoded.float().cpu()[0].clamp(-1, 1)
        recon = (recon + 1.0) / 2.0
        recon_pil = transforms.ToPILImage()(recon)
        recon_path = OUTPUT_DIR / "recon_test.png"
        recon_pil.save(recon_path)
        vprint(f"[save] Saved recon image: {recon_path}")
    except Exception as e:
        vprint("[error] Failed saving outputs")
        vprint(str(e))
        vprint(traceback.format_exc())
        raise

    vprint("========== PROBE SUCCESS ==========")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        vprint("========== PROBE FAILED ==========")
        vprint(str(e))
        vprint(traceback.format_exc())
        raise