import sys
sys.path.append("/mnt/e/purple/VideoX-Fun")

import json
import traceback
from pathlib import Path
from typing import List, Optional, Union

import torch
from transformers import AutoTokenizer, Qwen3ForCausalLM


# =========================
# Paths
# =========================
PRETRAINED_MODEL_NAME_OR_PATH = "/mnt/e/purple/VideoX-Fun/models/Z-Image-turbo"
CAPTIONS_JSONL = Path("/mnt/e/purple/datasets/prepare/jdb/raw/train/train_anno_realease_repath.jsonl")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# How many captions to probe
NUM_SAMPLES = 20


def vprint(msg: str):
    print(msg, flush=True)


def encode_prompt_verbose(
    prompt: Union[str, List[str]],
    device: Optional[torch.device] = None,
    text_encoder=None,
    tokenizer=None,
    max_sequence_length: int = 512,
):
    """
    Matches the logic from train.py but returns both:
    - full hidden states before masking: (B, 512, 2560)
    - masked list: variable-length tensors
    - attention mask
    """
    if isinstance(prompt, str):
        prompt = [prompt]

    prompt = list(prompt)

    templated_prompts = []
    for prompt_item in prompt:
        messages = [{"role": "user", "content": prompt_item}]
        prompt_item = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        templated_prompts.append(prompt_item)

    text_inputs = tokenizer(
        templated_prompts,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids.to(device)
    prompt_masks = text_inputs.attention_mask.to(device).bool()

    output = text_encoder(
        input_ids=text_input_ids,
        attention_mask=prompt_masks,
        output_hidden_states=True,
    )

    full_hidden = output.hidden_states[-2]  # (B, 512, D)

    masked_list = []
    for i in range(len(full_hidden)):
        masked_list.append(full_hidden[i][prompt_masks[i]])

    return {
        "templated_prompts": templated_prompts,
        "input_ids": text_input_ids,
        "prompt_masks": prompt_masks,
        "full_hidden": full_hidden,
        "masked_list": masked_list,
    }


def iter_valid_captions(jsonl_path: Path):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            try:
                row = json.loads(line)
            except Exception:
                continue

            text = row.get("text") or row.get("prompt") or row.get("caption") or ""
            if text.strip():
                yield line_idx, text, row


def main():
    vprint("========== CAPTION PROBE START ==========")
    vprint(f"[env] torch version: {torch.__version__}")
    vprint(f"[env] cuda available: {torch.cuda.is_available()}")
    vprint(f"[env] device: {DEVICE}")
    vprint(f"[env] dtype: {DTYPE}")
    if torch.cuda.is_available():
        vprint(f"[env] gpu: {torch.cuda.get_device_name(0)}")

    vprint(f"[paths] model root: {PRETRAINED_MODEL_NAME_OR_PATH}")
    vprint(f"[paths] captions jsonl: {CAPTIONS_JSONL}")
    vprint(f"[config] NUM_SAMPLES: {NUM_SAMPLES}")

    vprint("[load] Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        PRETRAINED_MODEL_NAME_OR_PATH,
        subfolder="tokenizer",
        use_fast=False,
        trust_remote_code=True,
    )
    vprint(f"[load] tokenizer type: {type(tokenizer)}")

    vprint("[load] Loading text encoder")
    text_encoder = Qwen3ForCausalLM.from_pretrained(
        PRETRAINED_MODEL_NAME_OR_PATH,
        subfolder="text_encoder",
        torch_dtype=DTYPE,
        trust_remote_code=True,
    ).to(DEVICE).eval()
    vprint(f"[load] text encoder type: {type(text_encoder)}")

    token_lengths = []
    truncated_count = 0

    for sample_idx, (line_idx, text, row) in enumerate(iter_valid_captions(CAPTIONS_JSONL)):
        if sample_idx >= NUM_SAMPLES:
            break

        try:
            with torch.no_grad():
                result = encode_prompt_verbose(
                    [text],
                    device=DEVICE,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                )

            full_hidden = result["full_hidden"]          # (1, 512, 2560)
            prompt_masks = result["prompt_masks"]       # (1, 512)
            masked = result["masked_list"][0]           # (seq_len, 2560)
            templated = result["templated_prompts"][0]

            token_count = int(prompt_masks[0].sum().item())
            token_lengths.append(token_count)

            # crude truncation signal: if mask sum == 512, sequence may be hitting max length
            if token_count == 512:
                truncated_count += 1

            vprint("--------------------------------------------------")
            vprint(f"[sample {sample_idx}] jsonl line: {line_idx}")
            vprint(f"[sample {sample_idx}] raw prompt length (chars): {len(text)}")
            vprint(f"[sample {sample_idx}] templated prompt length (chars): {len(templated)}")
            vprint(f"[sample {sample_idx}] full_hidden shape: {tuple(full_hidden.shape)} dtype={full_hidden.dtype}")
            vprint(f"[sample {sample_idx}] prompt_mask shape: {tuple(prompt_masks.shape)} dtype={prompt_masks.dtype}")
            vprint(f"[sample {sample_idx}] masked shape: {tuple(masked.shape)} dtype={masked.dtype}")
            vprint(f"[sample {sample_idx}] token_count from mask: {token_count}")
            vprint(f"[sample {sample_idx}] prompt preview: {text[:180]}")

        except Exception as e:
            vprint("--------------------------------------------------")
            vprint(f"[sample {sample_idx}] FAILED on jsonl line {line_idx}")
            vprint(str(e))
            vprint(traceback.format_exc())

    vprint("========== SUMMARY ==========")
    if len(token_lengths) == 0:
        vprint("No valid captions processed.")
        return

    unique_lengths = sorted(set(token_lengths))
    vprint(f"processed samples: {len(token_lengths)}")
    vprint(f"min tokens: {min(token_lengths)}")
    vprint(f"max tokens: {max(token_lengths)}")
    vprint(f"unique token lengths: {unique_lengths}")
    vprint(f"count hitting 512 tokens: {truncated_count}")

    vprint("========== CAPTION PROBE SUCCESS ==========")


if __name__ == "__main__":
    main()