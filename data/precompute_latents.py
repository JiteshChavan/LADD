import os
import json
from argparse import ArgumentParser

import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

from typing import Union, Optional, List

def encode_prompt(
    prompt: Union[str, List[str]],
    device: Optional[torch.device] = None,
    text_encoder = None, 
    tokenizer = None,
    max_sequence_length: int = 512,
) -> List[torch.FloatTensor]:
    if isinstance(prompt, str):
        prompt = [prompt]

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

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids.to(device)
    prompt_masks = text_inputs.attention_mask.to(device).bool()

    prompt_embeds = text_encoder(
        input_ids=text_input_ids,
        attention_mask=prompt_masks,
        output_hidden_states=True,
    ).hidden_states[-2]

    embeddings_list = []

    for i in range(len(prompt_embeds)):
        embeddings_list.append(prompt_embeds[i][prompt_masks[i]])

    return embeddings_list

# ---------- args ----------
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--captions_jsonl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--debug_max_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


# ---------- image transform (MATCH TRAINING) ----------
def build_image_transform(size=512):
    return transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


# ---------- main ----------
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("Loading models...")

    # ⚠️ adjust this path if needed
    model_root = "/mnt/e/purple/VideoX-Fun/models/Z-Image-turbo"

    from transformers import AutoTokenizer, AutoModel
    from diffusers import AutoencoderKL

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(model_root, "tokenizer"),
        trust_remote_code=True,
    )

    text_encoder = AutoModel.from_pretrained(
        os.path.join(model_root, "text_encoder"),
        trust_remote_code=True,
        torch_dtype=torch.float16,
    ).to(device).eval()

    vae = AutoencoderKL.from_pretrained(
        os.path.join(model_root, "vae"),
        torch_dtype=torch.float16,
    ).to(device).eval()

    transform = build_image_transform(512)

    print("Starting precompute...")

    written = 0
    skipped = 0
    records = []

    with open(args.captions_jsonl, "r", encoding="utf-8") as f:
        for line in tqdm(f):

            if args.debug_max_samples and written >= args.debug_max_samples:
                break

            try:
                d = json.loads(line)

                caption = str(d["prompt"])
                relpath = d["img_path"].strip("./")

                img_path = os.path.join(args.images_dir, relpath)

                if not os.path.isfile(img_path):
                    skipped += 1
                    continue

                # ----- image -----
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    pixel_values = transform(img).unsqueeze(0).to(device, dtype=torch.float16)

                # ----- encode -----
                with torch.no_grad():

                    # VAE
                    latent_dist = vae.encode(pixel_values)
                    latents = latent_dist.latent_dist.sample()[0].cpu()  # (C,H,W)

                    # TEXT
                    cap_feats = encode_prompt(
                        caption,
                        device=device,
                        text_encoder=text_encoder,
                        tokenizer=tokenizer,
                    )[0].cpu()  # (seq, dim)

                # ----- save -----
                fname = f"{written:09d}.pt"
                fpath = os.path.join(args.output_dir, fname)

                torch.save({
                    "latents": latents,
                    "cap_feats": cap_feats,
                    "text": caption,
                    "relpath": relpath,
                }, fpath)

                records.append({
                    "pt_path": fname,
                    "text": caption,
                    "relpath": relpath,
                })

                written += 1

            except Exception as e:
                skipped += 1
                print("Error:", e)

    # save index
    with open(os.path.join(args.output_dir, "index.json"), "w") as f:
        json.dump(records, f, indent=2)

    print("\nDONE")
    print("written:", written)
    print("skipped:", skipped)


if __name__ == "__main__":
    main()