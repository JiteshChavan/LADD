import os
import gc
import json
from argparse import ArgumentParser

import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--captions_jsonl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_root", type=str, required=True)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--vae_dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--text_dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--save_dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])

    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--debug_max_samples", type=int, default=None)
    parser.add_argument("--gc_every", type=int, default=10)

    parser.add_argument("--shard_name", type=str, default="shard_000.pt")

    return parser.parse_args()


def get_dtype(name: str):
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def build_image_transform(size=512):
    return transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def encode_prompt_batch(
    prompts,
    device,
    text_encoder,
    tokenizer,
    max_sequence_length=512,
):
    prompts = list(prompts)

    for i, prompt_item in enumerate(prompts):
        messages = [{"role": "user", "content": prompt_item}]
        prompt_item = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        prompts[i] = prompt_item

    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids.to(device)
    prompt_masks = text_inputs.attention_mask.to(device).bool()

    with torch.no_grad():
        prompt_embeds = text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_masks,
            output_hidden_states=True,
        ).hidden_states[-2]

    embeddings_list = []
    for i in range(len(prompt_embeds)):
        embeddings_list.append(prompt_embeds[i][prompt_masks[i]])

    return embeddings_list


def load_valid_records(images_dir, captions_jsonl, max_samples=None):
    records = []
    with open(captions_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if max_samples is not None and len(records) >= max_samples:
                break

            d = json.loads(line)

            caption = str(d["caption"])
            relpath = d["image"].strip("./")
            img_path = os.path.join(images_dir, relpath)

            if not os.path.isfile(img_path):
                continue

            records.append({
                "text": caption,
                "relpath": relpath,
                "img_path": img_path,
            })
    return records


def iterate_batches(records, batch_size):
    for i in range(0, len(records), batch_size):
        yield records[i:i + batch_size]


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    vae_dtype = get_dtype(args.vae_dtype)
    text_dtype = get_dtype(args.text_dtype)
    save_dtype = get_dtype(args.save_dtype)

    print("Loading models...")
    print(f"device={device}, vae_dtype={vae_dtype}, text_dtype={text_dtype}, save_dtype={save_dtype}, batch_size={args.batch_size}")

    from transformers import AutoTokenizer, AutoModel
    from diffusers import AutoencoderKL

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.model_root, "tokenizer"),
        trust_remote_code=True,
    )

    text_encoder = AutoModel.from_pretrained(
        os.path.join(args.model_root, "text_encoder"),
        trust_remote_code=True,
        torch_dtype=text_dtype,
    ).to(device).eval()

    vae = AutoencoderKL.from_pretrained(
        os.path.join(args.model_root, "vae"),
        torch_dtype=vae_dtype,
    ).to(device).eval()

    transform = build_image_transform(args.image_size)

    print("Collecting valid records...")
    valid_records = load_valid_records(
        images_dir=args.images_dir,
        captions_jsonl=args.captions_jsonl,
        max_samples=args.debug_max_samples,
    )
    print(f"Found {len(valid_records)} valid samples")

    written = 0
    skipped = 0
    index_records = []
    all_samples = []

    print("Starting precompute...")
    for batch_idx, batch_records in enumerate(tqdm(list(iterate_batches(valid_records, args.batch_size)))):
        try:
            pixel_values_list = []
            texts = []
            relpaths = []

            for rec in batch_records:
                try:
                    with Image.open(rec["img_path"]) as img:
                        img = img.convert("RGB")
                        pixel_values = transform(img)
                    pixel_values_list.append(pixel_values)
                    texts.append(rec["text"])
                    relpaths.append(rec["relpath"])
                except Exception as e:
                    print(f"Skipping bad image {rec['img_path']}: {e}")
                    skipped += 1

            if len(pixel_values_list) == 0:
                continue

            pixel_values = torch.stack(pixel_values_list, dim=0).to(device=device, dtype=vae_dtype)

            with torch.no_grad():
                latent_dist = vae.encode(pixel_values)
                latents = latent_dist.latent_dist.sample().to("cpu", dtype=save_dtype)

            del pixel_values, latent_dist
            if device.type == "cuda":
                torch.cuda.empty_cache()

            cap_feats_list = encode_prompt_batch(
                texts,
                device=device,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
            )
            cap_feats_list = [x.to("cpu", dtype=save_dtype) for x in cap_feats_list]

            for i in range(len(cap_feats_list)):
                sample = {
                    "latents": latents[i],          # raw VAE latent, unscaled
                    "cap_feats": cap_feats_list[i], # variable length
                    "text": texts[i],
                    "relpath": relpaths[i],
                }
                all_samples.append(sample)

                index_records.append({
                    "sample_idx": written,
                    "text": texts[i],
                    "relpath": relpaths[i],
                    "latents_shape": list(latents[i].shape),
                    "cap_feats_shape": list(cap_feats_list[i].shape),
                })
                written += 1

            del latents, cap_feats_list
            if (batch_idx + 1) % args.gc_every == 0:
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        except KeyboardInterrupt:
            raise
        except Exception as e:
            skipped += len(batch_records)
            print(f"Batch error: {e}")
            if device.type == "cuda":
                torch.cuda.empty_cache()

    shard_path = os.path.join(args.output_dir, args.shard_name)
    torch.save(
        {
            "samples": all_samples,
            "num_samples": len(all_samples),
        },
        shard_path,
    )

    index_path = os.path.join(args.output_dir, "index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_records, f, ensure_ascii=False, indent=2)

    print("\nDONE")
    print("written:", written)
    print("skipped:", skipped)
    print("shard:", shard_path)
    print("index:", index_path)


if __name__ == "__main__":
    main()