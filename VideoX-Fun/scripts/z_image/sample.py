import os
import argparse

import numpy as np
import wandb
import torch

from PIL import Image

import math


from videox_fun.models import AutoencoderKL, AutoTokenizer, Qwen3ForCausalLM, ZImageTransformer2DModel

from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union


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


def make_image_grid(images, rows=None, cols=None):
    if len(images) == 0:
        return None

    if rows is None and cols is None:
        cols = min(4, len(images))
        rows = math.ceil(len(images) / cols)
    elif rows is None:
        rows = math.ceil(len(images) / cols)
    elif cols is None:
        cols = math.ceil(len(images) / rows)

    w, h = images[0].size
    grid = Image.new("RGB", (cols * w, rows * h))

    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        grid.paste(img, (c * w, r * h))

    return grid

def main():
    args = parse_args()

    wandb.init(
        project="Distillation",
        name=args.run_name,
        config=vars(args),
    )

    print(f"\n\n\n{args.sample_prompts}\n\n\n")

    sample_dir = args.sample_dir
    os.makedirs(sample_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print("using device:", device, "dtype:", dtype)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        subfolder="tokenizer",
    )

    print("loading base model from:", args.model_path)
    text_encoder = Qwen3ForCausalLM.from_pretrained(args.model_path, subfolder="text_encoder", torch_dtype=dtype).to(device).eval()

    vae = AutoencoderKL.from_pretrained(args.model_path, subfolder="vae").to(device=device, dtype=dtype).eval()
    
    print("loading checkpoint from:", args.ckpt_path)
    model = ZImageTransformer2DModel.from_pretrained(args.ckpt_path, subfolder="transformer", torch_dtype=dtype)

    model = model.to(device, dtype=dtype).eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {total_params/1e9:.2f}B")

    if args.inference_nfe == 1:
        sigmas = [1.0]
    elif args.inference_nfe == 2:
        sigmas = [1.0, 0.5]
    elif args.inference_nfe == 4:
        sigmas = [1.0, 0.75, 0.5, 0.25]
    else:
        raise ValueError("Only NFE in {1,2,4} supported for now")
    
    latent_h = args.inference_resolution // 8
    latent_w = args.inference_resolution // 8
    print("latent shape:", (1, 16, 1, latent_h, latent_w))

    with torch.no_grad():
        generated_images = []
        for i, prompt in enumerate(args.sample_prompts):
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=True if device.type == "cuda" else False):
                prompt_embeds = encode_prompt(prompt, device=device, text_encoder=text_encoder, tokenizer=tokenizer, max_sequence_length=512)

                prompt_embeds = [x.to(device=device, dtype=dtype) for x in prompt_embeds]

                x = torch.randn((1, 16, 1, latent_h, latent_w), device=device, dtype=dtype)
                for step_idx, t in enumerate(sigmas):
                    t = torch.tensor([t], device=device, dtype=dtype)
                    t = 1.0 - t

                    u = model(x=x, t=t, cap_feats=prompt_embeds)[0]

                    if step_idx < len(sigmas) - 1:
                        dt = sigmas[step_idx] - sigmas[step_idx + 1]
                    else:
                        dt = sigmas[step_idx]

                    # euler step
                    x = x + dt * u

                latents = x
                latents = latents / vae.config.scaling_factor + vae.config.shift_factor
                latents_2d = latents.squeeze(2)

                image = vae.decode(latents_2d).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3 , 1).float().numpy()
            pil_image = Image.fromarray((image[0] * 255).astype(np.uint8))
            save_path = os.path.join(args.sample_dir, f"sample_{i}.png")
            pil_image.save(save_path)
            generated_images.append(pil_image)
            print(f"saved {save_path}")
        
        grid = make_image_grid(generated_images)
        log_dict = {
            "inference/num_images": len(generated_images),
        }
        if grid is not None:
            log_dict["inference/grid"] = wandb.Image(
                grid,
                caption=" | ".join(args.sample_prompts[:10])
            )
        wandb.log(log_dict)
    wandb.finish()


    


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sample_dir", type=str, default="./samples")
    parser.add_argument("--run_name", type=str, default="inference")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--inference_nfe", type=int, required=True)

    parser.add_argument("--sample_prompts", type=str, nargs="+", required=True)
    parser.add_argument("--inference_resolution", type=int, default=512)

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    main()