"""Modified from https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
"""
#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and




import argparse
import gc
import logging
import math
import os
import pickle
import random
import shutil
import sys
from typing import (Any, Callable, Dict, List, NamedTuple, Optional, Tuple,
                    Union)

import accelerate
import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms.functional as TF
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import DDIMScheduler, FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (EMAModel,
                                      compute_density_for_timestep_sampling,
                                      compute_loss_weighting_for_sd3)
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module
from einops import rearrange
from omegaconf import OmegaConf
from packaging import version
from PIL import Image
from torch.utils.data import RandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers.utils import ContextManagers

import datasets

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None


from videox_fun.dist import set_multi_gpus_devices, shard_model
from videox_fun.models import (AutoencoderKL, AutoTokenizer,
                               CLIPImageProcessor,
                               CLIPVisionModelWithProjection,
                               Qwen2_5_VLForConditionalGeneration,
                               Qwen2Tokenizer, Qwen3ForCausalLM,
                               QwenImageTransformer2DModel,
                               ZImageTransformer2DModel)
from videox_fun.pipeline import ZImagePipeline
from videox_fun.utils.discrete_sampler import DiscreteSampling
from videox_fun.utils.utils import get_image_to_video_latent

import torch
import torch.nn as nn
import torch.nn.functional as F


class CondFeatureDiscHead(nn.Module):
    def __init__(
        self,
        in_channels=3840,
        cond_dim=3840,
        # NOTE : Project features downwards to keep heads lightweight
        hidden_channels=512,
        time_hidden_dim=256,
    ):
        super().__init__()

        # Turn scalar t into a vector we can inject into conv features
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_hidden_dim),
            nn.SiLU(),
            nn.Linear(time_hidden_dim, hidden_channels),
        )

        # Turn pooled caption embedding into a vector we can inject too
        self.caption_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        # Main conv net
        self.conv_in = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
        )

        self.conv_mid = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
        )

        # NOTE : out put logits
        self.conv_out = nn.Conv2d(
            hidden_channels // 2,
            1,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x, t, caption_pooled):
        """
        x: [B, C, H, W]
        t: [B] or [B, 1]
        caption_pooled: [B, cond_dim]
        returns: [B, num_logits]
        """
        if t.ndim == 1:
            t = t.unsqueeze(-1)  # [B, 1] -> [B, time_dime] later

        # Force conditioning path to fp32 for stability
        t = t.float()
        caption_pooled = caption_pooled.float()

        # Main image feature path
        h = self.conv_in(x)

        # Build conditioning vectors
        t_embed = self.time_mlp(t)[:, :, None, None]              # [B, hidden, 1, 1]
        c_embed = self.caption_mlp(caption_pooled)[:, :, None, None]

        # Inject conditioning additively
        h = h + t_embed + c_embed

        # More convs
        h = self.conv_mid(h)

        # Patch logits
        logits = self.conv_out(h)   # [B, 1, H', W']
        logits = logits.reshape(logits.shape[0], -1)

        return logits

class MultiFeatureDiscriminator(nn.Module):
    def __init__(
        self,
        in_channels_list,
        cond_dim=3840,
        hidden_channels=512,
        time_hidden_dim=256,
    ):
        super().__init__()

        self.feature_heads = nn.ModuleList([
            CondFeatureDiscHead(
                in_channels=in_ch,
                cond_dim=cond_dim,
                hidden_channels=hidden_channels,
                time_hidden_dim=time_hidden_dim,
            )
            for in_ch in in_channels_list
        ])

    def forward(self, feature_maps, t, caption_pooled):
        """
        feature_maps: list of [B, C, H, W]
        t: [B] or [B, 1]
        caption_pooled: [B, cond_dim]
        returns: [B, total_num_logits]
        """
        assert len(feature_maps) == len(self.feature_heads), (
            f"Expected {len(self.feature_heads)} feature maps, got {len(feature_maps)}"
        )

        outs = []
        for fmap, head in zip(feature_maps, self.feature_heads):
            outs.append(head(fmap, t, caption_pooled))

        return torch.cat(outs, dim=1)
    
# Function to noise X0_student or X0_real
def add_noise_for_disc(x0: torch.Tensor, eps: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    sigma_b = broadcast_t_like_x(sigma, x0)
    return (1.0 - sigma_b) * x0 + sigma_b * eps

# Helper function to send noised(x0_student) or noised(x0_real) into teacher and then evaluate discriminator
# returns discriminator logits
def disc_forward_teacher_nograd(x_in, t_in, prompt_embeds, teacher, disc, caption_pooled, feature_layers):
    teacher_dtype = next(teacher.parameters()).dtype
    teacher_device = next(teacher.parameters()).device

    x_in = x_in.to(device=teacher_device, dtype=teacher_dtype)
    t_in = t_in.to(device=teacher_device, dtype=teacher_dtype)
    prompt_embeds = [p.to(device=teacher_device, dtype=teacher_dtype) for p in prompt_embeds]
    with torch.no_grad():
        _, teacher_aux = teacher(
            x=x_in,
            t=t_in,
            cap_feats=prompt_embeds,
            return_features=True,
            feature_layers=feature_layers,
        )

    teacher_maps = [
        #NOTE: convert to conv friendly BCHW
        tokens_to_image_maps(feat, teacher_aux["x_item_seqlens"]).float()
        for feat in teacher_aux["features"]
    ]

    pred = disc(
        teacher_maps,
        t_in.float(),
        caption_pooled,
    )
    return pred

def disc_forward_teacher_grad(x_in, t_in, prompt_embeds, teacher, disc, caption_pooled, feature_layers):
    teacher_dtype = next(teacher.parameters()).dtype
    teacher_device = next(teacher.parameters()).device

    x_in = x_in.to(device=teacher_device, dtype=teacher_dtype)
    t_in = t_in.to(device=teacher_device, dtype=teacher_dtype)
    prompt_embeds = [p.to(device=teacher_device, dtype=teacher_dtype) for p in prompt_embeds]
    _, teacher_aux = teacher(
        x=x_in,
        t=t_in,
        cap_feats=prompt_embeds,
        return_features=True,
        feature_layers=feature_layers,
    )

    teacher_maps = [
        tokens_to_image_maps(feat, teacher_aux["x_item_seqlens"]).float()
        for feat in teacher_aux["features"]
    ]

    pred = disc(
        teacher_maps,
        t_in.float(),
        caption_pooled,
    )
    return pred



# ------------------------------------------------- Custom Dataset --------------------------------
DATASET_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "datasets")
if DATASET_ROOT not in sys.path:
    sys.path.insert(0, DATASET_ROOT)
from data.latent_set import ShardedLatentsDataset, collate_precomputed
# ---------------------------------------------------------------------------------


import wandb

def filter_kwargs(cls, kwargs):
    import inspect
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self', 'cls'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return filtered_kwargs

def linear_decay(initial_value, final_value, total_steps, current_step):
    if current_step >= total_steps:
        return final_value
    current_step = max(0, current_step)
    step_size = (final_value - initial_value) / total_steps
    current_value = initial_value + step_size * current_step
    return current_value

def generate_timestep_with_lognorm(low, high, shape, device="cpu", generator=None):
    u = torch.normal(mean=0.0, std=1.0, size=shape, device=device, generator=generator)
    t = 1 / (1 + torch.exp(-u)) * (high - low) + low
    return torch.clip(t.to(torch.int32), low, high - 1)

def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        mu = a2 * image_seq_len + b2
        return float(mu)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1

    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    mu = a * num_steps + b

    return float(mu)

def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

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

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0.dev0")

logger = get_logger(__name__, log_level="INFO")

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

def log_image_grid_only(tracker, prefix, images, step, captions=None):
    if images is None or len(images) == 0:
        return

    grid = make_image_grid(images)
    tracker.log(
        {
            f"{prefix}/image_grid": wandb.Image(
                grid,
                caption=f"step={step}" if captions is None else f"step={step} | " + " | ".join(captions[:4]),
            ),
            f"{prefix}/num_images": len(images),
        },
        step=step,
    )


def log_teacher_LADD_inference(vae, text_encoder, tokenizer, teacher, args, accelerator, weight_dtype, global_step):
    try:
        generated_images = []
        prompt_captions = []

        teacher.eval()

        with torch.no_grad(), torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=accelerator.device):
            logger.info("Running TEACHER custom LADD inference...")

            # same few-step schedules as student test
            if args.inference_nfe == 1:
                sigmas = [1.0]
            elif args.inference_nfe == 2:
                sigmas = [1.0, 0.5]
            elif args.inference_nfe == 4:
                sigmas = [1.0, 0.75, 0.5, 0.25]
            else:
                raise ValueError("Teacher custom LADD inference currently supports NFE in {1, 2, 4}")

            vae.to(accelerator.device)
            text_encoder.to(accelerator.device)
            # tokenizer stays on CPU

            if args.seed is None:
                generator = None
            else:
                rank_seed = args.seed + accelerator.process_index
                generator = torch.Generator(device=accelerator.device).manual_seed(rank_seed)
                logger.info(f"Rank {accelerator.process_index} using seed: {rank_seed}")

            os.makedirs(os.path.join(args.output_dir, "sample_teacher_custom"), exist_ok=True)

            for i, prompt in enumerate(args.sample_prompts):
                # -----------------------------------------
                # FIrst embed prompt
                # -----------------------------------------
                prompt_embeds = encode_prompt(
                    prompt=prompt,
                    device=accelerator.device,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    max_sequence_length=args.tokenizer_max_length,
                )
                prompt_embeds = [x.to(device=accelerator.device, dtype=weight_dtype) for x in prompt_embeds]

                # -----------------------------------------
                # SECONd Start from pure latent noise
                # -----------------------------------------
                latent_h = args.image_sample_size // 8
                latent_w = args.image_sample_size // 8

                x = torch.randn(
                    (1, 16, 1, latent_h, latent_w),
                    device=accelerator.device,
                    dtype=weight_dtype,
                    generator=generator,
                )

                # -----------------------------------------
                # then run euler sampler 
                # -----------------------------------------
                for step_idx, sigma in enumerate(sigmas):
                    sigma_tensor = torch.tensor([sigma], device=x.device, dtype=weight_dtype)
                    model_t = 1.0 - sigma_tensor

                    u = teacher(
                        x=x,
                        t=model_t,
                        cap_feats=prompt_embeds,
                    )[0]

                    if step_idx < len(sigmas) - 1:
                        dt = sigmas[step_idx] - sigmas[step_idx + 1]
                    else:
                        dt = sigmas[step_idx]  # final step to 0

                    x = x + dt * u

                # -----------------------------------------
                # fix latent scaling before passing to vae decoder
                # -----------------------------------------
                latents = x
                latents = latents / vae.config.scaling_factor + vae.config.shift_factor
                latents_2d = latents.squeeze(2)  # [1, 16, H, W]

                image = vae.decode(latents_2d).sample
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.cpu().permute(0, 2, 3, 1).float().numpy()

                pil_image = Image.fromarray((image[0] * 255).astype(np.uint8))

                img_path = os.path.join(
                    args.output_dir,
                    "sample_teacher_custom",
                    f"teacher-custom-{global_step}-rank{accelerator.process_index}-image-{i}.jpg",
                )
                pil_image.save(img_path)

                generated_images.append(pil_image)
                prompt_captions.append(prompt)

            if accelerator.is_main_process and args.report_to in ["wandb", "all"] and len(generated_images) > 0:
                tracker = accelerator.get_tracker("wandb", unwrap=True)
                log_image_grid_only(
                    tracker=tracker,
                    prefix="validation_teacher_custom",
                    images=generated_images,
                    step=global_step,
                    captions=prompt_captions,
                )

            text_encoder.to("cpu")
            vae.to("cpu")

            del generated_images, prompt_captions
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    except Exception as e:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print(f"Teacher custom LADD eval error on rank {accelerator.process_index}: {e}")

def log_LADD_inference(vae, text_encoder, tokenizer, student, args, accelerator, weight_dtype, global_step):
    try:
        is_deepspeed = type(student).__name__ == "DeepSpeedEngine"
        if is_deepspeed:
            origin_config = student.config
            student.config = accelerator.unwrap_model(student).config

        generated_images = []
        prompt_captions = []

        # unwrap once
        student_model = accelerator.unwrap_model(student)
        student_model.eval()

        with torch.no_grad(), torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=accelerator.device):
            logger.info("Running LADD inference...")

            # paper few-step schedule
            if args.inference_nfe == 1:
                sigmas = [1.0]
            elif args.inference_nfe == 2:
                sigmas = [1.0, 0.5]
            elif args.inference_nfe == 4:
                sigmas = [1.0, 0.75, 0.5, 0.25]
            else:
                raise ValueError("LADD inference currently supports NFE in {1, 2, 4}")

            vae.to(accelerator.device)
            text_encoder.to(accelerator.device)
            # tokenizer stays on CPU; do NOT call tokenizer.to(...)

            if args.seed is None:
                generator = None
            else:
                rank_seed = args.seed + accelerator.process_index
                generator = torch.Generator(device=accelerator.device).manual_seed(rank_seed)
                logger.info(f"Rank {accelerator.process_index} using seed: {rank_seed}")

            os.makedirs(os.path.join(args.output_dir, "sample"), exist_ok=True)

            for i, prompt in enumerate(args.sample_prompts):
                # --------------------------------------------------
                # 1) Encode prompt
                # --------------------------------------------------
                prompt_embeds = encode_prompt(
                    prompt=prompt,
                    device=accelerator.device,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    max_sequence_length=args.tokenizer_max_length,
                )
                # encode_prompt returns a list[Tensor[L, C]]
                # student expects a list, same as training
                # here batch size = 1
                prompt_embeds = [x.to(device=accelerator.device, dtype=weight_dtype) for x in prompt_embeds]

                # --------------------------------------------------
                # 2) Sample x0
                # --------------------------------------------------
                latent_h = args.image_sample_size // 8
                latent_w = args.image_sample_size // 8

                # match training layout: [B, C, F, H, W]
                x = torch.randn(
                    (1, 16, 1, latent_h, latent_w),
                    device=accelerator.device,
                    dtype=weight_dtype,
                    generator=generator,
                )

                # --------------------------------------------------
                # 3) solve with euler approximation towards z
                #    x_{next} = x + dt * u
                # --------------------------------------------------
                for step_idx, sigma in enumerate(sigmas):
                    sigma_tensor = torch.tensor([sigma], device=x.device, dtype=weight_dtype)
                    model_t = 1.0 - sigma_tensor

                    u = student_model(
                        x=x,
                        t=model_t,
                        cap_feats=prompt_embeds,
                    )[0]

                    if step_idx < len(sigmas) - 1:
                        dt = sigmas[step_idx] - sigmas[step_idx + 1]
                    else:
                        dt = sigmas[step_idx]  # final step to t=0

                    x = x + dt * u

                # --------------------------------------------------
                #  decode z
                # --------------------------------------------------
                # x is now approximately x0 in the scaled latent training space
                latents = x

                # undo training latent normalization
                latents = latents / vae.config.scaling_factor + vae.config.shift_factor

                # remove temporal dim for image VAE decode
                latents_2d = latents.squeeze(2)  # [1, 16, H, W]

                image = vae.decode(latents_2d).sample
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.cpu().permute(0, 2, 3, 1).float().numpy()

                pil_image = Image.fromarray((image[0] * 255).astype(np.uint8))

                img_path = os.path.join(
                    args.output_dir,
                    "sample",
                    f"sample-{global_step}-rank{accelerator.process_index}-image-{i}.jpg",
                )
                pil_image.save(img_path)

                generated_images.append(pil_image)
                prompt_captions.append(prompt)

            
            if accelerator.is_main_process and args.report_to in ["wandb", "all"] and len(generated_images) > 0:
                tracker = accelerator.get_tracker("wandb", unwrap=True)
                log_image_grid_only(
                    tracker=tracker,
                    prefix="validation_student_custom",
                    images=generated_images,
                    step=global_step,
                    captions=prompt_captions,
                )

            # Move frozen modules back to RAM
            text_encoder.to("cpu")
            vae.to("cpu")

            del generated_images, prompt_captions
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        if is_deepspeed:
            student.config = origin_config

    except Exception as e:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print(f"LADD eval error on rank {accelerator.process_index}: {e}")


def log_inference(vae, text_encoder, tokenizer, student, args, accelerator, weight_dtype, global_step):
    try:
        is_deepspeed = type(student).__name__ == "DeepSpeedEngine"
        if is_deepspeed:
            origin_config = student.config
            student.config = accelerator.unwrap_model(student).config

        generated_images = []
        prompt_captions = []

        with torch.no_grad(), torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=accelerator.device):
            logger.info("Running inference...")

            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="scheduler",
            )

            pipeline = ZImagePipeline(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                transformer=accelerator.unwrap_model(student) if type(student).__name__ == "DistributedDataParallel" else student,
                scheduler=scheduler,
            )
            pipeline = pipeline.to(accelerator.device)

            if args.seed is None:
                generator = None
            else:
                rank_seed = args.seed + accelerator.process_index
                generator = torch.Generator(device=accelerator.device).manual_seed(rank_seed)
                logger.info(f"Rank {accelerator.process_index} using seed: {rank_seed}")

            os.makedirs(os.path.join(args.output_dir, "sample"), exist_ok=True)

            for i, prompt in enumerate(args.sample_prompts):
                sample = pipeline(
                    prompt,
                    negative_prompt="bad detailed",
                    height=args.image_sample_size,
                    width=args.image_sample_size,
                    generator=generator,
                    guidance_scale=0,
                    num_inference_steps=args.inference_nfe,
                ).images

                img = sample[0]
                img_path = os.path.join(
                    args.output_dir,
                    "sample",
                    f"sample-{global_step}-rank{accelerator.process_index}-image-{i}.jpg",
                )
                img.save(img_path)

                generated_images.append(img)
                prompt_captions.append(prompt)

            if accelerator.is_main_process and args.report_to in ["wandb", "all"] and len(generated_images) > 0:
                tracker = accelerator.get_tracker("wandb", unwrap=True)

                grid = make_image_grid(generated_images)

                tracker.log(
                    {
                        "validation_teacher_stock/image_grid": wandb.Image(
                            grid,
                            caption=f"step={global_step}",
                        ),
                        "validation_teacher_stock/num_images": len(generated_images),
                    },
                    step=global_step,
                )

            del pipeline
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        if is_deepspeed:
            student.config = origin_config

    except Exception as e:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print(f"Eval error on rank {accelerator.process_index} with info {e}")
        

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")

    parser.add_argument(
        "--use_ladd",
        action="store_true",
        help="Enable teacher-student denoised latent distillation loss."
    )

    parser.add_argument("--use_adv", action="store_true")
    parser.add_argument("--adv_weight", type=float, default=0.01)
    parser.add_argument("--disc_lr", type=float, default=1e-5)
    parser.add_argument("--disc_hidden_channels", type=int, default=512)
    parser.add_argument("--disc_time_hidden_dim", type=int, default=256)

    parser.add_argument(
        "--disc_feature_layers_teacher",
        type=int,
        nargs="+",
        default=[29], # teacher goes 0, 29
    )

    parser.add_argument(
        "--discrete_time_pdf",
        type=str,
        required=True,
        choices=["ladd_paper", "uniform"],
        help="PDF to sample discrete timesteps during training.",
    )

    parser.add_argument(
        "--inference_nfe",
        type=int,
        default=16,
        help="Number of inference steps used during validation sampling.",
    )

    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. "
        ),
    )
    parser.add_argument(
        "--train_data_meta",
        type=str,
        default=None,
        help=(
            "A csv containing the training data. "
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--sample_prompts",
        type=str,
        default=None,
        nargs="+",
        help=("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--use_came",
        action="store_true",
        help="whether to use came",
    )
    parser.add_argument(
        "--multi_stream",
        action="store_true",
        help="whether to use cuda multi-stream",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--vae_mini_batch", type=int, default=32, help="mini batch size for vae."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--teacher_init", action="store_true", help="init from teacher or not.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_model_info", action="store_true", help="Whether or not to report more info about model (such as norm, grad)."
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpoint_every`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--sample_every",
        type=int,
        default=200,
        help="Run validation every X steps.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--tracker_run_id",
        type=str,
        default=None,
        help="W&B run id for resuming the same run.",
    )
    parser.add_argument(
        "--tracker_run_resume",
        type=str,
        default="allow",
        choices=["allow", "must", "never", "auto"],
        help="W&B resume policy.",
    )

    parser.add_argument(
        "--tracker_run_name",
        type=str,
        default=None,
        required=True,
        help="W&B run name shown in the UI.",
    )
    
    parser.add_argument(
        "--snr_loss", action="store_true", help="Whether or not to use snr_loss."
    )
    parser.add_argument(
        "--uniform_sampling", action="store_true", help="Whether or not to use uniform_sampling."
    )
    parser.add_argument(
        "--enable_text_encoder_in_dataloader", action="store_true", help="Whether or not to use text encoder in dataloader."
    )
    parser.add_argument(
        "--enable_bucket", action="store_true", help="Whether enable bucket sample in datasets."
    )
    parser.add_argument(
        "--random_ratio_crop", action="store_true", help="Whether enable random ratio crop sample in datasets."
    )
    parser.add_argument(
        "--random_hw_adapt", action="store_true", help="Whether enable random adapt height and width in datasets."
    )
    parser.add_argument(
        "--train_sampling_steps",
        type=int,
        default=1000,
        help="Run train_sampling_steps.",
    )
    parser.add_argument(
        "--image_sample_size",
        type=int,
        default=512,
        help="Sample size of the image.",
    )
    parser.add_argument(
        "--fix_sample_size", 
        nargs=2, type=int, default=None,
        help="Fix Sample size [height, width] when using bucket and collate_fn."
    )
    parser.add_argument(
        "--transformer_path",
        type=str,
        default=None,
        help=("If you want to load the weight from other transformers, input its path."),
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help=("If you want to load the weight from other vaes, input its path."),
    )

    parser.add_argument(
        '--trainable_modules',
        nargs='+',
        default=[],
        help='Enter a list of trainable modules'
    )
    parser.add_argument(
        '--trainable_modules_low_learning_rate', 
        nargs='+', 
        default=[],
        help='Enter a list of trainable modules with lower learning rate'
    )
    parser.add_argument(
        '--tokenizer_max_length', 
        type=int,
        default=512,
        help='Max length of tokenizer'
    )
    parser.add_argument(
        "--use_deepspeed", action="store_true", help="Whether or not to use deepspeed."
    )
    parser.add_argument(
        "--use_fsdp", action="store_true", help="Whether or not to use fsdp."
    )
    parser.add_argument(
        "--low_vram", action="store_true", help="Whether enable low_vram mode."
    )
    parser.add_argument(
        "--prompt_template_encode",
        type=str,
        default="<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        help=(
            'The prompt template for text encoder.'
        ),
    )
    parser.add_argument(
        "--prompt_template_encode_start_idx",
        type=int,
        default=34,
        help=(
            'The start idx for prompt template.'
        ),
    )
    parser.add_argument(
        "--abnormal_norm_clip_start",
        type=int,
        default=1000,
        help=(
            'When do we start doing additional processing on abnormal gradients. '
        ),
    )
    parser.add_argument(
        "--initial_grad_norm_ratio",
        type=int,
        default=5,
        help=(
            'The initial gradient is relative to the multiple of the max_grad_norm. '
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def broadcast_t_like_x(timesteps: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    broadcast (B,) timesteps into shape (B, 1, 1, 1, 1) (or matching ndim of x)
    """
    t = timesteps.to(device=x.device, dtype=x.dtype)
    while t.ndim < x.ndim:
        t = t.unsqueeze(-1)
    return t

def pool_caption_feats(cap_feats, target_dtype=None, target_device=None):
    """
    cap_feats: list of [Li, C] #NOTE: variable seq len caption embeddings pool then stack!
    returns: [B, C]
    """
    pooled = []
    for x in cap_feats:
        pooled.append(x.mean(dim=0))
    pooled = torch.stack(pooled, dim=0)
    if target_device is not None:
        pooled = pooled.to(target_device)
    if target_dtype is not None:
        pooled = pooled.to(target_dtype)
    return pooled

def project_and_pool_caption_feats(model, cap_feats, target_dtype=None, target_device=None):
    """
    cap_feats: list of [Li, 2560]
    returns: [B, 3840]
    """
    pooled = []
    for cf in cap_feats:
        proj = model.cap_embedder(cf)
        pooled.append(proj.mean(dim=0))
    pooled = torch.stack(pooled, dim=0)

    if target_device is not None:
        pooled = pooled.to(target_device)
    if target_dtype is not None:
        pooled = pooled.to(target_dtype)

    return pooled

def pred_x0_from_u(x_t: torch.Tensor, t: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """
    Our training parameterization is classical flow matching objective (z - eps) or (x0 - eps):
        x_t = (1 - t) * z + t * eps # interpolation
        u   = z - eps
    Therefore:
        eps = z - u
        and hence
        z = x_t + t * u
    and z is the clean latent x0.
    """
    return x_t + t * u

# NOTE: reshape backbone features for conv net
def tokens_to_image_maps(feat, x_item_seqlens):
    """
    feat: [B, unified_len, C]
    x_item_seqlens: list[int]
    returns: [B, C, H, W]
    """
    img_maps = []
    B, _, C = feat.shape

    for i in range(B):
        x_len = x_item_seqlens[i]
        img_tokens = feat[i, :x_len, :]
        side = int(x_len ** 0.5)
        assert side * side == x_len, f"x_len={x_len} is not square"
        img_map = img_tokens.view(side, side, C).permute(2, 0, 1).contiguous()
        img_maps.append(img_map)

    return torch.stack(img_maps, dim=0)

def generator_adv_loss(pred_fake):
    return F.binary_cross_entropy_with_logits(
        pred_fake, torch.ones_like(pred_fake)
    )

def discriminator_loss(pred_real, pred_fake):
    loss_real = F.binary_cross_entropy_with_logits(
        pred_real, torch.ones_like(pred_real)
    )
    loss_fake = F.binary_cross_entropy_with_logits(
        pred_fake, torch.zeros_like(pred_fake)
    )
    return loss_real + loss_fake, loss_real, loss_fake

def main():
    args = parse_args()
    print(f"\n\n\n{args.sample_prompts}\n\n\n")
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision, # NOTE: torch.bf16 for now
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    deepspeed_plugin = accelerator.state.deepspeed_plugin if hasattr(accelerator.state, "deepspeed_plugin") else None
    fsdp_plugin = accelerator.state.fsdp_plugin if hasattr(accelerator.state, "fsdp_plugin") else None
    if deepspeed_plugin is not None:
        zero_stage = int(deepspeed_plugin.zero_stage)
        fsdp_stage = 0
        print(f"Using DeepSpeed Zero stage: {zero_stage}")

        args.use_deepspeed = True
        if zero_stage == 3:
            print(f"Auto set save_state to True because zero_stage == 3")
            args.save_state = True
    elif fsdp_plugin is not None:
        from torch.distributed.fsdp import ShardingStrategy
        zero_stage = 0
        if fsdp_plugin.sharding_strategy is ShardingStrategy.FULL_SHARD:
            fsdp_stage = 3
        elif fsdp_plugin.sharding_strategy is None: # The fsdp_plugin.sharding_strategy is None in FSDP 2.
            fsdp_stage = 3
        elif fsdp_plugin.sharding_strategy is ShardingStrategy.SHARD_GRAD_OP:
            fsdp_stage = 2
        else:
            fsdp_stage = 0
        print(f"Using FSDP stage: {fsdp_stage}")

        args.use_fsdp = True
        if fsdp_stage == 3:
            print(f"Auto set save_state to True because fsdp_stage == 3")
            args.save_state = True
    else:
        zero_stage = 0
        fsdp_stage = 0
        print("DeepSpeed is not enabled.")

    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=logging_dir)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        rng = np.random.default_rng(np.random.PCG64(args.seed + accelerator.process_index))
        torch_rng = torch.Generator(accelerator.device).manual_seed(args.seed + accelerator.process_index)
    else:
        rng = None
        torch_rng = None
    index_rng = np.random.default_rng(np.random.PCG64(43))
    print(f"Init rng with seed {args.seed + accelerator.process_index}. Process_index is {accelerator.process_index}")

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora student) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Load scheduler, tokenizer and models.
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="scheduler"
    )

    # Get Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer"
    )

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So Qwen3ForCausalLM and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        # Get Text encoder
        text_encoder = Qwen3ForCausalLM.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", torch_dtype=weight_dtype
        )
        text_encoder = text_encoder.eval()
        # Get Vae
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, 
            subfolder="vae"
        ).to(weight_dtype)
        vae.eval()
    


    # -----------------------------------------------------------------------
    # NOTE: Model creation!
    # -----------------------------------------------------------------------
    # TODO: Change to fp32 later if required
    teacher = ZImageTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )
    

    disc = MultiFeatureDiscriminator(
        in_channels_list=[teacher.dim] * len(args.disc_feature_layers_teacher),  # 3840
        cond_dim=teacher.dim,   # pooled caption also in model dim 3840 for z_image
        hidden_channels=args.disc_hidden_channels,
        time_hidden_dim=args.disc_time_hidden_dim,
    )

    print(f"\n\n\nTeacher config inspection")
    print(type(teacher.config))
    print(teacher.config)
    print(teacher.config.n_layers)
    print(f"Teacher config inspection end\n\n\n")

    teacher_cfg = dict(teacher.config)
    teacher_n_layers = teacher_cfg["n_layers"]

    student_cfg = dict(teacher_cfg)
    # TODO: add flag later
    if args.teacher_init:
        student_cfg["n_layers"] = 30
    else:
        student_cfg["n_layers"] = 6   # 1/5 of 30

    # IMPORTANT: remove config-only keys not accepted by __init__
    student_cfg = filter_kwargs(ZImageTransformer2DModel, student_cfg)
    print("filtered student_cfg keys:", student_cfg.keys())

    student = ZImageTransformer2DModel(**student_cfg)
    student = student.to(dtype=torch.bfloat16)

    print("teacher_n_layers:", teacher_n_layers)
    print("student_n_layers:", len(student.layers))

    
    # ---------------------------
    # NOTE: Copy weights from teacher to student!
    # ---------------------------
    print("\n\n\nCopying shared modules...")

    student.all_x_embedder.load_state_dict(teacher.all_x_embedder.state_dict())
    student.all_final_layer.load_state_dict(teacher.all_final_layer.state_dict())

    student.noise_refiner.load_state_dict(teacher.noise_refiner.state_dict())
    student.context_refiner.load_state_dict(teacher.context_refiner.state_dict())

    student.t_embedder.load_state_dict(teacher.t_embedder.state_dict())
    student.cap_embedder.load_state_dict(teacher.cap_embedder.state_dict())

    student.x_pad_token.data.copy_(teacher.x_pad_token.data)
    student.cap_pad_token.data.copy_(teacher.cap_pad_token.data)

    print("Copying transformer layers...")

    t_layers = len(teacher.layers)
    s_layers = len(student.layers)

    layer_map = [
        round(i * (t_layers - 1) / (s_layers - 1))
        for i in range(s_layers)
    ]

    print("layer_map:", layer_map)

    for s_idx, t_idx in enumerate(layer_map):
        print(f"copy teacher layer {t_idx} to student layer {s_idx}")
        student.layers[s_idx].load_state_dict(
            teacher.layers[t_idx].state_dict()
        )

    # TODO: remove checks later!
    print("teacher layers:", len(teacher.layers))
    print("student layers:", len(student.layers))
    # check one weight
    print(
        torch.allclose(
            teacher.layers[layer_map[0]].attention.to_q.weight,
            student.layers[0].attention.to_q.weight
        )
    )
    print("Weight copy DONE\n\n\n")
    accelerator.print(f"Teacher params: {sum(p.numel() for p in teacher.parameters())/1e9:.2f}B")
    accelerator.print(f"Student params: {sum(p.numel() for p in student.parameters())/1e9:.2f}B")



    # NOTE: Freeze vae and text_encoder and set student to trainable
    # FOR INFERENCE ONLY WE HAVE ALREADY PRECOMPUTED LATENTS
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    # INITIALLY SET AS FALSE AND LATER IF WE PASS AN EMPTY LIST FOR TRAINABLE PARAMETERS EVERY PARAMETER IS TRAINABLE
    student.requires_grad_(False)
    # SAVE VRAM ONLY LOAD DURING INFERENCE
    text_encoder.to("cpu")
    vae.to("cpu")
    teacher.requires_grad_(False)
    teacher.eval()

    teacher.to(device=accelerator.device, dtype=weight_dtype)
    # NOTE: KEEP disc in fp32, small, less capacity
    disc = disc.to(device=accelerator.device, dtype=torch.float32)
    #optimizer_D = torch.optim.AdamW(
    #    disc.parameters(),
    #    lr=args.disc_lr,
    #    betas=(0.0, 0.99),
    #    weight_decay=0.0,
    #    eps=1e-8,
    #)
    
    
    if args.transformer_path is not None:
        print(f"From checkpoint: {args.transformer_path}")
        if args.transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(args.transformer_path)
        else:
            state_dict = torch.load(args.transformer_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = student.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
        assert len(u) == 0

    if args.vae_path is not None:
        print(f"From checkpoint: {args.vae_path}")
        if args.vae_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(args.vae_path)
        else:
            state_dict = torch.load(args.vae_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = vae.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
        assert len(u) == 0
    
    # A good trainable modules is showed below now.
    # For 3D Patch: trainable_modules = ['ff.net', 'pos_embed', 'attn2', 'proj_out', 'timepositionalencoding', 'h_position', 'w_position']
    # For 2D Patch: trainable_modules = ['ff.net', 'attn2', 'timepositionalencoding', 'h_position', 'w_position']

    # NOTE: SET STUDENT TO TRAIN!
    student.train()
    # NOTE: If no module filters are provided, train everything.
    if not args.trainable_modules and not args.trainable_modules_low_learning_rate:
        # NOTE: the only branch that matters
        if accelerator.is_main_process:
            accelerator.print("No trainable_modules filters provided hence training ALL transformer parameters.")
        for param in student.parameters():
            param.requires_grad = True
    else:
        if accelerator.is_main_process:
            accelerator.print(
                f"Trainable modules '{args.trainable_modules}', "
                f"low-lr modules '{args.trainable_modules_low_learning_rate}'."
            )
        for name, param in student.named_parameters():
            for trainable_module_name in (args.trainable_modules or []) + (args.trainable_modules_low_learning_rate or []):
                if trainable_module_name in name:
                    param.requires_grad = True
                    break
    
    #matched = [(n, p) for n, p in student.named_parameters() if p.requires_grad]
    #print(f"matched trainable tensors: {len(matched)}")
    #print(f"matched trainable params: {sum(p.numel() for _, p in matched):,}")
    #if len(matched) == 0:
    #    print("\nNo matches. Showing first 200 parameter names:\n")
    #    for i, (n, p) in enumerate(student.named_parameters()):
    #        if i >= 200:
    #            break
    #        print(n)
    #    raise ValueError(
    #        f"No parameters matched trainable_modules={args.trainable_modules} "
    #        f"and trainable_modules_low_learning_rate={args.trainable_modules_low_learning_rate}"
    #    )

    # Create EMA for the student.
    if args.use_ema:
        if zero_stage == 3:
            raise NotImplementedError("FSDP does not support EMA.")

        ema_student = ZImageTransformer2DModel.from_pretrained(
            args.pretrained_model_name_or_path, 
            subfolder="transformer",
            torch_dtype=weight_dtype,
        ).to(weight_dtype)

        ema_student = EMAModel(ema_student.parameters(), model_cls=ZImageTransformer2DModel, model_config=ema_student.config)

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        if fsdp_stage != 0 or zero_stage == 3:
            def save_model_hook(models, weights, output_dir):
                accelerate_state_dict = accelerator.get_state_dict(models[-1], unwrap=True)
                if accelerator.is_main_process:
                    from safetensors.torch import save_file

                    safetensor_save_path = os.path.join(output_dir, f"diffusion_pytorch_model.safetensors")
                    accelerate_state_dict = {k: v.to(dtype=weight_dtype) for k, v in accelerate_state_dict.items()}
                    save_file(accelerate_state_dict, safetensor_save_path, metadata={"format": "pt"})


            def load_model_hook(models, input_dir):
                pkl_path = os.path.join(input_dir, "sampler_pos_start.pkl")
                if os.path.exists(pkl_path):
                    with open(pkl_path, 'rb') as file:
                        loaded_number, _ = pickle.load(file)
                    print(f"Load pkl from {pkl_path}. Get loaded_number = {loaded_number}.")
        else:
            # create custom saving & loading hooks so that `accelerator.save_state(.)` serializes in a nice format
            def save_model_hook(models, weights, output_dir):
                if accelerator.is_main_process:
                    if args.use_ema:
                        ema_student.save_pretrained(os.path.join(output_dir, "transformer_ema"))

                    student_model = accelerator.unwrap_model(student)
                    student_model.save_pretrained(os.path.join(output_dir, "transformer"))

                    disc_model = accelerator.unwrap_model(disc)
                    torch.save(disc_model.state_dict(), os.path.join(output_dir, "disc.pt"))

                while len(weights) > 0:
                    weights.pop()


            def load_model_hook(models, input_dir):
                if args.use_ema:
                    ema_path = os.path.join(input_dir, "transformer_ema")
                    if os.path.isdir(ema_path):
                        _, ema_kwargs = ZImageTransformer2DModel.load_config(
                            ema_path, return_unused_kwargs=True
                        )
                        load_model = ZImageTransformer2DModel.from_pretrained(
                            input_dir, subfolder="transformer_ema"
                        )
                        load_model = EMAModel(
                            load_model.parameters(),
                            model_cls=ZImageTransformer2DModel,
                            model_config=load_model.config,
                        )
                        load_model.load_state_dict(ema_kwargs)

                        ema_student.load_state_dict(load_model.state_dict())
                        ema_student.to(accelerator.device)
                        del load_model

                student_load = ZImageTransformer2DModel.from_pretrained(
                    input_dir, subfolder="transformer"
                )
                accelerator.unwrap_model(student).register_to_config(**student_load.config)
                accelerator.unwrap_model(student).load_state_dict(student_load.state_dict())
                del student_load

                disc_path = os.path.join(input_dir, "disc.pt")
                if os.path.exists(disc_path):
                    disc_state = torch.load(disc_path, map_location="cpu")
                    accelerator.unwrap_model(disc).load_state_dict(disc_state)

                while len(models) > 0:
                    models.pop()

                pkl_path = os.path.join(input_dir, "sampler_pos_start.pkl")
                if os.path.exists(pkl_path):
                    with open(pkl_path, "rb") as file:
                        loaded_number, _ = pickle.load(file)
                    print(f"Load pkl from {pkl_path}. Get loaded_number = {loaded_number}.")

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    #NOTE: IMPORTANT FOR PAPER FAITHFUL ADVERSARIAL SUPERVISION
    if args.gradient_checkpointing:
        student.enable_gradient_checkpointing()
        teacher.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    elif args.use_came:
        try:
            from came_pytorch import CAME
        except Exception:
            raise ImportError(
                "Please install came_pytorch to use CAME. You can do so by running `pip install came_pytorch`"
            )

        optimizer_cls = CAME
    else:
        optimizer_cls = torch.optim.AdamW

    trainable_params = list(filter(lambda p: p.requires_grad, student.parameters()))
    trainable_params_optim = [
        {'params': [], 'lr': args.learning_rate},
        {'params': [], 'lr': args.learning_rate / 2},
    ]
    
    in_already = []

    trainable_params = [p for p in student.parameters() if p.requires_grad]
    if not args.trainable_modules and not args.trainable_modules_low_learning_rate:
        trainable_params_optim = [{"params": trainable_params, "lr": args.learning_rate}]
        if accelerator.is_main_process:
            accelerator.print(f"Training ALL transformer params at lr={args.learning_rate}")
    else:
        trainable_params_optim = [
            {"params": [], "lr": args.learning_rate},
            {"params": [], "lr": args.learning_rate / 2},
        ]

        for name, param in student.named_parameters():
            if not param.requires_grad:
                continue

            placed = False
            for trainable_module_name in args.trainable_modules:
                if trainable_module_name in name:
                    trainable_params_optim[0]["params"].append(param)
                    placed = True
                    if accelerator.is_main_process:
                        accelerator.print(f"Set {name} to lr={args.learning_rate}")
                    break

            if placed:
                continue

            for trainable_module_name in args.trainable_modules_low_learning_rate:
                if trainable_module_name in name:
                    trainable_params_optim[1]["params"].append(param)
                    placed = True
                    if accelerator.is_main_process:
                        accelerator.print(f"Set {name} to lr={args.learning_rate / 2}")
                    break

    if args.use_came:
        optimizer = optimizer_cls(
            trainable_params_optim,
            lr=args.learning_rate,
            # weight_decay=args.adam_weight_decay,
            betas=(0.9, 0.999, 0.9999), 
            eps=(1e-30, 1e-16)
        )
        
    else:
        optimizer = optimizer_cls(
            trainable_params_optim,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    
    #optimizer_D = optimizer_cls(
    #    disc.parameters(),
    #    lr=args.disc_lr,
    #    betas=(args.adam_beta1, args.adam_beta2),
    #    weight_decay=args.adam_weight_decay,
    #    eps=args.adam_epsilon,
    #)
    # NOTE: 8bit adam only for student 6B model
    optimizer_D = torch.optim.AdamW(
        disc.parameters(),
        lr=args.disc_lr,
        betas=(0.0, 0.99),
        weight_decay=0.0,
        eps=1e-8,
    )

    # Get the training dataset
    if args.fix_sample_size is not None and args.enable_bucket:
        args.image_sample_size = max(max(args.fix_sample_size), args.image_sample_size)
        args.random_hw_adapt = False


    from torch.utils.data import Subset
    # NOTE: Precomputed dataset
    train_dataset = ShardedLatentsDataset(
        shard_paths=args.train_data_dir,
        load_into_memory=True,
    )

    if args.max_train_samples is not None:
        train_dataset = Subset(train_dataset, range(min(args.max_train_samples, len(train_dataset))))

    def worker_init_fn(_seed):
        _seed = _seed * 256
        def _worker_init_fn(worker_id):
            print(f"worker_init_fn with {_seed + worker_id}")
            np.random.seed(_seed + worker_id)
            random.seed(_seed + worker_id)
        return _worker_init_fn
    
    # NOTE: data loader 
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size, # batch_size per forward pass per GPU
        shuffle=True,
        num_workers=args.dataloader_num_workers, # NOTE: keep 0 for single shard for now, DDP worker contention for single shard causes crashes
        collate_fn=collate_precomputed, # Collate function returns batch of latent tensors and list of cap embedding tensors
        pin_memory=True,
        persistent_workers=True if args.dataloader_num_workers > 0 else False, # NOTE: WHY WHAT HOW ?
    )
        

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # NOTE: we use constant for now
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    student, disc, optimizer, optimizer_D, train_dataloader, lr_scheduler = accelerator.prepare(
        student, disc, optimizer, optimizer_D, train_dataloader, lr_scheduler
    )


    if fsdp_stage != 0 or zero_stage != 0:
        from functools import partial

        from videox_fun.dist import set_multi_gpus_devices, shard_model

        shard_fn = partial(shard_model, device_id=accelerator.device, param_dtype=weight_dtype, module_to_wrapper=text_encoder.model.layers)
        text_encoder = shard_fn(text_encoder)

    if args.use_ema:
        ema_student.to(accelerator.device)


    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    # NOTE: WANDB setup
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        keys_to_pop = [k for k, v in tracker_config.items() if isinstance(v, list)]
        for k in keys_to_pop:
            tracker_config.pop(k)
            print(f"Removed tracker_config['{k}']")

        init_kwargs = {}
        if args.report_to in ["wandb", "all"]:
            init_kwargs["wandb"] = {
                "name": args.tracker_run_name,
                "id": args.tracker_run_id,
                "resume": args.tracker_run_resume,
            }

        accelerator.init_trackers(
            project_name=args.tracker_project_name,
            config=tracker_config,
            init_kwargs=init_kwargs,
        )
        

    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            global_step = int(path.split("-")[1])

            initial_global_step = global_step

            first_epoch = global_step // num_update_steps_per_epoch
            print(f"Resuming from checkpoint {path}. first_epoch = {first_epoch}.")

            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on rank 0.
        disable=not accelerator.is_local_main_process,
    )



    # Calculate the index we need
    idx_sampling = DiscreteSampling(args.train_sampling_steps, uniform_sampling=args.uniform_sampling)

    # NOTE: Generator / Disc phase!
    phase = "G"
    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Data batch sanity check
            if epoch == first_epoch and step == 0:
                accelerator.print("\n[Sanity check] first precomputed batch")
                accelerator.print(f"batch keys: {list(batch.keys())}")
                accelerator.print(f"latents: {batch['latents'].shape}, {batch['latents'].dtype}")
                accelerator.print(f"cap_feats type: {type(batch['cap_feats'])}, len={len(batch['cap_feats'])}")
                accelerator.print(f"cap_feats[0]: {batch['cap_feats'][0].shape}, {batch['cap_feats'][0].dtype}")
                accelerator.print(f"text[0]: {batch['text'][0]}")
                accelerator.print(f"relpath[0]: {batch['relpath'][0]}")

                assert "latents" in batch
                assert "cap_feats" in batch
                assert batch["latents"].ndim == 4
                assert batch["latents"].shape[1:] == (16, 64, 64) # NOTE: (C, H, W) for 512x512
                assert isinstance(batch["cap_feats"], list)
                assert len(batch["cap_feats"]) == batch["latents"].shape[0]
                assert batch["cap_feats"][0].ndim == 2
                assert batch["cap_feats"][0].shape[1] == 2560

            
            # NOTE:  z ~ p_data(.) to device
            # NOTE: send latents batch to device
            assert "latents" in batch, f"Script currently requires precomputed latents and text embeddings."
            latents = batch["latents"].to(device=accelerator.device, dtype=weight_dtype)
            latents = latents.unsqueeze(2)  # [B, 16, 1 (time frames), 64, 64] NOTE: for shape compliance with z_image backbone forward
            if epoch == first_epoch and step == 0:
                accelerator.print(
                    f"latents: {latents.shape}, {latents.dtype}, {latents.device}"
                )

            # NOTE: send prompt embedding batch to device
            assert "cap_feats" in batch, f"Script currently requires precomputed latents and text embeddings."
            prompt_embeds = [
                x.to(device=accelerator.device, dtype=weight_dtype)
                for x in batch["cap_feats"]
            ]

            # NOTE: verify shapes!
            if epoch == first_epoch and step == 0:
                accelerator.print(
                    f"prompt_embeds[0]: {prompt_embeds[0].shape}, {prompt_embeds[0].dtype}, {prompt_embeds[0].device}"
                )
            

            bsz, channel, f, height, width = latents.size()

            # NOTE: apply VAE latent space scaling
            latents = ((latents - vae.config.shift_factor) * vae.config.scaling_factor).to(dtype=weight_dtype)
            # NOTE: sample from gaussian, same shape as latents
            eps = torch.randn(latents.size(), device=latents.device, generator=torch_rng, dtype=weight_dtype)

            if not args.uniform_sampling:
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler.config.num_train_timesteps).long()
            else:
                # Sample a random timestep for each image
                # timesteps = generate_timestep_with_lognorm(0, args.train_sampling_steps, (bsz,), device=latents.device, generator=torch_rng)
                # timesteps = torch.randint(0, args.train_sampling_steps, (bsz,), device=latents.device, generator=torch_rng)
                indices = idx_sampling(bsz, generator=torch_rng, device=latents.device)
                indices = indices.long().cpu()

            image_seq_len = latents.shape[1]
            mu = calculate_shift(
                image_seq_len,
                noise_scheduler.config.get("base_image_seq_len", 256),
                noise_scheduler.config.get("max_image_seq_len", 4096),
                noise_scheduler.config.get("base_shift", 0.5),
                noise_scheduler.config.get("max_shift", 1.15),
            )
            noise_scheduler.sigma_min = 0.0
        
            if not args.use_ladd:
                noise_scheduler.set_timesteps(args.train_sampling_steps, device=latents.device, mu=mu)
                timesteps = noise_scheduler.timesteps[indices].to(device=latents.device)

                def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
                    sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
                    schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
                    timesteps = timesteps.to(accelerator.device)
                    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

                    sigma = sigmas[step_indices].flatten()
                    while len(sigma.shape) < n_dim:
                        sigma = sigma.unsqueeze(-1)
                    return sigma

                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
                model_t = (1000 - timesteps) / 1000
            else:
                # NOTE: LADD Discrete time sampling
                t = torch.tensor(
                    [1.0, 0.75, 0.5, 0.25],
                    device=latents.device,
                    dtype=latents.dtype,
                )

                # NOTE: discrete time sampling PDF (from the LADD paper)
                if args.discrete_time_pdf == "ladd_paper":
                    if global_step < 500:
                        probs = torch.tensor([0.0, 0.0, 0.5, 0.5], device=latents.device)
                    else:
                        probs = torch.tensor([0.7, 0.1, 0.1, 0.1], device=latents.device)
                elif args.discrete_time_pdf == "uniform":
                    probs = torch.tensor([0.25, 0.25, 0.25, 0.25], device=latents.device)
                
                # NOTE: sample idx of discrete timestep specified by the PDF for time sampling
                idx = torch.multinomial(probs, num_samples=latents.shape[0], replacement=True)
                sigmas = t[idx]

                # NOTE: coefficient of X0 (Z)
                model_t = 1.0 - sigmas

            # NOTE: Broadcast t so it matches xt tensor shape                
            sigma_broadcast = broadcast_t_like_x(sigmas, latents)
            # NOTE: xt ~ p_t (x|z)
            noisy_latents = (1.0 - sigma_broadcast) * latents + sigma_broadcast * eps
            
            
            # NOTE: project 2560 to 3840 then average across caption tokens
            caption_pooled = project_and_pool_caption_feats(
                teacher,
                prompt_embeds,
                target_dtype=torch.float32,
                target_device=accelerator.device,
            )

            if epoch == first_epoch and step == 0:
                accelerator.print("DEBUG:")
                if not args.use_ladd:
                    accelerator.print(f"raw timesteps[:4] = {timesteps[:4]}")
                else:
                    accelerator.print(f"sigmas[:4] = {sigmas[:4]}")
                accelerator.print(f"model_t[:4] = {model_t[:4]}")
                accelerator.print(f"sigmas[:4] = {sigmas.flatten()[:4]}")
    
            # NOTE: MSE loss for base flow matching objective, useless for distillation
            def custom_mse_loss(u_theta, u_t, weighting=None, threshold=50):
                u_theta = u_theta.float()
                u_t = u_t.float()
                diff = u_theta - u_t
                mse_loss = F.mse_loss(u_theta, u_t, reduction='none')
                mask = (diff.abs() <= threshold).float()
                masked_loss = mse_loss * mask
                if weighting is not None:
                    masked_loss = masked_loss * weighting
                final_loss = masked_loss.mean()
                return final_loss

            current_phase = phase
            if args.use_ladd:
                if phase == "G":
                    with accelerator.accumulate(student):
                        student.train()
                        disc.eval()

                        # NOTE: dont hold backward activations for disc on VRAM
                        # NOTE: Grads do flow back to the generator
                        for p in disc.parameters():
                            p.requires_grad_(False)
                        
                        with torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=accelerator.device):
                            u_student = student(
                                x=noisy_latents,
                                t=model_t,
                                cap_feats=prompt_embeds,
                            )[0]

                            # convert both student and teacher vector fields into clean data representations
                            x0_student = pred_x0_from_u(noisy_latents, sigma_broadcast, u_student) # x_t + dt * u
                            # EXAMLPLE : at t = 1 at gaussian x_0 = x_1 + (1-1) * u

                            # NOTE: obsolete later according to the paper
                            recon_loss = F.smooth_l1_loss(
                                x0_student.float(),
                                latents.float(),
                                reduction="mean",
                            )

                            # NOTE: Convert B,T, C to B, C, H, W so that its compatible with conv net discriminator
                            
                            # NOTE: disc needs t/sigma detach from graph first
                            disc_sigmas = sigmas.detach()
                            disc_model_t = model_t.detach()

                            # NOTE: Noise students x0 prediction and then ask the discriminator if it looks real
                            # BASICALLY USE RENOISED x0_student and ask if it would look the same as renoised realX0
                            eps_fake_D = torch.randn_like(latents)
                            x_fake_D = add_noise_for_disc(x0_student, eps_fake_D, disc_sigmas)

                            pred_fake = disc_forward_teacher_grad(
                                x_in=x_fake_D,
                                t_in=disc_model_t,
                                prompt_embeds=prompt_embeds,
                                teacher=teacher,    
                                disc=disc,
                                caption_pooled=caption_pooled,
                                feature_layers=args.disc_feature_layers_teacher,
                            )

                            g_adv_loss = generator_adv_loss(pred_fake) # if pred fake logits are higher BCE loss goes down
                            loss = recon_loss + args.adv_weight * g_adv_loss

                            avg_loss = accelerator.gather(loss.repeat(bsz)).mean()
                            train_loss += avg_loss.item() / args.gradient_accumulation_steps
                            
                            accelerator.backward(loss)

                            if accelerator.sync_gradients:
                                norm_sum = accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                                # NOTE : only switch phase in very last micro step
                                optimizer.step()
                                lr_scheduler.step()
                                optimizer.zero_grad(set_to_none=True)
                                optimizer_D.zero_grad(set_to_none=True)
                                phase = "D"



                            if accelerator.sync_gradients:
                                log_dict = {
                                    "train/g_total_loss": loss.detach().item(),
                                    "train/recon_loss": recon_loss.detach().item(),
                                    "train/g_adv_loss": g_adv_loss.detach().item(),
                                    "train/lr": lr_scheduler.get_last_lr()[0],
                                }
                                if not args.use_deepspeed and not args.use_fsdp:
                                    try:
                                        log_dict["train/g_grad_norm"] = float(norm_sum)
                                    except Exception:
                                        pass
                                accelerator.log(log_dict, step=global_step)
                elif phase == "D":
                    with accelerator.accumulate(disc):
                        disc.train()
                        student.eval()

                        for p in disc.parameters():
                            p.requires_grad_(True)

                        # NOTE: RIGHT NOW WE ARE DOING SAME TIME LEVEL FOR BOTH G AND D PHASES LOWERS VARIANCE
                        with torch.no_grad():
                            u_student = student(x=noisy_latents, t=model_t, cap_feats=prompt_embeds)[0]
                            x0_student = pred_x0_from_u(noisy_latents, sigma_broadcast, u_student)
                        
                        disc_sigmas = sigmas.detach()
                        disc_model_t = model_t.detach()

                        eps_fake_D = torch.randn_like(latents)
                        eps_real_D = torch.randn_like(latents)
                        
                        # REVERT : fine for now
                        # eps_fake_D = eps_real_D

                        x_fake_D = add_noise_for_disc(x0_student, eps_fake_D, disc_sigmas)
                        x_real_D = add_noise_for_disc(latents, eps_real_D, disc_sigmas)

                        pred_fake = disc_forward_teacher_nograd(
                            x_in=x_fake_D.detach(),
                            t_in=disc_model_t,
                            prompt_embeds=prompt_embeds,
                            teacher=teacher,
                            disc=disc,
                            caption_pooled=caption_pooled,
                            feature_layers=args.disc_feature_layers_teacher,
                        )

                        pred_real = disc_forward_teacher_nograd(
                            x_in=x_real_D,
                            t_in=disc_model_t,
                            prompt_embeds=prompt_embeds,
                            teacher=teacher,
                            disc=disc,
                            caption_pooled=caption_pooled,
                            feature_layers=args.disc_feature_layers_teacher,
                        )

                        d_loss, loss_real, loss_fake = discriminator_loss(pred_real, pred_fake)
                        loss = d_loss

                        accelerator.backward(d_loss)

                        if accelerator.sync_gradients:
                            d_grad_norm = accelerator.clip_grad_norm_(disc.parameters(), args.max_grad_norm)
                            optimizer_D.step()
                            optimizer_D.zero_grad(set_to_none=True)

                            if args.use_ema:
                                ema_student.step(student.parameters())
                            phase = "G"
                            global_step += 1
                            progress_bar.update(1)
                    
                            if global_step % args.checkpoint_every == 0 and global_step > 0:
                                if args.use_deepspeed or args.use_fsdp or accelerator.is_main_process:
                                    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                                    print(f"[DEBUG] SAVING CHECKPOINT at global_step={global_step}")
                                    if args.checkpoints_total_limit is not None:
                                        checkpoints = os.listdir(args.output_dir)
                                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                                        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                                        if len(checkpoints) >= args.checkpoints_total_limit:
                                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                            removing_checkpoints = checkpoints[0:num_to_remove]

                                            logger.info(
                                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                            )
                                            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                            for removing_checkpoint in removing_checkpoints:
                                                removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                                shutil.rmtree(removing_checkpoint)

                                    gc.collect()
                                    torch.cuda.empty_cache()
                                    torch.cuda.ipc_collect()
                                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                                    accelerator.save_state(save_path)
                                    logger.info(f"Saved state to {save_path}")


                        
                        optimizer.zero_grad(set_to_none=True)
                        optimizer_D.zero_grad(set_to_none=True)

                        if accelerator.sync_gradients:
                            try:
                                log_dict["train/d_grad_norm"] = float(d_grad_norm)
                            except Exception:
                                pass
                            accelerator.log(log_dict, step=global_step)
            else:
                with accelerator.accumulate(student):
                    # NOTE: Base FM branch
                    with torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=accelerator.device):
                        u_student = student(
                            x=noisy_latents,
                            t=model_t,
                            cap_feats=prompt_embeds,
                        )[0]
                    # NOTE: from flow matching literature conventions
                    z = latents
                    # NOTE: target conditional vectorfield u_t = z - eps
                    u_t = z - eps
                    weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                    weighting = broadcast_t_like_x(weighting, u_student)
                    FM_loss = custom_mse_loss(u_student.float(), u_t.float(), weighting.float())
                    loss = FM_loss.mean()
                    avg_loss = accelerator.gather(loss.repeat(bsz)).mean()
                    train_loss += avg_loss.item() / args.gradient_accumulation_steps

                    optimizer.zero_grad(set_to_none=True)
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        if not args.use_deepspeed and not args.use_fsdp:
                            trainable_params_grads = [p.grad for p in trainable_params if p.grad is not None]
                            trainable_params_total_norm = torch.norm(
                                torch.stack([torch.norm(g.detach(), 2) for g in trainable_params_grads]), 2
                            )
                            max_grad_norm = linear_decay(
                                args.max_grad_norm * args.initial_grad_norm_ratio,
                                args.max_grad_norm,
                                args.abnormal_norm_clip_start,
                                global_step,
                            )
                            if trainable_params_total_norm / max_grad_norm > 5 and global_step > args.abnormal_norm_clip_start:
                                actual_max_grad_norm = max_grad_norm / min((trainable_params_total_norm / max_grad_norm), 10)
                            else:
                                actual_max_grad_norm = max_grad_norm
                        else:
                            actual_max_grad_norm = args.max_grad_norm

                        norm_sum = accelerator.clip_grad_norm_(trainable_params, actual_max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    if accelerator.sync_gradients:
                        if args.use_ema:
                            ema_student.step(student.parameters())

                        progress_bar.update(1)
                        global_step += 1

                train_loss = 0.0
                if accelerator.is_main_process:
                    print(f"[DEBUG] global_step={global_step}, checkpoint_every={args.checkpoint_every}")
                if global_step % args.checkpoint_every == 0 and global_step > 0:
                    if args.use_deepspeed or args.use_fsdp or accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        print(f"[DEBUG] SAVING CHECKPOINT at global_step={global_step}")
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        gc.collect()
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            
            if accelerator.is_main_process and args.sample_prompts is not None and global_step % args.sample_every == 0 and phase == "G":
                if args.use_ema:
                    # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                    ema_student.store(student.parameters())
                    ema_student.copy_to(student.parameters())
                
                if args.use_ladd:
                    log_LADD_inference(vae, text_encoder, tokenizer, student, args, accelerator, weight_dtype, global_step)
                    
                    if global_step == 100:
                        log_teacher_LADD_inference(vae, text_encoder, tokenizer, teacher, args, accelerator, weight_dtype, global_step)
                    #log_inference(
                    #    vae,
                    #    text_encoder,
                    #    tokenizer,
                    #    # REVERT ASAP
                    #    teacher,# student,
                    #    args,
                    #    accelerator,
                    #    weight_dtype,
                    #    global_step,
                    #)
                    
                else:
                    log_inference(
                        vae,
                        text_encoder,
                        tokenizer,
                        student,
                        args,
                        accelerator,
                        weight_dtype,
                        global_step,
                    )

                if args.use_ema:
                    # Switch back to the original student parameters.
                    ema_student.restore(student.parameters())
            
            # NOTE: Phase aware logging
            if args.use_ladd:
                if current_phase == "G":
                    logs = {"g_step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                else:
                    logs = {"d_step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            else:
                logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}

            progress_bar.set_postfix(**logs)
            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if args.use_deepspeed or args.use_fsdp or accelerator.is_main_process:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        accelerator.save_state(save_path)
        logger.info(f"Saved state to {save_path}")

    accelerator.end_training()


if __name__ == "__main__":
    main()