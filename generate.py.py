import os
import sys
import json

import torch
from diffusers import FlowMatchEulerDiscreteScheduler

current_file_path = os.path.abspath(__file__)
project_roots = [
    os.path.dirname(current_file_path),
    os.path.dirname(os.path.dirname(current_file_path)),
    os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))),
]
for project_root in project_roots:
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from videox_fun.dist import set_multi_gpus_devices, shard_model
from videox_fun.models import (
    AutoencoderKL,
    AutoTokenizer,
    Qwen3ForCausalLM,
    ZImageTransformer2DModel,
)
from videox_fun.pipeline import ZImagePipeline
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from videox_fun.utils.fp8_optimization import (
    convert_model_weight_to_float8,
    convert_weight_dtype_wrapper,
)
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora


def load_prompts(jsonl_path, max_samples=None):
    prompts = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            data = json.loads(line)
            prompt = data.get("prompt", data.get("text"))
            if prompt is None:
                raise KeyError(f"No 'prompt' or 'text' field in line {i}")
            prompts.append(prompt)
    return prompts


# =========================================================
# CONFIG
# =========================================================

# GPU memory mode:
# ["model_full_load", "model_full_load_and_qfloat8",
#  "model_cpu_offload", "model_cpu_offload_and_qfloat8",
#  "sequential_cpu_offload"]
GPU_memory_mode = "model_full_load"

# Multi-GPU config
ulysses_degree = 1
ring_degree = 1
fsdp_dit = False
fsdp_text_encoder = False
compile_dit = False

# Model path
model_name = "/root/Grace/VideoX-Fun/models/Z-Image"

# Sampler: "Flow", "Flow_Unipc", "Flow_DPM++"
sampler_name = "Flow"

# Optional checkpoints / lora
transformer_path = None
vae_path = None
lora_path = None
lora_weight = 0.55

# Data config
DATA_ROOT = "/root/Grace/VideoX-Fun/datasets/overfit"
JSONL_PATH = os.path.join(DATA_ROOT, "overfit.jsonl")
SAVE_PATH = os.path.join(DATA_ROOT, "imgs")
MAX_SAMPLES = 8  # set to None to use all prompts in the jsonl

# Generation params
sample_size = [512, 512]
batch_size = 64
weight_dtype = torch.bfloat16  
guidance_scale = 4.0
seed = 43
num_inference_steps = 40

print("\n===== CONFIG =====")
print(f"JSONL_PATH           : {JSONL_PATH}")
print(f"SAVE_PATH            : {SAVE_PATH}")
print(f"MAX_SAMPLES          : {MAX_SAMPLES}")
print(f"batch_size           : {batch_size}")
print(f"sample_size          : {sample_size}")
print(f"guidance_scale       : {guidance_scale}")
print(f"num_inference_steps  : {num_inference_steps}")
print("==================\n")

if not os.path.exists(JSONL_PATH):
    raise FileNotFoundError(f"JSONL file not found: {JSONL_PATH}")

os.makedirs(SAVE_PATH, exist_ok=True)

# =========================================================
# DEVICE
# =========================================================

device = set_multi_gpus_devices(ulysses_degree, ring_degree)

# =========================================================
# LOAD MODELS
# =========================================================

transformer = ZImageTransformer2DModel.from_pretrained(
    model_name,
    subfolder="transformer",
    low_cpu_mem_usage=True,
    torch_dtype=weight_dtype,
).to(weight_dtype)

if transformer_path is not None:
    print(f"Loading transformer from checkpoint: {transformer_path}")
    if transformer_path.endswith("safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(transformer_path)
    else:
        state_dict = torch.load(transformer_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
    m, u = transformer.load_state_dict(state_dict, strict=False)
    print(f"Transformer missing keys: {len(m)}, unexpected keys: {len(u)}")

vae = AutoencoderKL.from_pretrained(
    model_name,
    subfolder="vae",
).to(weight_dtype)

if vae_path is not None:
    print(f"Loading VAE from checkpoint: {vae_path}")
    if vae_path.endswith("safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(vae_path)
    else:
        state_dict = torch.load(vae_path, map_location="cpu")
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
    m, u = vae.load_state_dict(state_dict, strict=False)
    print(f"VAE missing keys: {len(m)}, unexpected keys: {len(u)}")

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    subfolder="tokenizer",
)

text_encoder = Qwen3ForCausalLM.from_pretrained(
    model_name,
    subfolder="text_encoder",
    torch_dtype=weight_dtype,
    low_cpu_mem_usage=True,
)

Chosen_Scheduler = {
    "Flow": FlowMatchEulerDiscreteScheduler,
    "Flow_Unipc": FlowUniPCMultistepScheduler,
    "Flow_DPM++": FlowDPMSolverMultistepScheduler,
}[sampler_name]

scheduler = Chosen_Scheduler.from_pretrained(
    model_name,
    subfolder="scheduler",
)

pipeline = ZImagePipeline(
    vae=vae,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    transformer=transformer,
    scheduler=scheduler,
)


if ulysses_degree > 1 or ring_degree > 1:
    from functools import partial

    transformer.enable_multi_gpus_inference()

    if fsdp_dit:
        shard_fn = partial(
            shard_model,
            device_id=device,
            param_dtype=weight_dtype,
            module_to_wrapper=list(transformer.layers),
        )
        pipeline.transformer = shard_fn(pipeline.transformer)
        print("Enabled FSDP for DIT")

    if fsdp_text_encoder:
        shard_fn = partial(
            shard_model,
            device_id=device,
            param_dtype=weight_dtype,
            module_to_wrapper=list(text_encoder.model.layers),
        )
        text_encoder = shard_fn(text_encoder)
        print("Enabled FSDP for text encoder")

if compile_dit:
    # Only keep this if your transformer actually has transformer_blocks
    for i in range(len(pipeline.transformer.transformer_blocks)):
        pipeline.transformer.transformer_blocks[i] = torch.compile(
            pipeline.transformer.transformer_blocks[i]
        )
    print("Enabled torch.compile")

if GPU_memory_mode == "sequential_cpu_offload":
    pipeline.enable_sequential_cpu_offload(device=device)
elif GPU_memory_mode == "model_cpu_offload_and_qfloat8":
    convert_model_weight_to_float8(
        transformer,
        exclude_module_name=["x_pad_token", "cap_pad_token"],
        device=device,
    )
    convert_weight_dtype_wrapper(transformer, weight_dtype)
    pipeline.enable_model_cpu_offload(device=device)
elif GPU_memory_mode == "model_cpu_offload":
    pipeline.enable_model_cpu_offload(device=device)
elif GPU_memory_mode == "model_full_load_and_qfloat8":
    convert_model_weight_to_float8(
        transformer,
        exclude_module_name=["x_pad_token", "cap_pad_token"],
        device=device,
    )
    convert_weight_dtype_wrapper(transformer, weight_dtype)
    pipeline.to(device=device)
else:
    pipeline.to(device=device)

if lora_path is not None:
    pipeline = merge_lora(
        pipeline,
        lora_path,
        lora_weight,
        device=device,
        dtype=weight_dtype,
    )

# =========================================================
# GENERATE
# =========================================================

prompts = load_prompts(JSONL_PATH, max_samples=MAX_SAMPLES)
print(f"Loaded {len(prompts)} prompts from {JSONL_PATH}")

with torch.no_grad():
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start:start + batch_size]
        end = start + len(batch_prompts) - 1
        print(f"Generating batch {start} -> {end}")

        try:
            images = pipeline(
                prompt=batch_prompts,
                height=sample_size[0],
                width=sample_size[1],
                generator=torch.Generator(device=device).manual_seed(seed + start),
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
            ).images

            for i, image in enumerate(images):
                global_idx = start + i
                filename = f"{global_idx:08d}.png"
                out_path = os.path.join(SAVE_PATH, filename)

                if os.path.exists(out_path):
                    continue

                image.save(out_path)

        except Exception as e:
            print(f"[ERROR] batch start={start} failed: {e}")
            continue

if lora_path is not None:
    pipeline = unmerge_lora(
        pipeline,
        lora_path,
        lora_weight,
        device=device,
        dtype=weight_dtype,
    )

print("\nDone.")
print(f"Images saved to: {SAVE_PATH}")