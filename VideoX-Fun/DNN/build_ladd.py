# videox_fun/models/build_ladd.py
import os
from typing import Optional, Dict, Any

import torch

from videox_fun.models import ZImageTransformer2DModel


def filter_kwargs(cls, kwargs):
    import inspect
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {"self", "cls"}
    return {k: v for k, v in kwargs.items() if k in valid_params}


def copy_teacher_to_student(teacher: ZImageTransformer2DModel,
                            student: ZImageTransformer2DModel) -> None:
    # shared modules
    student.all_x_embedder.load_state_dict(teacher.all_x_embedder.state_dict())
    student.all_final_layer.load_state_dict(teacher.all_final_layer.state_dict())

    student.noise_refiner.load_state_dict(teacher.noise_refiner.state_dict())
    student.context_refiner.load_state_dict(teacher.context_refiner.state_dict())

    student.t_embedder.load_state_dict(teacher.t_embedder.state_dict())
    student.cap_embedder.load_state_dict(teacher.cap_embedder.state_dict())

    student.x_pad_token.data.copy_(teacher.x_pad_token.data)
    student.cap_pad_token.data.copy_(teacher.cap_pad_token.data)

    # layer subsampling
    t_layers = len(teacher.layers)
    s_layers = len(student.layers)

    layer_map = [
        round(i * (t_layers - 1) / (s_layers - 1))
        for i in range(s_layers)
    ]

    for s_idx, t_idx in enumerate(layer_map):
        student.layers[s_idx].load_state_dict(
            teacher.layers[t_idx].state_dict()
        )


def build_teacher(pretrained_model_name_or_path: str,
                  torch_dtype: torch.dtype = torch.bfloat16) -> ZImageTransformer2DModel:
    teacher = ZImageTransformer2DModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=torch_dtype,
    )
    return teacher


def build_student_from_teacher(teacher: ZImageTransformer2DModel,
                               student_n_layers: int = 6,
                               torch_dtype: torch.dtype = torch.bfloat16) -> ZImageTransformer2DModel:
    teacher_cfg = dict(teacher.config)
    student_cfg = dict(teacher_cfg)
    student_cfg["n_layers"] = student_n_layers
    student_cfg = filter_kwargs(ZImageTransformer2DModel, student_cfg)

    student = ZImageTransformer2DModel(**student_cfg)
    student = student.to(dtype=torch_dtype)

    copy_teacher_to_student(teacher, student)
    return student


def load_student_checkpoint(student: ZImageTransformer2DModel,
                            transformer_path: Optional[str]) -> ZImageTransformer2DModel:
    if transformer_path is None:
        return student

    if transformer_path.endswith("safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(transformer_path)
    else:
        state_dict = torch.load(transformer_path, map_location="cpu")

    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
    missing, unexpected = student.load_state_dict(state_dict, strict=False)

    if len(unexpected) != 0:
        raise ValueError(f"Unexpected keys when loading student checkpoint: {unexpected}")

    print(f"Loaded student checkpoint. Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    return student


def mark_trainable_modules(student: ZImageTransformer2DModel,
                           trainable_modules,
                           trainable_modules_low_learning_rate):
    student.train()

    # first freeze all
    student.requires_grad_(False)

    if not trainable_modules and not trainable_modules_low_learning_rate:
        for p in student.parameters():
            p.requires_grad = True
        return student

    for name, param in student.named_parameters():
        for module_name in (trainable_modules or []) + (trainable_modules_low_learning_rate or []):
            if module_name in name:
                param.requires_grad = True
                break

    return student


def build_disc(pretrained_model_name_or_path: str,
               disc_type: str = "conv",
               **kwargs):
    """
    disc_type
      - 'conv'
    """
    if disc_type == "conv":
        from videox_fun.models.discriminator import LatentDiscriminator
        return LatentDiscriminator(**kwargs)

    raise ValueError(f"Unknown disc_type: {disc_type}")


def build_ladd_models(pretrained_model_name_or_path: str,
                      student_n_layers: int = 6,
                      transformer_path: Optional[str] = None,
                      torch_dtype: torch.dtype = torch.bfloat16,
                      trainable_modules=None,
                      trainable_modules_low_learning_rate=None):
    teacher = build_teacher(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        torch_dtype=torch_dtype,
    )

    student = build_student_from_teacher(
        teacher=teacher,
        student_n_layers=student_n_layers,
        torch_dtype=torch_dtype,
    )

    student = load_student_checkpoint(student, transformer_path)
    student = mark_trainable_modules(
        student,
        trainable_modules=trainable_modules,
        trainable_modules_low_learning_rate=trainable_modules_low_learning_rate,
    )

    teacher.requires_grad_(False)
    teacher.eval()

    return teacher, student