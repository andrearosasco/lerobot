#!/usr/bin/env python

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
# limitations under the License.
import collections
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torchvision.transforms import v2
from torchvision.transforms.v2 import (
    Transform,
    functional as F,  # noqa: N812
)


class RandomSubsetApply(Transform):
    """Apply a random subset of N transformations from a list of transformations.

    Args:
        transforms: list of transformations.
        p: represents the multinomial probabilities (with no replacement) used for sampling the transform.
            If the sum of the weights is not 1, they will be normalized. If ``None`` (default), all transforms
            have the same probability.
        n_subset: number of transformations to apply. If ``None``, all transforms are applied.
            Must be in [1, len(transforms)].
        random_order: apply transformations in a random order.
    """

    def __init__(
        self,
        transforms: Sequence[Callable],
        p: list[float] | None = None,
        n_subset: int | None = None,
        random_order: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(transforms, Sequence):
            raise TypeError("Argument transforms should be a sequence of callables")
        if p is None:
            p = [1] * len(transforms)
        elif len(p) != len(transforms):
            raise ValueError(
                f"Length of p doesn't match the number of transforms: {len(p)} != {len(transforms)}"
            )

        if n_subset is None:
            n_subset = len(transforms)
        elif not isinstance(n_subset, int):
            raise TypeError("n_subset should be an int or None")
        elif not (1 <= n_subset <= len(transforms)):
            raise ValueError(f"n_subset should be in the interval [1, {len(transforms)}]")

        self.transforms = transforms
        total = sum(p)
        self.p = [prob / total for prob in p]
        self.n_subset = n_subset
        self.random_order = random_order

        self.selected_transforms = None

    def forward(self, *inputs: Any) -> Any:
        needs_unpacking = len(inputs) > 1

        selected_indices = torch.multinomial(torch.tensor(self.p), self.n_subset)
        if not self.random_order:
            selected_indices = selected_indices.sort().values

        self.selected_transforms = [self.transforms[i] for i in selected_indices]

        for transform in self.selected_transforms:
            outputs = transform(*inputs)
            inputs = outputs if needs_unpacking else (outputs,)

        return outputs

    def extra_repr(self) -> str:
        return (
            f"transforms={self.transforms}, "
            f"p={self.p}, "
            f"n_subset={self.n_subset}, "
            f"random_order={self.random_order}"
        )


class SharpnessJitter(Transform):
    """Randomly change the sharpness of an image or video.

    Similar to a v2.RandomAdjustSharpness with p=1 and a sharpness_factor sampled randomly.
    While v2.RandomAdjustSharpness applies — with a given probability — a fixed sharpness_factor to an image,
    SharpnessJitter applies a random sharpness_factor each time. This is to have a more diverse set of
    augmentations as a result.

    A sharpness_factor of 0 gives a blurred image, 1 gives the original image while 2 increases the sharpness
    by a factor of 2.

    If the input is a :class:`torch.Tensor`,
    it is expected to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        sharpness: How much to jitter sharpness. sharpness_factor is chosen uniformly from
            [max(0, 1 - sharpness), 1 + sharpness] or the given
            [min, max]. Should be non negative numbers.
    """

    def __init__(self, sharpness: float | Sequence[float]) -> None:
        super().__init__()
        self.sharpness = self._check_input(sharpness)

    def _check_input(self, sharpness):
        if isinstance(sharpness, (int | float)):
            if sharpness < 0:
                raise ValueError("If sharpness is a single number, it must be non negative.")
            sharpness = [1.0 - sharpness, 1.0 + sharpness]
            sharpness[0] = max(sharpness[0], 0.0)
        elif isinstance(sharpness, collections.abc.Sequence) and len(sharpness) == 2:
            sharpness = [float(v) for v in sharpness]
        else:
            raise TypeError(f"{sharpness=} should be a single number or a sequence with length 2.")

        if not 0.0 <= sharpness[0] <= sharpness[1]:
            raise ValueError(f"sharpness values should be between (0., inf), but got {sharpness}.")

        return float(sharpness[0]), float(sharpness[1])

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        sharpness_factor = torch.empty(1).uniform_(self.sharpness[0], self.sharpness[1]).item()
        return {"sharpness_factor": sharpness_factor}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        sharpness_factor = params["sharpness_factor"]
        return self._call_kernel(F.adjust_sharpness, inpt, sharpness_factor=sharpness_factor)

import os
from pathlib import Path
from typing import Any

import torch
from torchvision.io import read_image
from torchvision.transforms import v2
from torchvision.transforms.v2 import Transform, functional as F


class GreenScreenReplace(Transform):
    """Replace green pixels with a random background sampled from a directory.

    Expected input: torch.Tensor with shape [..., 3, H, W] (uint8 or float).
    Backgrounds are read from `pool_dir` and resized to match HxW.

    Args:
        pool_dir: path to a directory containing background images.
        green_ratio: how much stronger G must be relative to max(R,B) to be considered "green".
            Rule: G > green_ratio * max(R,B)
        min_green: minimum G value (in [0,1] after conversion) to be considered for keying.
        spill: additional margin: (G - max(R,B)) > spill (in [0,1])
        extensions: allowed image extensions.
    """

    def __init__(
        self,
        pool_dir: str | os.PathLike,
        green_ratio: float = 1.25,
        min_green: float = 0.25,
        spill: float = 0.10,
        extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp"),
    ) -> None:
        super().__init__()
        self.pool_dir = Path(pool_dir)
        self.green_ratio = float(green_ratio)
        self.min_green = float(min_green)
        self.spill = float(spill)
        self.extensions = tuple(ext.lower() for ext in extensions)

        if not self.pool_dir.exists() or not self.pool_dir.is_dir():
            raise ValueError(f"{self.pool_dir=} must exist and be a directory.")

        self._bg_paths = self._list_backgrounds(self.pool_dir)
        if len(self._bg_paths) == 0:
            raise ValueError(f"No background images found under {self.pool_dir} with {self.extensions=}.")

    def _list_backgrounds(self, pool_dir: Path) -> list[Path]:
        files: list[Path] = []
        for p in pool_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in self.extensions:
                files.append(p)
        files.sort()
        return files

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        # Pick a random background index each call.
        idx = int(torch.randint(low=0, high=len(self._bg_paths), size=(1,)).item())
        return {"bg_index": idx}

    def _load_background(self, path: Path) -> torch.Tensor:
        # read_image returns uint8 tensor [C, H, W] in RGB.
        bg = read_image(str(path))  # uint8, CPU
        if bg.ndim != 3 or bg.shape[0] not in (1, 3):
            raise ValueError(f"Background image {path} has unsupported shape {tuple(bg.shape)}.")

        # Force 3 channels.
        if bg.shape[0] == 1:
            bg = bg.repeat(3, 1, 1)
        return bg

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        # We only support Tensor inputs here (matches your pipeline expectations).
        if not isinstance(inpt, torch.Tensor):
            raise TypeError(f"GreenScreenReplace expects torch.Tensor input, got {type(inpt)}")

        # Expect [..., 3, H, W]
        if inpt.ndim < 3:
            raise ValueError(f"Input must have at least 3 dims, got {inpt.ndim=}.")
        if inpt.shape[-3] != 3:
            raise ValueError(f"Expected 3 channels at dim -3, got {inpt.shape[-3]=}.")

        # Get spatial size
        _, h, w = inpt.shape[-3], inpt.shape[-2], inpt.shape[-1]

        # Load & resize background
        bg_path = self._bg_paths[params["bg_index"]]
        bg = self._load_background(bg_path)  # [3, Hb, Wb], uint8 on CPU
        bg = F.resize(bg, size=[h, w], antialias=True)

        # Match dtype/range for blending by going to float in [0,1]
        orig_dtype = inpt.dtype
        inpt_f = F.convert_image_dtype(inpt, dtype=torch.float32)
        bg_f = F.convert_image_dtype(bg, dtype=torch.float32)

        # Move bg to input device and expand to match leading dims of inpt
        bg_f = bg_f.to(device=inpt_f.device)
        # Expand to [..., 3, H, W]
        for _ in range(inpt_f.ndim - 3):
            bg_f = bg_f.unsqueeze(0)
        bg_f = bg_f.expand(*inpt_f.shape[:-3], 3, h, w)

        r = inpt_f[..., 0, :, :]
        g = inpt_f[..., 1, :, :]
        b = inpt_f[..., 2, :, :]

        max_rb = torch.maximum(r, b)

        # Chroma key mask: "green enough"
        green_mask = (g > (self.green_ratio * max_rb)) & (g > self.min_green) & ((g - max_rb) > self.spill)
        green_mask = green_mask.unsqueeze(-3).to(dtype=inpt_f.dtype)  # [..., 1, H, W] float 0/1

        out_f = inpt_f * (1.0 - green_mask) + bg_f * green_mask
        out = F.convert_image_dtype(out_f, dtype=orig_dtype)
        return out


@dataclass
class ImageTransformConfig:
    """
    For each transform, the following parameters are available:
      weight: This represents the multinomial probability (with no replacement)
            used for sampling the transform. If the sum of the weights is not 1,
            they will be normalized.
      type: The name of the class used. This is either a class available under torchvision.transforms.v2 or a
            custom transform defined here.
      kwargs: Lower & upper bound respectively used for sampling the transform's parameter
            (following uniform distribution) when it's applied.
    """

    weight: float = 1.0
    type: str = "Identity"
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageTransformsConfig:
    """
    These transforms are all using standard torchvision.transforms.v2
    You can find out how these transformations affect images here:
    https://pytorch.org/vision/0.18/auto_examples/transforms/plot_transforms_illustrations.html
    We use a custom RandomSubsetApply container to sample them.
    """

    # Set this flag to `true` to enable transforms during training
    enable: bool = False
    # This is the maximum number of transforms (sampled from these below) that will be applied to each frame.
    # It's an integer in the interval [1, number_of_available_transforms].
    max_num_transforms: int = 3
    # By default, transforms are applied in Torchvision's suggested order (shown below).
    # Set this to True to apply them in a random order.
    random_order: bool = False
    tfs: dict[str, ImageTransformConfig] = field(
        default_factory=lambda: {
            "brightness": ImageTransformConfig(
                weight=1.0,
                type="ColorJitter",
                kwargs={"brightness": (0.8, 1.2)},
            ),
            "contrast": ImageTransformConfig(
                weight=1.0,
                type="ColorJitter",
                kwargs={"contrast": (0.8, 1.2)},
            ),
            "saturation": ImageTransformConfig(
                weight=1.0,
                type="ColorJitter",
                kwargs={"saturation": (0.5, 1.5)},
            ),
            "hue": ImageTransformConfig(
                weight=1.0,
                type="ColorJitter",
                kwargs={"hue": (-0.05, 0.05)},
            ),
            "sharpness": ImageTransformConfig(
                weight=1.0,
                type="SharpnessJitter",
                kwargs={"sharpness": (0.5, 1.5)},
            ),
            "affine": ImageTransformConfig(
                weight=1.0,
                type="RandomAffine",
                kwargs={"degrees": (-5.0, 5.0), "translate": (0.05, 0.05)},
            ),
        }
    )


def make_transform_from_config(cfg: ImageTransformConfig):
    if cfg.type == "Identity":
        return v2.Identity(**cfg.kwargs)
    elif cfg.type == "ColorJitter":
        return v2.ColorJitter(**cfg.kwargs)
    elif cfg.type == "SharpnessJitter":
        return SharpnessJitter(**cfg.kwargs)
    elif cfg.type == "RandomAffine":
        return v2.RandomAffine(**cfg.kwargs)
    elif cfg.type == "GreenScreenReplace":
        return GreenScreenReplace(**cfg.kwargs)
    else:
        raise ValueError(f"Transform '{cfg.type}' is not valid.")


class ImageTransforms(Transform):
    """A class to compose image transforms based on configuration."""

    def __init__(self, cfg: ImageTransformsConfig) -> None:
        super().__init__()
        self._cfg = cfg

        self.weights = []
        self.transforms = {}
        for tf_name, tf_cfg in cfg.tfs.items():
            if tf_cfg.weight <= 0.0:
                continue

            self.transforms[tf_name] = make_transform_from_config(tf_cfg)
            self.weights.append(tf_cfg.weight)

        n_subset = min(len(self.transforms), cfg.max_num_transforms)
        if n_subset == 0 or not cfg.enable:
            self.tf = v2.Identity()
        else:
            self.tf = RandomSubsetApply(
                transforms=list(self.transforms.values()),
                p=self.weights,
                n_subset=n_subset,
                random_order=cfg.random_order,
            )

    def forward(self, *inputs: Any) -> Any:
        return self.tf(*inputs)
