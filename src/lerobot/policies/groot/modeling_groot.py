#!/usr/bin/env python

# Copyright 2024 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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

"""
Groot Policy Wrapper for LeRobot Integration

Minimal integration that delegates to Isaac-GR00T components where possible
without porting their code. The intent is to:

- Download and load the pretrained GR00T model via GR00TN15.from_pretrained
- Optionally align action horizon similar to gr00t_finetune.py
- Expose predict_action via GR00T model.get_action
- Provide a training forward that can call the GR00T model forward if batch
  structure matches.

Notes:
- Dataset loading and full training orchestration is handled by Isaac-GR00T
  TrainRunner in their codebase. If you want to invoke that flow end-to-end
  from LeRobot, see `GrootPolicy.finetune_with_groot_runner` below.
"""

import builtins
import os
from collections import deque
from pathlib import Path
from typing import TypeVar
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.groot.configuration_groot import GrootConfig
from lerobot.policies.groot.groot_n1 import GR00TN15
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES

T = TypeVar("T", bound="GrootPolicy")


class _GradReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, lambd: float) -> Tensor:  # type: ignore[override]
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor):  # type: ignore[override]
        return -ctx.lambd * grad_output, None


def _grad_reverse(x: Tensor, lambd: float) -> Tensor:
    return _GradReverseFn.apply(x, float(lambd))


class GrootPolicy(PreTrainedPolicy):
    """Wrapper around external Groot model for LeRobot integration."""

    name = "groot"
    config_class = GrootConfig

    def __init__(self, config: GrootConfig, **kwargs):
        """Initialize Groot policy wrapper."""
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Initialize GR00T model using ported components
        self._groot_model = self._create_groot_model()

        # Optional DANN head (domain classifier + GRL)
        self._dann_step = 0
        self._domain_head: nn.Module | None = None
        if getattr(self.config, "use_dann", False):
            self._domain_head = self._build_domain_head()

        self.reset()

    def _create_groot_model(self):
        """Create and initialize the GR00T model using Isaac-GR00T API.

        This is only called when creating a NEW policy (not when loading from checkpoint).

        Steps (delegating to Isaac-GR00T):
        1) Download and load pretrained model via GR00TN15.from_pretrained
        2) Align action horizon with data_config if provided
        """
        # Handle Flash Attention compatibility issues
        self._handle_flash_attention_compatibility()

        model = GR00TN15.from_pretrained(
            pretrained_model_name_or_path=self.config.base_model_path,
            tune_llm=self.config.tune_llm,
            tune_visual=self.config.tune_visual,
            tune_projector=self.config.tune_projector,
            tune_diffusion_model=self.config.tune_diffusion_model,
        )

        model.compute_dtype = "bfloat16" if self.config.use_bf16 else model.compute_dtype
        model.config.compute_dtype = model.compute_dtype

        return model

    def reset(self):
        """Reset policy state when environment resets."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: GrootConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = True,
        **kwargs,
    ) -> T:
        """Load Groot policy from pretrained model.

        Handles two cases:
        1. Base GR00T models (e.g., 'nvidia/GR00T-N1.5-3B') - loads the raw model
        2. Fine-tuned LeRobot checkpoints - loads config and weights from safetensors

        Args:
            pretrained_name_or_path: Path to the GR00T model or fine-tuned checkpoint
            config: Optional GrootConfig. If None, loads from checkpoint or creates default
            force_download: Force download even if cached
            resume_download: Resume interrupted download
            proxies: Proxy settings
            token: HuggingFace authentication token
            cache_dir: Cache directory path
            local_files_only: Only use local files
            revision: Specific model revision
            strict: Strict state dict loading
            **kwargs: Additional arguments (passed to config)

        Returns:
            Initialized GrootPolicy instance with loaded model
        """
        from huggingface_hub import hf_hub_download
        from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
        from huggingface_hub.errors import HfHubHTTPError

        print(
            "The Groot policy is a wrapper around Nvidia's GR00T N1.5 model.\n"
            f"Loading pretrained model from: {pretrained_name_or_path}"
        )

        model_id = str(pretrained_name_or_path)
        is_finetuned_checkpoint = False

        # Check if this is a fine-tuned LeRobot checkpoint (has model.safetensors)
        try:
            if os.path.isdir(model_id):
                is_finetuned_checkpoint = os.path.exists(os.path.join(model_id, SAFETENSORS_SINGLE_FILE))
            else:
                # Try to download the safetensors file to check if it exists
                try:
                    hf_hub_download(
                        repo_id=model_id,
                        filename=SAFETENSORS_SINGLE_FILE,
                        revision=revision,
                        cache_dir=cache_dir,
                        force_download=False,  # Just check, don't force download
                        proxies=proxies,
                        token=token,
                        local_files_only=local_files_only,
                    )
                    is_finetuned_checkpoint = True
                except HfHubHTTPError:
                    is_finetuned_checkpoint = False
        except Exception:
            is_finetuned_checkpoint = False

        if is_finetuned_checkpoint:
            # This is a fine-tuned LeRobot checkpoint - use parent class loading
            print("Detected fine-tuned LeRobot checkpoint, loading with state dict...")
            return super().from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                config=config,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                strict=strict,
                **kwargs,
            )

        # This is a base GR00T model - load it fresh
        print("Detected base GR00T model, loading from HuggingFace...")

        if config is None:
            # Create default config with the pretrained path
            config = GrootConfig(base_model_path=str(pretrained_name_or_path))

            # Add minimal visual feature required for validation
            # validate_features() will automatically add state and action features
            # These are placeholders - actual robot features come from the preprocessor
            if not config.input_features:
                config.input_features = {
                    f"{OBS_IMAGES}.camera": PolicyFeature(
                        type=FeatureType.VISUAL,
                        shape=(3, 224, 224),  # Default image size from config
                    ),
                }
        else:
            # Override the base_model_path with the provided path
            config.base_model_path = str(pretrained_name_or_path)

        # Pass through any additional config overrides from kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        # Create a fresh policy instance - this will automatically load the GR00T model
        # in __init__ via _create_groot_model()
        policy = cls(config)

        policy.eval()
        return policy

    def get_optim_params(self) -> dict:
        return self.parameters()

    def _build_domain_head(self) -> nn.Module:
        # Best-effort: infer backbone projection dim (defaults to 1536).
        feat_dim = 1536
        try:
            eagle_linear = getattr(self._groot_model.backbone, "eagle_linear", None)
            if isinstance(eagle_linear, nn.Linear):
                feat_dim = int(eagle_linear.out_features)
        except Exception:
            feat_dim = 1536

        hidden = int(getattr(self.config, "dann_hidden_dim", 256))
        p = float(getattr(self.config, "dann_dropout", 0.0))
        out_dim = int(getattr(self.config, "dann_num_domains", 2))
        return nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p) if p > 0 else nn.Identity(),
            nn.Linear(hidden, out_dim),
        )

    def _masked_mean_pool(self, feats: Tensor, mask: Tensor | None) -> Tensor:
        # feats: (B, T, D), mask: (B, T)
        if mask is None:
            return feats.mean(dim=1)
        if mask.dtype != torch.float32:
            mask_f = mask.to(dtype=torch.float32)
        else:
            mask_f = mask
        if mask_f.dim() == 3 and mask_f.shape[-1] == 1:
            mask_f = mask_f.squeeze(-1)
        mask_f = mask_f.clamp(min=0.0, max=1.0)
        denom = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (feats * mask_f.unsqueeze(-1)).sum(dim=1) / denom
        return pooled

    def _dann_lambda(self) -> float:
        # DANN schedule: lambda(p)=2/(1+exp(-gamma*p)) - 1, scaled by lambda_max
        total = int(getattr(self.config, "dann_total_steps", 100_000))
        gamma = float(getattr(self.config, "dann_gamma", 10.0))
        lam_max = float(getattr(self.config, "dann_lambda_max", 1.0))
        if total <= 0:
            return lam_max
        p = float(min(max(self._dann_step / float(total), 0.0), 1.0))
        lam = 2.0 / (1.0 + math.exp(-gamma * p)) - 1.0
        return lam_max * float(lam)

    def _select_batch(self, data: dict, idx: Tensor) -> dict:
        # Select along batch dimension for any tensors with matching leading dim.
        out: dict = {}
        for k, v in data.items():
            if isinstance(v, torch.Tensor) and v.shape[0] == idx.shape[0]:
                # idx is assumed to be a 1D index tensor over batch dimension.
                out[k] = v.index_select(0, idx)
            else:
                out[k] = v
        return out

    def forward(self, batch: dict[str, Tensor], **kwargs) -> tuple[Tensor, dict]:
        """Training forward pass.

        Delegates to Isaac-GR00T model.forward when inputs are compatible.
        """
        # Build a clean input dict for GR00T: keep only tensors GR00T consumes
        allowed_base = {"state", "state_mask", "action", "action_mask", "embodiment_id"}
        groot_inputs = {
            k: v
            for k, v in batch.items()
            if (k in allowed_base or k.startswith("eagle_")) and not (k.startswith("next.") or k == "info")
        }

        # Get device from model parameters
        device = next(self.parameters()).device

        use_dann = bool(getattr(self.config, "use_dann", False))
        domain_key = str(getattr(self.config, "dann_domain_key", "domain_id"))
        domain_id = batch.get(domain_key)
        if domain_id is None and domain_key != "domain_id":
            domain_id = batch.get("domain_id")

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self.config.use_bf16):
            if not use_dann or domain_id is None or self._domain_head is None:
                outputs = self._groot_model.forward(groot_inputs)
                loss = outputs.get("loss")
                loss_dict = {"loss": float(loss.item())}
                return loss, loss_dict

            # DANN path: compute task loss on source only + domain loss on all.
            if domain_id.dim() > 1:
                domain_labels = domain_id.view(-1).to(device=device, dtype=torch.long)
            else:
                domain_labels = domain_id.to(device=device, dtype=torch.long)

            # Forward backbone once on full batch
            backbone_inputs = self._groot_model.backbone.prepare_input(groot_inputs)
            backbone_outputs = self._groot_model.backbone(backbone_inputs)

            # Domain classifier on features AFTER the VLM→DiT bridge.
            # Rationale: GR00T finetuning often freezes the VLM; the bridge transformer lives in
            # FlowmatchingActionHead.process_backbone_output() (vLLN + vl_self_attention).
            bridge_outputs = {k: v for k, v in backbone_outputs.items()}
            bridge_outputs = self._groot_model.action_head.process_backbone_output(bridge_outputs)
            feats = bridge_outputs.get("backbone_features")
            mask = bridge_outputs.get("backbone_attention_mask")
            pooled = self._masked_mean_pool(feats, mask)

            if self.training:
                self._dann_step += 1
            lambd = self._dann_lambda()
            rev = _grad_reverse(pooled, lambd)
            logits = self._domain_head(rev)
            domain_loss = F.cross_entropy(logits, domain_labels)

            # Task loss computed only on source domain samples
            task_domain_val = int(getattr(self.config, "dann_task_domain", 0))
            source_mask = domain_labels == task_domain_val
            if bool(source_mask.any()):
                src_idx = source_mask.nonzero(as_tuple=False).view(-1)

                # Prepare action inputs and select source entries
                action_inputs = self._groot_model.action_head.prepare_input(groot_inputs)
                src_backbone = {
                    k: (
                        v.index_select(0, src_idx)
                        if isinstance(v, torch.Tensor) and v.shape[0] == feats.shape[0]
                        else v
                    )
                    for k, v in backbone_outputs.items()
                }
                src_action = {
                    k: (
                        v.index_select(0, src_idx)
                        if isinstance(v, torch.Tensor) and v.shape[0] == feats.shape[0]
                        else v
                    )
                    for k, v in action_inputs.items()
                }

                task_out = self._groot_model.action_head(src_backbone, src_action)
                task_loss = task_out.get("loss")
            else:
                task_loss = torch.zeros((), device=device, dtype=torch.float32)

            loss = task_loss + domain_loss

        loss_dict = {
            "loss": float(loss.item()),
            "task_loss": float(task_loss.item()),
            "domain_loss": float(domain_loss.item()),
            "dann_lambda": float(lambd),
            "source_frac": float((domain_labels == int(getattr(self.config, "dann_task_domain", 0))).float().mean().item()),
        }

        return loss, loss_dict

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions for inference by delegating to Isaac-GR00T.

        Returns a tensor of shape (B, n_action_steps, action_dim).
        """
        self.eval()

        # Build a clean input dict for GR00T: keep only tensors GR00T consumes
        # Preprocessing is handled by the processor pipeline, so we just filter the batch
        # NOTE: During inference, we should NOT pass action/action_mask (that's what we're predicting)
        allowed_base = {"state", "state_mask", "embodiment_id"}
        groot_inputs = {
            k: v
            for k, v in batch.items()
            if (k in allowed_base or k.startswith("eagle_")) and not (k.startswith("next.") or k == "info")
        }

        # Get device from model parameters
        device = next(self.parameters()).device

        # Use bf16 autocast for inference to keep memory low and match backbone dtype
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self.config.use_bf16):
            outputs = self._groot_model.get_action(groot_inputs)

        actions = outputs.get("action_pred")

        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :, :original_action_dim]

        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select single action from action queue."""
        self.eval()

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    # -------------------------
    # Internal helpers
    # -------------------------
    def _handle_flash_attention_compatibility(self) -> None:
        """Handle Flash Attention compatibility issues by setting environment variables.

        This addresses the common 'undefined symbol' error that occurs when Flash Attention
        is compiled against a different PyTorch version than what's currently installed.
        """

        # Set environment variables to handle Flash Attention compatibility
        # These help with symbol resolution issues
        os.environ.setdefault("FLASH_ATTENTION_FORCE_BUILD", "0")
        os.environ.setdefault("FLASH_ATTENTION_SKIP_CUDA_BUILD", "0")

        # Try to import flash_attn and handle failures gracefully
        try:
            import flash_attn

            print(f"[GROOT] Flash Attention version: {flash_attn.__version__}")
        except ImportError as e:
            print(f"[GROOT] Flash Attention not available: {e}")
            print("[GROOT] Will use fallback attention mechanism")
        except Exception as e:
            if "undefined symbol" in str(e):
                print(f"[GROOT] Flash Attention compatibility issue detected: {e}")
                print("[GROOT] This is likely due to PyTorch/Flash Attention version mismatch")
                print("[GROOT] Consider reinstalling Flash Attention with compatible version:")
                print("  pip uninstall flash-attn")
                print("  pip install --no-build-isolation flash-attn==2.6.3")
                print("[GROOT] Continuing with fallback attention mechanism")
            else:
                print(f"[GROOT] Flash Attention error: {e}")
                print("[GROOT] Continuing with fallback attention mechanism")
