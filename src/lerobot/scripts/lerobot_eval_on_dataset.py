#!/usr/bin/env python

from __future__ import annotations

import os

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.utils import cycle
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device


# Uncomment `dump_batch_images(batch)` inside the loop to save a few denormalized images for debugging.
def dump_batch_images(batch, out_dir: str = "debug_images", n: int = 4) -> None:
    from pathlib import Path; from torchvision.utils import save_image
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]; std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    for k, v in batch.items():
        if not (isinstance(v, torch.Tensor) and v.ndim == 4 and v.shape[1] == 3): continue
        x = (v[:n].detach().float().cpu() * std + mean).clamp(0, 1)
        for i, img in enumerate(x): save_image(img, out / f"{k.replace('.', '_')}_{i}.png")
    exit(0)


# Env toggles:
# - `LEROBOT_PRETRAINED_POLICY` (HF repo id)
# - `LEROBOT_GREEN_SCREEN` = auto|on|off
# - `LEROBOT_GREEN_SCREEN_POOL_DIR` = path to background pool (optional)
PRETRAINED_POLICY = "HSP-IIT/act-ladyplush_2"

def main() -> None:
    # Load the exact training configuration saved with the model.
    train_cfg = TrainPipelineConfig.from_pretrained(PRETRAINED_POLICY)
    replace_green_screen = False

    if not replace_green_screen:
        train_cfg.dataset.image_transforms.tfs['green_screen_replace'].weight = 0.0
        # train_cfg.dataset.image_transforms.tfs['brightness'].weight = 0.0
        # train_cfg.dataset.image_transforms.tfs['contrast'].weight = 0.0
        # train_cfg.dataset.image_transforms.tfs['saturation'].weight = 0.0
        # train_cfg.dataset.image_transforms.tfs['hue'].weight = 0.0
        # train_cfg.dataset.image_transforms.tfs['sharpness'].weight = 0.0

    if train_cfg.seed is not None:
        set_seed(train_cfg.seed)

    # Load the policy config + weights.
    policy_cfg = PreTrainedConfig.from_pretrained(PRETRAINED_POLICY)
    policy_cfg.pretrained_path = PRETRAINED_POLICY
    device = get_safe_torch_device(policy_cfg.device)
    policy_cfg.device = str(device)

    # Build dataset exactly like training (including image_transforms).
    train_cfg.policy = policy_cfg
    dataset = make_dataset(train_cfg)

    # Build policy and load the exact preprocessor used during training.
    policy = make_policy(policy_cfg, ds_meta=dataset.meta).to(device)
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=PRETRAINED_POLICY,
        preprocessor_overrides={"device_processor": {"device": device.type}},
    )

    del preprocessor.steps[0]
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=train_cfg.num_workers,
        batch_size=train_cfg.batch_size,
        shuffle=not train_cfg.dataset.streaming,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if train_cfg.num_workers > 0 else None,
    )
    dl_iter = cycle(dataloader)

    loss_sum = 0.0
    l1_sum = 0.0
    kld_sum = 0.0

    # Match `lerobot_train.py` logging: average loss over the last `log_freq` training batches.
    policy.train()
    with torch.no_grad():
        for _ in range(train_cfg.log_freq):
            batch = preprocessor(next(dl_iter))
            # dump_batch_images(batch)
            loss, loss_dict = policy.forward(batch)
            loss_sum += loss.item()
            l1_sum += float(loss_dict.get("l1_loss", 0.0))
            kld_sum += float(loss_dict.get("kld_loss", 0.0))

    print(
        f"train_loss={loss_sum / train_cfg.log_freq:.6f} "
        f"l1_loss_train={l1_sum / train_cfg.log_freq:.6f} "
        f"kld_loss_train={kld_sum / train_cfg.log_freq:.6f} "
        f"(avg over {train_cfg.log_freq} batches) "
        f"replace_green_screen={replace_green_screen}"
    )


if __name__ == "__main__":
    main()
