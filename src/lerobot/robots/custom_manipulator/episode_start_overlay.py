from types import SimpleNamespace

import numpy as np
import rerun as rr

from lerobot.datasets.video_utils import _default_decoder_cache, decode_video_frames


def _to_rgb_uint8(image):
    image = image.detach().cpu().numpy() if hasattr(image, "detach") else np.asarray(image)
    if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if np.issubdtype(image.dtype, np.floating):
        image = image * 255.0 if image.size and float(image.max()) <= 1.0 else image
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image[..., :3] if image.ndim == 3 and image.shape[-1] > 3 else image


def _resize_like(image, shape):
    if image.shape[:2] == shape[:2]:
        return image
    y = np.linspace(0, image.shape[0] - 1, shape[0]).astype(int)
    x = np.linspace(0, image.shape[1] - 1, shape[1]).astype(int)
    return image[y][:, x]
def _load_episode_start_frame(dataset, dataset_camera_key, episode):
    episode_value = lambda key: episode[key][0] if isinstance(episode[key], (list, tuple)) else episode[key]
    video_backend = getattr(dataset, "video_backend", None)
    if video_backend is None:
        video_backend = getattr(dataset, "_video_backend", None)
    if video_backend is None and getattr(dataset, "reader", None) is not None:
        video_backend = getattr(dataset.reader, "_video_backend", None)

    frame = decode_video_frames(
        dataset.root
        / dataset.meta.video_path.format(
            video_key=dataset_camera_key,
            chunk_index=int(episode_value(f"videos/{dataset_camera_key}/chunk_index")),
            file_index=int(episode_value(f"videos/{dataset_camera_key}/file_index")),
        ),
        [float(episode_value(f"videos/{dataset_camera_key}/from_timestamp"))],
        dataset.tolerance_s,
        video_backend,
    ).squeeze(0)
    return _to_rgb_uint8(frame).astype(np.float32)


def make_initial_state_composite(dataset, camera_key, episodes):
    if not episodes:
        return None
    dataset_camera_key = camera_key if camera_key.startswith("observation.images.") else f"observation.images.{camera_key}"
    frames = np.stack([_load_episode_start_frame(dataset, dataset_camera_key, episode) for episode in episodes])
    background = np.percentile(frames, 90.0, axis=0).astype(np.float32)
    darkness = np.clip(background[None] - frames, 0.0, 255.0)
    evidence = darkness.mean(axis=-1)
    best = np.argmax(evidence, axis=0)
    strength = np.take_along_axis(evidence, best[None], axis=0)[0]
    foreground = np.take_along_axis(frames, best[None, ..., None], axis=0)[0]
    alpha = np.clip(2.5 * (strength - 12.0) / 20.0, 0.0, 1.0)[..., None]
    return np.clip((1.0 - alpha) * background + alpha * foreground, 0.0, 255.0).astype(np.uint8)


def make_episode_start_overlay(dataset, camera_key="left_rgb", alpha=0.5, path="episode_start_overlay"):
    camera_keys = [key.removeprefix("observation.images.") for key in (dataset.meta.camera_keys or [])]
    camera_key = camera_key if camera_key in camera_keys else (camera_keys[0] if camera_keys else None)
    if camera_key is None:
        return None

    episodes = list(dataset.meta.episodes) if dataset.meta.episodes is not None else []
    composite = make_initial_state_composite(dataset, camera_key, episodes)
    base_path = f"{path}/{camera_key}"

    def show(observation):
        nonlocal composite
        if camera_key not in observation:
            return
        live = _to_rgb_uint8(observation[camera_key])
        # rr.log(f"{base_path}/live", rr.Image(live))
        if composite is None:
            return
        composite_for_live = _resize_like(composite, live.shape)
        rr.log(f"{base_path}/initial_state_composite", rr.Image(composite_for_live))
        rr.log(
            f"{base_path}/overlay",
            rr.Image(np.clip(alpha * composite_for_live + (1.0 - alpha) * live, 0.0, 255.0).astype(np.uint8)),
        )

    def on_episode_saved(dataset):
        nonlocal composite
        episodes.append(dataset.meta.latest_episode)
        _default_decoder_cache.clear()
        composite = make_initial_state_composite(dataset, camera_key, episodes)

    return SimpleNamespace(show=show, on_episode_saved=on_episode_saved, close=lambda: None)
