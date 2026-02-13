#!/usr/bin/env python
import math
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import tqdm
from matplotlib import colormaps

from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import add_envs_task, preprocess_observation
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION
from lerobot.utils.io_utils import write_video
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device, init_logging

# ---- user knobs (keep it simple) --------------------------------------------
SUITE = "libero_object"
TASK_ID = 2
OUT = None  # e.g. "/tmp/pop_viz.mp4"
CAMERA_NAME = "agentview_image,robot0_eye_in_hand_image"
LINE_THICK = 2
HUTCHINSON_SAMPLES = 5
DENSITY_COLOR_PERCENTILES = (5, 95)
DEBUG_STEPS = 20


def poly_pixels(pts, h, w):
    xs_all, ys_all = [], []
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:], strict=False):
        n = max(abs(x1 - x0), abs(y1 - y0))
        n = min(n, 2 * max(h, w)) + 1
        xs = np.linspace(x0, x1, n).astype(np.int32)
        ys = np.linspace(y0, y1, n).astype(np.int32)
        m = (0 <= xs) & (xs < w) & (0 <= ys) & (ys < h)
        xs_all.append(xs[m])
        ys_all.append(ys[m])
    return (np.concatenate(ys_all), np.concatenate(xs_all)) if xs_all else (np.array([], int), np.array([], int))


def project(world_pts, cam_pos, cam_xmat, fovy_deg, w, h):
    r = cam_xmat.reshape(3, 3)  # camera frame in world coords (camera->world)
    pc = (world_pts - cam_pos) @ r  # world->camera for row vectors
    z = pc[:, 2]
    if z[0] < 0:
        z = -z
    z = np.clip(z, 1e-6, None)
    f = 0.5 * h / math.tan(math.radians(fovy_deg) / 2)
    u = f * (pc[:, 0] / z) + (w * 0.5)
    v = f * (pc[:, 1] / z) + (h * 0.5)
    return np.stack([(w - 1) - u, (h - 1) - v], axis=1).astype(np.int32)  # render() flips H,W


def draw_x(img, x, y, color=(255, 255, 0)):
    h, w = img.shape[:2]
    x = int(np.clip(x, 0, w - 1))
    y = int(np.clip(y, 0, h - 1))
    img[max(0, y - 2) : min(h, y + 3), max(0, x - 2) : min(w, x + 3)] = color


def logp_to_colors(logp: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(logp, DENSITY_COLOR_PERCENTILES)
    t = (logp - lo) / (hi - lo + 1e-8)
    t = np.clip(t, 0.0, 1.0)
    rgba = colormaps["viridis"](t)
    return (rgba[:, :3] * 255).astype(np.uint8)


@parser.wrap()
def main(cfg: EvalPipelineConfig):
    init_logging()
    register_third_party_plugins()
    cfg.eval.batch_size = cfg.eval.n_episodes = 1
    cfg.eval.use_async_envs = False
    cfg.env.task, cfg.env.task_ids = SUITE, [TASK_ID]
    cfg.env.camera_name = CAMERA_NAME
    cfg.policy.device='cuda:1'

    device = get_safe_torch_device(cfg.policy.device, log=True)
    set_seed(cfg.seed)

    envs = make_env(cfg.env, n_envs=1, use_async_envs=False, lazy=True, trust_remote_code=cfg.trust_remote_code)
    env = envs[SUITE][TASK_ID]
    env = env() if callable(env) else env

    policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env, rename_map=cfg.rename_map).eval()
    policy.reset()
    pre, post = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides={
            "device_processor": {"device": str(policy.config.device)},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    )
    env_pre, env_post = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)

    frames, cached_px = [], None
    cached_strokes = None
    obs, _ = env.reset(seed=[cfg.seed])
    max_steps = env.call("_max_episode_steps")[0]
    lib = env.envs[0]
    ctrl = lib._env.robots[0].controller
    dt = getattr(lib._env, "env", lib._env).control_timestep
    pos_max = np.asarray(ctrl.output_max[:3], np.float32)
    cam_name = lib.camera_name[0].removesuffix("_image")
    cam_id = lib._env.sim.model.camera_name2id(cam_name)
    fovy = lib._env.sim.model.cam_fovy[cam_id]

    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        for _t in tqdm.tqdm(range(max_steps)):
            raw_obs = obs
            obs_t = pre(env_pre(add_envs_task(env, preprocess_observation(obs))))
            new_chunk = len(policy._action_queue) == 0
            del obs_t['observation.images.image']
            if new_chunk:
                policy.predict_action_chunk(obs_t, compute_densities=True, hutchinson_samples=HUTCHINSON_SAMPLES)
                pop = policy.model._last_population
                pop_logp = policy.model._last_population_logp
                idx = torch.argmax(pop_logp, dim=1) # torch.tensor([0], device='cuda:0')
                batch_idx = torch.arange(pop.shape[0], device=idx.device)
                act_dim = policy.config.output_features[ACTION].shape[0]
                best = pop[batch_idx, idx, : policy.config.n_action_steps, :act_dim]
                policy._action_queue.extend(best.transpose(0, 1))
            a = env_post({ACTION: post(policy._action_queue.popleft())})[ACTION]
            act_np = a.cpu().numpy()

            frame = raw_obs["pixels"]["image"][0][::-1, ::-1].copy()  # match LiberoEnv.render()
            data = lib._env.sim.data
            eef_pos = raw_obs["robot_state"]["eef"]["pos"][0].astype(np.float32)
            d_step = act_np[0, :3].astype(np.float32) * pos_max
            d_vel = d_step * dt
            eef_px = project(
                eef_pos[None], data.cam_xpos[cam_id], data.cam_xmat[cam_id], fovy, frame.shape[1], frame.shape[0]
            )[0]
            draw_x(frame, eef_px[0], eef_px[1])
            if new_chunk:
                pop = policy.model._last_population[0, :, : policy.config.n_action_steps, : act_np.shape[1]]
                n_pop, n_steps, act_dim = pop.shape
                pop_logp = policy.model._last_population_logp[0].detach().cpu().numpy()
                pop_colors = logp_to_colors(pop_logp)
                pop = env_post({ACTION: post(pop.reshape(-1, pop.shape[-1]))})[ACTION]
                pop = pop.cpu().numpy().reshape(n_pop, n_steps, act_dim)

                ctrl = lib._env.robots[0].controller
                pos_scale = np.asarray(ctrl.output_max[:3], dtype=np.float32)  # action in [-1,1] -> meters
                pts = eef_pos[None, None] + np.cumsum(pop[:, :, :3] * pos_scale, axis=1)
                pts = np.concatenate([np.repeat(eef_pos[None, None], n_pop, axis=0), pts], axis=1)

                cached_px = [
                    project(p, data.cam_xpos[cam_id], data.cam_xmat[cam_id], fovy, frame.shape[1], frame.shape[0])
                    for p in pts
                ]
                h, w = frame.shape[:2]
                cached_strokes = [
                    (*poly_pixels(px, h, w), tuple(map(int, pop_colors[i])))
                    for i, px in enumerate(cached_px)
                ]
                if LINE_THICK > 1:
                    offs = np.array([(dy, dx) for dy in range(-LINE_THICK + 1, LINE_THICK) for dx in range(-LINE_THICK + 1, LINE_THICK)], dtype=int)
                    cached_strokes = [
                        (
                            np.clip((ys[:, None] + offs[:, 0]).reshape(-1), 0, h - 1),
                            np.clip((xs[:, None] + offs[:, 1]).reshape(-1), 0, w - 1),
                            col,
                        )
                        for ys, xs, col in cached_strokes
                    ]

            if cached_strokes is not None:
                for ys, xs, col in cached_strokes:
                    frame[ys, xs] = col
            frames.append(frame)

            obs, _r, terminated, truncated, _info = env.step(act_np)
            if _t < DEBUG_STEPS:
                eef_next = obs["robot_state"]["eef"]["pos"][0].astype(np.float32)
                d_real = eef_next - eef_pos
                nr, ns, nv = map(np.linalg.norm, (d_real, d_step, d_vel))
                print(
                    f"t={_t:03d} dt={dt:.4f} |d_real|={nr:.4g} |d_step|={ns:.4g} |d_vel|={nv:.4g} "
                    f"r_step={nr/(ns+1e-9):.3f} r_vel={nr/(nv+1e-9):.3f}"
                )
            if bool(terminated[0] or truncated[0]):
                last = obs["pixels"]["image"][0][::-1, ::-1].copy()
                frames.append(last)
                break

    out = Path(OUT) if OUT else Path(cfg.output_dir) / "pop_viz.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_video(str(out), np.stack(frames), env.unwrapped.metadata["render_fps"])
    print(f"Wrote: {out}")
    env.close()


if __name__ == "__main__":
    main()
