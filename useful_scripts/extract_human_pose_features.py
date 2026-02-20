#!/usr/bin/env python

from __future__ import annotations

import argparse
import importlib
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Iterate a LeRobot dataset and extract human pose landmarks per frame. "
            "Saves one pose vector per frame (or None if no human is detected)."
        )
    )
    parser.add_argument("--dataset-repo-id", required=True, help="LeRobot dataset repo id (e.g. user/my_dataset)")
    parser.add_argument("--output-dir", default="dataset_analysis", help="Output directory")
    parser.add_argument("--dataset-root", default=None, help="Optional local dataset root")
    parser.add_argument("--revision", default=None, help="Optional dataset revision/tag")
    parser.add_argument("--episodes", type=int, nargs="+", default=None, help="Optional episode indices")
    parser.add_argument("--camera-key", default=None, help="Camera key (default: first available)")

    parser.add_argument(
        "--backend",
        choices=["auto", "yolo", "mediapipe"],
        default="auto",
        help="Pose backend. 'auto' prefers YOLO (GPU) and falls back to MediaPipe.",
    )

    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for fast mode")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers for fast mode")
    parser.add_argument("--debug", action="store_true", help="Debug mode: frame-by-frame + optional visualization")
    parser.add_argument("--no-show", action="store_true", help="Disable visualization in debug mode")

    parser.add_argument("--yolo-model", default="yolo11x-pose.pt", help="Ultralytics pose model")
    parser.add_argument("--yolo-conf", type=float, default=0.15, help="YOLO confidence threshold")
    parser.add_argument("--yolo-iou", type=float, default=0.7, help="YOLO NMS IoU threshold")
    parser.add_argument("--yolo-device", default="auto", help="YOLO device: auto|cpu|cuda|cuda:0|0")
    parser.add_argument("--yolo-imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument(
        "--yolo-lock-person",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep consistent person identity across consecutive frames (default: enabled)",
    )
    parser.add_argument(
        "--yolo-lock-weight",
        type=float,
        default=0.7,
        help="Weight for temporal consistency when selecting person in YOLO results",
    )
    parser.add_argument(
        "--yolo-min-kp-conf",
        type=float,
        default=0.2,
        help="Minimum keypoint confidence for temporal matching",
    )

    parser.add_argument("--min-detection-confidence", type=float, default=0.2, help="MediaPipe detection threshold")
    parser.add_argument("--min-tracking-confidence", type=float, default=0.2, help="MediaPipe tracking threshold")
    parser.add_argument(
        "--no-color-fallback",
        action="store_true",
        help="Disable second pass with swapped RGB/BGR channels",
    )

    parser.add_argument(
        "--stabilize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply temporal stabilization to reduce jumps (default: enabled)",
    )
    parser.add_argument("--stabilize-alpha", type=float, default=0.6, help="EMA smoothing factor [0,1]")
    parser.add_argument(
        "--stabilize-max-jump",
        type=float,
        default=0.12,
        help="Reject frame if mean xy jump exceeds this normalized threshold",
    )
    parser.add_argument(
        "--stabilize-min-vis",
        type=float,
        default=0.2,
        help="Min visibility/confidence to consider keypoint in jump check",
    )

    return parser.parse_args()


def _to_hwc_uint8(image: torch.Tensor) -> np.ndarray:
    if image.ndim != 3:
        raise ValueError(f"Expected image with shape (C,H,W), got {tuple(image.shape)}")

    if image.dtype.is_floating_point:
        image = (image.clamp(0, 1) * 255.0).to(torch.uint8)
    else:
        image = image.to(torch.uint8)

    img = image.permute(1, 2, 0).cpu().numpy()
    if img.shape[2] == 1:
        img = np.repeat(img, repeats=3, axis=2)
    return img


def _opencv_gui_available() -> bool:
    try:
        cv2.namedWindow("_pose_extract_gui_test_", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("_pose_extract_gui_test_")
        return True
    except cv2.error:
        return False


def _draw_landmarks_bgr(frame_bgr: np.ndarray, landmarks_xy: list[tuple[float, float]] | None) -> np.ndarray:
    out = frame_bgr.copy()
    if not landmarks_xy:
        return out

    h, w = out.shape[:2]
    for x_norm, y_norm in landmarks_xy:
        x = int(np.clip(x_norm * w, 0, w - 1))
        y = int(np.clip(y_norm * h, 0, h - 1))
        cv2.circle(out, (x, y), 3, (0, 0, 255), -1, lineType=cv2.LINE_AA)
    return out


def _resolve_yolo_device(device_arg: str) -> str | int:
    if device_arg == "auto":
        return 0 if torch.cuda.is_available() else "cpu"

    lower = device_arg.lower()
    if lower == "cpu":
        return "cpu"
    if lower in {"cuda", "gpu"}:
        return 0
    if lower.startswith("cuda:"):
        idx = lower.split(":", maxsplit=1)[1]
        return int(idx) if idx.isdigit() else 0
    if lower.isdigit():
        return int(lower)
    return device_arg


def _find_mediapipe_pose_task_model(mp_module) -> Path | None:
    mp_root = Path(mp_module.__file__).resolve().parent
    candidates = sorted(mp_root.rglob("pose_landmarker*.task"))
    if not candidates:
        return None

    preferred_keywords = ["full", "heavy", "lite"]
    for keyword in preferred_keywords:
        for path in candidates:
            if keyword in path.name:
                return path
    return candidates[0]


def _resolve_pose_task_model(mp_module) -> Path | None:
    model_path = _find_mediapipe_pose_task_model(mp_module)
    if model_path is not None:
        return model_path

    cache_dir = Path.home() / ".cache" / "mediapipe" / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    target_path = cache_dir / "pose_landmarker_lite.task"

    if target_path.exists() and target_path.stat().st_size > 0:
        return target_path

    model_urls = [
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    ]

    for url in model_urls:
        try:
            urllib.request.urlretrieve(url, target_path)
            if target_path.exists() and target_path.stat().st_size > 0:
                return target_path
        except Exception:
            continue

    return None


def _extract_pose_from_yolo_result(
    result: Any,
    prev_landmarks_xy: list[tuple[float, float]] | None,
    lock_person: bool,
    lock_weight: float,
    min_kp_conf: float,
) -> tuple[np.ndarray | None, list[tuple[float, float]] | None]:
    if result is None or getattr(result, "keypoints", None) is None:
        return None, None

    keypoints = result.keypoints
    if keypoints.xyn is None or len(keypoints.xyn) == 0:
        return None, None

    xyn_all = keypoints.xyn.detach().cpu().numpy()  # (N,K,2)
    conf_all = None
    if getattr(keypoints, "conf", None) is not None:
        conf_all = keypoints.conf.detach().cpu().numpy()  # (N,K)

    box_conf = np.ones((xyn_all.shape[0],), dtype=np.float32)
    if getattr(result, "boxes", None) is not None and getattr(result.boxes, "conf", None) is not None:
        box_conf = result.boxes.conf.detach().cpu().numpy().astype(np.float32)

    person_idx = int(np.argmax(box_conf))

    if lock_person and prev_landmarks_xy is not None and len(prev_landmarks_xy) > 0 and xyn_all.shape[0] > 1:
        prev_xy = np.asarray(prev_landmarks_xy, dtype=np.float32)
        match_scores = []
        for i in range(xyn_all.shape[0]):
            curr_xy = xyn_all[i]
            if curr_xy.shape[0] != prev_xy.shape[0]:
                match_scores.append(-1e6)
                continue

            if conf_all is not None:
                vis_mask = conf_all[i] >= min_kp_conf
            else:
                vis_mask = np.ones((curr_xy.shape[0],), dtype=bool)

            if np.any(vis_mask):
                mean_dist = float(np.mean(np.linalg.norm(curr_xy[vis_mask] - prev_xy[vis_mask], axis=1)))
            else:
                mean_dist = 1.0

            score = float(box_conf[i]) - lock_weight * mean_dist
            match_scores.append(score)

        person_idx = int(np.argmax(np.asarray(match_scores, dtype=np.float32)))

    xy = xyn_all[person_idx]
    kp_conf = conf_all[person_idx] if conf_all is not None else None

    vec = []
    landmarks_xy = []
    for i in range(xy.shape[0]):
        x, y = float(xy[i, 0]), float(xy[i, 1])
        vis = float(kp_conf[i]) if kp_conf is not None else 1.0
        vec.extend([x, y, 0.0, vis])
        landmarks_xy.append((x, y))

    return np.asarray(vec, dtype=np.float32), landmarks_xy


def _extract_pose_from_tasks_results(results) -> tuple[np.ndarray | None, list[tuple[float, float]] | None]:
    landmarks_list = results.pose_landmarks if hasattr(results, "pose_landmarks") else None
    if not landmarks_list:
        return None, None

    vec = []
    landmarks_xy = []
    for lm in landmarks_list[0]:
        vis = float(getattr(lm, "visibility", 1.0))
        vec.extend([float(lm.x), float(lm.y), float(lm.z), vis])
        landmarks_xy.append((float(lm.x), float(lm.y)))

    return np.asarray(vec, dtype=np.float32), landmarks_xy


def _extract_pose_from_solutions_results(results) -> tuple[np.ndarray | None, list[tuple[float, float]] | None]:
    if results.pose_landmarks is None:
        return None, None

    vec = []
    landmarks_xy = []
    for lm in results.pose_landmarks.landmark:
        vec.extend([lm.x, lm.y, lm.z, lm.visibility])
        landmarks_xy.append((float(lm.x), float(lm.y)))

    return np.asarray(vec, dtype=np.float32), landmarks_xy


def _detect_pose(
    image_rgb: np.ndarray,
    *,
    mp,
    yolo_model,
    yolo_conf: float,
    yolo_iou: float,
    yolo_device: str | int,
    yolo_imgsz: int,
    yolo_prev_landmarks_xy: list[tuple[float, float]] | None,
    yolo_lock_person: bool,
    yolo_lock_weight: float,
    yolo_min_kp_conf: float,
    using_tasks_backend: bool,
    pose_landmarker,
    pose_estimator,
    use_color_fallback: bool,
) -> tuple[np.ndarray | None, list[tuple[float, float]] | None]:
    if yolo_model is not None:
        results = yolo_model.predict(
            source=[image_rgb],
            conf=yolo_conf,
            iou=yolo_iou,
            imgsz=yolo_imgsz,
            device=yolo_device,
            verbose=False,
        )
        if len(results) == 0:
            return None, None
        return _extract_pose_from_yolo_result(
            results[0],
            prev_landmarks_xy=yolo_prev_landmarks_xy,
            lock_person=yolo_lock_person,
            lock_weight=yolo_lock_weight,
            min_kp_conf=yolo_min_kp_conf,
        )

    if using_tasks_backend:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        results = pose_landmarker.detect(mp_image)
        pose_vec, landmarks_xy = _extract_pose_from_tasks_results(results)

        if pose_vec is None and use_color_fallback:
            image_swapped = image_rgb[:, :, ::-1].copy()
            mp_image_swapped = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_swapped)
            results_swapped = pose_landmarker.detect(mp_image_swapped)
            pose_vec, landmarks_xy = _extract_pose_from_tasks_results(results_swapped)

        return pose_vec, landmarks_xy

    results = pose_estimator.process(image_rgb)
    pose_vec, landmarks_xy = _extract_pose_from_solutions_results(results)

    if pose_vec is None and use_color_fallback:
        image_swapped = image_rgb[:, :, ::-1].copy()
        results_swapped = pose_estimator.process(image_swapped)
        pose_vec, landmarks_xy = _extract_pose_from_solutions_results(results_swapped)

    return pose_vec, landmarks_xy


def _stabilize_pose_vectors(
    pose_vectors: list[np.ndarray | None],
    *,
    alpha: float,
    max_jump: float,
    min_vis: float,
) -> tuple[list[np.ndarray | None], int]:
    stabilized: list[np.ndarray | None] = []
    prev: np.ndarray | None = None
    corrected = 0

    for vec in pose_vectors:
        if vec is None:
            stabilized.append(None)
            continue

        curr = np.asarray(vec, dtype=np.float32).copy()
        if curr.ndim != 1 or curr.shape[0] % 4 != 0:
            stabilized.append(curr)
            prev = curr
            continue

        curr_kp = curr.reshape(-1, 4)

        if prev is not None and prev.shape == curr.shape:
            prev_kp = prev.reshape(-1, 4)
            mask = (curr_kp[:, 3] >= min_vis) & (prev_kp[:, 3] >= min_vis)

            if np.any(mask):
                jump = np.linalg.norm(curr_kp[mask, :2] - prev_kp[mask, :2], axis=1)
                mean_jump = float(np.mean(jump))
                if mean_jump > max_jump:
                    curr_kp[:, :3] = prev_kp[:, :3]
                    corrected += 1
                else:
                    curr_kp[:, :3] = alpha * curr_kp[:, :3] + (1.0 - alpha) * prev_kp[:, :3]

        curr = curr_kp.reshape(-1).astype(np.float32)
        stabilized.append(curr)
        prev = curr

    return stabilized, corrected


def main() -> None:
    args = parse_args()

    mp = None
    yolo_model = None
    yolo_device: str | int = "cpu"

    if args.backend in ("auto", "yolo"):
        try:
            ultralytics_mod = importlib.import_module("ultralytics")
            YOLO = getattr(ultralytics_mod, "YOLO")
            yolo_device = _resolve_yolo_device(args.yolo_device)
            yolo_model = YOLO(args.yolo_model)
            print(f"Using YOLO pose backend: model={args.yolo_model}, device={yolo_device}")
        except Exception as exc:
            if args.backend == "yolo":
                raise ImportError(
                    "YOLO backend requested but Ultralytics model could not be loaded. Install with: pip install ultralytics"
                ) from exc
            print("YOLO backend unavailable, falling back to MediaPipe.")

    if yolo_model is None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise ImportError(
                "No usable backend found. Install one of: `pip install ultralytics` or `pip install mediapipe`."
            ) from exc

    mp_pose_cls = None
    pose_landmarker = None
    using_tasks_backend = False
    no_detector_available = False

    if yolo_model is None and hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
        mp_pose_cls = mp.solutions.pose.Pose
    elif yolo_model is None:
        try:
            pose_mod = importlib.import_module("mediapipe.python.solutions.pose")
            mp_pose_cls = getattr(pose_mod, "Pose")
        except Exception:
            mp_pose_cls = None

    if yolo_model is None and mp_pose_cls is None:
        try:
            tasks_python_vision = None
            tasks_base_options = None

            for vision_mod in ("mediapipe.tasks.python.vision", "mediapipe.tasks.vision"):
                try:
                    tasks_python_vision = importlib.import_module(vision_mod)
                    break
                except Exception:
                    continue

            for base_mod in ("mediapipe.tasks.python.core.base_options", "mediapipe.tasks.core.base_options"):
                try:
                    tasks_base_options = importlib.import_module(base_mod)
                    break
                except Exception:
                    continue

            if tasks_python_vision is None or tasks_base_options is None:
                raise ImportError("Could not import MediaPipe Tasks modules")

            model_path = _resolve_pose_task_model(mp)
            if model_path is None:
                raise FileNotFoundError("No pose_landmarker model found and download failed")

            PoseLandmarker = tasks_python_vision.PoseLandmarker
            PoseLandmarkerOptions = tasks_python_vision.PoseLandmarkerOptions
            VisionRunningMode = tasks_python_vision.RunningMode
            BaseOptions = tasks_base_options.BaseOptions

            options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                running_mode=VisionRunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=args.min_detection_confidence,
                min_pose_presence_confidence=args.min_tracking_confidence,
                min_tracking_confidence=args.min_tracking_confidence,
            )
            pose_landmarker = PoseLandmarker.create_from_options(options)
            using_tasks_backend = True
            print(f"Using MediaPipe Tasks backend with model: {model_path}")
        except Exception:
            no_detector_available = True
            print("Warning: no usable pose backend found. Will save None for all frames.")

    dataset = LeRobotDataset(
        repo_id=args.dataset_repo_id,
        root=args.dataset_root,
        revision=args.revision,
        episodes=args.episodes,
    )

    if len(dataset.meta.camera_keys) == 0:
        raise ValueError(f"Dataset '{args.dataset_repo_id}' has no camera/image features.")

    camera_key = args.camera_key or dataset.meta.camera_keys[0]
    if camera_key not in dataset.meta.camera_keys:
        raise ValueError(f"Camera key '{camera_key}' not found. Available: {dataset.meta.camera_keys}")

    total_frames = len(dataset)
    dataset_name = args.dataset_repo_id.replace("/", "__")
    pose_vectors: list[np.ndarray | None] = []

    debug_mode = bool(args.debug)
    show = debug_mode and not args.no_show
    if show and not _opencv_gui_available():
        print("Warning: OpenCV GUI not available, disabling visualization.")
        show = False

    if debug_mode:
        print("Running in DEBUG mode (frame-by-frame).")
    else:
        print(f"Running in FAST mode (batch_size={args.batch_size}, num_workers={args.num_workers}).")

    if show:
        print("Live visualization enabled. Press 'q' to stop early.")

    pose_estimator = None
    if yolo_model is None and not using_tasks_backend and not no_detector_available:
        pose_estimator = mp_pose_cls(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
        )

    interrupted = False
    yolo_prev_landmarks_xy: list[tuple[float, float]] | None = None
    pbar = tqdm(total=total_frames, desc="Extracting human pose", unit="frame")
    try:
        if no_detector_available:
            pose_vectors = [None] * total_frames
            pbar.update(total_frames)
        elif debug_mode:
            for idx in range(total_frames):
                sample = dataset[idx]
                image = _to_hwc_uint8(sample[camera_key])

                pose_vec, landmarks_xy = _detect_pose(
                    image,
                    mp=mp,
                    yolo_model=yolo_model,
                    yolo_conf=args.yolo_conf,
                    yolo_iou=args.yolo_iou,
                    yolo_device=yolo_device,
                    yolo_imgsz=args.yolo_imgsz,
                    yolo_prev_landmarks_xy=yolo_prev_landmarks_xy,
                    yolo_lock_person=args.yolo_lock_person,
                    yolo_lock_weight=args.yolo_lock_weight,
                    yolo_min_kp_conf=args.yolo_min_kp_conf,
                    using_tasks_backend=using_tasks_backend,
                    pose_landmarker=pose_landmarker,
                    pose_estimator=pose_estimator,
                    use_color_fallback=not args.no_color_fallback,
                )
                yolo_prev_landmarks_xy = landmarks_xy if landmarks_xy is not None else yolo_prev_landmarks_xy
                pose_vectors.append(pose_vec)

                if show:
                    frame_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    frame_bgr = _draw_landmarks_bgr(frame_bgr, landmarks_xy)
                    detected_so_far = sum(v is not None for v in pose_vectors)
                    cv2.putText(frame_bgr, f"frame {idx + 1}/{total_frames}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, lineType=cv2.LINE_AA)
                    cv2.putText(frame_bgr, f"detected: {detected_so_far} missing: {(idx + 1) - detected_so_far}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, lineType=cv2.LINE_AA)
                    cv2.imshow("Human Pose Extraction", frame_bgr)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        interrupted = True
                        print("Stopping early on user request (q).")
                        break

                pbar.update(1)
        else:
            dataloader = DataLoader(
                dataset,
                batch_size=max(1, args.batch_size),
                shuffle=False,
                num_workers=max(0, args.num_workers),
                pin_memory=False,
                drop_last=False,
            )

            for batch in dataloader:
                images = batch[camera_key]
                bsz = int(images.shape[0])
                images_list = [_to_hwc_uint8(images[i]) for i in range(bsz)]

                if yolo_model is not None:
                    yolo_results = yolo_model.predict(
                        source=images_list,
                        conf=args.yolo_conf,
                        iou=args.yolo_iou,
                        imgsz=args.yolo_imgsz,
                        device=yolo_device,
                        verbose=False,
                    )
                    for result in yolo_results:
                        pose_vec, landmarks_xy = _extract_pose_from_yolo_result(
                            result,
                            prev_landmarks_xy=yolo_prev_landmarks_xy,
                            lock_person=args.yolo_lock_person,
                            lock_weight=args.yolo_lock_weight,
                            min_kp_conf=args.yolo_min_kp_conf,
                        )
                        yolo_prev_landmarks_xy = landmarks_xy if landmarks_xy is not None else yolo_prev_landmarks_xy
                        pose_vectors.append(pose_vec)
                else:
                    for image in images_list:
                        pose_vec, _ = _detect_pose(
                            image,
                            mp=mp,
                            yolo_model=None,
                            yolo_conf=args.yolo_conf,
                            yolo_iou=args.yolo_iou,
                            yolo_device=yolo_device,
                            yolo_imgsz=args.yolo_imgsz,
                            yolo_prev_landmarks_xy=None,
                            yolo_lock_person=False,
                            yolo_lock_weight=args.yolo_lock_weight,
                            yolo_min_kp_conf=args.yolo_min_kp_conf,
                            using_tasks_backend=using_tasks_backend,
                            pose_landmarker=pose_landmarker,
                            pose_estimator=pose_estimator,
                            use_color_fallback=not args.no_color_fallback,
                        )
                        pose_vectors.append(pose_vec)

                pbar.update(bsz)
    finally:
        pbar.close()
        if pose_estimator is not None:
            pose_estimator.close()
        if pose_landmarker is not None:
            pose_landmarker.close()
        if show:
            cv2.destroyAllWindows()

    if args.stabilize and len(pose_vectors) > 0:
        pose_vectors, corrected = _stabilize_pose_vectors(
            pose_vectors,
            alpha=float(np.clip(args.stabilize_alpha, 0.0, 1.0)),
            max_jump=max(0.0, float(args.stabilize_max_jump)),
            min_vis=max(0.0, float(args.stabilize_min_vis)),
        )
        print(f"Applied temporal stabilization (corrected jumpy frames: {corrected}).")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{dataset_name}_human_pose.npy"
    np.save(output_path, np.array(pose_vectors, dtype=object), allow_pickle=True)

    saved_frames = len(pose_vectors)
    num_detected = sum(v is not None for v in pose_vectors)
    print(
        f"Saved pose vectors for {saved_frames} frames to {output_path} "
        f"(detected: {num_detected}, missing: {saved_frames - num_detected})"
    )
    if interrupted:
        print(f"Note: extraction stopped early at {saved_frames} frames; saved file contains partial results.")


if __name__ == "__main__":
    main()
