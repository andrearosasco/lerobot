#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path

import numpy as np


FEATURES = [
    "density_accum",
    "density_mean",
    "density_std",
    "density_min",
    "density_max",
    "density_start",
    "density_end",
    "density_delta",
    "density_slope",
    "n_steps",
]


def auc_success_vs_fail(values: np.ndarray, labels: np.ndarray) -> float:
    success_values = values[labels == 1]
    fail_values = values[labels == 0]
    if len(success_values) == 0 or len(fail_values) == 0:
        return float("nan")
    greater = (success_values[:, None] > fail_values[None, :]).sum()
    equal = (success_values[:, None] == fail_values[None, :]).sum()
    return float((greater + 0.5 * equal) / (len(success_values) * len(fail_values)))


def partial_corr_with_control(values: np.ndarray, labels: np.ndarray, control: np.ndarray) -> float:
    if len(values) == 0 or labels.min() == labels.max() or np.std(values) == 0 or np.std(control) == 0:
        return float("nan")
    control = control.reshape(-1, 1)
    design = np.column_stack([np.ones(len(control)), control])
    beta_y = np.linalg.lstsq(design, labels, rcond=None)[0]
    beta_x = np.linalg.lstsq(design, values, rcond=None)[0]
    residual_y = labels - design @ beta_y
    residual_x = values - design @ beta_x
    if np.std(residual_y) == 0 or np.std(residual_x) == 0:
        return float("nan")
    return float(np.corrcoef(residual_x, residual_y)[0, 1])


def feature_stats(rows: list[dict]) -> list[dict]:
    labels = np.array([r["success"] for r in rows], dtype=int)
    n_steps = np.array([r["n_steps"] for r in rows], dtype=float)
    out = []
    for feature in FEATURES:
        values = np.array([r[feature] for r in rows], dtype=float)
        success_values = values[labels == 1]
        fail_values = values[labels == 0]
        out.append(
            {
                "feature": feature,
                "success_mean": float(success_values.mean()) if len(success_values) else float("nan"),
                "fail_mean": float(fail_values.mean()) if len(fail_values) else float("nan"),
                "mean_diff_success_minus_fail": float(success_values.mean() - fail_values.mean())
                if len(success_values) and len(fail_values)
                else float("nan"),
                "point_biserial_r": float(np.corrcoef(values, labels)[0, 1]) if labels.min() != labels.max() else float("nan"),
                "partial_r_given_n_steps": partial_corr_with_control(values, labels, n_steps),
                "auc_success_vs_fail": auc_success_vs_fail(values, labels),
            }
        )
    return out


def build_episode_rows(eval_info: dict) -> list[dict]:
    rows = []
    for task in eval_info["per_task"]:
        metrics = task["metrics"]
        successes = metrics["successes"]
        densities = metrics["log_density_per_step"]
        n_steps_list = metrics["n_steps"]
        for ep_id, (success, density_seq, n_steps) in enumerate(zip(successes, densities, n_steps_list, strict=True)):
            arr = np.asarray(density_seq, dtype=float)
            if arr.size == 0:
                continue
            x = np.arange(arr.size, dtype=float)
            slope = float(np.polyfit(x, arr, 1)[0]) if arr.size > 1 else 0.0
            rows.append(
                {
                    "task_group": task["task_group"],
                    "task_id": int(task["task_id"]),
                    "episode_id": ep_id,
                    "success": int(bool(success)),
                    "n_steps": int(n_steps),
                    "density_accum": float(arr.sum()),
                    "density_mean": float(arr.mean()),
                    "density_std": float(arr.std(ddof=0)),
                    "density_min": float(arr.min()),
                    "density_max": float(arr.max()),
                    "density_start": float(arr[0]),
                    "density_end": float(arr[-1]),
                    "density_delta": float(arr[-1] - arr[0]),
                    "density_slope": slope,
                }
            )
    return rows


def per_group_stats(rows: list[dict]) -> dict:
    groups = sorted({r["task_group"] for r in rows})
    out = {}
    for group in groups:
        group_rows = [r for r in rows if r["task_group"] == group]
        labels = np.array([r["success"] for r in group_rows], dtype=int)
        n_steps = np.array([r["n_steps"] for r in group_rows], dtype=float)
        features = {}
        for feature in FEATURES:
            values = np.array([r[feature] for r in group_rows], dtype=float)
            features[feature] = {
                "point_biserial_r": float(np.corrcoef(values, labels)[0, 1]) if labels.min() != labels.max() else float("nan"),
                "partial_r_given_n_steps": partial_corr_with_control(values, labels, n_steps),
                "auc_success_vs_fail": auc_success_vs_fail(values, labels),
            }
        out[group] = {
            "n_episodes": len(group_rows),
            "success_rate": float(labels.mean()),
            "features": features,
        }
    return out


def within_task_controlled_stats(rows: list[dict]) -> dict:
    by_task = {}
    for row in rows:
        key = (row["task_group"], row["task_id"])
        by_task.setdefault(key, []).append(row)

    diffs = {feature: [] for feature in FEATURES}
    used_tasks = 0
    for task_rows in by_task.values():
        labels = np.array([r["success"] for r in task_rows], dtype=int)
        if labels.min() == labels.max():
            continue
        used_tasks += 1
        for feature in FEATURES:
            values = np.array([r[feature] for r in task_rows], dtype=float)
            diffs[feature].append(float(values[labels == 1].mean() - values[labels == 0].mean()))

    return {
        "tasks_total": len(by_task),
        "tasks_with_both_success_and_failure": used_tasks,
        "avg_task_level_mean_diff_success_minus_fail": {
            feature: float(np.mean(values)) if values else float("nan") for feature, values in diffs.items()
        },
    }


def write_episode_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["task_group", "task_id", "episode_id", "success", *FEATURES]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in columns})


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze connection between log density and success in eval_info.json.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/eval_logs/libero_object,libero_goal,libero_10,libero_spatial/eval_info.json"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/eval_logs/libero_object,libero_goal,libero_10,libero_spatial/density_success_report.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/eval_logs/libero_object,libero_goal,libero_10,libero_spatial/density_episode_features.csv"),
    )
    args = parser.parse_args()

    with args.input.open() as f:
        eval_info = json.load(f)

    rows = build_episode_rows(eval_info)
    labels = np.array([r["success"] for r in rows], dtype=int)
    global_stats = feature_stats(rows)

    summary = {
        "n_episodes": len(rows),
        "n_success": int(labels.sum()),
        "n_fail": int((1 - labels).sum()),
        "success_rate": float(labels.mean()),
        "top_abs_r": sorted(
            [row for row in global_stats if not np.isnan(row["point_biserial_r"])],
            key=lambda row: abs(row["point_biserial_r"]),
            reverse=True,
        )[:5],
        "top_abs_partial_r_given_n_steps": sorted(
            [row for row in global_stats if not np.isnan(row["partial_r_given_n_steps"])],
            key=lambda row: abs(row["partial_r_given_n_steps"]),
            reverse=True,
        )[:5],
        "top_abs_auc_shift_from_0.5": sorted(
            [row for row in global_stats if not np.isnan(row["auc_success_vs_fail"])],
            key=lambda row: abs(row["auc_success_vs_fail"] - 0.5),
            reverse=True,
        )[:5],
    }

    report = {
        "input_path": str(args.input),
        "summary": summary,
        "global_feature_stats": global_stats,
        "per_group_feature_stats": per_group_stats(rows),
        "within_task_controlled_stats": within_task_controlled_stats(rows),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w") as f:
        json.dump(report, f, indent=2)
    write_episode_csv(rows, args.output_csv)

    print(f"Saved JSON report to {args.output_json}")
    print(f"Saved episode features to {args.output_csv}")
    print(f"Episodes={summary['n_episodes']} | Success rate={summary['success_rate']:.3f}")
    print("Top features by |point_biserial_r|:")
    for row in summary["top_abs_r"]:
        print(
            f"  {row['feature']}: r={row['point_biserial_r']:.4f}, "
            f"diff={row['mean_diff_success_minus_fail']:.4f}, auc={row['auc_success_vs_fail']:.4f}"
        )
    print("Top features by |partial_r_given_n_steps|:")
    for row in summary["top_abs_partial_r_given_n_steps"]:
        print(f"  {row['feature']}: partial_r={row['partial_r_given_n_steps']:.4f}")


if __name__ == "__main__":
    main()
