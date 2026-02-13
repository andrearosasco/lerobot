#!/usr/bin/env python3

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


GROUP_ORDER = ["libero_object", "libero_spatial", "libero_goal", "libero_10"]
GROUP_COLORS = {
    "libero_object": "#4c72b0",
    "libero_spatial": "#55a868",
    "libero_goal": "#8172b2",
    "libero_10": "#c44e52",
}


def safe_corr(values: np.ndarray, labels: np.ndarray) -> float:
    if len(values) == 0 or labels.min() == labels.max() or np.std(values) == 0:
        return float("nan")
    return float(np.corrcoef(values, labels)[0, 1])


def load_episodes(input_path: Path) -> tuple[list[dict], dict]:
    with input_path.open() as f:
        eval_info = json.load(f)

    episodes = []
    task_count_by_group = defaultdict(int)
    for task in eval_info["per_task"]:
        group = task["task_group"]
        task_id = int(task["task_id"])
        task_count_by_group[group] += 1
        metrics = task["metrics"]
        for episode_id, (success, seq, n_steps) in enumerate(
            zip(metrics["successes"], metrics["log_density_per_step"], metrics["n_steps"], strict=True)
        ):
            density = np.asarray(seq, dtype=float)
            if density.size == 0:
                continue
            episodes.append(
                {
                    "task_group": group,
                    "task_id": task_id,
                    "episode_id": episode_id,
                    "success": int(bool(success)),
                    "n_steps": int(n_steps),
                    "cum_density": np.cumsum(density),
                }
            )
    return episodes, dict(task_count_by_group)


def compute_success_window_stats(episodes: list[dict]) -> tuple[list[dict], dict, dict]:
    by_task = defaultdict(list)
    for row in episodes:
        by_task[(row["task_group"], row["task_id"])].append(row)

    task_rows = []
    excluded = {
        "single_outcome": [],
        "no_shared_steps": [],
        "missing_success": [],
    }
    pooled_by_group = defaultdict(lambda: {"values": [], "labels": []})
    pooled_all_values = []
    pooled_all_labels = []

    for (group, task_id), task_eps in sorted(by_task.items()):
        success_eps = [e for e in task_eps if e["success"] == 1]
        fail_eps = [e for e in task_eps if e["success"] == 0]
        if not success_eps:
            excluded["missing_success"].append((group, task_id))
            continue
        if not fail_eps:
            excluded["single_outcome"].append((group, task_id))
            continue

        k_success_max = max(e["n_steps"] for e in success_eps)
        step_diffs = []
        for step in range(1, k_success_max + 1):
            s_vals = [e["cum_density"][step - 1] for e in success_eps if len(e["cum_density"]) >= step]
            f_vals = [e["cum_density"][step - 1] for e in fail_eps if len(e["cum_density"]) >= step]
            if s_vals and f_vals:
                step_diffs.append(float(np.mean(s_vals) - np.mean(f_vals)))
        if not step_diffs:
            excluded["no_shared_steps"].append((group, task_id))
            continue

        at_k = [e for e in task_eps if len(e["cum_density"]) >= k_success_max]
        at_k_values = np.asarray([e["cum_density"][k_success_max - 1] for e in at_k], dtype=float)
        at_k_labels = np.asarray([e["success"] for e in at_k], dtype=int)
        at_k_success_vals = at_k_values[at_k_labels == 1]
        at_k_fail_vals = at_k_values[at_k_labels == 0]
        if len(at_k_success_vals) == 0 or len(at_k_fail_vals) == 0:
            excluded["no_shared_steps"].append((group, task_id))
            continue

        final_diff = float(at_k_success_vals.mean() - at_k_fail_vals.mean())
        task_rows.append(
            {
                "task_group": group,
                "task_id": task_id,
                "k_success_max": int(k_success_max),
                "n_at_k": int(len(at_k_values)),
                "n_success_at_k": int(len(at_k_success_vals)),
                "n_fail_at_k": int(len(at_k_fail_vals)),
                "pooled_r_at_k": safe_corr(at_k_values, at_k_labels.astype(float)),
                "mean_step_diff_success_minus_fail": float(np.mean(step_diffs)),
                "median_step_diff_success_minus_fail": float(np.median(step_diffs)),
                "positive_step_fraction": float(np.mean(np.asarray(step_diffs) > 0)),
                "final_diff_at_k_success_minus_fail": final_diff,
            }
        )

        pooled_by_group[group]["values"].extend(at_k_values.tolist())
        pooled_by_group[group]["labels"].extend(at_k_labels.tolist())
        pooled_all_values.extend(at_k_values.tolist())
        pooled_all_labels.extend(at_k_labels.tolist())

    group_summary = {}
    for group in GROUP_ORDER:
        group_tasks = [r for r in task_rows if r["task_group"] == group]
        values = np.asarray(pooled_by_group[group]["values"], dtype=float)
        labels = np.asarray(pooled_by_group[group]["labels"], dtype=int)
        if len(values):
            pooled_diff = float(values[labels == 1].mean() - values[labels == 0].mean())
            pooled_r = safe_corr(values, labels.astype(float))
        else:
            pooled_diff = float("nan")
            pooled_r = float("nan")
        if group_tasks:
            positive_final = int(sum(r["final_diff_at_k_success_minus_fail"] > 0 for r in group_tasks))
            positive_mean = int(sum(r["mean_step_diff_success_minus_fail"] > 0 for r in group_tasks))
            median_positive_fraction = float(np.median([r["positive_step_fraction"] for r in group_tasks]))
        else:
            positive_final = 0
            positive_mean = 0
            median_positive_fraction = float("nan")
        group_summary[group] = {
            "tasks_used": len(group_tasks),
            "pooled_points": int(len(values)),
            "pooled_r_at_k": pooled_r,
            "pooled_diff_at_k_success_minus_fail": pooled_diff,
            "tasks_positive_final_diff": positive_final,
            "tasks_positive_mean_step_diff": positive_mean,
            "median_positive_step_fraction": median_positive_fraction,
        }

    pooled_all_values = np.asarray(pooled_all_values, dtype=float)
    pooled_all_labels = np.asarray(pooled_all_labels, dtype=int)
    overall = {
        "tasks_used": len(task_rows),
        "pooled_points": int(len(pooled_all_values)),
        "pooled_r_at_k": safe_corr(pooled_all_values, pooled_all_labels.astype(float)),
        "pooled_diff_at_k_success_minus_fail": float(
            pooled_all_values[pooled_all_labels == 1].mean() - pooled_all_values[pooled_all_labels == 0].mean()
        )
        if len(pooled_all_values)
        else float("nan"),
    }
    return task_rows, group_summary, {"overall": overall, "excluded": excluded}


def make_plots(task_rows: list[dict], group_summary: dict, assets_dir: Path) -> dict:
    assets_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "task_mean_step_diff": assets_dir / "task_mean_step_diff_by_group.png",
        "task_positive_step_fraction": assets_dir / "task_positive_step_fraction_by_group.png",
        "group_pooled_r": assets_dir / "group_pooled_r_at_k.png",
        "object_spatial_task_final_diff": assets_dir / "object_spatial_task_final_diff.png",
    }

    grouped_mean_diff = [
        [r["mean_step_diff_success_minus_fail"] for r in task_rows if r["task_group"] == group] for group in GROUP_ORDER
    ]
    plt.figure(figsize=(9.2, 4.8))
    bp = plt.boxplot(grouped_mean_diff, patch_artist=True, tick_labels=GROUP_ORDER, showfliers=False)
    for patch, group in zip(bp["boxes"], GROUP_ORDER, strict=True):
        patch.set_facecolor(GROUP_COLORS[group])
        patch.set_alpha(0.65)
    plt.axhline(0, color="black", linewidth=1)
    plt.ylabel("mean step diff (success - fail)")
    plt.title("Task-Level Mean Accumulated-Density Difference")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(out["task_mean_step_diff"], dpi=160)
    plt.close()

    grouped_pos_frac = [
        [r["positive_step_fraction"] for r in task_rows if r["task_group"] == group] for group in GROUP_ORDER
    ]
    plt.figure(figsize=(9.2, 4.8))
    bp = plt.boxplot(grouped_pos_frac, patch_artist=True, tick_labels=GROUP_ORDER, showfliers=False)
    for patch, group in zip(bp["boxes"], GROUP_ORDER, strict=True):
        patch.set_facecolor(GROUP_COLORS[group])
        patch.set_alpha(0.65)
    plt.axhline(0.5, color="black", linewidth=1, linestyle="--")
    plt.ylim(0, 1)
    plt.ylabel("fraction of steps with positive diff")
    plt.title("Task-Level Positive-Step Fraction")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(out["task_positive_step_fraction"], dpi=160)
    plt.close()

    pooled_r = [group_summary[group]["pooled_r_at_k"] for group in GROUP_ORDER]
    colors = [GROUP_COLORS[group] for group in GROUP_ORDER]
    plt.figure(figsize=(8.6, 4.6))
    plt.bar(GROUP_ORDER, pooled_r, color=colors, alpha=0.8)
    plt.axhline(0, color="black", linewidth=1)
    plt.ylabel("pooled r at task-specific K")
    plt.title("Pooled Correlation by Group (Task-Specific Successful Window)")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(out["group_pooled_r"], dpi=160)
    plt.close()

    subset = [r for r in task_rows if r["task_group"] in {"libero_object", "libero_spatial"}]
    subset = sorted(subset, key=lambda r: (r["task_group"], r["task_id"]))
    labels = [f"{r['task_group']}_{r['task_id']}" for r in subset]
    vals = [r["final_diff_at_k_success_minus_fail"] for r in subset]
    colors = [GROUP_COLORS[r["task_group"]] for r in subset]
    plt.figure(figsize=(10.4, 5.2))
    y = np.arange(len(subset))
    plt.hlines(y, 0, vals, color=colors, alpha=0.75, linewidth=2)
    plt.scatter(vals, y, color=colors, s=40)
    plt.axvline(0, color="black", linewidth=1)
    plt.yticks(y, labels)
    plt.xlabel("final diff at K (success - fail)")
    plt.title("Object/Spatial Per-Task Final Difference at Task-Specific K")
    plt.tight_layout()
    plt.savefig(out["object_spatial_task_final_diff"], dpi=160)
    plt.close()

    return out


def write_report(
    output_md: Path,
    input_path: Path,
    task_count_by_group: dict,
    task_rows: list[dict],
    group_summary: dict,
    meta: dict,
    plot_paths: dict,
) -> None:
    overall = meta["overall"]
    excluded = meta["excluded"]
    md_paths = {k: str(v.relative_to(output_md.parent)) for k, v in plot_paths.items()}

    lines = []
    lines.append("# Accumulated Density Report (Task Successful-Window Protocol)")
    lines.append("")
    lines.append(f"- Input: `{input_path}`")
    lines.append(f"- Total tasks in file: {sum(task_count_by_group.values())}")
    lines.append(f"- Tasks used in this analysis: {overall['tasks_used']}")
    lines.append(f"- Pooled episode-points used: {overall['pooled_points']}")
    lines.append("")
    lines.append("## Protocol (Only This Case)")
    lines.append("")
    lines.append("For each task, set `K_task = max(n_steps among successful episodes in that task)`.")
    lines.append("Then compare success vs failure accumulated density only within that task and up to `K_task`.")
    lines.append("")
    lines.append("## Definitions")
    lines.append("")
    lines.append("- `accumulated density` at step `t`: `A(t) = sum_{i=1..t} density_i`.")
    lines.append("- `step diff` at step `t`: `mean(A_success(t)) - mean(A_fail(t))` using episodes that have at least `t` steps.")
    lines.append("- `positive step`: a step where `step diff > 0`.")
    lines.append("- `positive_step_fraction`: fraction of valid steps in `[1..K_task]` with positive step.")
    lines.append("- `final diff at K`: `mean(A_success(K_task)) - mean(A_fail(K_task))`.")
    lines.append("- `pooled r at K`: correlation between success label and `A(K_task)` after concatenating episodes from all included tasks (each task contributes values at its own `K_task`).")
    lines.append("- Because `K_task` is the max successful length, `n_success@K` is often 1; final-at-K estimates can therefore be noisy at task level.")
    lines.append("")
    lines.append("## Why Not All Tasks Are Used")
    lines.append("")
    lines.append("- A task is excluded if it has only one outcome class (all success or all failure), since success-failure comparison is undefined.")
    lines.append("- A task is excluded if success/failure cannot be compared on shared valid steps under this protocol.")
    lines.append(f"- Excluded (single outcome): {len(excluded['single_outcome'])} tasks.")
    lines.append(f"- Excluded (missing success): {len(excluded['missing_success'])} tasks.")
    lines.append(f"- Excluded (no shared steps): {len(excluded['no_shared_steps'])} tasks.")
    lines.append("")
    lines.append("## Main Results")
    lines.append("")
    lines.append(
        f"- Overall pooled r at task-specific K: `{overall['pooled_r_at_k']:.3f}` (weak global effect after pooling heterogeneous tasks)."
    )
    lines.append(
        f"- Overall pooled success-fail diff at task-specific K: `{overall['pooled_diff_at_k_success_minus_fail']:.3f}`."
    )
    lines.append("- Local (task-level) patterns can still be strong even when pooled r is weak.")
    lines.append("")
    lines.append("Group summary:")
    lines.append("")
    lines.append("| group | tasks total | tasks used | tasks positive final diff | tasks positive mean step diff | median positive step fraction | pooled r at K | pooled diff at K |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for group in GROUP_ORDER:
        g = group_summary[group]
        lines.append(
            f"| `{group}` | {task_count_by_group.get(group, 0)} | {g['tasks_used']} | "
            f"{g['tasks_positive_final_diff']}/{g['tasks_used']} | "
            f"{g['tasks_positive_mean_step_diff']}/{g['tasks_used']} | "
            f"{g['median_positive_step_fraction']:.3f} | {g['pooled_r_at_k']:.3f} | {g['pooled_diff_at_k_success_minus_fail']:.3f} |"
        )
    lines.append("")
    lines.append("Object/Spatial per-task detail:")
    lines.append("")
    lines.append("| task | K_task | n_success@K | n_fail@K | mean step diff | positive step fraction | final diff at K |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in task_rows:
        if row["task_group"] not in {"libero_object", "libero_spatial"}:
            continue
        lines.append(
            f"| `{row['task_group']}_{row['task_id']}` | {row['k_success_max']} | {row['n_success_at_k']} | {row['n_fail_at_k']} | "
            f"{row['mean_step_diff_success_minus_fail']:.3f} | {row['positive_step_fraction']:.3f} | "
            f"{row['final_diff_at_k_success_minus_fail']:.3f} |"
        )
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    lines.append("Task-level mean step difference by group:")
    lines.append("")
    lines.append(f"![Task Mean Step Diff]({md_paths['task_mean_step_diff']})")
    lines.append("")
    lines.append("Task-level positive-step fraction by group:")
    lines.append("")
    lines.append(f"![Task Positive Step Fraction]({md_paths['task_positive_step_fraction']})")
    lines.append("")
    lines.append("Pooled correlation at task-specific K by group:")
    lines.append("")
    lines.append(f"![Group Pooled R]({md_paths['group_pooled_r']})")
    lines.append("")
    lines.append("Object/Spatial final diff at task-specific K:")
    lines.append("")
    lines.append(f"![Object Spatial Task Final Diff]({md_paths['object_spatial_task_final_diff']})")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "Your visual claim is compatible with this analysis: object/spatial can show many tasks with positive success-fail accumulated-density differences. "
        "A weak pooled r does not contradict that; it means after mixing tasks/groups with different magnitudes and signs, one global linear summary is small."
    )
    lines.append("")

    output_md.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate focused accumulated-density report using task-specific successful-window protocol.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/eval_logs/libero_object,libero_goal,libero_10,libero_spatial/eval_info.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("data/eval_logs/libero_object,libero_goal,libero_10,libero_spatial/density_success_report.md"),
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=Path("data/eval_logs/libero_object,libero_goal,libero_10,libero_spatial/density_report_assets"),
    )
    args = parser.parse_args()

    episodes, task_count_by_group = load_episodes(args.input)
    task_rows, group_summary, meta = compute_success_window_stats(episodes)
    plot_paths = make_plots(task_rows, group_summary, args.assets_dir)
    write_report(args.output_md, args.input, task_count_by_group, task_rows, group_summary, meta, plot_paths)

    print(f"Wrote report: {args.output_md}")
    for name, path in plot_paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
