#!/usr/bin/env python3

import argparse
import shutil
from pathlib import Path
from collections import Counter

from lerobot.datasets.dataset_tools import add_features, merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.utils import init_logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge source+target LeRobot datasets, add domain_id (0=source, 1=target), "
            "and optionally push to Hugging Face Hub."
        )
    )
    parser.add_argument(
        "--source-repo-id",
        type=str,
        default="steb6/redball_sim_reactive",
        help="Source (in-domain) dataset repo_id.",
    )
    parser.add_argument(
        "--target-repo-id",
        type=str,
        default="steb6/eval_groot-redball_ood_2",
        help="Target (OOD) dataset repo_id.",
    )
    parser.add_argument(
        "--output-repo-id",
        type=str,
        required=True,
        help="Output merged dataset repo_id (must be different from inputs).",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Optional dataset cache root. Defaults to HF_LEROBOT_HOME.",
    )
    parser.add_argument(
        "--source-revision",
        type=str,
        default=None,
        help="Optional source dataset revision (branch/tag/commit).",
    )
    parser.add_argument(
        "--target-revision",
        type=str,
        default=None,
        help="Optional target dataset revision (branch/tag/commit).",
    )
    parser.add_argument(
        "--force-cache-sync",
        action="store_true",
        help="Force re-sync metadata/data from hub for source and target datasets.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push final dataset to hub.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create/push the dataset as private on hub.",
    )
    parser.add_argument(
        "--cleanup-tmp",
        action="store_true",
        help="Remove temporary merged dataset directory after completion.",
    )

    args = parser.parse_args()
    init_logging()

    cache_root = Path(args.root) if args.root else HF_LEROBOT_HOME

    tmp_source_repo_id = f"{args.output_repo_id}__tmp_source_domain"
    tmp_target_repo_id = f"{args.output_repo_id}__tmp_target_domain"
    tmp_source_dir = cache_root / tmp_source_repo_id
    tmp_target_dir = cache_root / tmp_target_repo_id
    output_dir = cache_root / args.output_repo_id

    for tmp_dir in (tmp_source_dir, tmp_target_dir):
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

    source_ds = LeRobotDataset(
        args.source_repo_id,
        root=args.root,
        revision=args.source_revision,
        force_cache_sync=args.force_cache_sync,
    )
    target_ds = LeRobotDataset(
        args.target_repo_id,
        root=args.root,
        revision=args.target_revision,
        force_cache_sync=args.force_cache_sync,
    )

    print(
        "resolved datasets:",
        {
            "source": {
                "repo_id": source_ds.repo_id,
                "root": str(source_ds.root),
                "revision": source_ds.revision,
                "episodes": int(source_ds.meta.total_episodes),
                "frames": int(source_ds.meta.total_frames),
            },
            "target": {
                "repo_id": target_ds.repo_id,
                "root": str(target_ds.root),
                "revision": target_ds.revision,
                "episodes": int(target_ds.meta.total_episodes),
                "frames": int(target_ds.meta.total_frames),
            },
        },
    )

    source_frame_count = int(source_ds.meta.total_frames)
    target_frame_count = int(target_ds.meta.total_frames)
    print(
        "input frame counts:",
        {
            "source": source_frame_count,
            "target": target_frame_count,
            "total": source_frame_count + target_frame_count,
        },
    )

    def domain_source(_row: dict, _episode_idx: int, _frame_in_ep: int) -> int:
        return 0

    def domain_target(_row: dict, _episode_idx: int, _frame_in_ep: int) -> int:
        return 1

    source_labeled = add_features(
        dataset=source_ds,
        features={
            "domain_id": (
                domain_source,
                {"dtype": "int64", "shape": (1,), "names": None},
            )
        },
        output_dir=tmp_source_dir,
        repo_id=tmp_source_repo_id,
    )

    target_labeled = add_features(
        dataset=target_ds,
        features={
            "domain_id": (
                domain_target,
                {"dtype": "int64", "shape": (1,), "names": None},
            )
        },
        output_dir=tmp_target_dir,
        repo_id=tmp_target_repo_id,
    )

    final_ds = merge_datasets(
        datasets=[source_labeled, target_labeled],
        output_repo_id=args.output_repo_id,
        output_dir=output_dir,
    )

    merged_frame_count = int(final_ds.meta.total_frames)
    expected_total = source_frame_count + target_frame_count
    print("merged frame count:", merged_frame_count)
    if merged_frame_count != expected_total:
        raise RuntimeError(
            f"Merged frame count mismatch: got {merged_frame_count}, expected {expected_total} "
            f"(source={source_frame_count}, target={target_frame_count})"
        )

    domain_counts = Counter(int(v) for v in final_ds.hf_dataset["domain_id"])
    if domain_counts.get(0, 0) != source_frame_count or domain_counts.get(1, 0) != target_frame_count:
        raise RuntimeError(
            "domain_id counts mismatch after merge: "
            f"got 0->{domain_counts.get(0, 0)}, 1->{domain_counts.get(1, 0)}; "
            f"expected 0->{source_frame_count}, 1->{target_frame_count}"
        )

    print(
        "domain_id frame counts:",
        {
            0: domain_counts.get(0, 0),
            1: domain_counts.get(1, 0),
            "total": sum(domain_counts.values()),
        },
    )

    if args.push_to_hub:
        final_ds.push_to_hub(private=args.private)

    if args.cleanup_tmp:
        for tmp_dir in (tmp_source_dir, tmp_target_dir):
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()
