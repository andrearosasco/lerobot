---
license: apache-2.0
task_categories:
- robotics
tags:
- LeRobot
configs:
- config_name: default
  data_files: data/*/*.parquet
---

This dataset was created using [LeRobot](https://github.com/huggingface/lerobot).

## Dataset Description



- **Homepage:** [More Information Needed]
- **Paper:** [More Information Needed]
- **License:** apache-2.0

## Dataset Structure

[meta/info.json](meta/info.json):
```json
{
    "codebase_version": "v3.0",
    "robot_type": "ergocub",
    "total_episodes": 150,
    "total_frames": 21981,
    "total_tasks": 2,
    "chunks_size": 1000,
    "data_files_size_in_mb": 100,
    "video_files_size_in_mb": 500,
    "fps": 10,
    "splits": {
        "train": "0:150"
    },
    "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
    "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    "features": {
        "action": {
            "dtype": "float32",
            "names": [
                "left_hand.position.x",
                "left_hand.position.y",
                "left_hand.position.z",
                "left_hand.orientation.d1",
                "left_hand.orientation.d2",
                "left_hand.orientation.d3",
                "left_hand.orientation.d4",
                "left_hand.orientation.d5",
                "left_hand.orientation.d6",
                "right_hand.position.x",
                "right_hand.position.y",
                "right_hand.position.z",
                "right_hand.orientation.d1",
                "right_hand.orientation.d2",
                "right_hand.orientation.d3",
                "right_hand.orientation.d4",
                "right_hand.orientation.d5",
                "right_hand.orientation.d6",
                "head.orientation.d1",
                "head.orientation.d2",
                "head.orientation.d3",
                "head.orientation.d4",
                "head.orientation.d5",
                "head.orientation.d6",
                "left_fingers.thumb_add",
                "left_fingers.thumb_oc",
                "left_fingers.index_add",
                "left_fingers.index_oc",
                "left_fingers.middle_oc",
                "left_fingers.ring_pinky_oc",
                "right_fingers.thumb_add",
                "right_fingers.thumb_oc",
                "right_fingers.index_add",
                "right_fingers.index_oc",
                "right_fingers.middle_oc",
                "right_fingers.ring_pinky_oc"
            ],
            "shape": [
                36
            ]
        },
        "observation.state": {
            "dtype": "float32",
            "names": [
                "left_hand.position.x",
                "left_hand.position.y",
                "left_hand.position.z",
                "left_hand.orientation.d1",
                "left_hand.orientation.d2",
                "left_hand.orientation.d3",
                "left_hand.orientation.d4",
                "left_hand.orientation.d5",
                "left_hand.orientation.d6",
                "right_hand.position.x",
                "right_hand.position.y",
                "right_hand.position.z",
                "right_hand.orientation.d1",
                "right_hand.orientation.d2",
                "right_hand.orientation.d3",
                "right_hand.orientation.d4",
                "right_hand.orientation.d5",
                "right_hand.orientation.d6",
                "head.orientation.d1",
                "head.orientation.d2",
                "head.orientation.d3",
                "head.orientation.d4",
                "head.orientation.d5",
                "head.orientation.d6",
                "left_fingers.thumb_add",
                "left_fingers.thumb_oc",
                "left_fingers.index_add",
                "left_fingers.index_oc",
                "left_fingers.middle_oc",
                "left_fingers.ring_pinky_oc",
                "right_fingers.thumb_add",
                "right_fingers.thumb_oc",
                "right_fingers.index_add",
                "right_fingers.index_oc",
                "right_fingers.middle_oc",
                "right_fingers.ring_pinky_oc"
            ],
            "shape": [
                36
            ]
        },
        "timestamp": {
            "dtype": "float32",
            "shape": [
                1
            ],
            "names": null
        },
        "frame_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        },
        "episode_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        },
        "index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        },
        "task_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        },
        "observation.images.egocentric": {
            "dtype": "video",
            "shape": [
                480,
                640,
                3
            ],
            "names": [
                "height",
                "width",
                "channels"
            ],
            "info": {
                "video.height": 480,
                "video.width": 640,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": false,
                "video.fps": 10,
                "video.channels": 3,
                "has_audio": false
            }
        }
    }
}
```


## Citation

**BibTeX:**

```bibtex
[More Information Needed]
```