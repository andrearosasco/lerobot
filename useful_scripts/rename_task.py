from pathlib import Path
import pandas as pd

root = Path("~/.cache/huggingface/lerobot/steb6/IROS_BOX").expanduser()
old = "wave"
new = "look at the box"

# 1) Update tasks.parquet (task list + indices)
tasks_path = root / "meta" / "tasks.parquet"
tasks = pd.read_parquet(tasks_path)

if old in tasks.index:
    tasks.rename(index={old: new}, inplace=True)
elif "task" in tasks.columns and old in tasks["task"].tolist():
    tasks.loc[tasks["task"] == old, "task"] = new
else:
    available = tasks.index.tolist()
    if "task" in tasks.columns:
        available = sorted(set(available + tasks["task"].tolist()))
    raise ValueError(
        f"'{old}' not found in tasks.parquet. Available tasks: {available}"
    )

tasks.to_parquet(tasks_path)

# 2) Update per-episode task strings
episodes_dir = root / "meta" / "episodes"
for p in episodes_dir.glob("chunk-*/file-*.parquet"):
    df = pd.read_parquet(p)
    if "tasks" in df.columns:
        df["tasks"] = df["tasks"].apply(
            lambda ts: [new if t == old else t for t in ts]
        )
        df.to_parquet(p)