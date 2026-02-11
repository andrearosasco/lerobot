import argparse
import shutil
import json
import sys
from pathlib import Path
from typing import Any
import copy

# Aggiungiamo src al path per caricare la versione locale
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    # IMPORT AGGIORNATI PER LEROBOT v3.0
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import (
        get_hf_features_from_features,
        load_info,
        to_parquet_with_hf_images,
        write_info,
    )
    from huggingface_hub import HfApi, create_repo
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"❌ Errore di importazione: {e}")
    sys.exit(1)


def _load_info(root: Path) -> dict[str, Any]:
    # Use LeRobot loader to normalize shapes to tuples
    return load_info(root)


def _update_info_for_fingers(info: dict[str, Any]) -> tuple[dict[str, Any], list[int], list[str]]:
    if "observation.state" not in info.get("features", {}):
        raise ValueError("observation.state non trovato in info.json")

    obs_state_info = info["features"]["observation.state"]
    original_names = obs_state_info.get("names", [])
    indices_to_keep = [i for i, name in enumerate(original_names) if "finger" not in name.lower()]
    indices_to_remove = [i for i, name in enumerate(original_names) if "finger" in name.lower()]

    if not indices_to_remove:
        raise ValueError("Nessun finger trovato in observation.state.names")

    filtered_names = [original_names[i] for i in indices_to_keep]
    info["features"]["observation.state"]["names"] = filtered_names
    info["features"]["observation.state"]["shape"] = [len(filtered_names)]
    return info, indices_to_keep, [original_names[i] for i in indices_to_remove]


def _filter_stats(stats: dict[str, Any], indices_to_keep: list[int], original_len: int) -> dict[str, Any]:
    if "observation.state" not in stats:
        return stats

    obs_stats = stats["observation.state"]
    for stat_key, stat_val in list(obs_stats.items()):
        if isinstance(stat_val, list) and len(stat_val) == original_len:
            obs_stats[stat_key] = [stat_val[i] for i in indices_to_keep]
    return stats


def _verify_no_fingers_in_info(info: dict[str, Any]) -> None:
    names = info.get("features", {}).get("observation.state", {}).get("names", [])
    if any("finger" in name.lower() for name in names):
        raise ValueError("Trovati 'finger' in observation.state.names dopo l'aggiornamento")


def _ensure_list_value(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist() if value.ndim > 0 else [value.item()]
    return [value]


def _normalize_list_columns(df: pd.DataFrame, features: dict[str, Any]) -> pd.DataFrame:
    for key, spec in features.items():
        if key not in df.columns:
            continue
        shape = spec.get("shape")
        if shape is None:
            continue
        if shape == (1,):
            df[key] = df[key].apply(
                lambda v: _ensure_list_value(v)[0]
            )
    return df


def remove_finger_observations(old_repo_id, new_repo_id, local_dir, revision=None):
    print(f"🚀 Caricamento dataset originale: {old_repo_id}")
    if revision is not None:
        print(f"🔖 Usando revision: {revision}")
    dataset = LeRobotDataset(old_repo_id, force_cache_sync=True, revision=revision)
    root = Path(dataset.root)
    print(f"DATASET ROOT: {dataset.root}")
    print("DATASET LENGTH:", len(dataset))

    info = _load_info(root)
    original_info = copy.deepcopy(info)
    original_len = len(info["features"]["observation.state"].get("names", []))
    info, indices_to_keep, removed_names = _update_info_for_fingers(info)

    print(f"🗑️ Rimozione {len(removed_names)} dimensioni finger da observation.state")
    print(f"   Nomi rimossi: {removed_names}")

    output_path = Path(local_dir)
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    # Copia meta, videos e file di repo
    meta_src = root / "meta"
    if meta_src.exists():
        shutil.copytree(meta_src, output_path / "meta", dirs_exist_ok=True)
    videos_src = root / "videos"
    if videos_src.exists():
        shutil.copytree(videos_src, output_path / "videos", dirs_exist_ok=True)
    for fname in [".gitattributes", "README.md"]:
        src = root / fname
        if src.exists():
            shutil.copy2(src, output_path / fname)

    # Scrivi parquet filtrati con schema aggiornato
    data_src = root / "data"
    parquet_files = sorted(data_src.glob("*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"Nessun parquet trovato in {data_src}")

    write_features = get_hf_features_from_features(info["features"])

    for src_path in parquet_files:
        df = pd.read_parquet(src_path).reset_index(drop=True)
        if "observation.state" not in df.columns:
            raise ValueError(f"Colonna observation.state non trovata in {src_path}")
        df["observation.state"] = df["observation.state"].apply(
            lambda row: [row[i] for i in indices_to_keep]
        )
        df = _normalize_list_columns(df, info["features"])

        rel_path = src_path.relative_to(root)
        dst_path = output_path / rel_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        to_parquet_with_hf_images(df, dst_path, features=write_features)

    # Sanity check: ensure data parquet exists
    written_parquets = sorted((output_path / "data").glob("*/*.parquet"))
    if not written_parquets:
        raise FileNotFoundError(f"Nessun parquet scritto in {output_path / 'data'}")

    # Aggiorna info.json
    write_info(info, output_path)

    _verify_no_fingers_in_info(info)

    # Aggiorna stats.json
    stats_src = root / "meta" / "stats.json"
    if stats_src.exists():
        with open(stats_src, "r") as f:
            stats = json.load(f)
        stats = _filter_stats(stats, indices_to_keep, original_len)
        with open(output_path / "meta" / "stats.json", "w") as f:
            json.dump(stats, f, indent=4)

    print("✅ Verifica: observation.state.names non contiene 'finger'")

    # Upload
    print(f"☁️ Uploading su HF: {new_repo_id}")
    api = HfApi()
    create_repo(repo_id=new_repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=str(output_path),
        repo_id=new_repo_id,
        repo_type="dataset",
    )

    print(f"\n✅ Dataset caricato: https://huggingface.co/datasets/{new_repo_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--old_repo", type=str, default="steb6/hri_ste_carmela")
    parser.add_argument("--new_repo", type=str, default="steb6/hri_ste_carmela_no_fingers")
    parser.add_argument("--local_dir", type=str, default="data/clean_temp")
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="HF revision/tag/commit to load (e.g. 'main' or 'v3.0')",
    )
    args = parser.parse_args()
    
    # Auto-append no_fingers if new_repo equals old_repo
    if args.new_repo == args.old_repo:
        args.new_repo = args.old_repo + "_no_fingers"
    
    remove_finger_observations(args.old_repo, args.new_repo, args.local_dir, args.revision)