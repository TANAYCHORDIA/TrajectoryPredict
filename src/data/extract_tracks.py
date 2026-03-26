import os
import json
from pathlib import Path
from typing import Iterable
import pandas as pd
from tqdm import tqdm


def _load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def extract_and_build_tracks(
    dataroot: str = "data/raw/v1.0-mini",
    output_file: str = "data/processed/tracks_raw.csv",
    expected_dt_us: int = 500_000,         # 0.5 sec in microseconds (2Hz)
    dt_tolerance_us: int = 10_000,         # ±10 ms tolerance
    min_segment_len: int = 10,             # need 4 obs + 6 future
    allowed_prefixes: Iterable[str] = ("human.pedestrian", "vehicle.bicycle"),
) -> None:
    """
    Build continuous trajectory tracks directly from NuScenes-style JSON tables.
    Splits trajectories on timestamp discontinuities and keeps only valid-length segments.
    """
    root = Path(dataroot)
    required = [
        "category.json",
        "instance.json",
        "sample.json",
        "scene.json",
        "sample_annotation.json",
    ]
    missing_files = [name for name in required if not (root / name).exists()]
    if missing_files:
        raise FileNotFoundError(
            f"Missing required files under {root}: {missing_files}"
        )

    print(f"Loading raw JSONs directly from {root}...")

    categories = {c["token"]: c["name"] for c in _load_json(root / "category.json")}
    instances = {
        i["token"]: categories.get(i["category_token"])
        for i in _load_json(root / "instance.json")
    }
    samples = {s["token"]: s for s in _load_json(root / "sample.json")}
    scenes = {s["token"]: s["name"] for s in _load_json(root / "scene.json")}
    annotations = _load_json(root / "sample_annotation.json")

    allowed_prefixes = tuple(allowed_prefixes)

    extracted = []
    stats = {
        "total_annotations": 0,
        "category_matched": 0,
        "missing_instance_or_category": 0,
        "missing_sample": 0,
        "missing_scene": 0,
        "bad_translation": 0,
    }

    print("Extracting target agent tracks...")
    for ann in tqdm(annotations):
        stats["total_annotations"] += 1

        inst_token = ann.get("instance_token")
        sample_token = ann.get("sample_token")
        category_name = instances.get(inst_token)

        if not category_name:
            stats["missing_instance_or_category"] += 1
            continue

        if not any(category_name.startswith(p) for p in allowed_prefixes):
            continue

        stats["category_matched"] += 1

        sample = samples.get(sample_token)
        if sample is None:
            stats["missing_sample"] += 1
            continue

        scene_name = scenes.get(sample.get("scene_token"))
        if scene_name is None:
            stats["missing_scene"] += 1
            continue

        translation = ann.get("translation")
        if not isinstance(translation, list) or len(translation) < 2:
            stats["bad_translation"] += 1
            continue

        extracted.append(
            {
                "scene_name": scene_name,
                "instance_token": inst_token,
                "sample_token": sample_token,
                "timestamp": sample["timestamp"],
                "x": translation[0],
                "y": translation[1],
                "category": category_name,
            }
        )

    if not extracted:
        raise ValueError("No candidate rows extracted. Check category filters and JSON integrity.")

    df = pd.DataFrame(extracted).sort_values(
        by=["scene_name", "instance_token", "timestamp"]
    ).reset_index(drop=True)

    print("Building continuous tracks and splitting discontinuities...")

    # Compute dt per (scene, instance)
    gcols = ["scene_name", "instance_token"]
    df["dt"] = df.groupby(gcols)["timestamp"].diff()

    lower = expected_dt_us - dt_tolerance_us
    upper = expected_dt_us + dt_tolerance_us

    # First row in each group is valid by definition
    first_in_group = df.groupby(gcols).cumcount() == 0
    dt_ok = df["dt"].abs().between(lower, upper, inclusive="both")
    df["valid_step"] = first_in_group | dt_ok

    # New segment starts where valid_step is False
    df["segment_id"] = (~df["valid_step"]).groupby([df[c] for c in gcols]).cumsum()

    # Keep only sufficiently long segments
    seg_cols = gcols + ["segment_id"]
    df["segment_len"] = df.groupby(seg_cols)["timestamp"].transform("size")
    clean_df = df[df["segment_len"] >= min_segment_len].copy()

    if clean_df.empty:
        raise ValueError("No valid tracks found after continuity and length filtering.")

    # Ensure unique trajectory IDs after splitting
    clean_df["instance_token"] = (
        clean_df["instance_token"] + "_seg" + clean_df["segment_id"].astype(str)
    )

    out_cols = [
        "scene_name",
        "instance_token",
        "sample_token",
        "timestamp",
        "x",
        "y",
        "category",
    ]
    clean_df = clean_df[out_cols].reset_index(drop=True)

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(out_path, index=False)

    # Diagnostics
    n_tracks = clean_df["instance_token"].nunique()
    print("\nExtraction summary")
    print(f"  Total annotations scanned: {stats['total_annotations']}")
    print(f"  Category matched:          {stats['category_matched']}")
    print(f"  Missing instance/category: {stats['missing_instance_or_category']}")
    print(f"  Missing sample:            {stats['missing_sample']}")
    print(f"  Missing scene:             {stats['missing_scene']}")
    print(f"  Bad translation:           {stats['bad_translation']}")
    print(f"\nClean tracks saved: {out_path}")
    print(f"Total valid waypoints: {len(clean_df)}")
    print(f"Total unique tracks:   {n_tracks}")


if __name__ == "__main__":
    extract_and_build_tracks()