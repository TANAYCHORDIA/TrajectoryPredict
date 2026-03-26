import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def compute_velocities(x, y):
    """Computes dx, dy as first-order differences. Frame 0 is (0,0)."""
    dx = np.zeros_like(x)
    dy = np.zeros_like(y)
    dx[1:] = x[1:] - x[:-1]
    dy[1:] = y[1:] - y[:-1]
    return dx, dy

def process_pipeline(
    input_csv="data/processed/tracks_raw.csv",
    output_dir="data/processed/",
    obs_len=4,
    pred_len=6,
    stride=1,
    random_state=42,
):
    print("Loading raw tracks...")
    df = pd.read_csv(input_csv)

    required_cols = {"scene_name", "timestamp", "x", "y"}
    missing_cols = required_cols.difference(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in {input_csv}: {sorted(missing_cols)}")

    id_col = "track_id" if "track_id" in df.columns else "instance_token"
    if id_col not in df.columns:
        raise ValueError("Expected either 'track_id' or 'instance_token' column in input CSV.")

    if stride <= 0:
        raise ValueError("stride must be >= 1")

    total_len = obs_len + pred_len
    if total_len <= 1:
        raise ValueError("obs_len + pred_len must be >= 2")

    # Deterministic ordering protects reproducibility of sliding windows.
    df = df.sort_values(["scene_name", id_col, "timestamp"]).reset_index(drop=True)

    # Scene-level split to prevent data leakage.
    scenes = np.sort(df["scene_name"].unique())
    if len(scenes) < 3:
        raise ValueError(
            f"Need at least 3 scenes for train/val/test split, found {len(scenes)}."
        )

    train_scenes, temp_scenes = train_test_split(
        scenes, test_size=0.30, random_state=random_state
    )
    val_scenes, test_scenes = train_test_split(
        temp_scenes, test_size=0.50, random_state=random_state
    )

    scene_splits = {
        "train": train_scenes,
        "val": val_scenes,
        "test": test_scenes,
    }

    os.makedirs(output_dir, exist_ok=True)

    for split_name, scene_list in scene_splits.items():
        print(f"Processing {split_name} split...")
        split_df = df[df["scene_name"].isin(scene_list)]

        inputs_list = []
        targets_list = []

        grouped = split_df.groupby(["scene_name", id_col], sort=False)

        for _, group in tqdm(grouped, leave=False):
            group = group.sort_values("timestamp").reset_index(drop=True)
            if len(group) < total_len:
                continue

            x = group["x"].to_numpy(dtype=np.float32)
            y = group["y"].to_numpy(dtype=np.float32)

            # Sliding window over each trajectory.
            for i in range(0, len(group) - total_len + 1, stride):
                win_x = x[i : i + total_len]
                win_y = y[i : i + total_len]

                # Anchor at the last observed frame.
                anchor_idx = obs_len - 1
                anchor_x = win_x[anchor_idx]
                anchor_y = win_y[anchor_idx]

                trans_x = win_x - anchor_x
                trans_y = win_y - anchor_y

                # Estimate heading from observed segment; fallback to latest non-zero motion.
                obs_tx = trans_x[:obs_len]
                obs_ty = trans_y[:obs_len]
                obs_dx, obs_dy = compute_velocities(obs_tx, obs_ty)
                non_zero = (obs_dx != 0) | (obs_dy != 0)
                if np.any(non_zero):
                    nz_idx = np.flatnonzero(non_zero)[-1]
                    heading_dx = obs_dx[nz_idx]
                    heading_dy = obs_dy[nz_idx]
                    theta = np.arctan2(heading_dy, heading_dx)
                else:
                    # Fully stationary observed history: keep canonical orientation.
                    theta = 0.0

                c, s = np.cos(-theta), np.sin(-theta)
                R = np.array([[c, -s], [s, c]], dtype=np.float32)

                positions = np.stack([trans_x, trans_y], axis=1)
                rot_positions = positions @ R.T

                # Compute velocity in the rotated frame for full consistency.
                rot_dx = np.zeros_like(rot_positions[:, 0])
                rot_dy = np.zeros_like(rot_positions[:, 1])
                rot_dx[1:] = rot_positions[1:, 0] - rot_positions[:-1, 0]
                rot_dy[1:] = rot_positions[1:, 1] - rot_positions[:-1, 1]
                rot_velocities = np.stack([rot_dx, rot_dy], axis=1)

                input_tensor = np.hstack([rot_positions[:obs_len], rot_velocities[:obs_len]])
                target_tensor = rot_positions[obs_len:]

                inputs_list.append(input_tensor)
                targets_list.append(target_tensor)

        inputs_path = os.path.join(output_dir, f"{split_name}_inputs.npy")
        targets_path = os.path.join(output_dir, f"{split_name}_targets.npy")

        if inputs_list:
            inputs_arr = np.asarray(inputs_list, dtype=np.float32)
            targets_arr = np.asarray(targets_list, dtype=np.float32)

            # Sanity checks before persisting.
            if inputs_arr.shape[1:] != (obs_len, 4):
                raise ValueError(
                    f"Unexpected input shape for {split_name}: {inputs_arr.shape}, expected (*, {obs_len}, 4)"
                )
            if targets_arr.shape[1:] != (pred_len, 2):
                raise ValueError(
                    f"Unexpected target shape for {split_name}: {targets_arr.shape}, expected (*, {pred_len}, 2)"
                )
            if np.isnan(inputs_arr).any() or np.isnan(targets_arr).any():
                raise ValueError(f"NaN detected in {split_name} arrays.")

            np.save(inputs_path, inputs_arr)
            np.save(targets_path, targets_arr)
            print(f"Saved {split_name}: {len(inputs_arr)} samples.")
        else:
            # Keep pipeline behavior explicit: write empty arrays with correct rank.
            empty_inputs = np.empty((0, obs_len, 4), dtype=np.float32)
            empty_targets = np.empty((0, pred_len, 2), dtype=np.float32)
            np.save(inputs_path, empty_inputs)
            np.save(targets_path, empty_targets)
            print(f"Saved {split_name}: 0 samples (empty arrays).")

if __name__ == "__main__":
    process_pipeline()