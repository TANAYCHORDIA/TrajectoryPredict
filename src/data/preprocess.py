import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def compute_velocities(x, y):
    dx = np.zeros_like(x, dtype=np.float32)
    dy = np.zeros_like(y, dtype=np.float32)
    dx[1:] = x[1:] - x[:-1]
    dy[1:] = y[1:] - y[:-1]
    return dx, dy


def process_pipeline(
    input_csv="data/processed/tracks_raw.csv",
    social_npz="data/processed/social_features.npz",
    output_dir="data/processed/",
    obs_len=4,
    pred_len=6,
    stride=1,
    random_state=42,
    expected_dt_us=500_000,
    dt_tolerance_us=10_000,
    min_motion_eps=1e-5,
):
    print("Loading raw tracks...")
    df = pd.read_csv(input_csv)

    # ---------- Validation ----------
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
    if total_len < 2:
        raise ValueError("obs_len + pred_len must be >= 2")

    # Deterministic ordering
    df = df.sort_values(["scene_name", id_col, "timestamp"]).reset_index(drop=True)

    # ---------- Load social features ----------
    print("Loading social features and building O(1) lookup dictionary...")
    soc = np.load(social_npz, allow_pickle=False)

    # Expected keys from social builder
    needed_soc_keys = {"scene_name", "timestamp", "instance_token", "neighbors", "mask", "max_neighbors"}
    missing_soc_keys = needed_soc_keys.difference(set(soc.files))
    if missing_soc_keys:
        raise ValueError(f"Missing keys in {social_npz}: {sorted(missing_soc_keys)}")

    K = int(np.asarray(soc["max_neighbors"]).item())

    soc_scene = soc["scene_name"]
    soc_ts = soc["timestamp"]
    soc_tok = soc["instance_token"]
    soc_neighbors = soc["neighbors"]  # [M, K, 2]
    soc_mask = soc["mask"]            # [M, K]

    if soc_neighbors.ndim != 3 or soc_neighbors.shape[1:] != (K, 2):
        raise ValueError(f"Unexpected social neighbor shape: {soc_neighbors.shape}, expected [M, {K}, 2]")
    if soc_mask.ndim != 2 or soc_mask.shape[1] != K:
        raise ValueError(f"Unexpected social mask shape: {soc_mask.shape}, expected [M, {K}]")

    # Key = (scene, token, timestamp) to avoid cross-scene collisions
    soc_dict = {}
    for sc, ts, tok, n, m in zip(soc_scene, soc_ts, soc_tok, soc_neighbors, soc_mask):
        soc_dict[(str(sc), str(tok), int(ts))] = (n.astype(np.float32), m.astype(np.float32))

    # ---------- Split by scene ----------
    scenes = np.sort(df["scene_name"].unique())
    if len(scenes) < 3:
        raise ValueError(f"Need at least 3 scenes for train/val/test split, found {len(scenes)}.")

    train_scenes, temp_scenes = train_test_split(scenes, test_size=0.30, random_state=random_state)
    val_scenes, test_scenes = train_test_split(temp_scenes, test_size=0.50, random_state=random_state)

    scene_splits = {"train": train_scenes, "val": val_scenes, "test": test_scenes}
    os.makedirs(output_dir, exist_ok=True)

    dt_low = expected_dt_us - dt_tolerance_us
    dt_high = expected_dt_us + dt_tolerance_us

    for split_name, scene_list in scene_splits.items():
        print(f"Processing {split_name} split...")
        split_df = df[df["scene_name"].isin(scene_list)]

        inputs_list, targets_list = [], []
        social_list, mask_list = [], []

        grouped = split_df.groupby(["scene_name", id_col], sort=False)

        for (scene_name, _), group in tqdm(grouped, leave=False):
            group = group.sort_values("timestamp").reset_index(drop=True)
            if len(group) < total_len:
                continue

            x = group["x"].to_numpy(dtype=np.float32)
            y = group["y"].to_numpy(dtype=np.float32)
            ts = group["timestamp"].to_numpy(dtype=np.int64)
            token_series = group["instance_token"].astype(str).to_numpy()
            token0 = token_series[0]  # within split tracks, token should be stable for social lookup

            # Prebuild social arrays aligned with the track timeline (avoid per-window dict loops)
            track_neighbors = np.zeros((len(group), K, 2), dtype=np.float32)
            track_masks = np.zeros((len(group), K), dtype=np.float32)
            for t_idx, tstamp in enumerate(ts):
                key = (str(scene_name), token0, int(tstamp))
                item = soc_dict.get(key)
                if item is not None:
                    track_neighbors[t_idx] = item[0]
                    track_masks[t_idx] = item[1]

            for i in range(0, len(group) - total_len + 1, stride):
                # Window slices
                win_x = x[i:i + total_len]
                win_y = y[i:i + total_len]
                win_ts = ts[i:i + total_len]

                # Continuity check to avoid crossing temporal gaps
                dts = np.diff(win_ts)
                if len(dts) > 0 and not np.all((dts >= dt_low) & (dts <= dt_high)):
                    continue

                # Anchor at last observed timestep
                anchor_idx = obs_len - 1
                trans_x = win_x - win_x[anchor_idx]
                trans_y = win_y - win_y[anchor_idx]

                positions = np.stack([trans_x, trans_y], axis=1)  # [T,2]

                # Heading from observed motion, robust fallback
                obs_pos = positions[:obs_len]
                obs_dx, obs_dy = compute_velocities(obs_pos[:, 0], obs_pos[:, 1])
                speed = np.sqrt(obs_dx**2 + obs_dy**2)
                nz = np.flatnonzero(speed > min_motion_eps)
                if len(nz) > 0:
                    j = nz[-1]
                    theta = np.arctan2(obs_dy[j], obs_dx[j])
                else:
                    theta = 0.0

                c, s = np.cos(-theta), np.sin(-theta)
                R = np.array([[c, -s], [s, c]], dtype=np.float32)

                # Rotate positions
                rot_positions = positions @ R.T  # [T,2]

                # Derive velocity AFTER rotation for frame consistency
                rot_dx, rot_dy = compute_velocities(rot_positions[:, 0], rot_positions[:, 1])
                rot_velocities = np.stack([rot_dx, rot_dy], axis=1)  # [T,2]

                # Social for observation horizon only
                win_neighbors = track_neighbors[i:i + obs_len].copy()  # [obs,K,2]
                win_masks = track_masks[i:i + obs_len].copy()          # [obs,K]

                # Rotate only valid neighbor vectors
                valid = win_masks > 0.5
                flat_n = win_neighbors.reshape(-1, 2)
                flat_v = valid.reshape(-1)
                if np.any(flat_v):
                    flat_n_valid = flat_n[flat_v] @ R.T
                    flat_n[flat_v] = flat_n_valid
                rot_neighbors = flat_n.reshape(obs_len, K, 2)

                # Final tensors
                input_tensor = np.hstack([rot_positions[:obs_len], rot_velocities[:obs_len]])  # [obs,4]
                target_tensor = rot_positions[obs_len:]                                           # [pred,2]

                inputs_list.append(input_tensor)
                targets_list.append(target_tensor)
                social_list.append(rot_neighbors)
                mask_list.append(win_masks)

        # Save split
        inputs_path = os.path.join(output_dir, f"{split_name}_inputs.npy")
        targets_path = os.path.join(output_dir, f"{split_name}_targets.npy")
        social_path = os.path.join(output_dir, f"{split_name}_social.npy")
        mask_path = os.path.join(output_dir, f"{split_name}_mask.npy")

        if inputs_list:
            inputs_arr = np.asarray(inputs_list, dtype=np.float32)
            targets_arr = np.asarray(targets_list, dtype=np.float32)
            social_arr = np.asarray(social_list, dtype=np.float32)
            mask_arr = np.asarray(mask_list, dtype=np.float32)

            # Sanity checks
            if inputs_arr.shape[1:] != (obs_len, 4):
                raise ValueError(f"{split_name} inputs shape bad: {inputs_arr.shape}")
            if targets_arr.shape[1:] != (pred_len, 2):
                raise ValueError(f"{split_name} targets shape bad: {targets_arr.shape}")
            if social_arr.shape[1:] != (obs_len, K, 2):
                raise ValueError(f"{split_name} social shape bad: {social_arr.shape}")
            if mask_arr.shape[1:] != (obs_len, K):
                raise ValueError(f"{split_name} mask shape bad: {mask_arr.shape}")

            if (
                np.isnan(inputs_arr).any()
                or np.isnan(targets_arr).any()
                or np.isnan(social_arr).any()
                or np.isnan(mask_arr).any()
            ):
                raise ValueError(f"NaN detected in {split_name} outputs.")

            np.save(inputs_path, inputs_arr)
            np.save(targets_path, targets_arr)
            np.save(social_path, social_arr)
            np.save(mask_path, mask_arr)
            print(f"Saved {split_name}: {len(inputs_arr)} samples.")
        else:
            # Keep empty files with consistent rank
            np.save(inputs_path, np.empty((0, obs_len, 4), dtype=np.float32))
            np.save(targets_path, np.empty((0, pred_len, 2), dtype=np.float32))
            np.save(social_path, np.empty((0, obs_len, K, 2), dtype=np.float32))
            np.save(mask_path, np.empty((0, obs_len, K), dtype=np.float32))
            print(f"Saved {split_name}: 0 samples (empty arrays).")


if __name__ == "__main__":
    process_pipeline()