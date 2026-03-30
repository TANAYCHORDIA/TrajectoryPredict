import os
import numpy as np
import pandas as pd
from tqdm import tqdm


def build_social_tensors(
    input_csv="data/processed/tracks_raw.csv",
    output_file="data/processed/social_features.npz",
    radius=2.0,
    max_neighbors=4,
):
    print("Loading raw tracks for social pooling...")
    df = pd.read_csv(input_csv)

    required_cols = {"scene_name", "timestamp", "instance_token", "x", "y"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("Input CSV is empty; cannot build social tensors.")

    # Drop rows with invalid coordinates/timestamps
    df = df.dropna(subset=["scene_name", "timestamp", "instance_token", "x", "y"]).copy()

    # Deterministic ordering
    df = df.sort_values(["scene_name", "timestamp", "instance_token"]).reset_index(drop=True)

    r2 = float(radius) ** 2
    K = int(max_neighbors)
    if K <= 0:
        raise ValueError("max_neighbors must be >= 1")

    key_scene = []
    key_timestamp = []
    key_token = []
    all_neighbors = []
    all_masks = []

    grouped = df.groupby(["scene_name", "timestamp"], sort=False)

    print(f"Extracting neighbors (Radius: {radius}m, Max K: {K})...")
    for (scene, ts), group in tqdm(grouped, leave=False):
        tokens = group["instance_token"].to_numpy()
        coords = group[["x", "y"]].to_numpy(dtype=np.float32)  # [N,2]
        N = coords.shape[0]

        if N == 0:
            continue

        # delta[i,j] = p_j - p_i  (FROM i TO j)
        deltas = coords[np.newaxis, :, :] - coords[:, np.newaxis, :]  # [N,N,2]
        dist_sq = np.sum(deltas * deltas, axis=2)  # [N,N]
        np.fill_diagonal(dist_sq, np.inf)


        # Number of possible non-self neighbors per agent
        avail = max(0, N - 1)

        if avail == 0:
            topk_deltas = np.zeros((N, K, 2), dtype=np.float32)
            topk_mask = np.zeros((N, K), dtype=np.float32)
        else:
            k_eff = min(K, avail)

            # Closest available neighbors
            topk_idx_eff = np.argsort(dist_sq, axis=1)[:, :k_eff]      # [N, k_eff]
            topk_dist_eff = np.take_along_axis(dist_sq, topk_idx_eff, axis=1)  # [N, k_eff]

            # Pad to fixed K with sentinel -1
            topk_idx = np.full((N, K), -1, dtype=np.int64)
            topk_idx[:, :k_eff] = topk_idx_eff

            topk_dist = np.full((N, K), np.inf, dtype=np.float32)
            topk_dist[:, :k_eff] = topk_dist_eff.astype(np.float32)

            # Safe gather (replace -1 temporarily by 0, then mask out)
            safe_idx = np.where(topk_idx < 0, 0, topk_idx)             # [N, K]
            row_idx = np.arange(N)[:, None]                            # [N, 1]
            gathered = deltas[row_idx, safe_idx, :]                    # [N, K, 2]

            valid = (topk_idx >= 0) & (topk_dist <= r2)                # [N, K]
            topk_deltas = np.where(valid[..., None], gathered, 0.0).astype(np.float32)
            topk_mask = valid.astype(np.float32)

        key_scene.extend([scene] * N)
        key_timestamp.extend([ts] * N)
        key_token.extend(tokens.tolist())
        all_neighbors.append(topk_deltas)
        all_masks.append(topk_mask)

    if not all_neighbors:
        raise ValueError("No frame-agent pairs processed. Check input filtering/grouping.")

    neighbors_arr = np.concatenate(all_neighbors, axis=0).astype(np.float32)  # [M,K,2]
    mask_arr = np.concatenate(all_masks, axis=0).astype(np.float32)            # [M,K]

    # Sanity checks
    if neighbors_arr.ndim != 3 or neighbors_arr.shape[1:] != (K, 2):
        raise ValueError(f"Unexpected neighbors shape: {neighbors_arr.shape}, expected [M,{K},2]")
    if mask_arr.ndim != 2 or mask_arr.shape[1] != K:
        raise ValueError(f"Unexpected mask shape: {mask_arr.shape}, expected [M,{K}]")
    if np.isnan(neighbors_arr).any() or np.isnan(mask_arr).any():
        raise ValueError("NaN detected in output tensors.")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"\nSaving social features to {output_file}...")
    np.savez_compressed(
        output_file,
        scene_name=np.asarray(key_scene),
        timestamp=np.asarray(key_timestamp),
        instance_token=np.asarray(key_token),
        neighbors=neighbors_arr,
        mask=mask_arr,
        radius=np.float32(radius),
        max_neighbors=np.int32(K),
    )

    print("\n--- SOCIAL TENSOR SANITY CHECK ---")
    print(f"Total frame-agent pairs processed: {neighbors_arr.shape[0]}")
    print(f"Neighbor Tensor Shape: {neighbors_arr.shape} -> Expected [M, {K}, 2]")
    print(f"Mask Tensor Shape:     {mask_arr.shape} -> Expected [M, {K}]")
    print("✅ SUCCESS: Vectorized social pooling complete.")


if __name__ == "__main__":
    build_social_tensors()