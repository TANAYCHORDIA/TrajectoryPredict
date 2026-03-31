from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.metrics import minade_minfde
from src.model import TrajectoryPredictor

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT_DIR / "outputs" / "checkpoints" / "best_model_final.pth"
DEFAULT_OBS_PATH = ROOT_DIR / "data" / "processed" / "val_inputs.npy"
DEFAULT_FUT_PATH = ROOT_DIR / "data" / "processed" / "val_targets.npy"
DEFAULT_SOCIAL_PATH = ROOT_DIR / "data" / "processed" / "val_social.npy"
DEFAULT_MASK_PATH = ROOT_DIR / "data" / "processed" / "val_mask.npy"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "outputs" / "predictions" / "latest_prediction.npz"


class EndToEndPredictor:
    """
    Production wrapper for Trajectory Prediction.
    Takes raw global map coordinates, normalizes them, runs inference, 
    and returns multimodal predictions in the original global map coordinate frame.
    """
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = TrajectoryPredictor().to(self.device)
        
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def _compute_velocities(self, positions: np.ndarray) -> np.ndarray:
        vel = np.zeros_like(positions, dtype=np.float32)
        vel[1:] = positions[1:] - positions[:-1]
        return vel

    def predict_global(
        self, 
        global_obs: np.ndarray, 
        global_social: np.ndarray = None, 
        mask: np.ndarray = None
    ) -> np.ndarray:
        """
        global_obs: [4, 2] array of past (x, y) map coordinates
        global_social: [4, K, 2] array of neighbors' (x, y) map coordinates
        mask: [4, K] boolean array of valid neighbors
        
        Returns: [3, 6, 2] array containing 3 multimodal future trajectories 
                 in global (x, y) map coordinates.
        """
        # 1. EXTRACT ANCHOR & HEADING
        anchor_idx = len(global_obs) - 1
        anchor_pos = global_obs[anchor_idx]
        
        obs_vel = self._compute_velocities(global_obs)
        anchor_vel = obs_vel[anchor_idx]
        
        # Calculate angle of motion
        theta = 0.0 if (anchor_vel[0] == 0 and anchor_vel[1] == 0) else np.arctan2(anchor_vel[1], anchor_vel[0])
        
        # 2. FORWARD ROTATION (World -> Local)
        c, s = np.cos(-theta), np.sin(-theta)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)
        
        local_obs_pos = (global_obs - anchor_pos) @ R.T
        local_obs_vel = obs_vel @ R.T
        local_obs = np.hstack([local_obs_pos, local_obs_vel]) # Shape: [4, 4]

        # Rotate Social Tensors
        local_social = None
        if global_social is not None and mask is not None:
            local_social = np.zeros_like(global_social, dtype=np.float32)
            valid = mask > 0.5
            flat_soc = global_social.reshape(-1, 2)
            flat_v = valid.reshape(-1)
            
            if np.any(flat_v):
                translated = flat_soc[flat_v] - anchor_pos
                flat_soc[flat_v] = translated @ R.T
            
            local_social = flat_soc.reshape(global_social.shape)

        # 3. MODEL INFERENCE
        obs_tensor = torch.tensor(local_obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        soc_tensor = None
        mask_tensor = None
        if local_social is not None:
            soc_tensor = torch.tensor(local_social, dtype=torch.float32).unsqueeze(0).to(self.device)
            mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Model outputs local normalized coordinates [1, 3, 6, 2] -> squeezed to [3, 6, 2]
            local_preds = self.model(obs_tensor, soc_tensor, mask_tensor).squeeze(0).cpu().numpy()

        # 4. INVERSE ROTATION (Local -> World)
        inv_c, inv_s = np.cos(theta), np.sin(theta)
        R_inv = np.array([[inv_c, -inv_s], [inv_s, inv_c]], dtype=np.float32)
        
        # Broadcasting applies [2, 2] rotation to the [3, 6, 2] tensor
        global_preds = (local_preds @ R_inv.T) + anchor_pos

        return global_preds


def run_inference(
    checkpoint_path: Path | str = DEFAULT_CHECKPOINT,
    obs_path: Path | str = DEFAULT_OBS_PATH,
    fut_path: Path | str = DEFAULT_FUT_PATH,
    sample_idx: int = 0,
    social_path: Path | str | None = DEFAULT_SOCIAL_PATH,
    mask_path: Path | str | None = DEFAULT_MASK_PATH,
    output_path: Path | str | None = DEFAULT_OUTPUT_PATH,
    device: str | None = None,
) -> dict[str, np.ndarray | float | int]:
    """
    Run model inference on one dataset sample (already normalized local frame).

    Returns a dictionary containing obs, gt, preds, min_ade, min_fde, sample_idx.
    """
    checkpoint_path = Path(checkpoint_path)
    obs_path = Path(obs_path)
    fut_path = Path(fut_path)
    social_path = Path(social_path) if social_path is not None else None
    mask_path = Path(mask_path) if mask_path is not None else None
    output_path = Path(output_path) if output_path is not None else None

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not obs_path.exists():
        raise FileNotFoundError(f"Observed input file not found: {obs_path}")
    if not fut_path.exists():
        raise FileNotFoundError(f"Future target file not found: {fut_path}")

    obs_all = np.load(obs_path, allow_pickle=False)
    fut_all = np.load(fut_path, allow_pickle=False)

    if obs_all.ndim != 3 or obs_all.shape[-1] != 4:
        raise ValueError(f"Invalid obs tensor shape: {obs_all.shape}; expected [N, obs_len, 4]")
    if fut_all.ndim != 3 or fut_all.shape[-1] != 2:
        raise ValueError(f"Invalid target tensor shape: {fut_all.shape}; expected [N, pred_len, 2]")
    if len(obs_all) != len(fut_all):
        raise ValueError(f"obs/target length mismatch: {len(obs_all)} vs {len(fut_all)}")

    if sample_idx < 0 or sample_idx >= len(obs_all):
        raise IndexError(f"sample_idx {sample_idx} is out of range [0, {len(obs_all)-1}]")

    social_all = None
    mask_all = None
    if social_path is not None and mask_path is not None and social_path.exists() and mask_path.exists():
        social_all = np.load(social_path, allow_pickle=False)
        mask_all = np.load(mask_path, allow_pickle=False)
        if len(social_all) != len(obs_all) or len(mask_all) != len(obs_all):
            raise ValueError(
                "social/mask length mismatch with obs: "
                f"social={len(social_all)}, mask={len(mask_all)}, obs={len(obs_all)}"
            )

    chosen_device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TrajectoryPredictor().to(chosen_device)
    state_dict = torch.load(checkpoint_path, map_location=chosen_device)
    model.load_state_dict(state_dict)
    model.eval()

    obs = obs_all[sample_idx].astype(np.float32)
    gt = fut_all[sample_idx].astype(np.float32)

    obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(chosen_device)

    social_tensor = None
    mask_tensor = None
    if social_all is not None and mask_all is not None:
        social_tensor = torch.from_numpy(social_all[sample_idx].astype(np.float32)).unsqueeze(0).to(chosen_device)
        mask_tensor = torch.from_numpy(mask_all[sample_idx].astype(np.float32)).unsqueeze(0).to(chosen_device)

    with torch.no_grad():
        preds = model(obs_tensor, social_tensor, mask_tensor).squeeze(0).cpu().numpy()

    gt_tensor = torch.from_numpy(gt)
    preds_tensor = torch.from_numpy(preds)
    min_ade, min_fde = minade_minfde(preds_tensor, gt_tensor)

    result = {
        "obs": obs,
        "gt": gt,
        "preds": preds,
        "min_ade": float(min_ade),
        "min_fde": float(min_fde),
        "sample_idx": int(sample_idx),
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            obs=obs,
            gt=gt,
            preds=preds,
            min_ade=np.float32(result["min_ade"]),
            min_fde=np.float32(result["min_fde"]),
            sample_idx=np.int32(sample_idx),
        )

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run trajectory model inference for one sample.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--obs-path", type=Path, default=DEFAULT_OBS_PATH)
    parser.add_argument("--fut-path", type=Path, default=DEFAULT_FUT_PATH)
    parser.add_argument("--social-path", type=Path, default=DEFAULT_SOCIAL_PATH)
    parser.add_argument("--mask-path", type=Path, default=DEFAULT_MASK_PATH)
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    out = run_inference(
        checkpoint_path=args.checkpoint,
        obs_path=args.obs_path,
        fut_path=args.fut_path,
        social_path=args.social_path,
        mask_path=args.mask_path,
        sample_idx=args.sample_idx,
        output_path=args.output_path,
    )
    print(f"Sample {out['sample_idx']} | minADE={out['min_ade']:.4f} | minFDE={out['min_fde']:.4f}")
    print(f"Saved prediction to: {args.output_path}")