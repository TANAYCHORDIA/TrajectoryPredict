from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.data.dataset import TrajectoryDataset
from src.metrics import minade_minfde
from src.model import TrajectoryPredictor

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT_DIR / "outputs" / "checkpoints" / "best_model.pth"
DEFAULT_DATA_DIR = ROOT_DIR / "data" / "processed"
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
        mask: np.ndarray = None,
    ) -> np.ndarray:
        """
        global_obs: [4, 2] array of past (x, y) map coordinates
        global_social: [4, K, 2] array of neighbors' (x, y) map coordinates
        mask: [4, K] boolean array of valid neighbors

        Returns: [3, 6, 2] array containing 3 multimodal future trajectories
        in global (x, y) map coordinates.
        """
        anchor_idx = len(global_obs) - 1
        anchor_pos = global_obs[anchor_idx]

        obs_vel = self._compute_velocities(global_obs)
        anchor_vel = obs_vel[anchor_idx]

        theta = (
            0.0
            if (anchor_vel[0] == 0 and anchor_vel[1] == 0)
            else np.arctan2(anchor_vel[1], anchor_vel[0])
        )

        c, s = np.cos(-theta), np.sin(-theta)
        rotation = np.array([[c, -s], [s, c]], dtype=np.float32)

        local_obs_pos = (global_obs - anchor_pos) @ rotation.T
        local_obs_vel = obs_vel @ rotation.T
        local_obs = np.hstack([local_obs_pos, local_obs_vel])

        local_social = None
        if global_social is not None and mask is not None:
            local_social = np.zeros_like(global_social, dtype=np.float32)
            valid = mask > 0.5
            flat_social = global_social.reshape(-1, 2).copy()
            flat_valid = valid.reshape(-1)

            if np.any(flat_valid):
                translated = flat_social[flat_valid] - anchor_pos
                flat_social[flat_valid] = translated @ rotation.T

            local_social = flat_social.reshape(global_social.shape)

        obs_tensor = torch.tensor(local_obs, dtype=torch.float32).unsqueeze(0).to(self.device)

        social_tensor = None
        mask_tensor = None
        if local_social is not None:
            social_tensor = (
                torch.tensor(local_social, dtype=torch.float32).unsqueeze(0).to(self.device)
            )
            mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            local_preds = self.model(obs_tensor, social_tensor, mask_tensor).squeeze(0).cpu().numpy()

        inv_c, inv_s = np.cos(theta), np.sin(theta)
        inv_rotation = np.array([[inv_c, -inv_s], [inv_s, inv_c]], dtype=np.float32)
        global_preds = (local_preds @ inv_rotation.T) + anchor_pos

        return global_preds


def load_dataset(data_dir: Path, split: str) -> TrajectoryDataset:
    return TrajectoryDataset(data_dir=str(data_dir), split=split)


def run_inference(
    checkpoint_path: Path | str = DEFAULT_CHECKPOINT,
    data_dir: Path | str = DEFAULT_DATA_DIR,
    split: str = "val",
    sample_idx: int = 0,
    output_path: Path | str | None = DEFAULT_OUTPUT_PATH,
    device: str | None = None,
) -> dict[str, np.ndarray | float | int]:
    """
    Run model inference on one dataset sample (already normalized local frame).

    Returns a dictionary containing obs, gt, preds, min_ade, min_fde, sample_idx.
    """
    checkpoint_path = Path(checkpoint_path)
    data_dir = Path(data_dir)
    output_path = Path(output_path) if output_path is not None else None

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    dataset = load_dataset(data_dir=data_dir, split=split)

    if sample_idx < 0 or sample_idx >= len(dataset):
        raise IndexError(f"sample_idx {sample_idx} is out of range [0, {len(dataset) - 1}]")

    obs, fut, social, mask = dataset[sample_idx]

    chosen_device = (
        torch.device(device)
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    model = TrajectoryPredictor().to(chosen_device)
    state_dict = torch.load(checkpoint_path, map_location=chosen_device)
    model.load_state_dict(state_dict)
    model.eval()

    obs_batch = obs.unsqueeze(0).to(chosen_device)
    social_batch = social.unsqueeze(0).to(chosen_device)
    mask_batch = mask.unsqueeze(0).to(chosen_device)

    with torch.no_grad():
        preds = model(obs_batch, social_batch, mask_batch).squeeze(0).cpu()

    min_ade, min_fde = minade_minfde(preds, fut)

    result = {
        "sample_idx": int(sample_idx),
        "obs": obs.cpu().numpy(),
        "gt": fut.cpu().numpy(),
        "preds": preds.cpu().numpy(),
        "min_ade": float(min_ade),
        "min_fde": float(min_fde),
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            sample_idx=np.int32(result["sample_idx"]),
            obs=result["obs"],
            gt=result["gt"],
            preds=result["preds"],
            min_ade=np.float32(result["min_ade"]),
            min_fde=np.float32(result["min_fde"]),
        )

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run trajectory model inference for one sample.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = run_inference(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        split=args.split,
        sample_idx=args.sample_idx,
        output_path=args.output_path,
    )
    print(f"Sample {out['sample_idx']} | minADE={out['min_ade']:.4f} | minFDE={out['min_fde']:.4f}")
    print(f"Saved prediction to: {args.output_path}")
