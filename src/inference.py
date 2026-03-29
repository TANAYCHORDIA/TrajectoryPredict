from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.data.dataset import TrajectoryDataset
from src.metrics import minade_minfde
from src.model import TrajectoryPredictor
from src.utils import get_device


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT_DIR / "outputs" / "checkpoints" / "best_model.pth"
DEFAULT_OBS_PATH = ROOT_DIR / "data" / "processed" / "obs_val.npy"
DEFAULT_FUT_PATH = ROOT_DIR / "data" / "processed" / "fut_val.npy"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "outputs" / "predictions" / "latest_prediction.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run trajectory inference for a single sample."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to a trained model checkpoint.",
    )
    parser.add_argument(
        "--obs-path",
        type=Path,
        default=DEFAULT_OBS_PATH,
        help="Path to observed trajectory numpy file.",
    )
    parser.add_argument(
        "--fut-path",
        type=Path,
        default=DEFAULT_FUT_PATH,
        help="Path to future trajectory numpy file for evaluation.",
    )
    parser.add_argument(
        "--social-path",
        type=Path,
        default=None,
        help="Optional path to social trajectory numpy file.",
    )
    parser.add_argument(
        "--mask-path",
        type=Path,
        default=None,
        help="Optional path to social mask numpy file.",
    )
    parser.add_argument(
        "--sample-idx",
        type=int,
        default=0,
        help="Dataset sample index to run inference on.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Where to save the inference result as an .npz file.",
    )
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: torch.device) -> TrajectoryPredictor:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = TrajectoryPredictor().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_sample(
    obs_path: Path,
    fut_path: Path,
    sample_idx: int,
    social_path: Path | None = None,
    mask_path: Path | None = None,
) -> tuple[torch.Tensor, ...]:
    dataset = TrajectoryDataset(
        str(obs_path),
        str(fut_path),
        social_path=str(social_path) if social_path is not None else None,
        mask_path=str(mask_path) if mask_path is not None else None,
    )

    if sample_idx < 0 or sample_idx >= len(dataset):
        raise IndexError(
            f"sample_idx {sample_idx} is out of range for dataset size {len(dataset)}"
        )

    return dataset[sample_idx]


def run_inference(
    checkpoint_path: Path,
    obs_path: Path,
    fut_path: Path,
    sample_idx: int,
    social_path: Path | None = None,
    mask_path: Path | None = None,
    output_path: Path | None = None,
) -> dict[str, np.ndarray | float | int]:
    device = get_device()
    model = load_model(checkpoint_path, device)
    sample = load_sample(
        obs_path=obs_path,
        fut_path=fut_path,
        sample_idx=sample_idx,
        social_path=social_path,
        mask_path=mask_path,
    )

    if len(sample) == 2:
        obs, fut = sample
        social = None
        mask = None
    else:
        obs, fut, social, mask = sample

    obs_batch = obs.unsqueeze(0).to(device)
    social_batch = social.unsqueeze(0).to(device) if social is not None else None
    mask_batch = mask.unsqueeze(0).to(device) if mask is not None else None

    with torch.no_grad():
        preds = model(obs_batch, social_batch, mask_batch).squeeze(0).cpu()

    min_ade, min_fde = minade_minfde(preds, fut)

    result = {
        "sample_idx": sample_idx,
        "obs": obs.cpu().numpy(),
        "gt": fut.cpu().numpy(),
        "preds": preds.cpu().numpy(),
        "min_ade": float(min_ade),
        "min_fde": float(min_fde),
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_path,
            sample_idx=result["sample_idx"],
            obs=result["obs"],
            gt=result["gt"],
            preds=result["preds"],
            min_ade=result["min_ade"],
            min_fde=result["min_fde"],
        )

    return result


def main() -> None:
    args = parse_args()

    result = run_inference(
        checkpoint_path=args.checkpoint,
        obs_path=args.obs_path,
        fut_path=args.fut_path,
        sample_idx=args.sample_idx,
        social_path=args.social_path,
        mask_path=args.mask_path,
        output_path=args.output,
    )

    print(f"Sample index: {result['sample_idx']}")
    print(f"Observed shape: {result['obs'].shape}")
    print(f"Ground truth shape: {result['gt'].shape}")
    print(f"Predictions shape: {result['preds'].shape}")
    print(f"minADE: {result['min_ade']:.4f}")
    print(f"minFDE: {result['min_fde']:.4f}")

    if args.output is not None:
        print(f"Saved inference output to: {args.output}")


if __name__ == "__main__":
    main()
