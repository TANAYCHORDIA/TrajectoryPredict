from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.inference import (
    DEFAULT_CHECKPOINT,
    DEFAULT_FUT_PATH,
    DEFAULT_OBS_PATH,
    DEFAULT_OUTPUT_PATH,
    run_inference,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PLOT_PATH = ROOT_DIR / "outputs" / "plots" / "trajectory_prediction.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize observed, ground-truth, and predicted trajectories."
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
        help="Dataset sample index to visualize.",
    )
    parser.add_argument(
        "--prediction-output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Where to save the intermediate inference result.",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=DEFAULT_PLOT_PATH,
        help="Where to save the generated plot image.",
    )
    return parser.parse_args()


def plot_trajectories(
    obs: np.ndarray,
    gt: np.ndarray,
    preds: np.ndarray,
    plot_path: Path,
    sample_idx: int,
    min_ade: float,
    min_fde: float,
) -> None:
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(obs[:, 0], obs[:, 1], marker="o", linewidth=2, label="Observed")
    ax.plot(gt[:, 0], gt[:, 1], marker="o", linewidth=2, label="Ground truth")

    for mode_idx, pred in enumerate(preds):
        ax.plot(
            pred[:, 0],
            pred[:, 1],
            marker="x",
            linestyle="--",
            linewidth=1.5,
            label=f"Prediction mode {mode_idx +1}",
        )

    ax.scatter(obs[-1, 0], obs[-1, 1], color="black", s=50, label="Obs endpoint")
    ax.set_title(
        f"Sample {sample_idx} | minADE={min_ade:.3f}, minFDE={min_fde:.3f}"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    result = run_inference(
        checkpoint_path=args.checkpoint,
        obs_path=args.obs_path,
        fut_path=args.fut_path,
        sample_idx=args.sample_idx,
        social_path=args.social_path,
        mask_path=args.mask_path,
        output_path=args.prediction_output,
    )

    plot_trajectories(
        obs=result["obs"],
        gt=result["gt"],
        preds=result["preds"],
        plot_path=args.plot_output,
        sample_idx=int(result["sample_idx"]),
        min_ade=float(result["min_ade"]),
        min_fde=float(result["min_fde"]),
    )

    print(f"Saved visualization to: {args.plot_output}")


if __name__ == "__main__":
    main()
