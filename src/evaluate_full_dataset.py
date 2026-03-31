import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.metrics import minade_minfde
from src.model import TrajectoryPredictor


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
	checkpoint = torch.load(checkpoint_path, map_location=device)

	# Support both plain state_dict and wrapped checkpoints.
	if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
		state_dict = checkpoint["state_dict"]
	else:
		state_dict = checkpoint

	model.load_state_dict(state_dict)


def evaluate_full_dataset(
	checkpoint_path: str = "outputs/checkpoints/best_model_final.pth",
	data_dir: str = "data/processed",
	split: str = "val",
) -> int:
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	ckpt_path = Path(checkpoint_path)
	if not ckpt_path.exists():
		print(f"❌ Checkpoint not found at {ckpt_path}")
		return 1

	print(f"Using device: {device}")
	model = TrajectoryPredictor().to(device)
	_load_checkpoint(model, ckpt_path, device)
	model.eval()

	print(f"Loading {split} tensors...")
	try:
		obs_all = np.load(Path(data_dir) / f"{split}_inputs.npy", allow_pickle=False)
		fut_all = np.load(Path(data_dir) / f"{split}_targets.npy", allow_pickle=False)
		soc_all = np.load(Path(data_dir) / f"{split}_social.npy", allow_pickle=False)
		mask_all = np.load(Path(data_dir) / f"{split}_mask.npy", allow_pickle=False)
	except Exception as e:
		print(f"❌ Failed to load dataset arrays: {e}")
		return 1

	total_samples = len(obs_all)
	if total_samples == 0:
		print(f"❌ No samples found in split '{split}'.")
		return 1

	if not (len(fut_all) == len(soc_all) == len(mask_all) == total_samples):
		print(
			"❌ Length mismatch: "
			f"inputs={len(obs_all)}, targets={len(fut_all)}, social={len(soc_all)}, mask={len(mask_all)}"
		)
		return 1

	ades, fdes = [], []

	print(f"\nRunning inference on {total_samples} samples...")
	with torch.no_grad():
		for i in tqdm(range(total_samples)):
			obs = torch.from_numpy(obs_all[i]).unsqueeze(0).float().to(device)
			fut = torch.from_numpy(fut_all[i]).float().to(device)
			soc = torch.from_numpy(soc_all[i]).unsqueeze(0).float().to(device)
			mask = torch.from_numpy(mask_all[i]).unsqueeze(0).float().to(device)

			preds = model(obs, soc, mask).squeeze(0)  # [3, 6, 2]
			min_ade, min_fde = minade_minfde(preds, fut)
			ades.append(float(min_ade.item()))
			fdes.append(float(min_fde.item()))

	mean_ade = float(np.mean(ades))
	mean_fde = float(np.mean(fdes))

	print("\n" + "=" * 50)
	print(f"📊 TRUE {split.upper()} METRICS (FULL DATASET)")
	print("=" * 50)
	print(f"Total Samples Analyzed : {total_samples}")
	print(f"True Mean ADE          : {mean_ade:.4f} meters")
	print(f"True Mean FDE          : {mean_fde:.4f} meters")
	print("-" * 50)

	if mean_ade < 1.5 and mean_fde < 3.0:
		print("✅ QUALIFIED: Model passes the hackathon constraints.")
	else:
		print("⚠️ WARNING: Metrics missed the threshold on global average.")
	print("=" * 50)

	return 0


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Evaluate trajectory model on a full dataset split.")
	parser.add_argument(
		"--checkpoint",
		type=str,
		default="outputs/checkpoints/best_model_final.pth",
		help="Path to model checkpoint (.pth).",
	)
	parser.add_argument(
		"--data-dir",
		type=str,
		default="data/processed",
		help="Directory containing *_inputs.npy, *_targets.npy, *_social.npy, *_mask.npy.",
	)
	parser.add_argument(
		"--split",
		type=str,
		default="val",
		choices=["train", "val", "test"],
		help="Dataset split to evaluate.",
	)
	return parser.parse_args()


if __name__ == "__main__":
	args = _parse_args()
	raise SystemExit(
		evaluate_full_dataset(
			checkpoint_path=args.checkpoint,
			data_dir=args.data_dir,
			split=args.split,
		)
	)
