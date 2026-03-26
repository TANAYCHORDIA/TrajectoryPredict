from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import TrajectoryDataset
from metrics import minade_minfde
from model import TrajectoryPredictor
from utils import get_device, set_seed, wta_loss


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OBS_TRAIN_PATH = ROOT_DIR / "data" / "processed" / "obs_train.npy"
DEFAULT_FUT_TRAIN_PATH = ROOT_DIR / "data" / "processed" / "fut_train.npy"
DEFAULT_OBS_VAL_PATH = ROOT_DIR / "data" / "processed" / "obs_val.npy"
DEFAULT_FUT_VAL_PATH = ROOT_DIR / "data" / "processed" / "fut_val.npy"
DEFAULT_CHECKPOINT_PATH = ROOT_DIR / "outputs" / "checkpoints" / "best_model.pth"
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_EPOCHS = 10
DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(description="Train trajectory prediction model.")
    parser.add_argument(
        "--obs-train",
        type=Path,
        default=DEFAULT_OBS_TRAIN_PATH,
        help="Path to observed training trajectories.",
    )
    parser.add_argument(
        "--fut-train",
        type=Path,
        default=DEFAULT_FUT_TRAIN_PATH,
        help="Path to future training trajectories.",
    )
    parser.add_argument(
        "--obs-val",
        type=Path,
        default=DEFAULT_OBS_VAL_PATH,
        help="Path to observed validation trajectories.",
    )
    parser.add_argument(
        "--fut-val",
        type=Path,
        default=DEFAULT_FUT_VAL_PATH,
        help="Path to future validation trajectories.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Path to save the best model checkpoint.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Mini-batch size for training and validation.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Learning rate for the Adam optimizer.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs.",
    )
    return parser.parse_args()


def validate_required_files(paths: list[Path]) -> None:
    """Raise a helpful error if any required data file is missing."""
    missing_paths = [path for path in paths if not path.exists()]
    if missing_paths:
        missing_list = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(
            f"Missing required file(s):\n{missing_list}\n"
            "Ask Data Engineer to provide processed .npy files first."
        )


def create_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    """Build training and validation dataloaders."""
    train_dataset = TrajectoryDataset(str(args.obs_train), str(args.fut_train))
    val_dataset = TrajectoryDataset(str(args.obs_val), str(args.fut_val))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    return train_loader, val_loader


def train_one_epoch(
    model: TrajectoryPredictor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one training epoch and return average loss."""
    model.train()
    total_loss = 0.0

    for obs, fut in loader:
        obs = obs.to(device)
        fut = fut.to(device)

        optimizer.zero_grad()
        preds = model(obs)
        loss = wta_loss(preds, fut)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def validate_one_epoch(
    model: TrajectoryPredictor,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float]:
    """Run one validation epoch and return loss, ADE, and FDE."""
    model.eval()
    total_loss = 0.0
    total_ade = 0.0
    total_fde = 0.0
    total_samples = 0

    with torch.no_grad():
        for obs, fut in loader:
            obs = obs.to(device)
            fut = fut.to(device)

            preds = model(obs)
            loss = wta_loss(preds, fut)
            total_loss += loss.item()

            batch_size = obs.size(0)
            for index in range(batch_size):
                min_ade, min_fde = minade_minfde(preds[index], fut[index])
                total_ade += float(min_ade)
                total_fde += float(min_fde)

            total_samples += batch_size

    avg_loss = total_loss / len(loader)
    avg_ade = total_ade / total_samples
    avg_fde = total_fde / total_samples
    return avg_loss, avg_ade, avg_fde


def train(args: argparse.Namespace) -> None:
    """Train the model and save the best checkpoint by validation loss."""
    set_seed(DEFAULT_SEED)
    device = get_device()
    print(f"Using device: {device}")

    validate_required_files(
        [
            args.obs_train,
            args.fut_train,
            args.obs_val,
            args.fut_val,
        ]
    )

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = create_dataloaders(args)
    model = TrajectoryPredictor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_ade, val_fde = validate_one_epoch(model, val_loader, device)

        print(
            f"Epoch {epoch + 1:02d}/{args.epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val ADE: {val_ade:.4f} | "
            f"Val FDE: {val_fde:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.checkpoint)
            print(f"Saved best model to {args.checkpoint}")

    print("Training complete.")


def main() -> None:
    """CLI entrypoint for model training."""
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
