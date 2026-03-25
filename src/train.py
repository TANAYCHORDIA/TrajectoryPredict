import os
import torch
from torch.utils.data import DataLoader

from dataset import TrajectoryDataset
from model import TrajectoryPredictor
from utils import set_seed, get_device, wta_loss
from metrics import minade_minfde


OBS_TRAIN_PATH = "data/processed/obs_train.npy"
FUT_TRAIN_PATH = "data/processed/fut_train.npy"
OBS_VAL_PATH = "data/processed/obs_val.npy"
FUT_VAL_PATH = "data/processed/fut_val.npy"

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 10
CHECKPOINT_PATH = "outputs/checkpoints/best_model.pth"


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for obs, fut in loader:
        obs = obs.to(device)
        fut = fut.to(device)

        optimizer.zero_grad()
        preds = model(obs)
        loss = wta_loss(preds, fut)
        loss.backward()

        # Prevent unstable gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def validate_one_epoch(model, loader, device):
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

            for i in range(batch_size):
                min_ade, min_fde = minade_minfde(preds[i], fut[i])
                total_ade += float(min_ade)
                total_fde += float(min_fde)

            total_samples += batch_size

    avg_loss = total_loss / len(loader)
    avg_ade = total_ade / total_samples
    avg_fde = total_fde / total_samples

    return avg_loss, avg_ade, avg_fde


def main():
    set_seed(42)
    device = get_device()
    print("Using device:", device)

    os.makedirs("outputs/checkpoints", exist_ok=True)

    required_files = [
        OBS_TRAIN_PATH, FUT_TRAIN_PATH,
        OBS_VAL_PATH, FUT_VAL_PATH
    ]

    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"Missing file: {file_path}")
            print("Ask Data Engineer to provide processed .npy files first.")
            return

    train_dataset = TrajectoryDataset(OBS_TRAIN_PATH, FUT_TRAIN_PATH)
    val_dataset = TrajectoryDataset(OBS_VAL_PATH, FUT_VAL_PATH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = TrajectoryPredictor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_ade, val_fde = validate_one_epoch(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val ADE: {val_ade:.4f} | "
            f"Val FDE: {val_fde:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"Saved best model to {CHECKPOINT_PATH}")

    print("Training complete.")


if __name__ == "__main__":
    main()