import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        obs_path,
        fut_path,
        social_path=None,
        mask_path=None,
    ):
        """
        Load preprocessed trajectory arrays.

        obs_path: path to [N, 4, 4] npy file
        fut_path: path to [N, 6, 2] npy file
        social_path: optional path to [N, 4, 4, 2] npy file
        mask_path: optional path to [N, 4, 4] npy file
        """
        self.obs = np.load(obs_path)
        self.fut = np.load(fut_path)
        self.social = np.load(social_path) if social_path is not None else None
        self.mask = np.load(mask_path) if mask_path is not None else None

        if len(self.obs) != len(self.fut):
            raise ValueError("Observed inputs and targets length mismatch.")

        if (self.social is None) != (self.mask is None):
            raise ValueError("social_path and mask_path must be provided together.")

        if self.social is not None and len(self.social) != len(self.obs):
            raise ValueError("Observed inputs and social inputs length mismatch.")

        if self.mask is not None and len(self.mask) != len(self.obs):
            raise ValueError("Observed inputs and masks length mismatch.")

        self.has_social = self.social is not None

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        obs_tensor = torch.tensor(self.obs[idx], dtype=torch.float32)
        fut_tensor = torch.tensor(self.fut[idx], dtype=torch.float32)

        if not self.has_social:
            return obs_tensor, fut_tensor

        social_tensor = torch.tensor(self.social[idx], dtype=torch.float32)
        mask_tensor = torch.tensor(self.mask[idx], dtype=torch.float32)
        return obs_tensor, fut_tensor, social_tensor, mask_tensor


def get_dataloaders(data_dir="data/processed", batch_size=64, num_workers=2):
    train_dataset = TrajectoryDataset(
        os.path.join(data_dir, "train_inputs.npy"),
        os.path.join(data_dir, "train_targets.npy"),
    )
    val_dataset = TrajectoryDataset(
        os.path.join(data_dir, "val_inputs.npy"),
        os.path.join(data_dir, "val_targets.npy"),
    )
    test_dataset = TrajectoryDataset(
        os.path.join(data_dir, "test_inputs.npy"),
        os.path.join(data_dir, "test_targets.npy"),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()

    print(f"Total Train Batches: {len(train_loader)}")
    print(f"Total Val Batches: {len(val_loader)}")

    for obs, fut in train_loader:
        print("\n--- Shape Sanity Check (Train) ---")
        print(f"Input shape: {obs.shape}  --> Expected: torch.Size([batch, 4, 4])")
        print(f"Target shape: {fut.shape} --> Expected: torch.Size([batch, 6, 2])")
        print(f"Input dtype: {obs.dtype}")
        break
