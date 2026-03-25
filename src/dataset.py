import os
import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    def __init__(self, obs_path, fut_path):
        if not os.path.exists(obs_path):
            raise FileNotFoundError(f"Missing file: {obs_path}")
        if not os.path.exists(fut_path):
            raise FileNotFoundError(f"Missing file: {fut_path}")

        self.obs = np.load(obs_path)   # expected shape [N, 8, 4]
        self.fut = np.load(fut_path)   # expected shape [N, 12, 2]

        if len(self.obs) != len(self.fut):
            raise ValueError("obs and fut must have same number of samples")

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        obs_tensor = torch.tensor(self.obs[idx], dtype=torch.float32)
        fut_tensor = torch.tensor(self.fut[idx], dtype=torch.float32)
        return obs_tensor, fut_tensor