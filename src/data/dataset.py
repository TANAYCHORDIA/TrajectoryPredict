import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class TrajectoryDataset(Dataset):
    def __init__(self, inputs_path, targets_path):
        """
        Loads preprocessed numpy arrays.
        inputs_path: path to [N, 4, 4] npy file
        targets_path: path to [N, 6, 2] npy file
        """
        self.inputs = np.load(inputs_path)
        self.targets = np.load(targets_path)
        
        assert len(self.inputs) == len(self.targets), "Inputs and targets length mismatch!"

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Convert to float32 explicitly for PyTorch
        input_tensor = torch.tensor(self.inputs[idx], dtype=torch.float32)
        target_tensor = torch.tensor(self.targets[idx], dtype=torch.float32)
        return input_tensor, target_tensor

def get_dataloaders(data_dir="data/processed", batch_size=64, num_workers=2):
    train_dataset = TrajectoryDataset(
        os.path.join(data_dir, "train_inputs.npy"),
        os.path.join(data_dir, "train_targets.npy")
    )
    val_dataset = TrajectoryDataset(
        os.path.join(data_dir, "val_inputs.npy"),
        os.path.join(data_dir, "val_targets.npy")
    )
    test_dataset = TrajectoryDataset(
        os.path.join(data_dir, "test_inputs.npy"),
        os.path.join(data_dir, "test_targets.npy")
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Sanity Check
    train_loader, val_loader, test_loader = get_dataloaders()
    
    print(f"Total Train Batches: {len(train_loader)}")
    print(f"Total Val Batches: {len(val_loader)}")
    
    for inputs, targets in train_loader:
        print("\n--- Shape Sanity Check (Train) ---")
        print(f"Input shape: {inputs.shape}  --> Expected: torch.Size([batch, 4, 4])")
        print(f"Target shape: {targets.shape} --> Expected: torch.Size([batch, 6, 2])")
        print(f"Input dtype: {inputs.dtype}")
        break