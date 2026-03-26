import os
import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    """
    Loads trajectory tensors for one split.

    Expected shapes:
      inputs:  [N, obs_len, 4]
      targets: [N, pred_len, 2]
      social:  [N, obs_len, max_neighbors, 2]
      mask:    [N, obs_len, max_neighbors]
    """

    def __init__(
        self,
        data_dir: str = "data/processed",
        split: str = "train",
        obs_len: int = 4,
        pred_len: int = 6,
        max_neighbors: int = 4,
        memory_map: bool = False,
        validate_nans: bool = True,
    ):
        self.data_dir = data_dir
        self.split = split
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.max_neighbors = max_neighbors

        paths = {
            "inputs": os.path.join(data_dir, f"{split}_inputs.npy"),
            "targets": os.path.join(data_dir, f"{split}_targets.npy"),
            "social": os.path.join(data_dir, f"{split}_social.npy"),
            "mask": os.path.join(data_dir, f"{split}_mask.npy"),
        }

        missing = [name for name, p in paths.items() if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                f"Missing {split} dataset files in {data_dir}: {missing}"
            )

        mmap_mode = "r" if memory_map else None

        # Safe numpy loading
        inputs_np = np.load(paths["inputs"], allow_pickle=False, mmap_mode=mmap_mode)
        targets_np = np.load(paths["targets"], allow_pickle=False, mmap_mode=mmap_mode)
        social_np = np.load(paths["social"], allow_pickle=False, mmap_mode=mmap_mode)
        mask_np = np.load(paths["mask"], allow_pickle=False, mmap_mode=mmap_mode)

        # Shape checks
        exp_inputs = (obs_len, 4)
        exp_targets = (pred_len, 2)
        exp_social = (obs_len, max_neighbors, 2)
        exp_mask = (obs_len, max_neighbors)

        if inputs_np.ndim != 3 or tuple(inputs_np.shape[1:]) != exp_inputs:
            raise ValueError(
                f"{split}_inputs.npy has shape {inputs_np.shape}, expected [N, {obs_len}, 4]"
            )
        if targets_np.ndim != 3 or tuple(targets_np.shape[1:]) != exp_targets:
            raise ValueError(
                f"{split}_targets.npy has shape {targets_np.shape}, expected [N, {pred_len}, 2]"
            )
        if social_np.ndim != 4 or tuple(social_np.shape[1:]) != exp_social:
            raise ValueError(
                f"{split}_social.npy has shape {social_np.shape}, expected [N, {obs_len}, {max_neighbors}, 2]"
            )
        if mask_np.ndim != 3 or tuple(mask_np.shape[1:]) != exp_mask:
            raise ValueError(
                f"{split}_mask.npy has shape {mask_np.shape}, expected [N, {obs_len}, {max_neighbors}]"
            )

        n = inputs_np.shape[0]
        if not (targets_np.shape[0] == n == social_np.shape[0] == mask_np.shape[0]):
            raise ValueError(
                f"Sample-count mismatch for split='{split}': "
                f"inputs={inputs_np.shape[0]}, targets={targets_np.shape[0]}, "
                f"social={social_np.shape[0]}, mask={mask_np.shape[0]}"
            )

        # Optional NaN checks
        if validate_nans:
            if np.isnan(inputs_np).any():
                raise ValueError(f"NaN detected in {split}_inputs.npy")
            if np.isnan(targets_np).any():
                raise ValueError(f"NaN detected in {split}_targets.npy")
            if np.isnan(social_np).any():
                raise ValueError(f"NaN detected in {split}_social.npy")
            if np.isnan(mask_np).any():
                raise ValueError(f"NaN detected in {split}_mask.npy")

        # Convert to torch tensors once (fast __getitem__)
        # If memory_map=True, asarray materializes clean contiguous arrays for from_numpy.
        self.inputs = torch.from_numpy(np.asarray(inputs_np, dtype=np.float32))
        self.targets = torch.from_numpy(np.asarray(targets_np, dtype=np.float32))
        self.social = torch.from_numpy(np.asarray(social_np, dtype=np.float32))
        self.mask = torch.from_numpy(np.asarray(mask_np, dtype=np.float32))

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, idx: int):
        # inputs: [obs_len, 4], targets: [pred_len, 2], social: [obs_len, K, 2], mask: [obs_len, K]
        return self.inputs[idx], self.targets[idx], self.social[idx], self.mask[idx]
    
from torch.utils.data import DataLoader

if __name__ == "__main__":
    print("Sanity Check: PyTorch DataLoader with Social Features")
    try:
        dataset = TrajectoryDataset(split="train")
        loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True) 

        inputs, targets, social, mask = next(iter(loader))
        print(f"Input shape:  {list(inputs.shape)} -> Expected [64, 4, 4]")
        print(f"Target shape: {list(targets.shape)} -> Expected [64, 6, 2]")
        print(f"Social shape: {list(social.shape)} -> Expected [64, 4, 4, 2]")
        print(f"Mask shape:   {list(mask.shape)}   -> Expected [64, 4, 4]")
        
        if (list(inputs.shape) == [64, 4, 4] and 
            list(targets.shape) == [64, 6, 2] and 
            list(social.shape) == [64, 4, 4, 2] and 
            list(mask.shape) == [64, 4, 4]):
            print("\n✅ SUCCESS: Pipeline verified. All 4 tensors are ready for the ML Lead.")
        else:
            print("\n❌ FAILURE: Tensor dimensions do not match specifications.")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")