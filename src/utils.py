import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def wta_loss(preds, gt):
    """
    Winner-Takes-All loss

    preds: [B, K, T, 2]
    gt:    [B, T, 2]
    """
    gt_expanded = gt.unsqueeze(1)  # [B, 1, T, 2]
    per_mode_error = torch.norm(preds - gt_expanded, dim=-1).mean(dim=-1)  # [B, K]
    best_mode_error, _ = per_mode_error.min(dim=1)  # [B]
    return best_mode_error.mean()