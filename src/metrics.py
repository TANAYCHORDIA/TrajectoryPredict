import torch


def ade_one(pred, gt):
    return torch.norm(pred - gt, dim=-1).mean()


def fde_one(pred, gt):
    return torch.norm(pred[-1] - gt[-1], dim=-1)


def minade_minfde(preds, gt):
    ades = torch.stack([ade_one(preds[k], gt) for k in range(preds.shape[0])])
    fdes = torch.stack([fde_one(preds[k], gt) for k in range(preds.shape[0])])
    return ades.min(), fdes.min()


if __name__ == "__main__":
    gt = torch.tensor([[0., 0.], [1., 1.], [2., 2.]])
    preds = torch.tensor([
        [[0., 0.], [1., 1.], [2., 2.]],
        [[0., 0.], [1., 0.], [2., 0.]],
        [[0., 1.], [1., 2.], [2., 3.]]
    ])
    min_ade, min_fde = minade_minfde(preds, gt)
    print("minADE:", float(min_ade))
    print("minFDE:", float(min_fde))