import torch
import torch.nn as nn


class TrajectoryPredictor(nn.Module):
    def __init__(
        self,
        input_dim=4,
        hidden_size=128,
        num_layers=2,
        pred_len=6,
        num_modes=3,
        social_input_dim=2,
        social_hidden_size=64,
    ):
        super().__init__()

        self.pred_len = pred_len
        self.num_modes = num_modes

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1,
        )

        self.social_mlp = nn.Sequential(
            nn.Linear(social_input_dim, social_hidden_size),
            nn.ReLU(),
            nn.Linear(social_hidden_size, social_hidden_size),
            nn.ReLU(),
        )

        fused_dim = hidden_size + social_hidden_size
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(fused_dim, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, pred_len * 2),
                )
                for _ in range(num_modes)
            ]
        )

    def encode_social(self, social, mask):
        """
        social: [B, 4, 4, 2]
        mask: [B, 4, 4]
        returns: [B, social_hidden_size]
        """
        if social is None or mask is None:
            batch_size = social.shape[0] if social is not None else mask.shape[0]
            device = social.device if social is not None else mask.device
            return torch.zeros(
                batch_size,
                self.social_mlp[0].out_features,
                device=device,
            )

        masked_social = social * mask.unsqueeze(-1)
        valid_counts = mask.sum(dim=(1, 2)).unsqueeze(-1).clamp_min(1.0)
        pooled_social = masked_social.sum(dim=(1, 2)) / valid_counts
        return self.social_mlp(pooled_social)

    def forward(self, x, social=None, mask=None):
        """
        x: [B, 4, 4]
        social: [B, 4, 4, 2] or None
        mask: [B, 4, 4] or None
        returns: [B, 3, 6, 2]
        """
        _, (h_n, _) = self.encoder(x)
        main_embedding = h_n[-1]  # [B, hidden_size]

        if social is None and mask is None:
            social_embedding = torch.zeros(
                x.size(0),
                self.social_mlp[0].out_features,
                device=x.device,
            )
        else:
            social_embedding = self.encode_social(social, mask)

        fused_embedding = torch.cat([main_embedding, social_embedding], dim=-1)

        outputs = []
        for head in self.heads:
            out = head(fused_embedding)            # [B, 12]
            out = out.view(-1, self.pred_len, 2)  # [B, 6, 2]
            outputs.append(out)

        preds = torch.stack(outputs, dim=1)       # [B, 3, 6, 2]
        return preds


if __name__ == "__main__":
    model = TrajectoryPredictor()
    x = torch.randn(5, 4, 4)
    social = torch.randn(5, 4, 4, 2)
    mask = torch.ones(5, 4, 4)
    y = model(x, social, mask)
    print("Output shape:", y.shape)
