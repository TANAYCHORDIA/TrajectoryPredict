import torch
import torch.nn as nn


class TrajectoryPredictor(nn.Module):
    def __init__(
        self,
        input_dim=4,
        hidden_size=128,
        num_layers=2,
        pred_len=12,
        num_modes=3
    ):
        super().__init__()

        self.pred_len = pred_len
        self.num_modes = num_modes

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1
        )

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, pred_len * 2)
            )
            for _ in range(num_modes)
        ])

    def forward(self, x):
        """
        x: [B, 8, 4]
        returns: [B, 3, 12, 2]
        """
        _, (h_n, _) = self.encoder(x)
        final_hidden = h_n[-1]  # [B, hidden_size]

        outputs = []
        for head in self.heads:
            out = head(final_hidden)               # [B, 24]
            out = out.view(-1, self.pred_len, 2)  # [B, 12, 2]
            outputs.append(out)

        preds = torch.stack(outputs, dim=1)       # [B, 3, 12, 2]
        return preds


if __name__ == "__main__":
    model = TrajectoryPredictor()
    x = torch.randn(5, 8, 4)
    y = model(x)
    print("Output shape:", y.shape)