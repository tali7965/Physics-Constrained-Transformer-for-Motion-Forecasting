"""Forecasting models. Everything works in the focal-agent frame, where the focal agent sits at the
origin at t=49 heading along +x; predictions are future positions of shape (N, 60, 2)."""

import numpy as np
import torch
from torch import nn

from preprocess import FUTURE

DT = 0.1  # seconds per step


def pick_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def constant_velocity(history, estimate):
    """Extrapolate from the t=49 position at constant velocity.

    history: (N, 50, 5) numpy x, y, vx, vy, heading. estimate: "tracker" uses the dataset velocity at
    t=49; "displacement" uses the last observed step, (p49 - p48) / 0.1 s.
    """
    history = np.asarray(history, dtype=np.float64)
    if estimate == "tracker":
        velocity = history[:, -1, 2:4]
    elif estimate == "displacement":
        velocity = (history[:, -1, :2] - history[:, -2, :2]) / DT
    else:
        raise ValueError(f"unknown velocity estimate {estimate!r}")
    t = DT * np.arange(1, FUTURE + 1)[:, None]
    return history[:, -1, None, :2] + velocity[:, None, :] * t


class LSTMForecaster(nn.Module):
    """LSTM encoder-decoder over the focal history only, predicting one trajectory (K=1).

    Input per step: x/20, y/20, vx/10, vy/10, cos(heading), sin(heading). The decoder is unrolled for
    60 steps; each step outputs a displacement in metres that is fed back as the next input, starting
    from the last observed displacement. Positions are the cumulative sum of displacements.
    """

    POS_SCALE, VEL_SCALE = 20.0, 10.0

    def __init__(self, hidden=128, encoder_layers=2):
        super().__init__()
        self.encoder = nn.LSTM(6, hidden, num_layers=encoder_layers, batch_first=True)
        self.decoder = nn.LSTMCell(2, hidden)
        self.head = nn.Linear(hidden, 2)

    def features(self, history):
        heading = history[..., 4:5]
        return torch.cat(
            [history[..., 0:2] / self.POS_SCALE, history[..., 2:4] / self.VEL_SCALE,
             torch.cos(heading), torch.sin(heading)],
            dim=-1,
        )

    def forward(self, history):
        """history: (B, 50, 5) tensor -> (B, 60, 2) future positions."""
        _, (h, c) = self.encoder(self.features(history))
        h, c = h[-1], c[-1]
        step = history[:, -1, 0:2] - history[:, -2, 0:2]
        steps = []
        for _ in range(FUTURE):
            h, c = self.decoder(step, (h, c))
            step = self.head(h)
            steps.append(step)
        return history[:, -1, None, 0:2] + torch.cumsum(torch.stack(steps, dim=1), dim=1)
