"""Forecasting models. Everything works in the focal-agent frame, where the focal agent sits at the
origin at t=49 heading along +x; predictions are future positions of shape (N, 60, 2), or
(N, K, 60, 2) with K mode logits for the multimodal Transformer."""

import numpy as np
import torch
from torch import nn

from data import SCENE_FIELDS
from preprocess import FUTURE, HISTORY

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


def mlp(d_in, d, d_out):
    return nn.Sequential(nn.Linear(d_in, d), nn.LayerNorm(d), nn.ReLU(), nn.Linear(d, d_out))


class TransformerForecaster(nn.Module):
    """Vectorized scene encoder and K-mode coordinate decoder, for the focal agent.

    Each agent (history steps) and each lane (centerline segments) is encoded PointNet-style into one
    token. A Transformer encoder runs joint self-attention over all agent and lane tokens, so
    agent-agent and agent-map attention share the same layers; empty slots are masked out. K mode
    queries -- the focal token plus a learned mode embedding -- attend to the scene in a Transformer
    decoder. traj_head maps each mode embedding to 60 future positions, score_head to a logit.
    """

    POS_SCALE, VEL_SCALE = LSTMForecaster.POS_SCALE, LSTMForecaster.VEL_SCALE
    N_OBJECT_TYPES, N_LANE_TYPES = 10, 3

    def __init__(self, d=128, heads=8, encoder_layers=4, decoder_layers=2, ffn=512, dropout=0.1, modes=6):
        super().__init__()
        self.agent_in = nn.Linear(6, d)
        self.time_emb = nn.Parameter(torch.zeros(HISTORY, d))
        self.agent_mlp = nn.Sequential(nn.LayerNorm(d), nn.ReLU(), nn.Linear(d, d))
        self.agent_type_emb = nn.Embedding(self.N_OBJECT_TYPES, d)
        self.lane_mlp = mlp(4, d, d)
        self.lane_type_emb = nn.Embedding(self.N_LANE_TYPES, d)
        self.intersection_emb = nn.Embedding(2, d)

        def layer(cls):
            return cls(d, heads, ffn, dropout, batch_first=True, norm_first=True)

        self.encoder = nn.TransformerEncoder(layer(nn.TransformerEncoderLayer), encoder_layers,
                                             norm=nn.LayerNorm(d), enable_nested_tensor=False)
        self.mode_emb = nn.Parameter(torch.randn(modes, d))  # token scale, so the K queries start distinct
        self.decoder = nn.TransformerDecoder(layer(nn.TransformerDecoderLayer), decoder_layers, norm=nn.LayerNorm(d))
        self.traj_head = mlp(d, 2 * d, FUTURE * 2)
        self.score_head = nn.Linear(d, 1)
        nn.init.normal_(self.time_emb, std=0.02)

    def encode_agents(self, agents, valid, agent_type):
        heading = agents[..., 4:5]
        x = torch.cat([agents[..., 0:2] / self.POS_SCALE, agents[..., 2:4] / self.VEL_SCALE,
                       torch.cos(heading), torch.sin(heading)], dim=-1)
        h = self.agent_mlp(self.agent_in(x) + self.time_emb)  # (B, A, 50, d)
        h = h.masked_fill(~valid[..., None], float("-inf")).amax(dim=2)
        h = torch.where(valid.any(-1, keepdim=True), h, torch.zeros_like(h))  # empty slots stay finite
        return h + self.agent_type_emb(agent_type.long().clamp(min=0))

    def encode_lanes(self, lanes, lane_type, intersection):
        seg = torch.cat([lanes[:, :, :-1], lanes[:, :, 1:]], dim=-1) / self.POS_SCALE  # (B, L, 9, 4)
        h = self.lane_mlp(seg).amax(dim=2)
        return h + self.lane_type_emb(lane_type.long().clamp(min=0)) + self.intersection_emb(intersection.long())

    def forward(self, batch):
        """batch: dict of SCENE_FIELDS tensors -> trajectories (B, K, 60, 2), logits (B, K)."""
        tokens = torch.cat([self.encode_agents(batch["agents"], batch["agent_valid"], batch["agent_type"]),
                            self.encode_lanes(batch["lanes"], batch["lane_type"], batch["lane_intersection"])], dim=1)
        pad = torch.cat([batch["agent_type"] < 0, batch["lane_type"] < 0], dim=1)
        scene = self.encoder(tokens, src_key_padding_mask=pad)
        queries = scene[:, 0:1] + self.mode_emb
        modes = self.decoder(queries, scene, memory_key_padding_mask=pad)  # (B, K, d)
        traj = self.traj_head(modes).unflatten(-1, (FUTURE, 2)) * self.POS_SCALE
        return traj, self.score_head(modes).squeeze(-1)


MODELS = {"lstm": LSTMForecaster, "transformer": TransformerForecaster}


def build_model(model_cfg):
    """Model from the config's model block; a block without `type` is the Phase 2 LSTM."""
    kwargs = dict(model_cfg)
    return MODELS[kwargs.pop("type", "lstm")](**kwargs)


def scene_batch(data, idx, device):
    """Scene tensors for rows idx of an in-RAM data dict (see data.load_focal with scene=True)."""
    return {name: torch.from_numpy(data[name][idx]).to(device) for name in SCENE_FIELDS}
