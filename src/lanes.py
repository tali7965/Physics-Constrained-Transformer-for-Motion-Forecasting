"""Distance from trajectory points to the nearest lane centerline, from the processed scene arrays.

The model already sees every lane segment within 150 m as a 10-point centerline (lanes.npy), so
"how far is this point from any lane?" needs no raw map and is available at training and at run
time. The drivable area is wider than the lanes, so this is a proxy for off-road, not the metric
itself: a point a few metres from every centerline is almost always outside the drivable area,
while the evaluation's off-road flag still reads the raw drivable-area polygons.

lane_distance picks candidate segments without gradient by a lower bound on the distance to each
(distance to its midpoint minus half its length, one torch.cdist), measures the exact
point-to-segment distance to the TOPK best candidates, and keeps the smallest with gradient. The
result is never below the true distance and equals it in practice (tested against brute force).
"""

import torch

TOPK = 8  # candidate segments per point


def lane_segments(lanes, lane_type):
    """(B, L, 10, 2) centerlines -> segment starts and ends (B, S, 2) and validity (B, S), S = 9 L."""
    a = lanes[:, :, :-1].flatten(1, 2)
    b = lanes[:, :, 1:].flatten(1, 2)
    valid = (lane_type >= 0)[:, :, None].expand(-1, -1, lanes.shape[2] - 1).flatten(1, 2)
    return a, b, valid


def _point_segment(p, a, b):
    """Distance from points p (..., 2) to segments a-b (..., 2), broadcasting."""
    ab = b - a
    t = (((p - a) * ab).sum(-1) / (ab * ab).sum(-1).clamp(min=1e-6)).clamp(0.0, 1.0)
    closest = a + t[..., None] * ab
    return torch.sqrt(((p - closest) ** 2).sum(-1) + 1e-9)


def lane_distance(points, lanes, lane_type):
    """points (B, ..., 2) in each scene's focal frame -> distance to the nearest lane centerline (B, ...).

    Scenes without any lane get distance 0 (no penalty)."""
    shape = points.shape
    p = points.reshape(shape[0], -1, 2)
    a, b, valid = lane_segments(lanes.to(p.dtype), lane_type)
    with torch.no_grad():
        bound = torch.cdist(p, 0.5 * (a + b)) - 0.5 * (b - a).norm(dim=-1)[:, None]  # (B, P, S)
        bound = bound.masked_fill(~valid[:, None], float("inf"))
        idx = bound.topk(min(TOPK, bound.shape[-1]), dim=-1, largest=False).indices  # (B, P, k)
    k = idx.shape[-1]
    gather = lambda x: torch.gather(x, 1, idx.flatten(1)[..., None].expand(-1, -1, 2)).unflatten(1, (p.shape[1], k))
    dist = _point_segment(p[:, :, None], gather(a), gather(b)).amin(-1)  # (B, P)
    dist = torch.where(valid.any(-1, keepdim=True), dist, torch.zeros_like(dist))
    return dist.reshape(shape[:-1])


def off_lane_penalty(traj, lanes, lane_type, margin):
    """Mean metres beyond `margin` from the nearest lane centerline, per scene and mode: (B, K)."""
    return torch.relu(lane_distance(traj, lanes, lane_type) - margin).mean(-1)


def rerank(logits, traj, lanes, lane_type, margin, weight):
    """Mode logits lowered by `weight` per metre that a mode's farthest point lies beyond `margin`
    from every lane centerline. Uses only model inputs, so it is a legitimate run-time step."""
    worst = torch.relu(lane_distance(traj, lanes, lane_type).amax(-1) - margin)  # (B, K)
    return logits - weight * worst
