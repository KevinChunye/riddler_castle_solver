"""Stage 2: the scoring engine, written once and reused by every game.

Notation (used throughout the package):
  values      castle point values, e.g. [1, 2, 3, 4, 5]
  allocation  soldiers sent to each castle, one integer per castle, summing to 100
  group       the allocations of every player in one contest, shape (..., n_players, n_castles)
  unit        points are kept as integers in units of 1/unit, so tie splits never use floats.
              unit = lcm(1..n_players): with up to 4 players, a castle tied by 2, 3 or 4
              players splits evenly (12 is divisible by 1, 2, 3 and 4).
"""
from math import lcm

import numpy as np


def point_unit(n_players):
    return lcm(*range(1, n_players + 1))


def castle_points(group, values, unit):
    """Integer castle points (in 1/unit) for every player in every group.

    A castle goes to whoever sends the most soldiers; players tied at the top split it.
    group: int array (..., n_players, n_castles). Returns int64 array (..., n_players).
    """
    group = np.asarray(group)
    scaled_values = np.asarray(values, dtype=np.int64) * unit
    top = group.max(axis=-2, keepdims=True)
    at_top = group == top
    n_at_top = at_top.sum(axis=-2, keepdims=True)
    share = np.where(at_top, scaled_values // n_at_top, 0)
    return share.sum(axis=-1)


def place_payouts(points, payouts):
    """Payout for every player given total points, splitting tied places evenly.

    points: (..., n_players) totals. payouts: list, payouts[k] for finishing place k+1.
    A player with `better` opponents strictly ahead and `equal` players tied with it
    (itself included) occupies places better .. better+equal-1 and gets their average.
    """
    points = np.asarray(points)
    cum = np.concatenate([[0.0], np.cumsum(payouts, dtype=float)])
    diff = points[..., None, :] - points[..., :, None]   # [i, j] = points_j - points_i
    better = (diff > 0).sum(axis=-1)
    equal = (diff == 0).sum(axis=-1)
    return (cum[better + equal] - cum[better]) / equal


def head_to_head(allocations, values, chunk=256):
    """Full round robin for a two-player game.

    Returns (points, wars):
      points[a, b] = castle points a scores against b (float, real points)
      wars[a, b]   = 1 if a wins the war, 0.5 tie, 0 loss  ("most total points wins")
    """
    allocations = np.asarray(allocations)
    n = len(allocations)
    unit = point_unit(2)
    points = np.empty((n, n))
    wars = np.empty((n, n))
    for start in range(0, n, chunk):
        a = allocations[start:start + chunk]
        group = np.stack(np.broadcast_arrays(a[:, None, :], allocations[None, :, :]), axis=2)
        pts = castle_points(group, values, unit)          # (chunk, n, 2)
        points[start:start + chunk] = pts[..., 0] / unit
        wars[start:start + chunk] = place_payouts(pts, [1.0, 0.0])[..., 0]
    return points, wars


def round_robin_summary(allocations, values):
    """Per-entry record against every *other* entry (self-play excluded)."""
    points, wars = head_to_head(allocations, values)
    n = len(allocations)
    off_diagonal = ~np.eye(n, dtype=bool)
    wins = ((wars == 1) & off_diagonal).sum(axis=1)
    ties = ((wars == 0.5) & off_diagonal).sum(axis=1)
    losses = ((wars == 0) & off_diagonal).sum(axis=1)
    mean_points = (points * off_diagonal).sum(axis=1) / (n - 1)
    win_rate = (wins + 0.5 * ties) / (n - 1)
    return {"wins": wins, "ties": ties, "losses": losses,
            "win_rate": win_rate, "mean_points": mean_points}
