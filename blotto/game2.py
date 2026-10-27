"""Stage 4b: Game 2 (four-way battle, payout 9/4/2/0 by finishing place, ties split).

Unlike Game 1 this does not split by castle: your payout depends on your total relative to
three other totals, and theirs depend on what you take from them. So the expected payout is
estimated by simulating groups, with these safeguards:

  * fixed seeds, so every number is reproducible;
  * common random numbers: every candidate allocation faces the *same* sampled groups, so
    differences between candidates are measured with far less noise than their levels;
  * an exact-enumeration path (every possible group) for small populations, used in tests
    to confirm the simulation is unbiased.

Outcome tables: for a sampled group of three opponents, castle i's outcome depends only on
how many soldiers we send there (0..100). table[i, x, g, p] stores the points player p gets
at castle i in group g when we send x. Scoring an allocation is then 5 lookups and a sum.
"""
from itertools import combinations

import numpy as np

from blotto.engine import castle_points, place_payouts, point_unit

VALUES_5 = np.array([1, 2, 3, 4, 5])
PAYOUTS = np.array([9.0, 4.0, 2.0, 0.0])
TOTAL = 100
N_PLAYERS = 4
UNIT = point_unit(N_PLAYERS)  # 12


def sample_groups(n_population, n_groups, seed, n_opponents=N_PLAYERS - 1):
    """Indices of n_opponents distinct population members per group (reproducible)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_population, size=(n_groups, n_opponents))
    while True:
        s = np.sort(idx, axis=1)
        dup = (np.diff(s, axis=1) == 0).any(axis=1)
        if not dup.any():
            return idx
        idx[dup] = rng.integers(0, n_population, size=(dup.sum(), n_opponents))


def outcome_tables(opponents, values=VALUES_5, total=TOTAL, chunk=8192):
    """opponents: (G, 3, n_castles). Returns int16 table (n_castles, total+1, G, 4).

    Built in chunks of groups so temporary arrays stay small (~30 MB) whatever G is.
    """
    n_groups, _, n_castles = opponents.shape
    tables = np.empty((n_castles, total + 1, n_groups, N_PLAYERS), dtype=np.int16)
    mine = np.arange(total + 1, dtype=np.int16)
    for start in range(0, n_groups, chunk):
        opp = opponents[start:start + chunk].astype(np.int16)
        g = len(opp)
        for i in range(n_castles):
            theirs = np.broadcast_to(opp[None, :, :, i], (total + 1, g, 3))
            ours = np.broadcast_to(mine[:, None, None], (total + 1, g, 1))
            group = np.concatenate([ours, theirs], axis=2)[..., None]   # one castle at a time
            tables[i, :, start:start + g] = castle_points(group, [values[i]], UNIT)
    return tables


def _payout_lookup(payouts=PAYOUTS):
    """lookup[better, equal]: average payout over places better .. better+equal-1."""
    cum = np.concatenate([[0.0], np.cumsum(payouts)])
    n = len(payouts)
    lookup = np.zeros((n + 1, n + 1))
    for better in range(n):
        for equal in range(1, n - better + 1):
            lookup[better, equal] = (cum[better + equal] - cum[better]) / equal
    return lookup


def payout_of_player0(points, payouts=PAYOUTS):
    """points: (..., 4) totals with our player first. Returns our payout (...).

    better = opponents strictly ahead of us, equal = players tied with us (ourselves included).
    """
    ours = points[..., :1]
    better = (points[..., 1:] > ours).sum(axis=-1, dtype=np.int8)
    equal = 1 + (points[..., 1:] == ours).sum(axis=-1, dtype=np.int8)
    return _payout_lookup(payouts)[better, equal]


def evaluate(allocations, tables, chunk=4096, per_group=False):
    """Mean payout of each allocation over the groups in `tables` (same groups for all)."""
    allocations = np.asarray(allocations)
    n_groups = tables.shape[2]
    out = np.empty((len(allocations), n_groups) if per_group else len(allocations))
    for start in range(0, len(allocations), chunk):
        a = allocations[start:start + chunk]
        pts = tables[0][a[:, 0]].astype(np.int16)
        for i in range(1, tables.shape[0]):
            pts += tables[i][a[:, i]]
        pay = payout_of_player0(pts)
        out[start:start + chunk] = pay if per_group else pay.mean(axis=1)
    return out


def evaluate_direct(allocation, opponents, values=VALUES_5, payouts=PAYOUTS):
    """Same quantity without tables: build every group and run the generic engine."""
    mine = np.broadcast_to(np.asarray(allocation)[None, None, :], (len(opponents), 1, len(values)))
    groups = np.concatenate([mine, opponents], axis=1)
    pts = castle_points(groups, values, UNIT)
    return place_payouts(pts, payouts)[:, 0]


def diagnostics(allocation, opponents, values=VALUES_5):
    """Where the points and the payout come from, for one allocation vs sampled groups.

    Returns a dict: share of each finishing place (tied places split), mean castle points,
    and the share of each castle we take (1 = won outright, 1/k when tied k ways).
    """
    mine = np.broadcast_to(np.asarray(allocation)[None, None, :], (len(opponents), 1, len(values)))
    groups = np.concatenate([mine, opponents], axis=1)
    pts = castle_points(groups, values, UNIT)
    places = {f"P(place {k + 1})": place_payouts(pts, np.eye(N_PLAYERS)[k])[:, 0].mean()
              for k in range(N_PLAYERS)}
    top = groups.max(axis=1)
    at_top = groups == top[:, None, :]
    castle_share = (at_top[:, 0, :] / at_top.sum(axis=1)).mean(axis=0)
    return {**places, "mean_points": pts[:, 0].mean() / UNIT,
            **{f"castle{i + 1}_share": castle_share[i] for i in range(len(values))}}


def exact_expected_payout(allocation, population, values=VALUES_5):
    """Exact expectation over every set of 3 distinct opponents (small populations only)."""
    population = np.asarray(population)
    triples = np.array(list(combinations(range(len(population)), N_PLAYERS - 1)))
    return evaluate_direct(allocation, population[triples], values).mean()
