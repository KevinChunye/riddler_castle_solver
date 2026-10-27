from math import comb

import numpy as np

from blotto.engine import castle_points, point_unit
from blotto.game1 import all_allocations, castle_value_table, dp_best, score_all


def test_enumeration_is_complete_and_unique():
    small = all_allocations(3, 4)
    assert len(small) == comb(6, 2) and (small.sum(axis=1) == 4).all()
    assert len({tuple(r) for r in small.tolist()}) == len(small)
    full = all_allocations()
    assert len(full) == comb(104, 4) == 4_598_126
    assert (full.sum(axis=1) == 100).all() and full.min() >= 0


def test_table_score_equals_direct_head_to_head_average():
    rng = np.random.default_rng(4)
    pop = rng.multinomial(100, [0.1, 0.15, 0.2, 0.25, 0.3], size=300)
    mine = rng.multinomial(100, [0.2] * 5, size=50)
    table = castle_value_table(pop)
    fast = score_all(table, mine)
    unit = point_unit(2)
    for alloc, f in zip(mine, fast):
        groups = np.stack(np.broadcast_arrays(alloc[None, :], pop), axis=1)  # (300, 2, 5)
        direct = castle_points(groups, [1, 2, 3, 4, 5], unit)[:, 0].mean() / unit
        assert abs(direct - f) < 1e-12


def test_dp_agrees_with_full_enumeration():
    rng = np.random.default_rng(5)
    pop = rng.multinomial(100, [0.05, 0.1, 0.25, 0.3, 0.3], size=400)
    table = castle_value_table(pop)
    full = all_allocations()
    scores = score_all(table, full)
    dp_score, dp_alloc = dp_best(table)
    assert abs(dp_score - scores.max()) < 1e-12
    assert abs(score_all(table, np.array([dp_alloc]))[0] - dp_score) < 1e-12
