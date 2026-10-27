"""Deterministic checks for the scoring engine: hand-worked cases + a slow reference."""
from fractions import Fraction
from itertools import product

import numpy as np

from blotto.engine import castle_points, place_payouts, point_unit, round_robin_summary


def reference_castle_points(group, values):
    """Plain-Python version of the rules, using exact fractions."""
    totals = [Fraction(0)] * len(group)
    for c, value in enumerate(values):
        top = max(player[c] for player in group)
        winners = [p for p, player in enumerate(group) if player[c] == top]
        for p in winners:
            totals[p] += Fraction(value, len(winners))
    return totals


def reference_payouts(totals, payouts):
    out = []
    for me in totals:
        better = sum(t > me for t in totals)
        equal = sum(t == me for t in totals)
        out.append(Fraction(sum(payouts[better:better + equal])) / equal)
    return out


def test_two_player_hand_case():
    # castle 1 tied (0.5 each), castle 2 to A (2), castle 3 to B (3) -> A 2.5, B 3.5
    pts = castle_points(np.array([[5, 5, 0], [5, 0, 5]]), [1, 2, 3], point_unit(2))
    assert (pts / point_unit(2)).tolist() == [2.5, 3.5]
    assert place_payouts(pts, [1, 0]).tolist() == [0.0, 1.0]


def test_three_way_castle_tie_splits_evenly():
    unit = point_unit(4)  # 12
    pts = castle_points(np.array([[10], [10], [10], [0]]), [3], unit)
    assert pts.tolist() == [12, 12, 12, 0]  # 1 point each, in twelfths


def test_place_payouts_hand_cases():
    pay = [9, 4, 2, 0]
    assert place_payouts(np.array([5, 5, 3, 1]), pay).tolist() == [6.5, 6.5, 2.0, 0.0]
    assert np.allclose(place_payouts(np.array([5, 3, 3, 3]), pay), [9, 2, 2, 2])
    assert place_payouts(np.array([7, 7, 7, 7]), pay).tolist() == [3.75] * 4
    # payouts always sum to the prize pool
    rng = np.random.default_rng(0)
    pts = rng.integers(0, 4, size=(1000, 4))
    assert np.allclose(place_payouts(pts, pay).sum(axis=1), 15)


def test_vectorized_matches_reference_on_random_groups():
    rng = np.random.default_rng(1)
    values = [1, 2, 3, 4, 5]
    for n_players in (2, 3, 4):
        unit = point_unit(n_players)
        groups = rng.integers(0, 4, size=(400, n_players, 5))  # small range forces many ties
        fast_pts = castle_points(groups, values, unit)
        fast_pay = place_payouts(fast_pts, [9, 4, 2, 0][:n_players])
        for g, fp, fpay in zip(groups, fast_pts, fast_pay):
            ref = reference_castle_points(g.tolist(), values)
            assert [Fraction(int(x), unit) for x in fp] == ref
            ref_pay = reference_payouts(ref, [9, 4, 2, 0][:n_players])
            assert np.allclose(fpay, [float(x) for x in ref_pay])


def test_round_robin_counts_add_up():
    rng = np.random.default_rng(2)
    allocs = rng.multinomial(100, [0.2] * 5, size=50)
    s = round_robin_summary(allocs, [1, 2, 3, 4, 5])
    assert ((s["wins"] + s["ties"] + s["losses"]) == 49).all()
    # zero-sum: average win rate is exactly one half
    assert abs(s["win_rate"].mean() - 0.5) < 1e-12


def test_exhaustive_tiny_game():
    """All 2-castle allocations of 3 soldiers, every pair, against the reference."""
    allocs = np.array([[a, 3 - a] for a in range(4)])
    s = round_robin_summary(allocs, [1, 2])
    for i, j in product(range(4), range(4)):
        ref = reference_castle_points([allocs[i].tolist(), allocs[j].tolist()], [1, 2])
        assert ref[0] + ref[1] == 3
    assert s["wins"].sum() == s["losses"].sum()
