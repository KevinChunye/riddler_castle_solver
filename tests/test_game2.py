import numpy as np

from blotto.game2 import (evaluate, evaluate_direct, exact_expected_payout, outcome_tables,
                          payout_of_player0, sample_groups)


def test_payout_hand_cases():
    # points in twelfths; we are player 0
    assert payout_of_player0(np.array([60, 10, 20, 30])) == 9
    assert payout_of_player0(np.array([30, 30, 10, 0])) == 6.5        # tied first
    assert payout_of_player0(np.array([0, 30, 30, 30])) == 0
    assert abs(payout_of_player0(np.array([10, 30, 10, 10])) - 2) < 1e-12  # tied 2nd-4th: (4+2+0)/3


def test_groups_are_distinct_and_reproducible():
    a = sample_groups(50, 5000, seed=7)
    b = sample_groups(50, 5000, seed=7)
    assert (a == b).all()
    s = np.sort(a, axis=1)
    assert (np.diff(s, axis=1) > 0).all()


def test_table_path_equals_direct_engine():
    rng = np.random.default_rng(8)
    pop = rng.multinomial(100, [0.1, 0.15, 0.2, 0.25, 0.3], size=200)
    opponents = pop[sample_groups(len(pop), 3000, seed=9)]
    tables = outcome_tables(opponents)
    mine = np.vstack([rng.multinomial(100, [0.2] * 5, size=30), [[3, 11, 21, 28, 37]],
                      [[20, 20, 20, 20, 20]], [[0, 0, 0, 0, 100]]])
    fast = evaluate(mine, tables, per_group=True)
    for alloc, f in zip(mine, fast):
        assert np.allclose(evaluate_direct(alloc, opponents), f)


def test_simulation_matches_exact_enumeration():
    """Small population: exact expectation over all C(14,3)=364 groups vs 200k sampled groups."""
    rng = np.random.default_rng(10)
    pop = rng.multinomial(100, [0.1, 0.15, 0.2, 0.25, 0.3], size=14)
    tables = outcome_tables(pop[sample_groups(len(pop), 200_000, seed=11)])
    for alloc in ([3, 11, 21, 28, 37], [6, 16, 1, 36, 41], [20, 20, 20, 20, 20]):
        sim = evaluate(np.array([alloc]), tables, per_group=True)[0]
        exact = exact_expected_payout(alloc, pop)
        se = sim.std() / np.sqrt(len(sim))
        assert abs(sim.mean() - exact) < 4 * se + 1e-9


def test_payouts_conserve_prize_pool():
    """Rotating who is 'player 0' over the same group must hand out exactly 15 points."""
    rng = np.random.default_rng(12)
    groups = rng.integers(0, 5, size=(2000, 4, 5))
    from blotto.engine import castle_points, place_payouts
    pts = castle_points(groups, [1, 2, 3, 4, 5], 12)
    rotated = sum(payout_of_player0(np.roll(pts, -k, axis=1)) for k in range(4))
    assert np.allclose(rotated, 15)
    assert np.allclose(place_payouts(pts, [9, 4, 2, 0]).sum(axis=1), 15)


def test_diagnostics_are_internally_consistent():
    from blotto.game2 import diagnostics, evaluate
    rng = np.random.default_rng(13)
    pop = rng.multinomial(100, [0.1, 0.15, 0.2, 0.25, 0.3], size=300)
    opponents = pop[sample_groups(len(pop), 20_000, seed=14)]
    alloc = np.array([0, 33, 0, 67, 0])
    d = diagnostics(alloc, opponents)
    places = [d[f"P(place {k})"] for k in (1, 2, 3, 4)]
    assert abs(sum(places) - 1) < 1e-9
    # castle points identity: sum of value x share taken = mean points
    assert abs(sum(v * d[f"castle{i + 1}_share"] for i, v in enumerate([1, 2, 3, 4, 5])) - d["mean_points"]) < 1e-9
    # expected payout identity: 9 P1 + 4 P2 + 2 P3 = payout from the evaluator
    payout = evaluate(alloc[None, :], outcome_tables(opponents))[0]
    assert abs(9 * places[0] + 4 * places[1] + 2 * places[2] - payout) < 1e-9
