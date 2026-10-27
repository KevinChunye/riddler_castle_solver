import numpy as np

from blotto.transfer import largest_remainder_round, pair_merge, snap_to_5, value_cdf


def test_pair_merge_hand_case():
    vatter = np.array([[0, 1, 2, 16, 21, 3, 2, 1, 32, 22]])
    assert pair_merge(vatter).tolist() == [[1, 18, 24, 3, 54]]


def test_largest_remainder_hand_cases():
    assert largest_remainder_round([[33.4, 33.3, 33.3]]).tolist() == [[34, 33, 33]]
    # equal remainders: lower castle index gets the extra soldier
    assert largest_remainder_round([[50.5, 49.5]]).tolist() == [[51, 49]]
    assert largest_remainder_round([[21, 22, 19, 38]], step=5).tolist() == [[20, 20, 20, 40]]


def test_value_cdf_identity_when_games_match():
    rng = np.random.default_rng(0)
    allocs = rng.multinomial(100, [0.1, 0.15, 0.2, 0.25, 0.3], size=200)
    assert (value_cdf(allocs, [1, 2, 3, 4, 5]) == allocs).all()


def test_value_cdf_hand_case():
    # all 100 soldiers on castle 10 (value share 10/55 = 0.1818 at the top end)
    # 5-castle castle 5 covers value share 10/15..1 = 0.667..1, which contains all of castle 10
    top_only = np.zeros((1, 10), dtype=int)
    top_only[0, 9] = 100
    assert value_cdf(top_only, np.arange(1, 11)).tolist() == [[0, 0, 0, 0, 100]]
    # proportional-to-value entries stay proportional-to-value
    prop = np.arange(1, 11) / 55 * 100
    assert value_cdf(prop[None, :], np.arange(1, 11)).tolist() == [[7, 13, 20, 27, 33]]


def test_mappings_keep_totals_and_signs():
    rng = np.random.default_rng(3)
    allocs = rng.multinomial(100, np.ones(10) / 10, size=500)
    for out in (pair_merge(allocs), value_cdf(allocs, np.arange(1, 11)),
                snap_to_5(pair_merge(allocs))):
        assert (out.sum(axis=1) == 100).all() and (out >= 0).all() and out.shape[1] == 5
    assert (snap_to_5(pair_merge(allocs)) % 5 == 0).all()
