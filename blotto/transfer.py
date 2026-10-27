"""Stage 3: turn Riddler entries (10 or 13 castles) into opponents for the 5-castle game.

No mapping is "the truth" — the applicants for this tournament never played the 10-castle
game. Each mapping below is deterministic and keeps a different feature of how people
played, so the solver can be run against all of them and the answer checked for robustness.

pair_merge   castles (1,2)->1, (3,4)->2, ..., (9,10)->5. The merged values 3,7,11,15,19
             are almost exactly 3.8 x (1,2,3,4,5), so the coarsened game is nearly the
             same game. Keeps which castles a player abandoned.
value_cdf    keep the share of soldiers sent per share of the point pool: interpolate the
             cumulative-soldiers curve at the 5-castle value breakpoints. Works for any
             source castle count (used for the 13-castle round 5).
snap_to_5    re-round an already mapped entry to multiples of 5 — a stress test for a
             crowd that likes round numbers more than the mapped data shows.
"""
import numpy as np

TOTAL = 100


def largest_remainder_round(real_allocs, total=TOTAL, step=1):
    """Round each row to multiples of `step` summing to `total` (Hamilton's method).

    Floors every entry, then hands the leftover units to the largest fractional parts;
    equal remainders go to the lower castle index (np.argsort is stable).
    """
    units = np.asarray(real_allocs, dtype=float) / step
    base = np.floor(units + 1e-9).astype(np.int64)
    leftover = total // step - base.sum(axis=1)
    remainder = units - base
    order = np.argsort(-remainder, axis=1, kind="stable")
    rank = np.argsort(order, axis=1, kind="stable")      # rank of each castle's remainder
    base += (rank < leftover[:, None]).astype(np.int64)
    return base * step


def pair_merge(allocs10):
    allocs10 = np.asarray(allocs10)
    if allocs10.shape[1] != 10:
        raise ValueError("pair_merge needs 10-castle entries")
    return allocs10.reshape(len(allocs10), 5, 2).sum(axis=2)


def value_cdf(allocs, src_values, dst_values=(1, 2, 3, 4, 5)):
    allocs = np.asarray(allocs, dtype=float)
    src_cum_value = np.concatenate([[0], np.cumsum(src_values)]) / np.sum(src_values)
    dst_cum_value = np.concatenate([[0], np.cumsum(dst_values)]) / np.sum(dst_values)
    cum_soldiers = np.concatenate([np.zeros((len(allocs), 1)), np.cumsum(allocs, axis=1)], axis=1)
    at_dst = np.stack([np.interp(dst_cum_value, src_cum_value, row) for row in cum_soldiers])
    return largest_remainder_round(np.diff(at_dst, axis=1))


def snap_to_5(allocs5):
    return largest_remainder_round(allocs5, step=5)


def build_populations(rounds, top_responses=None):
    """Named 5-castle opponent populations from the Riddler rounds.

    rounds: {round_id: allocations}. Returns {name: int array (n, 5)}.
    """
    pops = {}
    for r in (1, 2, 3, 4):
        pops[f"R{r}-pair"] = pair_merge(rounds[r])
        pops[f"R{r}-cdf"] = value_cdf(rounds[r], np.arange(1, 11))
    pops["R5-cdf"] = value_cdf(rounds[5], np.arange(1, 14))
    pops["R1-pair-snap5"] = snap_to_5(pops["R1-pair"])
    pops["R2-pair-snap5"] = snap_to_5(pops["R2-pair"])
    return pops
