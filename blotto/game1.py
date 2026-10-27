"""Stage 4a: Game 1 (head to head, score = average castle points over all opponents).

Because the score is an average of castle points, it splits castle by castle:

  score(x) = sum over castles i of  v_i * [ P(opponent sends < x_i) + 1/2 P(opponent sends = x_i) ]

where x_i is our soldiers at castle i, v_i its value, and P is over the opponent population.
So the whole strategy space (every way to split 100 soldiers over 5 castles,
C(104, 4) = 4,598,126 allocations) can be scored exactly — no sampling, no search heuristics.
"""
import numpy as np
import pandas as pd

TOTAL = 100
VALUES_5 = np.array([1, 2, 3, 4, 5])


def all_allocations(n_castles=5, total=TOTAL):
    """Every non-negative integer vector of length n_castles summing to total (int16).

    Built column by column: each partial row with running sum s is repeated once for every
    value 0..total-s the next castle can take; the last castle gets whatever is left.
    """
    prefix = np.zeros((1, 0), dtype=np.int16)
    used = np.zeros(1, dtype=np.int64)
    for _ in range(n_castles - 1):
        options = total - used + 1                               # choices for the next castle
        starts = np.repeat(np.cumsum(options) - options, options)
        next_col = np.arange(options.sum()) - starts             # 0..options-1 within each row
        prefix = np.column_stack([np.repeat(prefix, options, axis=0), next_col.astype(np.int16)])
        used = np.repeat(used, options) + next_col
    return np.column_stack([prefix, (total - used).astype(np.int16)])


def castle_value_table(population, values=VALUES_5, total=TOTAL):
    """f[i, x] = expected points at castle i when sending x soldiers vs one random opponent."""
    population = np.asarray(population)
    n = len(population)
    table = np.empty((len(values), total + 1))
    for i, v in enumerate(values):
        counts = np.bincount(population[:, i], minlength=total + 1)[: total + 1]
        below = np.concatenate([[0], np.cumsum(counts)[:-1]])  # opponents sending fewer than x
        table[i] = v * (below + 0.5 * counts) / n
    return table


def score_all(table, allocations):
    """Exact Game 1 score for every allocation (vectorised table lookup)."""
    return sum(table[i][allocations[:, i]] for i in range(table.shape[0]))


def dp_best(table, total=TOTAL):
    """Independent check: dynamic program over castles for the single best allocation."""
    n_castles = table.shape[0]
    best = np.full((n_castles + 1, total + 1), -np.inf)
    choice = np.zeros((n_castles + 1, total + 1), dtype=int)
    best[0, 0] = 0.0
    for i in range(n_castles):
        for used in range(total + 1):
            for x in range(used + 1):
                cand = best[i, used - x] + table[i, x]
                if cand > best[i + 1, used]:
                    best[i + 1, used], choice[i + 1, used] = cand, x
    alloc, used = [], total
    for i in range(n_castles, 0, -1):
        alloc.append(choice[i, used])
        used -= choice[i, used]
    return best[n_castles, total], tuple(int(a) for a in reversed(alloc))


def solve(population, allocations, values=VALUES_5, top_k=20):
    """Score the full space; return (scores array, top-k DataFrame)."""
    table = castle_value_table(population, values)
    scores = score_all(table, allocations)
    top_idx = np.argpartition(-scores, top_k)[:top_k]
    top = pd.DataFrame({"allocation": [tuple(int(v) for v in a) for a in allocations[top_idx]],
                        "score": scores[top_idx]}).sort_values("score", ascending=False)
    return scores, table, top.reset_index(drop=True)
