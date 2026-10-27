"""Stage 5a: Game 1 robustness — data scenarios plus sophisticated-crowd (level-k) scenarios.

1. Level-k drift: pooled data with a share w in {10%, 25%, 50%} of players best-responding
   to the previous level, levels 0-4. Shows where an increasingly clever crowd pushes play.
2. Robust pick over 16 scenarios (12 data + level 1/2 at 10%/25%), exactly, over the full
   4,598,126-allocation space: min worst-case regret and min average regret.
3. Candidate table: each candidate's score as a share of the best achievable, per scenario.
"""
import numpy as np
import pandas as pd

from blotto.game1 import all_allocations, castle_value_table, score_all
from blotto.scenarios import data_populations, level_k_scenarios, mix_in

CANDIDATES = [(3, 11, 21, 28, 37), (6, 16, 1, 36, 41), (7, 16, 1, 36, 40),
              (6, 6, 3, 42, 43), (8, 17, 32, 0, 43)]


def fmt(a):
    return "/".join(str(int(v)) for v in a)


def main():
    pd.set_option("display.width", 220)
    allocs = all_allocations()

    def ranked(pop):
        scores = score_all(castle_value_table(pop), allocs)
        return allocs[np.argsort(-scores, kind="stable")[:10]]

    def find(a):
        return int(np.flatnonzero((allocs == np.array(a)).all(axis=1))[0])

    pops = data_populations()

    # 1. level-k drift
    rows = []
    for share in (0.1, 0.25, 0.5):
        pop, responders = pops["pooled"], None
        for level in range(5):
            if level > 0:
                pop = mix_in(pops["pooled"], responders, share)
            scores = score_all(castle_value_table(pop), allocs)
            responders = allocs[np.argsort(-scores, kind="stable")[:10]]
            row = {"share": share, "level": level, "best": fmt(responders[0]), "best_score": scores.max()}
            row.update({fmt(c): scores[find(c)] / scores.max() for c in CANDIDATES})
            rows.append(row)
    drift = pd.DataFrame(rows)
    print("1. Level-k drift (candidate score / best achievable):")
    print(drift.round(3).to_string(index=False))

    # 2. robust pick over all scenarios
    scenarios = dict(pops)
    scenarios.update(level_k_scenarios(pops["pooled"], ranked))
    regret = np.empty((len(scenarios), len(allocs)), dtype=np.float32)
    share_table = []
    for k, (name, pop) in enumerate(scenarios.items()):
        s = score_all(castle_value_table(pop), allocs)
        regret[k] = s.max() - s
        share_table.append({"scenario": name, "best": fmt(allocs[np.argmax(s)]), "best_score": s.max(),
                            **{fmt(c): s[find(c)] / s.max() for c in CANDIDATES}})
    worst, average = regret.max(axis=0), regret.mean(axis=0)
    for label, key in (("worst-case", worst), ("average", average)):
        idx = np.argsort(key, kind="stable")[:8]
        print(f"\n2. Min {label} regret over {len(scenarios)} scenarios:")
        print(pd.DataFrame({"allocation": [fmt(a) for a in allocs[idx]],
                            "worst_regret": worst[idx], "avg_regret": average[idx]}).round(3).to_string(index=False))

    # 3. candidate table
    share_table = pd.DataFrame(share_table)
    print("\n3. Candidates, score / best achievable, per scenario:")
    print(share_table.round(3).to_string(index=False))
    cols = [fmt(c) for c in CANDIDATES]
    is_data = ~share_table["scenario"].str.startswith("level")
    print("\nsummary:")
    print(pd.DataFrame({"data mean": share_table.loc[is_data, cols].mean(),
                        "data min": share_table.loc[is_data, cols].min(),
                        "all mean": share_table[cols].mean(),
                        "all min": share_table[cols].min()}).round(3).to_string())

    drift.to_csv("results/game1_level_k_drift.csv", index=False)
    share_table.to_csv("results/game1_candidates_by_scenario.csv", index=False)


if __name__ == "__main__":
    main()
