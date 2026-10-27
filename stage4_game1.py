"""Stage 4a run: exact Game 1 solution against every opponent population.

For each population: score all 4,598,126 allocations exactly, cross-check the best one with
the dynamic program, and report where the reference picks rank. Then pick allocations over
the whole space: minimum worst-case regret and minimum average regret across populations
(regret = best possible score in that population minus this allocation's score).
"""
import numpy as np
import pandas as pd

from blotto.game1 import all_allocations, dp_best, solve, score_all
from blotto.scenarios import data_populations

REFERENCE = {"intuition pick": (3, 11, 21, 28, 37), "intuition +1": (4, 12, 22, 29, 33)}


def fmt(a):
    return "/".join(str(int(v)) for v in a)


def main():
    pops = data_populations()
    allocs = all_allocations()

    summary, regrets, tops = [], {}, []
    for name, pop in pops.items():
        scores, table, top = solve(pop, allocs, top_k=10)
        dp_score, dp_alloc = dp_best(table)
        assert abs(dp_score - scores.max()) < 1e-9, f"DP/enumeration mismatch in {name}"
        regrets[name] = (scores.max() - scores).astype(np.float32)
        row = {"population": name, "n_opponents": len(pop),
               "best_allocation": fmt(top.loc[0, "allocation"]), "best_score": scores.max()}
        for label, alloc in REFERENCE.items():
            s = score_all(table, np.array([alloc]))[0]
            row[f"{label} score"] = s
            row[f"{label} pct of best"] = s / scores.max()
            row[f"{label} rank"] = int((scores > s).sum()) + 1
        summary.append(row)
        top["population"] = name
        tops.append(top)

    summary = pd.DataFrame(summary)
    names = list(regrets)
    stacked = np.stack([regrets[n] for n in names])            # (populations, allocations)
    worst = stacked.max(axis=0)
    average = stacked.mean(axis=0)
    robust = pd.DataFrame({"allocation": [fmt(a) for a in allocs[np.argsort(worst)[:10]]],
                           "worst_regret": np.sort(worst)[:10],
                           "avg_regret": average[np.argsort(worst)[:10]]})
    by_avg = np.argsort(average)[:10]
    robust_avg = pd.DataFrame({"allocation": [fmt(a) for a in allocs[by_avg]],
                               "avg_regret": average[by_avg], "worst_regret": worst[by_avg]})
    ref_idx = {label: int(np.flatnonzero((allocs == np.array(a)).all(axis=1))[0])
                for label, a in REFERENCE.items()}

    pd.set_option("display.width", 200)
    print(summary[["population", "n_opponents", "best_allocation", "best_score",
                   "intuition pick score", "intuition pick pct of best", "intuition pick rank",
                   "intuition +1 score", "intuition +1 rank"]].round(3).to_string(index=False))
    print("\nRobust (min worst-case regret over all populations):")
    print(robust.round(3).to_string(index=False))
    print("\nRobust (min average regret):")
    print(robust_avg.round(3).to_string(index=False))
    for label, i in ref_idx.items():
        print(f"{label} {fmt(allocs[i])}: worst regret {worst[i]:.3f}, avg regret {average[i]:.3f}, "
              f"rank by worst regret {int((worst < worst[i]).sum()) + 1:,} of {len(allocs):,}")

    summary.to_csv("results/game1_by_population.csv", index=False)
    pd.concat(tops).to_csv("results/game1_top10_by_population.csv", index=False)
    robust.to_csv("results/game1_robust_minimax.csv", index=False)
    robust_avg.to_csv("results/game1_robust_average.csv", index=False)


if __name__ == "__main__":
    main()
