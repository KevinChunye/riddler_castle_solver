"""Stage 6: Game 2 robustness — data scenarios plus sophisticated-crowd (level-k) scenarios.

1. Level-k scenarios: pooled data + share w (10%, 25%) of players using the Game 2 best
   responses to the previous level; each level found with the full stage-4b search.
2. Candidate set: top 3 from every scenario's search + Game 1 candidates. All candidates are
   scored in every scenario on one fresh, seeded sample of 131,072 groups (common random
   numbers). "Share of best" uses the best candidate in that scenario, which includes the
   scenario's own search winner.
3. Diagnostics for the finalists: finishing-place odds and castle-by-castle take.
"""
from multiprocessing import Pool

import numpy as np
import pandas as pd

from blotto.game2 import diagnostics, evaluate, outcome_tables, sample_groups
from blotto.scenarios import data_populations, mix_in
from stage4_game2 import search

EVAL_GROUPS = 131_072
GAME1_CANDIDATES = ["3/11/21/28/37", "6/16/1/36/41", "7/16/1/36/40", "6/6/3/42/43", "8/17/32/0/43"]


def parse(key):
    return np.array([int(v) for v in key.split("/")])


def top_allocations(final, n=10):
    return np.array([parse(k) for k in final.sort_values("payout", ascending=False)["allocation"].head(n)])


def level_k_searches(pooled, pooled_final):
    """Run the level-1 and level-2 searches for both shares (two at a time)."""
    finals, pops = {}, {}
    responders0 = top_allocations(pooled_final)
    level1 = {f"level1-{int(s * 100)}pct": mix_in(pooled, responders0, s) for s in (0.1, 0.25)}
    pops.update(level1)
    with Pool(2) as pool:
        for name, final, log in pool.imap_unordered(
                search, [(n, p, 50 + i) for i, (n, p) in enumerate(level1.items())]):
            print("\n".join(log), flush=True)
            finals[name] = final
    level2 = {f"level2-{int(s * 100)}pct": mix_in(pooled, top_allocations(finals[f"level1-{int(s * 100)}pct"]), s)
              for s in (0.1, 0.25)}
    pops.update(level2)
    with Pool(2) as pool:
        for name, final, log in pool.imap_unordered(
                search, [(n, p, 60 + i) for i, (n, p) in enumerate(level2.items())]):
            print("\n".join(log), flush=True)
            finals[name] = final
    for name, final in finals.items():
        final.to_csv(f"results/game2_{name}.csv", index=False)
    return pops, finals


def main():
    pd.set_option("display.width", 240)
    pops = data_populations()
    finals = {name: pd.read_csv(f"results/game2_{name}.csv") for name in pops}
    level_pops, level_finals = level_k_searches(pops["pooled"], finals["pooled"])
    pops.update(level_pops)
    finals.update(level_finals)

    keys = []
    for final in finals.values():
        keys += final.sort_values("payout", ascending=False)["allocation"].head(3).tolist()
    keys += GAME1_CANDIDATES
    keys = list(dict.fromkeys(keys))           # unique, first-seen order
    cands = np.array([parse(k) for k in keys])

    rows = []
    for s, (name, pop) in enumerate(pops.items()):
        tables = outcome_tables(pop[sample_groups(len(pop), EVAL_GROUPS, seed=9000 + s)])
        payout = evaluate(cands, tables, chunk=8)
        rows.append(pd.DataFrame({"scenario": name, "allocation": keys, "payout": payout}))
        del tables
    long = pd.concat(rows)
    long["share_of_best"] = long["payout"] / long.groupby("scenario")["payout"].transform("max")
    long["is_data"] = ~long["scenario"].str.startswith("level")
    long.to_csv("results/game2_candidates_by_scenario.csv", index=False)

    data_part = long[long["is_data"]].groupby("allocation")["share_of_best"].agg(["mean", "min"])
    all_part = long.groupby("allocation")["share_of_best"].agg(["mean", "min"])
    summary = pd.concat([data_part.add_prefix("data_"), all_part.add_prefix("all_"),
                         long[long["scenario"] == "pooled"].set_index("allocation")["payout"]
                         .rename("pooled_payout")], axis=1).sort_values("all_min", ascending=False)
    print("\nCandidates ranked by worst-case share of best over all scenarios:")
    print(summary.round(3).head(15).to_string())
    print("\nGame 1 candidates in Game 2:")
    print(summary.loc[GAME1_CANDIDATES].round(3).to_string())
    summary.to_csv("results/game2_robust_summary.csv")

    wide = long.pivot(index="scenario", columns="allocation", values="share_of_best")
    show = summary.head(4).index.tolist() + GAME1_CANDIDATES[:2]
    print("\nShare of best by scenario:")
    print(wide.loc[list(pops), show].round(3).to_string())

    pooled = pops["pooled"]
    opponents = pooled[sample_groups(len(pooled), 200_000, seed=777)]
    diag = pd.DataFrame([{"allocation": k, **diagnostics(parse(k), opponents)} for k in show])
    print("\nDiagnostics vs pooled data crowd:")
    print(diag.round(3).to_string(index=False))
    diag.to_csv("results/game2_diagnostics.csv", index=False)


if __name__ == "__main__":
    main()
