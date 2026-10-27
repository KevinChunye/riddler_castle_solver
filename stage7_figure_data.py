"""Stage 7: the data behind the README figures. Writes tidy CSVs to results/figure_data/.

Definitions used in the figures
  score_s(x)        expected result of allocation x against scenario s's crowd
                    Game 1: average castle points per head-to-head match (exact)
                    Game 2: average payout per four-player group (9/4/2/0), from 131,072
                            simulated groups (the same groups stage 6 used)
  best_s            best score any allocation reaches in scenario s
                    Game 1: exact, over all 4,598,126 allocations
                    Game 2: best allocation found by the stage-4b/6 search
  regret_s(x)       best_s - score_s(x): what x gives up against the best choice for that crowd
  average regret    mean of regret_s(x) over the scenarios   (the pick minimises this)
  worst-case regret max of regret_s(x) over the scenarios

Simulated rounds ("epochs"): per scenario, 20,000 rounds; in each round every strategy meets
the *same* opponent(s) (common random numbers), so curves differ only because strategies do.
Saved: the running average after n rounds, and the average within each epoch of 1,000 rounds.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from blotto.game1 import all_allocations, castle_value_table, score_all
from blotto.game2 import diagnostics, evaluate, outcome_tables, sample_groups
from blotto.scenarios import data_populations, level_k_scenarios, mix_in

OUT = Path("results/figure_data")
VALUES = np.array([1, 2, 3, 4, 5])
N_ROUNDS = 20_000
EPOCH = 1_000
CHECKPOINTS = np.unique(np.geomspace(1, N_ROUNDS, 160).astype(int))

GAME1 = [
    ("6/16/1/36/41", "Robust pick: lowest average regret"),
    ("7/16/1/36/40", "Robust pick: lowest worst-case regret"),
    ("0/0/0/40/60", "All-in on castles 4 and 5"),
    ("3/11/21/28/37", "First intuition pick"),
    ("8/17/32/0/43", "Robust incl. sophisticated crowds"),
    ("7/13/20/27/33", "Proportional to castle value"),
    ("20/20/20/20/20", "Even split"),
]


def parse(key):
    return np.array([int(v) for v in key.split("/")])


def running_and_epochs(scenario, keys, per_round):
    """per_round: (n_strategies, N_ROUNDS) -> tidy running-average and epoch-average frames."""
    running = np.cumsum(per_round, axis=1)[:, CHECKPOINTS - 1] / CHECKPOINTS
    epochs = per_round.reshape(len(keys), -1, EPOCH).mean(axis=2)
    run = pd.DataFrame(running, index=keys, columns=CHECKPOINTS).rename_axis(index="allocation", columns="round")
    ep = pd.DataFrame(epochs, index=keys, columns=np.arange(1, epochs.shape[1] + 1)).rename_axis(
        index="allocation", columns="epoch")
    run = run.stack().rename("running_avg").reset_index().assign(scenario=scenario)
    ep = ep.stack().rename("epoch_avg").reset_index().assign(scenario=scenario)
    return run, ep


def regret_summary(scores):
    is_data = ~scores["scenario"].str.startswith("level")
    data = scores[is_data].groupby("allocation")["regret"].agg(["mean", "max"])
    every = scores.groupby("allocation")["regret"].agg(["mean", "max"])
    out = pd.concat([data.add_prefix("data_"), every.add_prefix("all_")], axis=1)
    out.columns = ["avg_regret_data", "worst_regret_data", "avg_regret_all", "worst_regret_all"]
    return out.reset_index()


def game1(scenarios, allocs):
    keys = [k for k, _ in GAME1]
    strategies = np.array([parse(k) for k in keys])
    runs, eps, scores, curves, castles = [], [], [], [], []
    for s, (name, pop) in enumerate(scenarios.items()):
        table = castle_value_table(pop)                     # table[i, x]: exact expected points
        full = score_all(table, allocs)
        best = int(np.argmax(full))
        exact = score_all(table, strategies)
        rng = np.random.default_rng(100 + s)
        opp = pop[rng.integers(0, len(pop), N_ROUNDS)]
        mine = strategies[:, None, :]
        per_round = ((mine > opp) + 0.5 * (mine == opp)) @ VALUES       # castle points per round
        run, ep = running_and_epochs(name, keys, per_round)
        runs.append(run)
        eps.append(ep)
        scores.append(pd.DataFrame({"scenario": name, "allocation": keys, "score": exact,
                                    "best_score": full[best],
                                    "best_allocation": "/".join(map(str, allocs[best])),
                                    "regret": full[best] - exact}))
        for i in range(5):
            curves.append(pd.DataFrame({"scenario": name, "castle": i + 1, "soldiers": np.arange(101),
                                        "exp_points": table[i]}))
            castles.append(pd.DataFrame({"scenario": name, "allocation": keys, "castle": i + 1,
                                         "soldiers": strategies[:, i],
                                         "exp_points": table[i][strategies[:, i]]}))
    scores = pd.concat(scores)
    return {"g1_running": pd.concat(runs), "g1_epochs": pd.concat(eps), "g1_scores": scores,
            "g1_curves": pd.concat(curves), "g1_castles": pd.concat(castles),
            "g1_regret": regret_summary(scores)}


def game2(scenarios, keys):
    strategies = np.array([parse(k) for k in keys])
    cand = pd.read_csv("results/game2_candidates_by_scenario.csv")
    best_rows = cand.loc[cand.groupby("scenario")["payout"].idxmax()].set_index("scenario")
    runs, eps, scores, places, castles = [], [], [], [], []
    for s, (name, pop) in enumerate(scenarios.items()):
        # the same evaluation groups as stage 6 (seed 9000 + s), so scores line up with best_s
        tables = outcome_tables(pop[sample_groups(len(pop), 131_072, seed=9000 + s)])
        score = evaluate(strategies, tables, chunk=8)
        del tables
        best_score = max(best_rows.loc[name, "payout"], score.max())
        best_alloc = best_rows.loc[name, "allocation"] if best_rows.loc[name, "payout"] >= score.max() \
            else keys[int(np.argmax(score))]
        opponents = pop[sample_groups(len(pop), N_ROUNDS, seed=200 + s)]
        per_round = evaluate(strategies, outcome_tables(opponents), chunk=8, per_group=True)
        run, ep = running_and_epochs(name, keys, per_round)
        runs.append(run)
        eps.append(ep)
        scores.append(pd.DataFrame({"scenario": name, "allocation": keys, "score": score,
                                    "best_score": best_score, "best_allocation": best_alloc,
                                    "regret": best_score - score}))
        for k in keys:
            d = diagnostics(parse(k), opponents)
            for p in range(1, 5):
                places.append({"scenario": name, "allocation": k, "place": p, "share": d[f"P(place {p})"]})
            for i in range(5):
                castles.append({"scenario": name, "allocation": k, "castle": i + 1,
                                "soldiers": int(parse(k)[i]), "take_share": d[f"castle{i + 1}_share"],
                                "exp_points": (i + 1) * d[f"castle{i + 1}_share"]})
    scores = pd.concat(scores)
    return {"g2_running": pd.concat(runs), "g2_epochs": pd.concat(eps), "g2_scores": scores,
            "g2_places": pd.DataFrame(places), "g2_castles": pd.DataFrame(castles),
            "g2_regret": regret_summary(scores)}


def top10(name):
    final = pd.read_csv(f"results/game2_{name}.csv")
    return np.array([parse(k) for k in final.sort_values("payout", ascending=False)["allocation"].head(10)])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    allocs = all_allocations()
    pops = data_populations()

    def ranked(pop):
        sc = score_all(castle_value_table(pop), allocs)
        return allocs[np.argsort(-sc, kind="stable")[:10]]

    g1_scenarios = dict(pops)
    g1_scenarios.update(level_k_scenarios(pops["pooled"], ranked))
    frames = game1(g1_scenarios, allocs)
    print("game 1 done", flush=True)

    g2_scenarios = dict(pops)          # same order as stage 6, so evaluation seeds line up
    for share in (10, 25):
        g2_scenarios[f"level1-{share}pct"] = mix_in(pops["pooled"], top10("pooled"), share / 100)
    for share in (10, 25):
        g2_scenarios[f"level2-{share}pct"] = mix_in(pops["pooled"], top10(f"level1-{share}pct"), share / 100)
    summary = pd.read_csv("results/game2_robust_summary.csv", index_col=0)
    robust = summary.sort_values("all_min", ascending=False).index[0]
    robust_avg = summary.sort_values("all_mean", ascending=False).index[0]
    game2_list = [(robust, "Robust pick: best worst case"), (robust_avg, "Robust pick: best average"),
                  ("10/29/0/61/0", "Best vs pooled Riddler crowd"),
                  ("0/33/0/0/67", "Same idea, big stack on castle 5"), ("6/16/1/36/41", "Game 1 robust pick"),
                  ("0/0/0/40/60", "All-in on castles 4 and 5"), ("3/11/21/28/37", "First intuition pick"),
                  ("20/20/20/20/20", "Even split")]
    game2_list = list(pd.DataFrame(game2_list).drop_duplicates(0).itertuples(index=False, name=None))
    frames.update(game2(g2_scenarios, [k for k, _ in game2_list]))
    print("game 2 done", flush=True)

    strategies = pd.concat([
        pd.DataFrame({"game": 1, "allocation": [k for k, _ in GAME1], "label": [l for _, l in GAME1],
                      "order": range(len(GAME1))}),
        pd.DataFrame({"game": 2, "allocation": [k for k, _ in game2_list], "label": [l for _, l in game2_list],
                      "order": range(len(game2_list))})])
    scenarios = pd.DataFrame({
        "scenario": list(g1_scenarios),
        "kind": ["sophisticated crowd" if n.startswith("level") else "Riddler data" for n in g1_scenarios],
        "opponents": [len(p) for p in g1_scenarios.values()]})
    frames["g1_castles"]["take_share"] = frames["g1_castles"]["exp_points"] / frames["g1_castles"]["castle"]

    def both(name):
        return pd.concat([frames[f"g1_{name}"].assign(game=1), frames[f"g2_{name}"].assign(game=2)])

    def labelled(frame, game):
        return frame.merge(strategies[strategies["game"] == game][["allocation", "label"]], on="allocation")

    files = {
        "running": both("running"), "epochs": both("epochs"),
        "scores": pd.concat([labelled(frames["g1_scores"], 1).assign(game=1),
                             labelled(frames["g2_scores"], 2).assign(game=2)]),
        "castles": both("castles")[["game", "scenario", "allocation", "castle", "soldiers",
                                    "take_share", "exp_points"]],
        "g1_regret": labelled(frames["g1_regret"], 1), "g2_regret": labelled(frames["g2_regret"], 2),
        "places": frames["g2_places"], "curves": frames["g1_curves"],
        "strategies": strategies, "scenarios": scenarios,
    }
    for name, frame in files.items():
        frame.to_csv(OUT / f"{name}.csv", index=False, float_format="%.5g")
        print(f"{name}: {len(frame):,} rows, {(OUT / f'{name}.csv').stat().st_size / 1e3:.0f} KB")


if __name__ == "__main__":
    main()
