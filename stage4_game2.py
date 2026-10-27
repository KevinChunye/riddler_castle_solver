"""Stage 4b run: Game 2 search, one per data population.

Search per population (successive halving with fresh, seeded groups at every stage):
  A  every one of the 4,598,126 allocations vs  256 groups   -> keep best 20,000
  B  20,000                                 vs  8,192 groups -> keep best 1,000
  C  1,000                                  vs 131,072 groups -> keep best 50
  D  hill-climb from those 50: move 1-3 soldiers between any two castles, all candidates
     scored on one fixed set of 131,072 groups, until the leader stops changing
  E  final top 10 re-scored on 262,144 brand-new groups (removes the winner's-curse bias
     of having selected them on the same sample), with paired standard errors vs the leader
"""
import sys
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd

from blotto.game1 import all_allocations
from blotto.game2 import evaluate, outcome_tables, sample_groups
from blotto.scenarios import data_populations

STAGES = [(256, 20_000), (8_192, 1_000), (131_072, 50)]
LOCAL_GROUPS = 131_072
FINAL_GROUPS = 262_144
REFERENCE = [(3, 11, 21, 28, 37), (4, 12, 22, 29, 33), (6, 16, 1, 36, 41), (7, 16, 1, 36, 40)]


def fmt(a):
    return "/".join(str(int(v)) for v in a)


def tables_for(pop, n_groups, seed):
    idx = sample_groups(len(pop), n_groups, seed)
    return outcome_tables(pop[idx])


def neighbours(alloc, steps=(1, 2, 3)):
    out = []
    for i in range(5):
        for j in range(5):
            if i == j:
                continue
            for s in steps:
                if alloc[i] >= s:
                    b = alloc.copy()
                    b[i] -= s
                    b[j] += s
                    out.append(b)
    return out


def search(args):
    name, pop, seed = args
    t0 = time.time()
    log = []
    all_allocs = all_allocations()
    cands = all_allocs
    for k, (n_groups, keep) in enumerate(STAGES):
        tab = tables_for(pop, n_groups, seed * 100 + k)
        scores = evaluate(cands, tab, chunk=max(16, 2_000_000 // n_groups))
        order = np.argsort(-scores, kind="stable")[:keep]
        if k == 0:
            stage_a_scores = scores
        log.append(f"{name} stage {'ABC'[k]}: {len(cands):,} x {n_groups:,} groups, "
                   f"best {scores[order[0]]:.3f} {fmt(cands[order[0]])}, cutoff {scores[order[-1]]:.3f}")
        cands = cands[order]
        del tab

    # D: hill-climb on one fixed sample (sample-average approximation)
    tab = tables_for(pop, LOCAL_GROUPS, seed * 100 + 7)
    pool = {fmt(a): a for a in cands}
    ranked = pd.DataFrame(columns=["allocation", "score"])
    leader = None
    for it in range(10):
        scored = set(ranked["allocation"])
        new_keys = [k for k in pool if k not in scored]
        if new_keys:
            new_scores = evaluate(np.array([pool[k] for k in new_keys]), tab, chunk=16)
            ranked = pd.concat([ranked, pd.DataFrame({"allocation": new_keys, "score": new_scores})])
        ranked = ranked.sort_values(["score", "allocation"], ascending=[False, True])
        top = ranked.head(10)["allocation"].tolist()
        log.append(f"{name} stage D iter {it}: pool {len(ranked):,}, leader {top[0]} {ranked.iloc[0]['score']:.3f}")
        if top[0] == leader:
            break
        leader = top[0]
        for key in top:
            for nb in neighbours(pool[key]):
                if fmt(nb) not in pool:
                    pool[fmt(nb)] = nb
    del tab

    # E: fresh sample for the final top 10 plus reference picks
    finalists = ranked.head(10)["allocation"].tolist()
    finalists += [fmt(h) for h in REFERENCE if fmt(h) not in finalists]
    allocs = np.array([[int(v) for v in f.split("/")] for f in finalists])
    tab = tables_for(pop, FINAL_GROUPS, seed * 100 + 9)
    per_group = evaluate(allocs, tab, chunk=8, per_group=True)
    means = per_group.mean(axis=1)
    lead = int(np.argmax(means))
    diff = per_group - per_group[lead]
    final = pd.DataFrame({
        "population": name, "allocation": finalists, "payout": means,
        "se": per_group.std(axis=1) / np.sqrt(FINAL_GROUPS),
        "gap_to_leader": diff.mean(axis=1),
        "gap_se": diff.std(axis=1) / np.sqrt(FINAL_GROUPS),
    }).sort_values("payout", ascending=False)
    leader_key = final.iloc[0]["allocation"]
    leader_idx = np.flatnonzero((all_allocs == np.array([int(v) for v in leader_key.split("/")])).all(axis=1))[0]
    stage_a_rank = int((stage_a_scores > stage_a_scores[leader_idx]).sum()) + 1
    log.append(f"{name}: final leader {leader_key} {final.iloc[0]['payout']:.3f}; its stage-A rank was "
               f"{stage_a_rank:,}; {time.time() - t0:.0f}s")
    return name, final, log


def main(names=None, redo=False):
    """Search every data population; populations with a saved result are skipped unless redo."""
    pops = data_populations()
    names = names or list(pops)
    jobs = [(n, pops[n], i + 1) for i, n in enumerate(pops)
            if n in names and (redo or not Path(f"results/game2_{n}.csv").exists())]
    with Pool(2) as pool:
        for name, final, log in pool.imap_unordered(search, jobs):
            print("\n".join(log), flush=True)
            final.to_csv(f"results/game2_{name}.csv", index=False)
    finals = [pd.read_csv(f"results/game2_{n}.csv") for n in pops if Path(f"results/game2_{n}.csv").exists()]
    pd.concat(finals).to_csv("results/game2_finalists_all.csv", index=False)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--redo"]
    main(args or None, redo="--redo" in sys.argv)
