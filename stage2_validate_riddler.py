"""Stage 2 check: does the engine reproduce the published Riddler results?

Published reference points (2017):
  Round 2 winner, Vince Vatter, (0,1,2,16,21,3,2,1,32,22): 751 W / 175 L / 5 T among 931 entries
    (as reported in "Who Survives Riddler Nation?", R-bloggers, June 2017).
  Round 1, Peter Norvig's pytudes notebook (Riddler Battle Royale), top of 1,202 cleaned plans:
    (0,3,4,7,16,24,4,34,4,4) 85.6%, (5,7,9,11,15,21,25,2,2,3) 84.1%, (3,5,8,10,13,1,26,30,2,2) 83.3%.
The public CSVs hold 1,349 (round 1) and 902 (round 2) valid entries, so exact counts will
differ slightly from the published pools; the test is that the same entries land at the top.
"""
import numpy as np
import pandas as pd

from blotto.data import load_all_rounds
from blotto.engine import round_robin_summary

VALUES_10 = np.arange(1, 11)
PUBLISHED = {
    2: [((0, 1, 2, 16, 21, 3, 2, 1, 32, 22), "Vatter, published 751-175-5 of 931 (80.9% w/ ties half)")],
    1: [((0, 3, 4, 7, 16, 24, 4, 34, 4, 4), "Norvig #1, 85.6% of 1,202"),
        ((5, 7, 9, 11, 15, 21, 25, 2, 2, 3), "Norvig #2, 84.1%"),
        ((3, 5, 8, 10, 13, 1, 26, 30, 2, 2), "Norvig #3, 83.3%")],
}


def main():
    rounds, report = load_all_rounds()
    print(report.to_string(index=False))
    for round_id in (1, 2, 3, 4):
        allocs = rounds[round_id]
        s = round_robin_summary(allocs, VALUES_10)
        table = pd.DataFrame({"allocation": [tuple(a) for a in allocs], **s})
        table["rank"] = table["win_rate"].rank(ascending=False, method="min").astype(int)
        table = table.sort_values("win_rate", ascending=False)
        print(f"\n=== Round {round_id}: {len(allocs)} entries — top 5 by war win rate (ties = half)")
        print(table.head(5)[["rank", "allocation", "wins", "losses", "ties", "win_rate"]].to_string(index=False))
        for alloc, note in PUBLISHED.get(round_id, []):
            row = table[table["allocation"] == alloc]
            if row.empty:
                print(f"  {alloc}: NOT FOUND in public CSV  ({note})")
                continue
            r = row.iloc[0]
            print(f"  {alloc}: rank {r['rank']}, {r['wins']}-{r['losses']}-{r['ties']}, "
                  f"win rate {r['win_rate']:.1%}  <- {note}")
    return 0


if __name__ == "__main__":
    main()
