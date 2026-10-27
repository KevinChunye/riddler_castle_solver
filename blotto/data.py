"""Stage 1: load and validate the FiveThirtyEight Riddler castle submissions.

Each round is a CSV whose first columns are "Castle 1".."Castle K" (soldier counts).
Rounds 1-4 have 10 castles worth 1..10 points; round 5 has 13 castles worth 1..13.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data_raw"
REPO_URL = "https://raw.githubusercontent.com/fivethirtyeight/data/master/riddler-castles/"
ROUND_FILES = {
    1: "castle-solutions.csv",
    2: "castle-solutions-2.csv",
    3: "castle-solutions-3.csv",
    4: "castle-solutions-4.csv",
    5: "castle-solutions-5.csv",
}
TOTAL_SOLDIERS = 100


def download_rounds(data_dir=DATA_DIR):
    """Fetch any missing round CSVs from the fivethirtyeight/data repo."""
    import urllib.request

    data_dir.mkdir(parents=True, exist_ok=True)
    for name in ROUND_FILES.values():
        path = data_dir / name
        if not path.exists():
            urllib.request.urlretrieve(REPO_URL + name, path)


def load_round(round_id, data_dir=DATA_DIR):
    """Return (allocations, report) for one round.

    allocations: int array, shape (n_valid_entries, n_castles).
    report: dict with row counts, so every dropped row is accounted for.
    """
    df = pd.read_csv(data_dir / ROUND_FILES[round_id])
    castle_cols = [c for c in df.columns if c.startswith("Castle")]
    soldiers = df[castle_cols].apply(pd.to_numeric, errors="coerce")

    has_missing = soldiers.isna().any(axis=1)
    filled = soldiers.fillna(-1)
    is_negative = (filled < 0).any(axis=1) & ~has_missing
    not_integer = (filled % 1 != 0).any(axis=1)
    wrong_total = filled.sum(axis=1) != TOTAL_SOLDIERS
    valid = ~(has_missing | is_negative | not_integer | wrong_total)

    report = {
        "round": round_id,
        "n_castles": len(castle_cols),
        "rows": len(df),
        "valid": int(valid.sum()),
        "dropped_missing": int(has_missing.sum()),
        "dropped_negative": int(is_negative.sum()),
        "dropped_non_integer": int((not_integer & ~has_missing).sum()),
        "dropped_wrong_total": int((wrong_total & ~has_missing).sum()),
    }
    allocations = soldiers[valid].to_numpy(dtype=np.int64)
    return allocations, report


def load_all_rounds(data_dir=DATA_DIR):
    """Return {round_id: allocations} and a DataFrame of validation reports."""
    allocations, reports = {}, []
    for round_id in ROUND_FILES:
        allocations[round_id], report = load_round(round_id, data_dir)
        reports.append(report)
    return allocations, pd.DataFrame(reports)
