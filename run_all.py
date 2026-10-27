"""Run the whole pipeline in order; each stage only runs if the one before it succeeded.

    python run_all.py            # everything (~35 min on 2 cores; Game 2 searches dominate)
    python run_all.py --fast     # after one full run: skip the Game 2 searches, reuse results/game2_*.csv
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGES = [
    ("unit tests", [sys.executable, "-m", "pytest", "-q", "tests"], None),
    ("download data", [sys.executable, "-c", "from blotto.data import download_rounds; download_rounds()"], None),
    ("stage 2: replicate Riddler results", [sys.executable, "stage2_validate_riddler.py"], "stage2_validation.txt"),
    ("stage 4a: Game 1 exact solve", [sys.executable, "stage4_game1.py"], "stage4_game1.txt"),
    ("stage 5a: Game 1 robustness", [sys.executable, "stage5_game1_robust.py"], "stage5_game1_robust.txt"),
    ("stage 4b: Game 2 search", [sys.executable, "stage4_game2.py"], "stage4_game2.log"),
    ("stage 6: Game 2 robustness", [sys.executable, "stage6_game2_robust.py"], "stage6_game2_robust.txt"),
    ("stage 7: figure data", [sys.executable, "stage7_figure_data.py"], "stage7_figure_data.txt"),
]


def main():
    fast = "--fast" in sys.argv
    (ROOT / "results").mkdir(exist_ok=True)
    for name, cmd, log in STAGES:
        if fast and name.startswith("stage 4b"):
            print(f"-- skipping {name} (--fast)")
            continue
        print(f"-- {name}", flush=True)
        out = open(ROOT / "results" / log, "w") if log else None
        result = subprocess.run(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            sys.exit(f"stage failed: {name} (see results/{log})")
    print("done; outputs in results/")


if __name__ == "__main__":
    main()
