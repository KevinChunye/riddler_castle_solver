"""Opponent scenarios shared by every stage.

data scenarios      the 11 Riddler-derived populations from blotto.transfer + "pooled"
                    (all 9 un-snapped data populations concatenated)
level-k scenarios   pooled data mixed with a share w of "sophisticated" players who play the
                    best responses to the previous level (see mix_in)
"""
import numpy as np

from blotto.data import load_all_rounds
from blotto.transfer import build_populations


def data_populations():
    rounds, _ = load_all_rounds()
    pops = build_populations(rounds)
    base = [k for k in pops if not k.endswith("snap5")]
    pops["pooled"] = np.concatenate([pops[k] for k in base])
    return pops


def mix_in(base, responders, share):
    """Append equal numbers of copies of each responder so they make up `share` of the crowd."""
    copies = int(round(len(base) * share / (1 - share) / len(responders)))
    return np.concatenate([base, np.repeat(np.asarray(responders), copies, axis=0)])


def level_k_scenarios(base, best_response_fn, shares=(0.1, 0.25), levels=(1, 2), n_responders=10):
    """{name: population}. best_response_fn(pop) -> array of the top allocations against pop."""
    out = {}
    for share in shares:
        responders = best_response_fn(base)[:n_responders]
        for level in levels:
            pop = mix_in(base, responders, share)
            out[f"level{level}-{int(share * 100)}pct"] = pop
            responders = best_response_fn(pop)[:n_responders]
    return out
