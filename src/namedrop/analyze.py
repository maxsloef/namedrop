"""Every number in the write-up, from the data directory.

Human results: how often the rater preferred the page built with the model's own
reference (Wilson 95% intervals, exact two-sided binomial test against 50%), and
how consistent the rater was on sides-swapped repeats.

AI judges: for each setup and panel, the majority vote over the panel's judges
(and both page orders, where run) against the rater's picks, as Cohen's kappa
with a 95% bootstrap interval over pairs. Ties are broken by a coin flip seeded
by the pair.
"""

import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

from .data import NAMES, load_ratings, rated, read_lines
from .judge import FRONTIER, SMALL

PANELS = {"small panel": SMALL, "frontier panel": FRONTIER, "all six": FRONTIER + SMALL}
CRITERION_LABEL = {"fidelity": "fidelity", "craft": "craft", "overall": '"which would you ship?"', "pick": ""}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def binomial_p(k: int, n: int) -> float:
    tail = sum(math.comb(n, i) for i in range(max(k, n - k), n + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def kappa(pairs: list[tuple[str, str]]) -> float:
    """Cohen's kappa for two raters choosing "name" or "plain"."""
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    pa = sum(a == "name" for a, _ in pairs) / n
    pb = sum(b == "name" for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 0.0


def bootstrap(pairs: list[tuple[str, str]], reps: int = 4000) -> tuple[float, float]:
    rng = random.Random(0)
    ks = sorted(kappa([rng.choice(pairs) for _ in pairs]) for _ in range(reps))
    return ks[int(0.025 * reps)], ks[int(0.975 * reps) - 1]


def vote(picks: list[str], seed: str) -> str:
    n = sum(p == "name" for p in picks)
    if 2 * n != len(picks):
        return "name" if 2 * n > len(picks) else "plain"
    return random.Random(seed).choice(["name", "plain"])


def human(data: Path) -> None:
    rows = rated(data)
    if not rows:
        print("no ratings yet")
        return
    print(f"HUMAN RATINGS: how often the page with the model's own reference was preferred\n")
    by = defaultdict(list)
    for p, w in rows:
        by[p.model].append(w)
    for model, ws in by.items():
        k, n = ws.count("name"), len(ws)
        lo, hi = wilson(k, n)
        print(f"  {NAMES.get(model, model):18} {k:3}/{n:<3} {k / n:5.0%}   95% CI {lo:4.0%}–{hi:4.0%}")
    k, n = sum(w == "name" for _, w in rows), len(rows)
    lo, hi = wilson(k, n)
    print(f"  {'All models':18} {k:3}/{n:<3} {k / n:5.0%}   95% CI {lo:4.0%}–{hi:4.0%}   p = {binomial_p(k, n):.2g} (two-sided binomial vs 50%)")

    ratings = load_ratings(data)
    reps = [(ratings[(pid, False)]["winner"], r["winner"]) for (pid, rep), r in ratings.items()
            if rep and r.get("winner") and ratings.get((pid, False), {}).get("winner")]
    skipped = sum(1 for (_, rep), r in ratings.items() if not rep and r.get("winner") is None)
    times = sorted(r["t"] for r in ratings.values())
    gaps = [b - a for a, b in zip(times, times[1:]) if b - a < 300]
    print(f"\n  repeats (sides swapped): same pick {sum(a == b for a, b in reps)}/{len(reps)}"
          f" · can't decide: {skipped} · median {statistics.median(gaps):.1f} s per decision" if gaps else "")


def judges(data: Path) -> None:
    rows = rated(data)
    verdicts = read_lines(data / "judgments.jsonl")
    if not rows or not verdicts:
        return
    you = {p.id: w for p, w in rows}
    by = defaultdict(lambda: defaultdict(list))  # (setup, criterion) -> pair -> [(judge, order, winner)]
    for v in verdicts:
        if v["pair"] in you:
            for criterion, winner in v["winner"].items():
                by[(v["setup"], criterion)][v["pair"]].append((v["judge"], v["order"], winner))
    meta = json.loads((data / "meta.json").read_text()) if (data / "meta.json").exists() else {}
    cut = meta.get("first_batch_until")
    ratings = load_ratings(data)
    first = {pid for pid in you if cut and ratings[(pid, False)]["t"] <= cut}

    print(f"\nAI JUDGES: agreement with the rater (Cohen's κ; 0 = chance, 1 = perfect) on {len(you)} rated pairs\n")
    head = f"  {'setup':38} {'panel':15} {'orders':>6} {'κ':>6}  {'95% CI':>15}"
    if first:
        head += f" {'first ' + str(len(first)):>9} {'rest ' + str(len(you) - len(first)):>8}"
    print(head)
    for (setup, criterion), per_pair in sorted(by.items()):
        for panel, members in PANELS.items():
            picks = {pid: [w for j, _, w in vs if j in members] for pid, vs in per_pair.items()}
            covered = [pid for pid in you if all(any(j == m for j, _, _ in per_pair.get(pid, [])) for m in members)]
            if len(covered) < len(you):
                continue
            orders = max(len({o for j, o, _ in per_pair[pid] if j == members[0]}) for pid in covered)
            pairs = [(you[pid], vote(picks[pid], pid)) for pid in covered]
            lo, hi = bootstrap(pairs)
            label = setup + (f" · {CRITERION_LABEL[criterion]}" if CRITERION_LABEL[criterion] else "")
            line = f"  {label:38} {panel:15} {orders:6} {kappa(pairs):6.2f}  [{lo:5.2f}, {hi:5.2f}]"
            if first:
                a = [(you[pid], vote(picks[pid], pid)) for pid in covered if pid in first]
                b = [(you[pid], vote(picks[pid], pid)) for pid in covered if pid not in first]
                line += f" {kappa(a):9.2f} {kappa(b):8.2f}"
            print(line)

    # Self-consistency: the same judge, the same pair, the pages in swapped order
    print("\nJUDGE SELF-CONSISTENCY: same pick when the page order is swapped (setups run in both orders)\n")
    for (setup, criterion), per_pair in sorted(by.items()):
        for panel, members in PANELS.items():
            if panel == "all six":
                continue
            same = []
            for vs in per_pair.values():
                for judge in members:
                    picks = {o: w for j, o, w in vs if j == judge}
                    if len(picks) == 2:
                        same.append(picks["plain-first"] == picks["name-first"])
            if same:
                print(f"  {setup:38} {panel:15} {sum(same) / len(same):4.0%}  ({len(same)} judge-pair comparisons)")

    # What each judge setup would have concluded, per model
    print("\nWHAT THE JUDGES WOULD HAVE CONCLUDED: how often the reference page wins, per model\n")
    compare = [("three-questions", "overall", "frontier panel", '"ship?", frontier'), ("mood", "pick", "small panel", "mood, small")]
    models = sorted({p.model for p, _ in rows})
    print(f"  {'model':18} {'rater':>6}" + "".join(f" {lab:>18}" for *_, lab in compare))
    for m in models:
        ids = [p.id for p, _ in rows if p.model == m]
        line = f"  {NAMES.get(m, m):18} {sum(you[i] == 'name' for i in ids) / len(ids):6.0%}"
        for setup, criterion, panel, _ in compare:
            per_pair = by.get((setup, criterion), {})
            votes = {i: [w for j, _, w in per_pair.get(i, []) if j in PANELS[panel]] for i in ids}
            got = [vote(v, i) for i, v in votes.items() if v]
            line += f" {sum(g == 'name' for g in got) / len(got):18.0%}" if len(got) == len(ids) else f" {'–':>18}"
        print(line)


def main(data: Path) -> None:
    human(data)
    judges(data)
