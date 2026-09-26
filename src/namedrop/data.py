"""Reading the data directory: pairs, human ratings, judge verdicts.

    data/pairs/<model>/<NN-adjectives>/   one generated pair (see generate.py)
    data/ratings.jsonl                     human picks, one line per decision
    data/judgments.jsonl                   AI judge verdicts, one line per call
"""

import json
from dataclasses import dataclass
from pathlib import Path

from .prompts import adjective_sets

NAMES = {"claude-opus-5-5": "Claude Opus 5.5", "openai-gpt-6-sol": "GPT-6 Sol", "moonshotai-kimi-k3": "Kimi K3"}


@dataclass(frozen=True)
class Pair:
    id: str  # "<model>/<NN-adjectives>"
    dir: Path
    adjectives: str
    reference: str

    @property
    def model(self) -> str:
        return self.id.split("/")[0]

    @property
    def model_name(self) -> str:
        return NAMES.get(self.model, self.model)

    def shot(self, condition: str) -> Path:
        return self.dir / f"{condition}.jpg"


def load_pairs(data: Path) -> list[Pair]:
    """Pairs with a reference and both first screens captured."""
    sets = adjective_sets()
    out = []
    for d in sorted(data.glob("pairs/*/*")):
        ref = d / "reference.json"
        if ref.exists() and (d / "plain.jpg").exists() and (d / "name.jpg").exists():
            out.append(Pair(f"{d.parent.name}/{d.name}", d, sets[int(d.name[:2])], json.loads(ref.read_text())["reference"]))
    return out


def read_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def append_line(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_ratings(data: Path) -> dict[tuple[str, bool], dict]:
    """Latest decision per (pair, is_repeat). An "undo" line removes the earlier one;
    a winner of None means "can't decide"."""
    ratings: dict[tuple[str, bool], dict] = {}
    for r in read_lines(data / "ratings.jsonl"):
        key = (r["pair"], r.get("repeat", False))
        if r.get("undo"):
            ratings.pop(key, None)
        else:
            ratings[key] = r
    return ratings


def rated(data: Path) -> list[tuple[Pair, str]]:
    """(pair, "plain" | "name") for every pair with a first (non-repeat) decision."""
    ratings = load_ratings(data)
    return [(p, ratings[(p.id, False)]["winner"]) for p in load_pairs(data)
            if ratings.get((p.id, False), {}).get("winner")]
