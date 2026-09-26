"""The AI-judge setups from the write-up's appendix.

Each judge sees the brief and the two first screens (in a stated order) and picks
one. Verdicts are appended to data/judgments.jsonl; calls already there are skipped.
"""

import asyncio
import hashlib
import io
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from . import prompts
from .data import Pair, append_line, load_pairs, rated, read_lines
from .llm import LLM
from .rate import _sides

SMALL = ["openai/gpt-6-luna", "google/gemini-3.5-flash-lite", "z-ai/glm-5.3-flash"]
FRONTIER = ["claude-opus-5-5", "openai/gpt-6-sol", "moonshotai/kimi-k3"]
EFFORT = "medium"
MAX_TOKENS = 8000


@dataclass(frozen=True)
class Setup:
    question: str | None = None  # None: the three-question prompt (fidelity, craft, "which would you ship?")
    framing: str = "client"      # "client", "neutral", or "critic"; see prompts.judge_intro
    quick: bool = False          # ask for a gut call: the pick only
    thumb: int | None = None     # shrink each first screen to this width, so body copy can't be read
    context: int = 0             # first show this many other pages from the batch
    examples: int = 0            # first show this many of the rater's own picks


SETUPS = {
    "three-questions": Setup(),
    "distinctive": Setup(prompts.DISTINCTIVE),
    "expressive": Setup(prompts.EXPRESSIVE),
    "rubric": Setup(prompts.RUBRIC, framing="critic"),
    "mood": Setup(prompts.MOOD, framing="neutral"),
    "glance": Setup(prompts.GLANCE, framing="neutral", quick=True),
    "mood-thumbnail": Setup(prompts.MOOD, framing="neutral", thumb=320),
    "expressive-context": Setup(prompts.EXPRESSIVE, context=6),
    "examples": Setup(examples=8),
}


def get_setup(name: str) -> Setup:
    """A named setup, or "as:<Name>" to judge through a named person's or publication's taste."""
    if name.startswith("as:"):
        return Setup(prompts.PERSONA.format(name=name[3:]), framing="neutral")
    return SETUPS[name]


def keys(name: str) -> tuple[str, ...]:
    return ("fidelity", "craft", "overall") if get_setup(name).question is None and not get_setup(name).examples else ("pick",)


def default_order(pair_id: str, judge: str) -> str:
    return ("plain-first", "name-first")[hashlib.sha256(f"{pair_id}|{judge}".encode()).digest()[0] % 2]


def _shrink(jpeg: bytes, width: int) -> bytes:
    im = Image.open(io.BytesIO(jpeg))
    im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    out = io.BytesIO()
    im.save(out, "JPEG", quality=85)
    return out.getvalue()


def parse_verdict(text: str, fields: tuple[str, ...]) -> dict | None:
    candidates = [prompts.extract_tag(text, "verdict")] + re.findall(r"\{[^{}]*\"%s\"[^{}]*\}" % fields[0], text)[::-1]
    for c in candidates:
        m = re.search(r"\{.*\}", c or "", re.S)
        try:
            v = json.loads(m.group(0)) if m else None
        except json.JSONDecodeError:
            continue
        if v and all(v.get(f) in (1, 2, "1", "2") for f in fields):
            return v
    return None


def _parts(name: str, pair: Pair, order: str, all_pairs: list[Pair], labelled: list[tuple[Pair, str]]) -> list:
    s = get_setup(name)
    first, second = ("plain", "name") if order == "plain-first" else ("name", "plain")
    shot = lambda p, c: _shrink(p.shot(c).read_bytes(), s.thumb) if s.thumb else p.shot(c).read_bytes()
    parts: list = []
    if s.examples:
        pool = [(p, w) for p, w in labelled if p.adjectives != pair.adjectives]  # never the pair itself or its brief
        parts.append(prompts.examples_intro(s.examples))
        for i, (ex, winner) in enumerate(random.Random("examples|" + pair.id).sample(pool, min(s.examples, len(pool))), 1):
            sides = _sides(ex.id, False)  # as the rater saw it
            before, middle = prompts.example(i, ex.adjectives)
            parts += [before, shot(ex, sides["left"]), middle, shot(ex, sides["right"]),
                      prompts.example_label(1 if winner == sides["left"] else 2)]
        before, middle, outro = prompts.examples_target(pair.adjectives)
        return parts + [before, shot(pair, first), middle, shot(pair, second), outro]
    if s.context:
        others = [(p, c) for p in all_pairs if p.adjectives != pair.adjectives for c in ("plain", "name")]
        parts.append(prompts.context_preamble(s.context))
        parts += [shot(p, c) for p, c in random.Random(pair.id).sample(others, s.context)]
    outro = prompts.three_questions(pair.adjectives) if s.question is None else prompts.one_question(s.question, pair.adjectives, s.quick)
    return parts + [prompts.judge_intro(pair.adjectives, s.framing, thumbs=s.thumb is not None),
                    shot(pair, first), "Page 2:", shot(pair, second), outro]


async def run(data: Path, setups: list[str], judges: list[str], both_orders: bool) -> None:
    llm = LLM()
    all_pairs = load_pairs(data)
    labelled = rated(data)
    targets = [p for p, _ in labelled]  # agreement is measured on rated pairs
    path = data / "judgments.jsonl"
    done = {(r["setup"], r["judge"], r["pair"], r["order"]) for r in read_lines(path)}
    jobs = []
    for name in setups:
        for judge in judges:
            for p in targets:
                orders = ["plain-first", "name-first"] if both_orders else [default_order(p.id, judge)]
                for order in orders:
                    if (name, judge, p.id, order) not in done:
                        jobs.append((name, judge, p, order))

    async def one(name, judge, p, order):
        async with llm.images:
            parts = _parts(name, p, order, all_pairs, labelled)
            try:
                r = await llm.call(judge, parts, effort=EFFORT, max_tokens=MAX_TOKENS)
            except Exception as e:  # noqa: BLE001 -- report and keep going; a rerun retries it
                print(f"  ✗ {name} / {judge} / {p.id}: {e!r}")
                return
        v = parse_verdict(r["text"], keys(name))
        if not v:
            print(f"  ✗ {name} / {judge} / {p.id}: no verdict (stop_reason={r['stop_reason']})")
            return
        first, second = ("plain", "name") if order == "plain-first" else ("name", "plain")
        append_line(path, {"setup": name, "judge": judge, "pair": p.id, "order": order,
                           "winner": {k: first if int(v[k]) == 1 else second for k in keys(name)},
                           "reason": v.get("reason", ""), "cost": r["cost"]})

    print(f"judging: {len(jobs)} calls ({len(setups)} setups × {len(judges)} judges × {len(targets)} rated pairs"
          f"{' × 2 orders' if both_orders else ''}, minus finished ones)")
    await asyncio.gather(*(one(*j) for j in jobs))
