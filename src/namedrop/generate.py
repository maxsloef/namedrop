"""Build every brief twice per model: from the adjectives alone ("plain"), and after
the model picks its own reference ("name").

Output, per model and adjective set, under data/pairs/<model>/<NN-adjectives>/:
    reference.json      the reference the model picked, and its raw reply
    plain.html / name.html
    plain.json / name.json    token counts and cost of the build call

Finished files are skipped, so an interrupted run resumes where it stopped.
"""

import asyncio
import json
import re
import traceback
from pathlib import Path

from . import prompts
from .llm import LLM

MODELS = ["claude-opus-5-5", "openai/gpt-6-sol", "moonshotai/kimi-k3"]
PLAN_EFFORT = "high"
BUILD_EFFORT = "medium"
MAX_TOKENS = 64000


def model_dir(model: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "-", model.lower())


def pair_dir(index: int, adjectives: str) -> str:
    return f"{index:02d}-" + re.sub(r"[^a-z]+", "-", adjectives.lower()).strip("-")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


class Failed(Exception):
    pass


async def _one(llm: LLM, model: str, adjectives: str, d: Path) -> None:
    ref_path = d / "reference.json"
    if not ref_path.exists():
        r = await llm.call(model, prompts.plan(adjectives), effort=PLAN_EFFORT, max_tokens=MAX_TOKENS)
        reference = prompts.extract_reference(r["text"])
        if not reference:
            raise Failed(f"no reference in reply (stop_reason={r['stop_reason']})")
        write_json(ref_path, {"reference": reference, "reply": r["text"], "model": r["model"], "cost": r["cost"]})
        print(f"  {d.parent.name}/{d.name}: picked {reference!r}")
    reference = json.loads(ref_path.read_text())["reference"]

    for cond, ref in (("plain", None), ("name", reference)):
        if (d / f"{cond}.html").exists():
            continue
        for _ in range(3):  # providers occasionally drop a long generation midway
            r = await llm.call(model, prompts.build(adjectives, ref), effort=BUILD_EFFORT, max_tokens=MAX_TOKENS)
            if r["stop_reason"] != "error":
                break
        html = prompts.extract_html(r["text"])
        if html is None or r["stop_reason"] != "end_turn":
            raise Failed(f"{cond} build: stop_reason={r['stop_reason']}, html={'yes' if html else 'no'}")
        (d / f"{cond}.html").write_text(html)
        write_json(d / f"{cond}.json", {k: r[k] for k in ("model", "stop_reason", "input_tokens", "output_tokens", "cost")})
        print(f"  {d.parent.name}/{d.name}: built {cond} ({len(html) // 1000}k chars, ${r['cost']:.2f})")


async def generate(data: Path, models: list[str], sets: int) -> None:
    llm = LLM()
    jobs = []
    for model in models:
        for i, adjectives in enumerate(prompts.adjective_sets()[:sets]):
            jobs.append((model, adjectives, data / "pairs" / model_dir(model) / pair_dir(i, adjectives)))

    async def run(model, adjectives, d):
        try:
            await _one(llm, model, adjectives, d)
        except Exception as e:  # noqa: BLE001 -- report and keep going; a rerun retries it
            print(f"  ✗ {d.parent.name}/{d.name}: {e!r}")
            if not isinstance(e, Failed):
                traceback.print_exc()

    print(f"generating {len(jobs)} pairs ({', '.join(models)}); finished files are skipped")
    await asyncio.gather(*(run(*j) for j in jobs))
