"""namedrop: do models design better when they name their own visual reference?

    uv run namedrop analyze                   # reproduce the write-up's numbers from data/
    uv run namedrop generate                  # build every brief twice per model (API keys needed)
    uv run namedrop render                    # capture each page's first screen
    uv run namedrop rate                      # rate pairs blind in your browser
    uv run namedrop judge --setup mood --panel small --both-orders
"""

import argparse
import asyncio
import os
import subprocess
from pathlib import Path

from .generate import MODELS


def load_env() -> None:
    env = Path(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep and not key.strip().startswith("#") and value.strip():
                os.environ.setdefault(key.strip(), value.strip())


def main() -> None:
    load_env()
    ap = argparse.ArgumentParser(prog="namedrop", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"), help="data directory (default: data)")
    sub = ap.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="build every brief twice per model")
    g.add_argument("--models", default=",".join(MODELS), help="comma-separated (default: the three from the write-up)")
    g.add_argument("--sets", type=int, default=30, help="how many adjective sets to use (default: all 30)")

    sub.add_parser("render", help="capture each page's first screen")

    r = sub.add_parser("rate", help="rate pairs blind in your browser")
    r.add_argument("--port", type=int, default=8765)
    r.add_argument("--repeats", type=int, default=10, help="already-rated pairs to show again, sides swapped (default: 10)")

    j = sub.add_parser("judge", help="run an AI-judge setup on the rated pairs")
    j.add_argument("--setup", required=True, help='comma-separated setups, e.g. "mood", "three-questions", "as:Paula Scher"')
    j.add_argument("--panel", choices=["small", "frontier", "all"], default="small")
    j.add_argument("--judges", help="comma-separated judge models (overrides --panel)")
    j.add_argument("--both-orders", action="store_true", help="judge each pair in both page orders")

    sub.add_parser("analyze", help="reproduce the write-up's numbers")
    sub.add_parser("probe", help="CLIP classifier on the rater's picks (needs: uv sync --extra probe)")

    args = ap.parse_args()
    if args.command == "generate":
        from .generate import generate
        asyncio.run(generate(args.data, args.models.split(","), args.sets))
    elif args.command == "render":
        from .render import render
        asyncio.run(render(args.data))
    elif args.command == "rate":
        from .rate import serve
        server = serve(args.data, args.port, args.repeats)
        url = f"http://127.0.0.1:{args.port}/"
        print(f"rating at {url} (Ctrl-C to stop); picks go to {args.data / 'ratings.jsonl'}")
        subprocess.run(["open", url], check=False) if os.uname().sysname == "Darwin" else None
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    elif args.command == "judge":
        from .judge import FRONTIER, SETUPS, SMALL, run
        setups = args.setup.split(",")
        for s in setups:
            if s not in SETUPS and not s.startswith("as:"):
                ap.error(f"unknown setup {s!r}; choose from {', '.join(SETUPS)} or as:<Name>")
        judges = args.judges.split(",") if args.judges else {"small": SMALL, "frontier": FRONTIER, "all": FRONTIER + SMALL}[args.panel]
        asyncio.run(run(args.data, setups, judges, args.both_orders))
    elif args.command == "analyze":
        from .analyze import main as analyze
        analyze(args.data)
    elif args.command == "probe":
        from .probe import main as probe
        probe(args.data)
