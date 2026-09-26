"""A local page for rating pairs blind.

Shows the two first screens side by side, sides shuffled per pair, models
interleaved, and appends each decision to data/ratings.jsonl. Which side is the
reference condition stays on the server.

Each session also mixes back in up to --repeats pairs you rated in an earlier
session, sides swapped, one after every three new pairs, to measure how
consistent your picks are.
"""

import hashlib
import json
import random
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import zip_longest
from pathlib import Path

from .data import Pair, append_line, load_pairs, load_ratings


def _sides(pair_id: str, repeat: bool) -> dict:
    name_left = hashlib.sha256(f"human|{pair_id}".encode()).digest()[0] % 2 == 1
    if repeat:
        name_left = not name_left
    return {"left": "name" if name_left else "plain", "right": "plain" if name_left else "name"}


def queue(data: Path, repeats: int) -> list[dict]:
    pairs = load_pairs(data)
    by_model: dict[str, list[Pair]] = {}
    for p in pairs:
        by_model.setdefault(p.model, []).append(p)
    for model, ps in by_model.items():
        random.Random(model).shuffle(ps)
    ordered = [p for group in zip_longest(*by_model.values()) for p in group if p]  # any stretch covers every model
    ratings = load_ratings(data)
    items = [{"pair": p, "repeat": False} for p in ordered]
    done = [i for i in items if (i["pair"].id, False) in ratings]
    todo = [i for i in items if (i["pair"].id, False) not in ratings]

    # Repeats: pairs rated in earlier sessions and never repeated, balanced across models.
    pool: dict[str, list[Pair]] = {}
    for i in done:
        if (i["pair"].id, True) not in ratings and ratings[(i["pair"].id, False)]["winner"]:
            pool.setdefault(i["pair"].model, []).append(i["pair"])
    for ps in pool.values():
        random.Random("repeats").shuffle(ps)
    picked = [p for group in zip_longest(*pool.values()) for p in group if p][:repeats]
    already = sum(1 for (_, rep) in ratings if rep)
    picked = picked[: max(0, repeats - already)]
    reps = [{"pair": p, "repeat": True} for p in picked]
    done_reps = [{"pair": next(p for p in pairs if p.id == pid), "repeat": True} for (pid, rep) in ratings if rep]

    mixed = []
    for n, item in enumerate(todo):
        mixed.append(item)
        if n % 3 == 2 and reps:
            mixed.append(reps.pop(0))
    return done + done_reps + mixed + reps


class _Handler(BaseHTTPRequestHandler):
    data: Path
    items: list[dict]
    page: bytes

    def log_message(self, *args):
        pass

    def _send(self, body: bytes, ctype: str, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._send(self.page, "text/html; charset=utf-8")
        if self.path == "/api/state":
            ratings = load_ratings(self.data)
            state = {"items": [{"adjectives": i["pair"].adjectives} for i in self.items],
                     "rated": [n for n, i in enumerate(self.items) if (i["pair"].id, i["repeat"]) in ratings]}
            return self._send(json.dumps(state).encode(), "application/json")
        parts = self.path.strip("/").split("/")
        if len(parts) == 3 and parts[0] == "img" and parts[2] in ("left.jpg", "right.jpg"):
            try:
                item = self.items[int(parts[1])]
            except (ValueError, IndexError):
                return self._send(b"not found", "text/plain", 404)
            side = parts[2].removesuffix(".jpg")
            cond = _sides(item["pair"].id, item["repeat"])[side]
            return self._send(item["pair"].shot(cond).read_bytes(), "image/jpeg")
        self._send(b"not found", "text/plain", 404)

    def do_POST(self):
        if self.path != "/api/rate":
            return self._send(b"not found", "text/plain", 404)
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        item = self.items[int(body["item"])]
        record = {"pair": item["pair"].id, "repeat": item["repeat"], "t": time.time()}
        if body["choice"] == "undo":
            record["undo"] = True
        else:
            record["winner"] = _sides(item["pair"].id, item["repeat"])[body["choice"]] if body["choice"] in ("left", "right") else None
        append_line(self.data / "ratings.jsonl", record)
        self._send(b'{"ok": true}', "application/json")


def serve(data: Path, port: int, repeats: int) -> ThreadingHTTPServer:
    items = queue(data, repeats)
    handler = type("Handler", (_Handler,), {"data": data, "items": items,
                                            "page": Path(__file__).with_name("rate.html").read_bytes()})
    return ThreadingHTTPServer(("127.0.0.1", port), handler)
