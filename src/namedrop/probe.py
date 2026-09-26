"""Can a plain image model learn the rater's taste? A logistic-regression
classifier on CLIP embeddings of the two first screens, trained on the rater's
picks and scored leave-one-out, so every prediction is for a pair the classifier
never saw.

    uv sync --extra probe && uv run namedrop probe
"""

from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression

from .analyze import bootstrap, kappa
from .data import rated
from .rate import _sides


def main(data: Path) -> None:
    rows = rated(data)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k", device=device)
    with torch.no_grad():
        imgs = torch.stack([preprocess(Image.open(p.shot(c)).convert("RGB")) for p, _ in rows for c in ("plain", "name")]).to(device)
        emb = model.encode_image(imgs).float()
    emb = (emb / emb.norm(dim=-1, keepdim=True)).cpu().numpy()
    E = {(p.id, c): emb[2 * i + j] for i, (p, _) in enumerate(rows) for j, c in enumerate(("plain", "name"))}

    def xy(subset):
        X, y = [], []
        for p, winner in subset:
            s = _sides(p.id, False)
            d = E[(p.id, s["left"])] - E[(p.id, s["right"])]
            X += [d, -d]  # both orientations, so the classifier can't learn a side
            y += [int(winner == s["left"]), int(winner != s["left"])]
        return np.array(X), np.array(y)

    preds = []
    for i, (p, winner) in enumerate(rows):
        X, y = xy(rows[:i] + rows[i + 1:])
        clf = LogisticRegression(max_iter=2000).fit(X, y)
        s = _sides(p.id, False)
        left = clf.predict_proba([E[(p.id, s["left"])] - E[(p.id, s["right"])]])[0, 1] > 0.5
        preds.append((winner, s["left"] if left else s["right"]))
    lo, hi = bootstrap(preds)
    print(f"CLIP classifier trained on the rater's picks, leave-one-out over {len(rows)} pairs: "
          f"κ {kappa(preds):.2f} [{lo:.2f}, {hi:.2f}]")
