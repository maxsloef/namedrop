# Namedrop

**Do models design better when they name their own visual reference?**

When you want a model to design something with a particular feel, naming a reference ("in the style of Charles and Ray Eames") works far better than adjectives ("warm, precise, unhurried"). But the model already knows those references. If making it name one first still improves its work, the model has taste it isn't using by default: an *elicitation gap*.

Namedrop measures that gap. The same model builds the same brief twice, once from three adjectives alone and once after picking its own reference for those adjectives, and a person compares the two blind.

**[Write-up](https://claude.ai/artifact/L5zgmiVkxZcoUvwKMjctjd)** · **[All 90 pairs](https://claude.ai/artifact/SjwNiWSySE1tHgktQBnqGs)**

## Results

| Model | Reference page preferred | 95% CI |
|---|---|---|
| Claude Opus 5.5 | 63% (19 of 30) | 46–78% |
| GPT-6 Sol | 73% (22 of 30) | 56–86% |
| Kimi K3 | 67% (20 of 30) | 49–81% |
| **All models** | **68% (61 of 90)** | **58–77%**, p < 0.001 |

One rater, fully blind to condition and model, rating the first screen of each page. On 10 pairs shown again later with the sides swapped, the rater made the same pick all 10 times.

We also tried to replace the rater with AI judges and couldn't: across 20 setups, the best agreement with the rater was κ ≈ 0.4, and the standard "which would you ship?" judge was barely above chance. See the write-up's appendix.

## Reproduce the numbers

No API keys needed; everything from the study is in [`data/`](data).

```sh
uv sync
uv run namedrop analyze
```

This prints every figure in the write-up: the win rates above, the rater's consistency, each AI-judge setup's agreement with the rater (overall and per batch of ratings), the judges' own consistency, and what each judge would have concluded per model. `uv sync --extra probe && uv run namedrop probe` reproduces the CLIP-classifier result.

## Run it yourself

```sh
uv sync
uv run playwright install chromium
cp .env.example .env       # add ANTHROPIC_API_KEY and OPENROUTER_API_KEY

uv run namedrop --data mydata generate    # 30 briefs × 2 conditions × 3 models = 180 pages
uv run namedrop --data mydata render      # the first screen of every page
uv run namedrop --data mydata rate        # rate the pairs blind in your browser
uv run namedrop --data mydata judge --setup mood --panel small --both-orders
uv run namedrop --data mydata analyze
```

Every step skips finished work, so an interrupted run picks up where it stopped. `generate --models` takes Anthropic model ids (`claude-…`) or any OpenRouter model id, so you can benchmark other models the same way.

**What it cost in the study:**

| Step | Cost |
|---|---|
| `generate`, all three models (180 pages) | ~$42: Opus 5.5 ~$27, GPT-6 Sol ~$7.50, Kimi K3 ~$7 |
| `judge`, small panel, one setup in one page order | ~$0.10–0.35 |
| `judge`, frontier panel, one setup in one page order | ~$3 |
| every judge setup in the appendix | ~$18 |

## How it works

**Generation** ([`prompts.py`](src/namedrop/prompts.py), [`generate.py`](src/namedrop/generate.py)). For each of 30 adjective sets, the model first picks a reference in a separate call (reasoning effort high), replying with a single name: an artist, designer, studio, or movement. It then builds the page twice from one builder prompt (reasoning effort medium). The name condition adds one sentence, `Build it in the style of <reference>.` Each page is a single self-contained HTML file; Google Fonts are allowed, and imagery must be CSS, inline SVG, or canvas.

**Capture** ([`render.py`](src/namedrop/render.py)). Each page is rendered in headless Chromium at 1440 × 900 and its first screen saved as a 960 × 600 JPEG.

**Human rating** ([`rate.py`](src/namedrop/rate.py)). A local page shows the two first screens side by side with the three adjectives and the question "Which page would you ship?". Left and right are shuffled per pair, models are interleaved, and nothing reveals the condition. Keys: <kbd>←</kbd> / <kbd>→</kbd> to pick, <kbd>↓</kbd> for can't decide, <kbd>U</kbd> to undo. Each new session mixes up to `--repeats` already-rated pairs back in with the sides swapped.

**AI judges** ([`judge.py`](src/namedrop/judge.py)). A judge sees the brief and the two first screens and picks one. Agreement is measured on the pairs you've rated. The setups from the write-up's appendix:

| `--setup` | In the write-up |
|---|---|
| `three-questions` | "Which better embodies the feel?", "Which is better crafted?", and "Which would you ship?", asked in one call |
| `distinctive` | "Which is more distinctive?" |
| `expressive` | Expressivity (direction × magnitude) |
| `rubric` | A rubric of the rater's criteria |
| `mood` | Visual mood only, no sales framing |
| `glance` | Gut call "at a glance" |
| `mood-thumbnail` | Mood, 320 px thumbnails (copy unreadable) |
| `expressive-context` | Expressivity, after 6 typical pages |
| `examples` | 8 of the rater's own picks as examples |
| `as:<Name>` | "Judge as <Name> would" |

Panels: `small` is GPT-6 Luna, Gemini 3.5 Flash-Lite, and GLM-5.3 Flash; `frontier` is Claude Opus 5.5, GPT-6 Sol, and Kimi K3; `all` is both. To regenerate every verdict in the appendix:

```sh
uv run namedrop judge --setup three-questions,distinctive --panel all
uv run namedrop judge --setup expressive,rubric,expressive-context,examples --panel small
uv run namedrop judge --setup mood --panel all --both-orders
uv run namedrop judge --setup glance,mood-thumbnail --panel small --both-orders
uv run namedrop judge --setup "as:Paula Scher,as:Stefan Sagmeister,as:the editors of It's Nice That,as:Tibor Kalman" --panel small --both-orders
```

**Analysis** ([`analyze.py`](src/namedrop/analyze.py)). Win rates use 95% Wilson intervals and an exact two-sided binomial test against 50%. Judge agreement is Cohen's κ between the rater's picks and the panel's majority vote (pooled over both page orders where run), with a 95% bootstrap interval over pairs. Ties are broken by a coin flip seeded by the pair, which can move the both-orders rows by about ±0.05.

## Data

| Path | Contents |
|---|---|
| `data/pairs/<model>/<NN-adjectives>/` | `plain.html` and `name.html`; their first screens `plain.jpg` and `name.jpg`; `reference.json` (the reference the model picked, and its raw reply); `plain.json` and `name.json` (tokens and cost of each build) |
| `data/ratings.jsonl` | One line per decision: `pair`, `repeat`, `winner` (`"name"`, `"plain"`, or `null` for can't decide), `t`. The latest line per pair and repeat wins; a line with `"undo": true` removes the one before it. |
| `data/judgments.jsonl` | One line per judge call: `setup`, `judge`, `pair`, `order`, `winner`, `reason`, `cost` |
| `data/meta.json` | When the first batch of ratings ended; the appendix reports agreement for each batch |

## Notes

- Model outputs aren't deterministic, so a rerun produces different pages and verdicts.
- In the original run, GPT-6 Sol's pages (and some of its judge calls) went through OpenAI's API directly. This repo sends every non-Claude model through OpenRouter.
- The ratings are one person's: the author, who was blind to condition and model but knew the hypothesis. More raters is the obvious next step, and `rate` makes it easy to add your own.

## License

MIT, covering the code and the data. See [LICENSE](LICENSE).
