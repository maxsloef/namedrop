"""Every prompt Namedrop sends, verbatim.

Generation has two conditions that share one builder prompt:

    plain: adjectives -> build
    name:  adjectives -> the model picks a reference -> adjectives + "Build it in the style of <reference>." -> build
"""

import re
from pathlib import Path

PRODUCT = """\
Acme is a B2B SaaS company that sells widgets. Its platform lets operations \
teams at mid-sized manufacturers configure, order, and track custom widgets \
across their supply chain. The landing page's job is to get visitors to book \
a demo."""


def adjective_sets() -> list[str]:
    path = Path(__file__).with_name("adjectives.txt")
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


# ── generation ────────────────────────────────────────────────────────────────


def plan(adjectives: str) -> str:
    """The model picks its own reference for the adjectives."""
    return f"""\
You're the art director for a landing page.

<product>
{PRODUCT}
</product>

The client describes the feel they want in three words: {adjectives}.

Choose one visual reference for the page: a specific artist, designer, studio, \
or art or design movement whose style best captures that feel. The builder \
will get the client's three words plus your reference.

Reply with just the name inside <reference></reference> tags: no description, \
qualifiers, or explanation."""


def build(adjectives: str, reference: str | None = None) -> str:
    """The builder prompt; the name condition adds one sentence."""
    style = f" Build it in the style of {reference}." if reference else ""
    return f"""\
Build a landing page.

<product>
{PRODUCT}
</product>

The client wants the page to feel {adjectives}.{style}

Deliver a single, complete, self-contained HTML file with inline CSS and JS. \
Write real copy, not placeholder text. You may load fonts from Google Fonts; \
don't load anything else from the network, so any imagery must be made with \
CSS, inline SVG, or canvas. The page will be viewed at a 1440px-wide desktop \
viewport. Reply with only the HTML, in one ```html code block."""


# ── AI judges (the write-up's appendix) ───────────────────────────────────────

SCREENS_FIRST = """\
Below is the first screen of Page 1 and of Page 2, at a 1440px-wide desktop \
viewport: what a visitor sees before scrolling."""

SCREENS_THUMB = """\
Below is a small thumbnail of the first screen of Page 1 and of Page 2: what a \
visitor sees before scrolling."""


def judge_intro(adjectives: str, framing: str, thumbs: bool = False) -> str:
    """Text before Page 1. Framings: "client" (the product and its demo goal),
    "neutral" (product and brief only), "critic" (a design-annual judge)."""
    screens = SCREENS_THUMB if thumbs else SCREENS_FIRST
    if framing == "neutral":
        return f"""\
You're comparing two landing page designs for Acme, whose platform helps \
manufacturers configure, order, and track custom widgets. Both were made from \
the same brief: the page should feel {adjectives}.

{screens}

Page 1:"""
    if framing == "critic":
        return f"""\
You're a design director choosing work for a design annual. Two designers got \
the same brief: a landing page for Acme, whose platform helps manufacturers \
configure, order, and track custom widgets. The page should feel: {adjectives}.

{screens}

Page 1:"""
    return f"""\
You're judging two landing page designs for the same product, made from the \
same client request.

<product>
{PRODUCT}
</product>

The client asked for a page that feels: {adjectives}.

{screens}

Page 1:"""


def three_questions(adjectives: str) -> str:
    """Fidelity, craft, and "which would you ship?" in one call."""
    return f"""\
Compare the two pages on three questions:

1. fidelity: Which page better embodies the feel the client asked for ({adjectives})?
2. craft: Which page is better designed and executed? Consider typography, \
layout, color, hierarchy, polish, and originality.
3. overall: If you were the client, which page would you ship?

Think it through, then reply with JSON inside <verdict></verdict> tags: \
{{"fidelity": 1 or 2, "craft": 1 or 2, "overall": 1 or 2, "reason": "one or two sentences"}}"""


def one_question(question: str, adjectives: str, quick: bool = False) -> str:
    """A single question. Quick asks for a gut call with no explanation."""
    if quick:
        reply = 'Reply right away with JSON inside <verdict></verdict> tags: {"pick": 1 or 2}'
    else:
        reply = ('Think it through, then reply with JSON inside <verdict></verdict> tags: '
                 '{"pick": 1 or 2, "reason": "one or two sentences"}')
    return f"{question.format(adjectives=adjectives)}\n\n{reply}"


DISTINCTIVE = "Which page is more distinctive and memorable?"

EXPRESSIVE = """\
Which page expresses the feel the client asked for ({adjectives}) more strongly?

Think of it as a vector. Direction: do the typography, color, imagery, and \
composition point toward those qualities? Magnitude: how fully and boldly does \
the design commit to them? A page that intensely embodies the feel beats one \
that is merely consistent with it or only gestures at it. Judge the visual \
design, not what the copy says about itself."""

RUBRIC = """\
Judge them as design, not as marketing:

- A central visual idea. Does the page have an idea of its own (an image, a \
typographic gesture, a world), or is it assembled from stock landing-page parts?
- Commitment to the feel. Do typography, color, imagery, and copy all serve \
"{adjectives}"?
- Stock SaaS parts are defaults, not decisions: product-dashboard cards floating \
in the hero, stat rows, logo strips, pill badges, grids of feature cards. A page \
built mostly from them is generic, however polished.
- Small slips (an awkward line break, one busy section) matter less than the \
strength of the idea.

Which page is the better piece of design?"""

MOOD = """\
Which page's visual design more strongly conveys the feel ({adjectives})? Judge \
only the mood the design creates: typography, color, imagery, composition. The \
imagery doesn't need to depict the product; an artistic or decorative image that \
carries the feel counts fully. Ignore calls to action, product clarity, trust \
signals, and fit for a B2B audience."""

GLANCE = """\
At a glance, which page's look conveys the feel ({adjectives}) more strongly? Go \
with your first impression of the whole screen (color, scale, contrast, imagery) \
rather than reading the text."""

PERSONA = """\
Which page would {name} prefer? Judge it the way {name} would, from their own \
taste and sensibility, not from marketing concerns."""


def context_preamble(n: int) -> str:
    return f"""\
Before the two pages you'll judge: here are the first screens of {n} other \
landing pages from the same batch, made for the same product from other briefs, \
so you know what's typical here."""


def examples_intro(k: int) -> str:
    return f"""\
You'll learn one reviewer's taste from examples, then predict their pick.

The reviewer compares two landing pages for the same product: Acme, a platform \
for configuring, ordering, and tracking custom widgets. Each pair was made from \
a brief of three words describing how the page should feel. The reviewer looks \
at the first screen of each page and picks the one they prefer.

Here are {k} of their past picks."""


def example(i: int, adjectives: str) -> tuple[str, str]:
    return f"Example {i}. Brief: {adjectives}.\nPage 1:", "Page 2:"


def example_label(picked: int) -> str:
    return f"The reviewer picked Page {picked}."


def examples_target(adjectives: str) -> tuple[str, str, str]:
    outro = """\
Which page would the reviewer pick? Think about what their past picks have in \
common, then reply with JSON inside <verdict></verdict> tags: \
{"pick": 1 or 2, "reason": "one or two sentences"}"""
    return f"Now a new pair. Brief: {adjectives}.\nPage 1:", "Page 2:", outro


# ── parsing ───────────────────────────────────────────────────────────────────


def extract_tag(text: str, tag: str) -> str | None:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.S) or re.search(rf"<{tag}>(.*)", text, re.S)
    return m.group(1).strip() or None if m else None


def extract_reference(text: str) -> str | None:
    out = extract_tag(text, "reference")
    bare = text.strip()
    if out is None and bare and "\n" not in bare and len(bare) <= 80:  # some models drop the tags
        out = bare.strip("*_`\"' .")
    return out


def extract_html(text: str) -> str | None:
    m = re.search(r"```html\s*\n(.*?)```", text, re.S)
    if m:
        return m.group(1).strip()
    m = re.search(r"(<!DOCTYPE html.*</html>|<html.*</html>)", text, re.S | re.I)
    return m.group(1).strip() if m else None
