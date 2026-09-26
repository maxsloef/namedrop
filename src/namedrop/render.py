"""Capture the first screen of every page: headless Chromium, a 1440 × 900
viewport, saved as a 960 × 600 JPEG next to the HTML (plain.jpg / name.jpg)."""

import asyncio
from pathlib import Path

from playwright.async_api import async_playwright

VIEWPORT = {"width": 1440, "height": 900}
SHOT_WIDTH = 960
TIMEOUT = 90  # seconds; a page whose script hangs the browser is captured again with JavaScript off

# Some pages scroll <body> or a wrapper instead of the window. Find what scrolls.
FIND_SCROLLER = """() => {
  const tall = el => el && el.scrollHeight > el.clientHeight + 10 && el.clientHeight >= innerHeight * 0.8;
  const scrolls = el => el === document.scrollingElement || /(auto|scroll)/.test(getComputedStyle(el).overflowY);
  const found = [document.scrollingElement, ...document.querySelectorAll('body, body *')].filter(el => tall(el) && scrolls(el));
  found.sort((a, b) => b.scrollHeight - a.scrollHeight);
  const el = found[0] || document.scrollingElement;
  el.setAttribute('data-namedrop-scroller', '');
  return el.scrollHeight;
}"""
SCROLLER = "document.querySelector('[data-namedrop-scroller]')"


async def _capture(browser, html: Path, out: Path, javascript: bool = True) -> None:
    page = await browser.new_page(viewport=VIEWPORT, device_scale_factor=SHOT_WIDTH / VIEWPORT["width"],
                                  java_script_enabled=javascript)
    try:
        try:
            await page.goto(html.resolve().as_uri(), wait_until="networkidle", timeout=20000)
        except Exception:  # noqa: BLE001 -- a slow font or a busy loop shouldn't sink the capture
            pass
        await page.wait_for_timeout(1500)
        height = await page.evaluate(FIND_SCROLLER)
        # Scroll through once so scroll-triggered animations settle, then return to the top.
        for y in range(0, min(height, 6 * VIEWPORT["height"]), VIEWPORT["height"] // 2):
            await page.evaluate(f"{SCROLLER}.scrollTop = {y}")
            await page.wait_for_timeout(100)
        await page.evaluate(f"{SCROLLER}.scrollTop = 0")
        await page.wait_for_timeout(600)
        await page.screenshot(path=out, type="jpeg", quality=85)
    finally:
        await page.close()


async def render(data: Path) -> None:
    todo = [(h, h.with_suffix(".jpg")) for h in sorted(data.glob("pairs/*/*/*.html")) if not h.with_suffix(".jpg").exists()]
    print(f"capturing {len(todo)} first screens")
    sem = asyncio.Semaphore(8)
    async with async_playwright() as p:
        browser = await p.chromium.launch()

        async def one(html: Path, out: Path):
            async with sem:
                try:
                    await asyncio.wait_for(_capture(browser, html, out), TIMEOUT)
                except TimeoutError:
                    print(f"  {html.parent.name}/{html.name}: page hung, capturing with JavaScript off")
                    await asyncio.wait_for(_capture(browser, html, out, javascript=False), TIMEOUT)

        await asyncio.gather(*(one(*t) for t in todo))
        await browser.close()
