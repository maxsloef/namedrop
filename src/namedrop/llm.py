"""Async model calls. Claude models go to the Anthropic API; every other model
goes through OpenRouter. A message is a list of parts: str for text, bytes for a
JPEG image."""

import asyncio
import base64
import os

import anthropic
from openai import AsyncOpenAI

# $ per million tokens (input, output) for Claude models, used only to report spend.
ANTHROPIC_PRICES = {"claude-opus-5-5": (4, 20)}

# Streams may pause while a model thinks; this is the longest silence tolerated.
READ_TIMEOUT = 300

STOP = {"stop": "end_turn", "length": "max_tokens", "content_filter": "refusal"}

Parts = list[str | bytes]


def _b64(jpeg: bytes) -> str:
    return base64.standard_b64encode(jpeg).decode()


class LLM:
    def __init__(self, concurrency: int = 200):
        self.sem = asyncio.Semaphore(concurrency)
        self.images = asyncio.Semaphore(64)  # judge calls carry images; cap how many are in memory
        self._anthropic = None
        self._openrouter = None

    @property
    def anthropic(self) -> anthropic.AsyncAnthropic:
        if self._anthropic is None:
            self._anthropic = anthropic.AsyncAnthropic(max_retries=8, timeout=1800)
        return self._anthropic

    @property
    def openrouter(self) -> AsyncOpenAI:
        if self._openrouter is None:
            key = os.environ.get("OPENROUTER_API_KEY")
            if not key:
                raise SystemExit("OPENROUTER_API_KEY is not set (needed for non-Claude models)")
            self._openrouter = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=key,
                                           max_retries=4, timeout=READ_TIMEOUT)
        return self._openrouter

    async def call(self, model: str, parts: Parts | str, *, effort: str, max_tokens: int) -> dict:
        parts = [parts] if isinstance(parts, str) else parts
        async with self.sem:
            if model.startswith("claude-"):
                return await self._claude(model, parts, effort, max_tokens)
            return await self._openrouter_call(model, parts, effort, max_tokens)

    async def _claude(self, model, parts, effort, max_tokens) -> dict:
        content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _b64(p)}}
                   if isinstance(p, bytes) else {"type": "text", "text": p} for p in parts]
        async with self.anthropic.messages.stream(
            model=model,
            max_tokens=max_tokens,
            thinking={"type": "adaptive"},
            output_config={"effort": effort},
            messages=[{"role": "user", "content": content}],
        ) as stream:
            msg = await stream.get_final_message()
        price_in, price_out = ANTHROPIC_PRICES.get(model, (0, 0))
        return {
            "text": "".join(b.text for b in msg.content if b.type == "text"),
            "stop_reason": msg.stop_reason,
            "model": msg.model,
            "input_tokens": msg.usage.input_tokens,
            "output_tokens": msg.usage.output_tokens,
            "cost": (msg.usage.input_tokens * price_in + msg.usage.output_tokens * price_out) / 1e6,
        }

    async def _openrouter_call(self, model, parts, effort, max_tokens) -> dict:
        content = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + _b64(p)}}
                   if isinstance(p, bytes) else {"type": "text", "text": p} for p in parts]
        stream = await self.openrouter.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
            extra_body={"reasoning": {"effort": effort}},
        )
        text, finish, usage, served = [], None, None, model
        async for chunk in stream:
            served = chunk.model or served
            usage = chunk.usage or usage
            for choice in chunk.choices:
                if choice.delta.content:
                    text.append(choice.delta.content)
                finish = choice.finish_reason or finish
        return {
            "text": "".join(text),
            "stop_reason": STOP.get(finish, finish),
            "model": served,
            "input_tokens": usage.prompt_tokens if usage else 0,
            "output_tokens": usage.completion_tokens if usage else 0,
            "cost": (getattr(usage, "cost", None) or 0.0) if usage else 0.0,
        }
