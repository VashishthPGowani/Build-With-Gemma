"""Thin client for the Spur OpenAI-compatible API.

The only module that knows the remote wire format. `chat()` returns raw text;
`chat_json()` adds a defensive JSON-parsing ladder so callers never crash on a
sloppy model response.
"""

import json
import re

from openai import OpenAI

from .config import settings

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=settings.spur_api_key,
            base_url=settings.spur_base_url,
            timeout=settings.spur_timeout_s,
        )
    return _client


def chat(system: str, user: str, *, model: str | None = None,
         temperature: float | None = None, max_tokens: int = 1024) -> str:
    resp = get_client().chat.completions.create(
        model=model or settings.spur_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=settings.temperature if temperature is None else temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


def parse_json_loose(text: str) -> dict | None:
    """json.loads -> strip code fences -> first {...} block -> None."""
    for candidate in _candidates(text):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def _candidates(text: str):
    yield text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        yield fenced.group(1).strip()
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        yield brace.group(0)


def chat_json(system: str, user: str, **kwargs) -> tuple[dict | None, str]:
    """Returns (parsed_dict_or_None, raw_text)."""
    raw = chat(system, user, **kwargs)
    return parse_json_loose(raw), raw
