"""Shared LLM completion helpers.

Comet (and official DeepSeek V4) enable thinking by default. Disable it on
every call and strip leftover chain-of-thought if a proxy still leaks it.
"""

from __future__ import annotations

import re
from typing import Any, Dict


_THINK_TAG_RE = re.compile(
    r"<(think|thinking|thought|reasoning)\b[^>]*>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)
_THINK_OPEN_RE = re.compile(
    r"<(think|thinking|thought|reasoning)\b[^>]*>[\s\S]*",
    re.IGNORECASE,
)
_THINK_ALT_RE = re.compile(
    r"(?:\[/?think\]|\[/thinking\]|◁/?think▷)",
    re.IGNORECASE,
)


def completion_kwargs(**kwargs: Any) -> Dict[str, Any]:
    """Force non-thinking mode for DeepSeek V4 via Comet / OpenAI-compatible APIs."""
    extra = dict(kwargs.get("extra_body") or {})
    extra["thinking"] = {"type": "disabled"}
    extra["reasoning_effort"] = "none"
    kwargs["extra_body"] = extra
    kwargs["reasoning_effort"] = "none"
    return kwargs


def create_chat_completion(client: Any, **kwargs: Any) -> Any:
    """chat.completions.create with thinking disabled.

    Comet reads thinking / reasoning_effort from extra_body. Official DeepSeek
    also accepts top-level reasoning_effort. If the SDK rejects that kwarg,
    retry with extra_body only.
    """
    payload = completion_kwargs(**kwargs)
    try:
        return client.chat.completions.create(**payload)
    except TypeError as exc:
        if "reasoning_effort" not in str(exc):
            raise
        payload.pop("reasoning_effort", None)
        return client.chat.completions.create(**payload)


def extract_completion_text(response: Any) -> str:
    """Take only the visible assistant text. Never include reasoning fields."""
    try:
        message = response.choices[0].message
    except Exception:
        return ""

    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")

    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                part_type = str(part.get("type") or "").lower()
                if part_type in {"thinking", "reasoning", "thought"}:
                    continue
                parts.append(str(part.get("text") or part.get("content") or ""))
        content = "".join(parts)

    return (content or "").strip()


def strip_think_tags(text: str) -> str:
    """Remove <think> / reasoning tags a proxy may still paste into content."""
    if not text:
        return ""

    cleaned = text.replace("\r\n", "\n")
    cleaned = _THINK_TAG_RE.sub("", cleaned)
    cleaned = _THINK_OPEN_RE.sub("", cleaned)
    cleaned = _THINK_ALT_RE.sub("", cleaned)
    return cleaned.strip()


def visible_completion_text(response: Any) -> str:
    """User-facing reply: visible content only, with leftover think tags stripped."""
    return strip_think_tags(extract_completion_text(response))
