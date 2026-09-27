"""Ash Chat Completions adapter; stdlib + openai >= 1.x, Python >= 3.11.

No credentials, clients, environment changes or network calls at import time.
See INTEGRATION.md for model capability limits and buffered stream semantics.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal
from urllib.parse import urlsplit

from openai import APIError, BadRequestError

Provider = Literal["openrouter", "deepseek", "cometapi", "openai", "generic"]
_LOG = logging.getLogger(__name__)
_CONTROLS = ("reasoning", "thinking", "reasoning_effort", "enable_thinking")
_TEMPLATE_CONTROLS = ("enable_thinking", "thinking", "reasoning", "reasoning_effort")
_REASONING_FIELDS = ("reasoning_content", "reasoning", "reasoning_text")


class ReasoningPolicy(str, Enum):
    OFF = "off"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    PASSTHROUGH = "passthrough"


@dataclass
class ReasoningResult:
    text: str = ""
    # Opaque provider data: never stringify, concatenate with text, or reorder.
    details: list[Any] = field(default_factory=list)


@dataclass
class CompletionResult:
    text: str
    reasoning: ReasoningResult
    raw: Any
    provider: Provider
    retried_without_reasoning: bool = False


def detect_provider(base_url: Any) -> Provider:
    """Exact hostname matching, not a substring match on paths/userinfo."""
    if base_url is None:
        return "generic"
    try:
        url = urlsplit(str(base_url))
        if url.scheme.lower() not in {"http", "https"}:
            return "generic"
        host = (url.hostname or "").lower().rstrip(".")
    except ValueError:
        return "generic"
    return {
        "openrouter.ai": "openrouter",
        "api.deepseek.com": "deepseek",
        "api.cometapi.com": "cometapi",
        "api.openai.com": "openai",
    }.get(host, "generic")


def _without_controls(body: Mapping[str, Any]) -> dict[str, Any]:
    result = {k: v for k, v in body.items() if k not in _CONTROLS}
    template = result.get("chat_template_kwargs")
    if isinstance(template, Mapping):
        template = {k: v for k, v in template.items() if k not in _TEMPLATE_CONTROLS}
        if template:
            result["chat_template_kwargs"] = template
        else:
            result.pop("chat_template_kwargs", None)
    return result


def _validate_passthrough(extra: Mapping[str, Any]) -> None:
    """Fail closed for multiple control dialects rather than guessing precedence."""
    families = sum(key in extra for key in ("reasoning", "enable_thinking"))
    families += int("thinking" in extra or "reasoning_effort" in extra)
    template = extra.get("chat_template_kwargs")
    families += int(isinstance(template, Mapping) and any(k in template for k in _TEMPLATE_CONTROLS))
    if isinstance(template, Mapping):
        enabled = template.get("enable_thinking")
        effort = template.get("reasoning_effort")
        if (enabled is False and effort not in (None, "none")) or (enabled is True and effort == "none"):
            raise ValueError("Contradictory template thinking controls")
    if families > 1:
        raise ValueError("Mixed reasoning control dialects; select an explicit reasoning_policy")
    thinking = extra.get("thinking")
    effort = extra.get("reasoning_effort")
    if isinstance(thinking, Mapping) and effort is not None:
        disabled = thinking.get("type") == "disabled"
        enabled = thinking.get("type") == "enabled"
        if (disabled and effort != "none") or (enabled and effort == "none"):
            raise ValueError("Contradictory thinking and reasoning_effort")
    reasoning = extra.get("reasoning")
    if isinstance(reasoning, Mapping):
        if "effort" in reasoning and "max_tokens" in reasoning:
            raise ValueError("reasoning.effort and reasoning.max_tokens are mutually exclusive")
        effort = reasoning.get("effort")
        budget = reasoning.get("max_tokens")
        if reasoning.get("enabled") is False and isinstance(budget, (int, float)) and budget > 0:
            raise ValueError("Contradictory reasoning.enabled and reasoning.max_tokens")
        if (reasoning.get("enabled") is False and effort not in (None, "none")) or (
            reasoning.get("enabled") is True and effort == "none"
        ):
            raise ValueError("Contradictory reasoning.enabled and reasoning.effort")


def completion_kwargs(
    *,
    base_url: Any = None,
    reasoning_policy: ReasoningPolicy | str = ReasoningPolicy.PASSTHROUGH,
    supports_reasoning_effort: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build SDK kwargs. All custom controls go through extra_body.

    Explicit policies replace old controls, including template controls.
    Passthrough retains explicit controls (moving SDK kwargs to extra_body),
    otherwise uses server defaults. Generic endpoints never receive controls.
    OpenAI and non-DeepSeek Comet models require an explicit capability flag.
    """
    policy = ReasoningPolicy(reasoning_policy)
    provider = detect_provider(base_url)
    payload = dict(kwargs)
    incoming_extra = payload.pop("extra_body", None)
    if incoming_extra is not None and not isinstance(incoming_extra, Mapping):
        raise TypeError("extra_body must be a mapping or None")
    extra = dict(incoming_extra or {})
    # Never expose these as SDK kwargs, including on older openai 1.x.
    for key in (*_CONTROLS, "chat_template_kwargs"):
        if key in payload:
            value = payload.pop(key)
            if policy is ReasoningPolicy.PASSTHROUGH and provider != "generic":
                if key in extra and extra[key] != value:
                    raise ValueError(f"Conflicting {key} in kwargs and extra_body")
            extra.setdefault(key, value)

    if provider == "generic" or policy is not ReasoningPolicy.PASSTHROUGH:
        extra = _without_controls(extra)
    elif provider == "openrouter" and "reasoning_effort" in extra:
        # Bifrost proxy normalization. Never send top-level effort to OpenRouter.
        effort = extra.pop("reasoning_effort")
        effort = {"off": "none", "adaptive": "medium", "max": "xhigh"}.get(effort, effort)
        reasoning = extra.get("reasoning")
        if reasoning is not None and not isinstance(reasoning, Mapping):
            raise ValueError("reasoning must be a mapping")
        reasoning = dict(reasoning or {})
        if "effort" in reasoning and reasoning["effort"] != effort:
            raise ValueError("Conflicting OpenRouter reasoning efforts")
        reasoning["effort"] = effort
        extra["reasoning"] = reasoning

    if policy is not ReasoningPolicy.PASSTHROUGH:
        effort = "none" if policy is ReasoningPolicy.OFF else policy.value
        if provider == "openrouter":
            extra["reasoning"] = {"effort": effort}
        elif provider == "deepseek":
            extra["reasoning_effort"] = "high" if effort == "medium" else effort
        elif provider == "cometapi" and str(payload.get("model", "")).lower().split("/")[-1].startswith("deepseek"):
            # Comet's documented DeepSeek dialect; not a policy for every model.
            if policy is ReasoningPolicy.OFF:
                extra["thinking"] = {"type": "disabled"}
            else:
                extra["thinking"] = {"type": "enabled"}
                extra["reasoning_effort"] = "high" if effort == "medium" else effort
        elif provider in {"openai", "cometapi"} and supports_reasoning_effort:
            extra["reasoning_effort"] = effort
    _validate_passthrough(extra)
    if extra:
        payload["extra_body"] = extra
    return payload


# Port of Bifrost reasoning-tags.ts and code-regions.ts, plus Ash's legacy
# <reasoning>, [think]/[thinking], and triangular tag spellings.
_FENCED_RE = re.compile(r"(^|\n)(```|~~~)[^\n]*\n[\s\S]*?(?:\n\2|$)")
_INLINE_RE = re.compile(r"`+[^`]+`+")
_NAMES = r"think(?:ing)?|thought|antthinking|reasoning"
_TAG_RE = re.compile(
    rf"<\s*(/?)\s*(?:{_NAMES})\b[^<>]*>"
    rf"|\[\s*(/?)\s*think(?:ing)?\s*\]"
    rf"|◁\s*(/?)\s*think(?:ing)?\s*▷", re.IGNORECASE,
)
_FINAL_RE = re.compile(r"<\s*/?\s*final\b[^<>]*>", re.IGNORECASE)


def _code_regions(text: str) -> list[tuple[int, int]]:
    regions = [(m.start() + len(m[1]), m.end()) for m in _FENCED_RE.finditer(text)]
    for match in _INLINE_RE.finditer(text):
        start, end = match.span()
        if not any(start >= a and end <= b for a, b in regions):
            regions.append((start, end))
    return sorted(regions)


def _in_code(pos: int, regions: list[tuple[int, int]]) -> bool:
    return any(start <= pos < end for start, end in regions)


def strip_reasoning_tags(
    text: str,
    mode: Literal["strict", "preserve"] = "strict",
    *,
    trim: Literal["none", "start", "both"] = "both",
) -> str:
    """Remove reasoning spans outside code; final wrappers keep their contents.

    Strict discards an unclosed thinking tail. Preserve retains that tail,
    NOT the contents of closed thinking spans. Matches Bifrost's state machine
    (boolean state, not an XML parser). Fast path leaves tag-free text unchanged.
    """
    if mode not in {"strict", "preserve"} or trim not in {"none", "start", "both"}:
        raise ValueError("Invalid reasoning tag mode or trim")
    if not text or not (_TAG_RE.search(text) or _FINAL_RE.search(text)):
        return text
    regions = _code_regions(text)
    cleaned = _FINAL_RE.sub(lambda m: m[0] if _in_code(m.start(), regions) else "", text)
    regions = _code_regions(cleaned)
    parts: list[str] = []
    last = 0
    in_thinking = False
    for match in _TAG_RE.finditer(cleaned):
        if _in_code(match.start(), regions):
            continue
        is_close = "/" in match.groups()
        if not in_thinking:
            parts.append(cleaned[last:match.start()])
            if not is_close:
                in_thinking = True
        elif is_close:
            in_thinking = False
        last = match.end()
    if not in_thinking or mode == "preserve":
        parts.append(cleaned[last:])
    result = "".join(parts)
    return result if trim == "none" else result.lstrip() if trim == "start" else result.strip()


def _get(value: Any, key: str, default: Any = None) -> Any:
    found = value.get(key) if isinstance(value, Mapping) else getattr(value, key, None)
    if found is not None:
        return found
    extra = value.get("model_extra") if isinstance(value, Mapping) else getattr(value, "model_extra", None)
    return extra.get(key, default) if isinstance(extra, Mapping) else default


def _choice_part(response: Any, part: str) -> Any:
    choices = _get(response, "choices", []) or []
    if not choices:
        return None
    # Only choice zero; never concatenate alternative completions.
    for choice in choices:
        if _get(choice, "index", 0) == 0:
            return _get(choice, part)
    return None


def _content_text(message: Any) -> str:
    content = _get(message, "content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif _get(part, "type") in {"text", "output_text"}:
                value = _get(part, "text", "")
                if isinstance(value, str):
                    parts.append(value)
        return "".join(parts)
    return ""


def _reasoning(message: Any) -> ReasoningResult:
    text = ""
    for name in _REASONING_FIELDS:
        value = _get(message, name)
        if not isinstance(value, str) or not value:
            extra = _get(message, "model_extra", {})
            value = extra.get(name) if isinstance(extra, Mapping) else None
        if isinstance(value, str) and value:
            text = value
            break
    details = _get(message, "reasoning_details")
    return ReasoningResult(text, list(details) if isinstance(details, list) else [] if details is None else [details])


def visible_text(
    response: Any, *, provider: Provider = "generic", retried_without_reasoning: bool = False,
) -> CompletionResult:
    """Parse raw SDK/dict completion into visible text + separate reasoning."""
    if isinstance(response, CompletionResult):
        return response
    message = _choice_part(response, "message")
    return CompletionResult(
        strip_reasoning_tags(_content_text(message), mode="strict").strip(),
        _reasoning(message), response, provider, retried_without_reasoning,
    )


def extract_completion_text(response: Any) -> str:
    """Legacy Ash helper; also accepts the new CompletionResult."""
    if isinstance(response, CompletionResult):
        return response.text
    return _content_text(_choice_part(response, "message")).strip()


def strip_think_tags(text: str) -> str:
    """Legacy alias with the same outer whitespace trimming as Ash."""
    return strip_reasoning_tags(text).strip()


def visible_completion_text(response: Any) -> str:
    return visible_text(response).text


_UNKNOWN_PARAM_RE = re.compile(
    r"\b(?:unknown|unrecognized|unrecognised|unsupported|unexpected|not\s+supported)"
    r"[\s_-]+(?:request[\s_-]+)?(?:parameter|param|argument|field|keyword)\b"
    r"|\b(?:parameter|param|argument|field|keyword)\b[^\n;:]{0,90}"
    r"\b(?:unknown|unrecognized|unrecognised|unsupported|not\s+(?:supported|allowed|recognized))\b"
    r"|\bextra\s+(?:inputs|fields)\s+are\s+not\s+permitted\b", re.IGNORECASE,
)
_PARAM_NAME = r"(?:reasoning_effort|reasoning|thinking)(?:\.[a-z_]+)*"
_REASONING_PARAM_RE = re.compile(rf"^(?:(?:body|extra_body)\.)?{_PARAM_NAME}$", re.IGNORECASE)
_NAMED_UNKNOWN_RE = re.compile(
    rf"\b(?:unknown|unrecognized|unrecognised|unsupported|unexpected)[\s_-]+"
    rf"(?:request[\s_-]+)?(?:parameters?|params?|arguments?|fields?|keywords?)\b"
    rf"(?:\s+(?:supplied|provided))?\s*[:=]?\s*[\[\"'`]*"
    rf"(?:(?:extra_body|body)\.)?{_PARAM_NAME}\b"
    rf"|\b(?:parameter|param|argument|field|keyword)\s*:?\s*[\"'`]*"
    rf"{_PARAM_NAME}[\"'`]*\s+(?:is\s+)?"
    rf"(?:unknown|unrecognized|unrecognised|unsupported|not\s+(?:supported|allowed|recognized))\b"
    rf"|\bextra\s+(?:inputs|fields)\s+are\s+not\s+permitted\s*:\s*{_PARAM_NAME}\b",
    re.IGNORECASE,
)


def _should_retry(exc: BadRequestError, payload: Mapping[str, Any]) -> bool:
    extra = payload.get("extra_body") or {}
    if not any(k in extra for k in ("reasoning", "thinking", "reasoning_effort")):
        return False
    body = getattr(exc, "body", None)
    if body is None:  # Early openai 1.x declares .body but does not assign it.
        try:
            body = exc.response.json()
        except (ValueError, AttributeError):
            body = None
    error = body.get("error", body) if isinstance(body, Mapping) else {}
    if not isinstance(error, Mapping):
        error = {}
    message = str(error.get("message") or getattr(exc, "message", "") or exc)
    param = str(error.get("param") or getattr(exc, "param", None) or "")
    code = str(error.get("code") or getattr(exc, "code", None) or "").lower()
    # Name and error must be related, not just two words anywhere in a 400.
    # Invalid effort VALUE errors must not silently fall back to defaults.
    if param:
        return bool(_REASONING_PARAM_RE.fullmatch(param) and (
            _UNKNOWN_PARAM_RE.search(message)
            or code in {"unknown_parameter", "unsupported_parameter", "unrecognized_parameter"}
        ))
    return bool(_NAMED_UNKNOWN_RE.search(message))


def _fallback_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = _without_controls(payload)
    extra = _without_controls(result.pop("extra_body", {}) or {})
    if extra:
        result["extra_body"] = extra
    return result


def _warn_retry(provider: Provider) -> None:
    # No exception body, request, prompt, reasoning or credentials in logs.
    _LOG.warning("%s rejected a reasoning parameter; retrying once without reasoning controls (server defaults may enable thinking)", provider)


def create_chat_completion(
    client: Any, *, reasoning_policy: ReasoningPolicy | str = ReasoningPolicy.PASSTHROUGH,
    base_url: Any = None, supports_reasoning_effort: bool = False, **kwargs: Any,
) -> CompletionResult:
    """Synchronous drop-in for Ash call sites; raw SDK result is in .raw."""
    if kwargs.get("stream") or (kwargs.get("extra_body") or {}).get("stream"):
        raise ValueError("Use stream_chat_completion() for stream=True")
    base_url = base_url if base_url is not None else getattr(client, "base_url", None)
    provider = detect_provider(base_url)
    payload = completion_kwargs(base_url=base_url, reasoning_policy=reasoning_policy,
                                supports_reasoning_effort=supports_reasoning_effort, **kwargs)
    retried = False
    try:
        raw = _sdk_completion(client, payload)
    except BadRequestError as exc:
        if not _should_retry(exc, payload):
            raise
        _warn_retry(provider)
        retried = True
        raw = _sdk_completion(client, _fallback_payload(payload))
    return visible_text(raw, provider=provider, retried_without_reasoning=retried)


def stream_chat_completion(
    client: Any, *, reasoning_policy: ReasoningPolicy | str = ReasoningPolicy.PASSTHROUGH,
    base_url: Any = None, supports_reasoning_effort: bool = False, **kwargs: Any,
) -> Iterator[CompletionResult]:
    """Consume SSE deltas, yield ONE buffered result after successful exhaustion.

    No unsafe provisional text: Markdown code regions can be established by
    later chunks. raw is the ordered chunk list, including usage-only chunks;
    reasoning.details holds ordered fragments, not reconstructed tool history.
    Never replay a request after receiving any chunk. Close the SDK stream on
    success and failure. Async clients are deliberately outside this adapter.
    """
    base_url = base_url if base_url is not None else getattr(client, "base_url", None)
    provider = detect_provider(base_url)
    kwargs["stream"] = True
    extra = kwargs.get("extra_body") or {}
    if "stream" in extra and extra["stream"] is not True:
        raise ValueError("extra_body.stream conflicts with streaming")
    payload = completion_kwargs(base_url=base_url, reasoning_policy=reasoning_policy,
                                supports_reasoning_effort=supports_reasoning_effort, **kwargs)
    for attempt in range(2):
        chunks: list[Any] = []
        content: list[str] = []
        reasoning_parts: list[str] = []
        details: list[Any] = []
        stream = None
        try:
            stream = _sdk_completion(client, payload)
            for chunk in stream:
                chunks.append(chunk)
                delta = _choice_part(chunk, "delta")
                content.append(_content_text(delta))
                reasoning = _reasoning(delta)
                reasoning_parts.append(reasoning.text)
                details.extend(reasoning.details)
        except BadRequestError as exc:
            if attempt or chunks or not _should_retry(exc, payload):
                raise
            _warn_retry(provider)
            payload = _fallback_payload(payload)
            continue
        finally:
            if stream is not None and callable(getattr(stream, "close", None)):
                stream.close()
        yield CompletionResult(
            strip_reasoning_tags("".join(content)).strip(),
            ReasoningResult("".join(reasoning_parts), details), chunks, provider, bool(attempt),
        )
        return


def log_completion_error(exc: Exception) -> None:
    """Never log response bodies, prompts, URLs or exception messages."""
    status = getattr(exc, "status_code", None)
    code = getattr(exc, "code", None)
    # Codes are untrusted server data; admit identifiers only.
    code = code if isinstance(code, str) and re.fullmatch(r"[a-zA-Z0-9_]{1,64}", code) else None
    _LOG.warning("LLM failure type=%s status=%s code=%s", type(exc).__name__,
                 status if isinstance(status, int) else None, code)


def _sdk_completion(client: Any, payload: Mapping[str, Any]) -> Any:
    try:
        return client.chat.completions.create(**payload)
    except APIError as exc:
        log_completion_error(exc)
        raise
