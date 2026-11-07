"""Tests for shared DeepSeek / Comet completion helpers."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
import logging

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from llm_completion import (
    completion_kwargs,
    create_chat_completion,
    extract_completion_text,
    strip_think_tags,
    visible_completion_text,
)


def test_completion_kwargs_disable_thinking():
    kwargs = completion_kwargs(model="deepseek-v4-pro", temperature=0.7)
    assert kwargs["extra_body"]["thinking"] == {"type": "disabled"}
    assert kwargs["extra_body"]["reasoning_effort"] == "none"
    assert kwargs["reasoning_effort"] == "none"
    assert kwargs["model"] == "deepseek-v4-pro"
    assert kwargs["temperature"] == 0.7


def test_completion_kwargs_overwrite_existing_thinking():
    kwargs = completion_kwargs(
        extra_body={"thinking": {"type": "enabled"}, "foo": 1},
        reasoning_effort="high",
    )
    assert kwargs["extra_body"]["thinking"] == {"type": "disabled"}
    assert kwargs["extra_body"]["reasoning_effort"] == "none"
    assert kwargs["extra_body"]["foo"] == 1
    assert kwargs["reasoning_effort"] == "none"


def test_create_chat_completion_retries_without_top_level_reasoning_effort():
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        TypeError("create() got an unexpected keyword argument 'reasoning_effort'"),
        SimpleNamespace(ok=True),
    ]
    result = create_chat_completion(client, model="deepseek-v4-pro", messages=[])
    assert result.ok is True
    assert client.chat.completions.create.call_count == 2
    first_kwargs = client.chat.completions.create.call_args_list[0].kwargs
    second_kwargs = client.chat.completions.create.call_args_list[1].kwargs
    assert first_kwargs["reasoning_effort"] == "none"
    assert "reasoning_effort" not in second_kwargs
    assert second_kwargs["extra_body"]["thinking"] == {"type": "disabled"}
    assert second_kwargs["extra_body"]["reasoning_effort"] == "none"


def test_extract_completion_ignores_reasoning_fields():
    message = SimpleNamespace(
        content="Visible answer",
        reasoning="hidden",
        reasoning_content="also hidden",
    )
    response = SimpleNamespace(choices=[SimpleNamespace(message=message)])
    assert extract_completion_text(response) == "Visible answer"

    multipart = SimpleNamespace(
        content=[
            {"type": "thinking", "text": "I will inspect this now."},
            {"type": "text", "text": "Final line"},
        ]
    )
    response = SimpleNamespace(choices=[SimpleNamespace(message=multipart)])
    assert extract_completion_text(response) == "Final line"


def test_strip_and_visible_text_drop_think_tags():
    raw = "<think>secret plan</think>\nHello there"
    assert strip_think_tags(raw) == "Hello there"
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=raw))]
    )
    assert visible_completion_text(response) == "Hello there"


def test_writer_chat_rejects_empty_and_preserves_history(monkeypatch):
    from writer_bot import WriterBot
    monkeypatch.setattr(WriterBot, "_init_llm", lambda self: None)
    bot = WriterBot(use_rag=False)
    bot.client = MagicMock()
    bot.client.chat.completions.create.return_value = SimpleNamespace(choices=[])
    assert bot.chat("private text").startswith("Error:")
    assert bot.history == []
    assert bot.messages_since_summary == 0


def test_writer_chat_adds_history_only_after_success(monkeypatch):
    from writer_bot import WriterBot
    monkeypatch.setattr(WriterBot, "_init_llm", lambda self: None)
    bot = WriterBot(use_rag=False)
    bot.client = MagicMock()
    bot.client.chat.completions.create.side_effect = [
        SimpleNamespace(choices=[]),
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="hello"))]),
    ]
    assert bot.chat("failed input").startswith("Error:")
    assert bot.chat("successful input") == "hello"
    assert [(item.role, item.content) for item in bot.history] == [
        ("user", "successful input"), ("assistant", "hello")
    ]


def test_writer_error_is_sanitized_and_not_recorded(monkeypatch, caplog):
    from writer_bot import WriterBot
    monkeypatch.setattr(WriterBot, "_init_llm", lambda self: None)
    bot = WriterBot(use_rag=False)
    bot.client = MagicMock()
    bot.client.chat.completions.create.side_effect = TimeoutError("secret key and private document")
    with caplog.at_level(logging.INFO):
        response = bot.chat("private text")
    assert response == "Error: LLM request failed."
    assert not bot.history
    assert "secret key" not in caplog.text
    assert "private document" not in caplog.text
    assert "private text" not in caplog.text
    assert '"event": "llm_completion"' in caplog.text


def test_openai_client_configures_timeout_and_retries(monkeypatch):
    import openai
    from writer_bot import WriterBot
    constructor = MagicMock()
    monkeypatch.setattr(openai, "OpenAI", constructor)
    WriterBot(api_key="fake", use_rag=False)
    assert constructor.call_args.kwargs["timeout"] == 60.0
    assert constructor.call_args.kwargs["max_retries"] == 2


def test_completion_does_not_retry_unrelated_type_error():
    client = MagicMock()
    client.chat.completions.create.side_effect = TypeError("unrelated problem")
    import pytest
    with pytest.raises(TypeError):
        create_chat_completion(client, model="fake", messages=[])
    assert client.chat.completions.create.call_count == 1


def test_openai_sdk_retries_transient_timeout_without_live_api(monkeypatch):
    import httpx
    import openai
    import openai._base_client
    from llm_completion import LLM_MAX_RETRIES, LLM_TIMEOUT_SECONDS

    attempts = []

    def send(request):
        attempts.append(request)
        if len(attempts) <= 2:
            raise httpx.ReadTimeout("private body", request=request)
        return httpx.Response(200, request=request, json={
            "id": "test", "object": "chat.completion", "created": 0,
            "model": "fake", "choices": [
                {"index": 0, "finish_reason": "stop",
                 "message": {"role": "assistant", "content": "ok"}}
            ],
        })

    monkeypatch.setattr(openai._base_client.time, "sleep", lambda _: None)
    http_client = httpx.Client(transport=httpx.MockTransport(send))
    with openai.OpenAI(
        api_key="fake", base_url="https://example.invalid/v1",
        http_client=http_client, timeout=LLM_TIMEOUT_SECONDS, max_retries=LLM_MAX_RETRIES,
    ) as client:
        result = create_chat_completion(client, model="fake", messages=[])
    assert visible_completion_text(result) == "ok"
    assert len(attempts) == 3


def test_openai_sdk_stops_after_timeout_retry_limit(monkeypatch):
    import httpx
    import openai
    import openai._base_client
    import pytest

    attempts = []

    def send(request):
        attempts.append(request)
        raise httpx.ReadTimeout("private body", request=request)

    monkeypatch.setattr(openai._base_client.time, "sleep", lambda _: None)
    http_client = httpx.Client(transport=httpx.MockTransport(send))
    with openai.OpenAI(
        api_key="fake", base_url="https://example.invalid/v1",
        http_client=http_client, timeout=60.0, max_retries=2,
    ) as client:
        with pytest.raises(openai.APITimeoutError):
            create_chat_completion(client, model="fake", messages=[])
    assert len(attempts) == 3


def test_completion_process_wide_limit():
    import concurrent.futures
    import threading

    guard = threading.Lock()
    release = threading.Event()
    four_started = threading.Event()
    active = 0
    peak = 0

    def create(**kwargs):
        nonlocal active, peak
        with guard:
            active += 1
            peak = max(peak, active)
            if active == 4:
                four_started.set()
        release.wait(5)
        with guard:
            active -= 1
        return SimpleNamespace()

    client = MagicMock()
    client.chat.completions.create.side_effect = create
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(create_chat_completion, client, model="fake") for _ in range(6)]
        try:
            assert four_started.wait(3)
            assert peak == 4
        finally:
            release.set()
        for future in futures:
            future.result()
    assert peak == 4
