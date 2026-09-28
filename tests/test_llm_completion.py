"""Tests for shared DeepSeek / Comet completion helpers."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

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
    kwargs = completion_kwargs(base_url="https://api.cometapi.com/v1",
        reasoning_policy="off", model="deepseek-v4-pro", temperature=0.7)
    assert kwargs["extra_body"]["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in kwargs
    assert kwargs["temperature"] == 0.7


def test_completion_kwargs_overwrite_existing_thinking():
    kwargs = completion_kwargs(base_url="https://api.cometapi.com/v1",
        reasoning_policy="off", model="deepseek-v4-pro",
        extra_body={"thinking": {"type": "enabled"}, "foo": 1}, reasoning_effort="high")
    assert kwargs["extra_body"] == {"thinking": {"type": "disabled"}, "foo": 1}
    assert "reasoning_effort" not in kwargs


def test_create_chat_completion_does_not_retry_programming_errors():
    import pytest
    client = MagicMock()
    client.base_url = "https://api.cometapi.com/v1"
    client.chat.completions.create.side_effect = TypeError("unexpected keyword")
    with pytest.raises(TypeError):
        create_chat_completion(client, reasoning_policy="off", model="deepseek-v4-pro", messages=[])
    assert client.chat.completions.create.call_count == 1


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


def test_http_error_logs_status_code_not_message(caplog):
    import openai
    try:
        import httpx
    except ImportError:
        import httpx2 as httpx
    client = SimpleNamespace(base_url='https://api.cometapi.com/v1', chat=SimpleNamespace(completions=MagicMock()))
    client.chat.completions.create.side_effect = openai.BadRequestError(
        'PRIVATE TEXT', response=httpx.Response(400, request=httpx.Request('POST', 'https://offline.invalid')),
        body={'code': 'invalid_request', 'message': 'PRIVATE TEXT'})
    import pytest
    with pytest.raises(openai.BadRequestError):
        create_chat_completion(client, reasoning_policy='off', model='deepseek-v4-pro')
    assert 'status=400' in caplog.text
    assert 'code=invalid_request' in caplog.text
    assert 'PRIVATE' not in caplog.text
