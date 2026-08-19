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
