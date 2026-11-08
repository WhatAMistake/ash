"""Offline contracts for persona configuration, not model-behavior tests."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from i18n import t
from writer_bot import Message, WriterBot


@pytest.fixture
def bot_factory(monkeypatch):
    monkeypatch.setattr(WriterBot, "_init_llm", lambda self: None)
    return lambda language: WriterBot(language=language, use_rag=False)


@pytest.mark.parametrize("language", ["ru", "en"])
def test_loaded_prompt_and_chat_message_assembly(bot_factory, language):
    bot = bot_factory(language)
    expected = (ROOT / "prompts" / f"system_prompt.{language}.md").read_text(encoding="utf-8")
    assert bot.system_prompt == expected
    assert "хухни" in expected and "хух" in expected and "/dev_feedback" in expected
    assert "двадцатилет" not in expected.lower()
    assert "twenty years" not in expected.lower()
    bot.client = MagicMock()
    bot.client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))]
    )
    assert bot.chat("Hello") == "answer"
    messages = bot.client.chat.completions.create.call_args.kwargs["messages"]
    assert messages[0] == {"role": "system", "content": expected}
    assert messages[-1] == {"role": "user", "content": "Hello"}
    assert [(m.role, m.content) for m in bot.history] == [("user", "Hello"), ("assistant", "answer")]


def test_generic_and_emergency_fallback(bot_factory, monkeypatch):
    bot = bot_factory("fr")
    assert bot.system_prompt == (ROOT / "prompts" / "system_prompt.md").read_text(encoding="utf-8")
    monkeypatch.setattr(Path, "exists", lambda self: False)
    fallback = bot._load_system_prompt()
    for term in ("ordinary topics", "sources", "dangerous", "/dev_feedback", "хухни"):
        assert term in fallback
    assert "twenty years" not in fallback.lower()


@pytest.mark.parametrize("language", ["ru", "en"])
def test_command_instructions_do_not_invent_experience_or_quotes(bot_factory, language):
    bot = bot_factory(language)
    bot.client = MagicMock()
    bot.client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))]
    )
    bot.generate_fresh_prompt()
    fresh = bot.client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "двадцатилет" not in fresh.lower() and "twenty years" not in fresh.lower()
    for key in ("instr_fresh_prompt", "instr_methodique", "instr_methodique_brief",
                "instr_methodique_standard", "instr_methodique_detailed", "instr_methodique_full",
                "instr_cite_llm"):
        instruction = t(language, key)
        assert "двадцатилет" not in instruction.lower()
        assert "twenty years" not in instruction.lower()


@pytest.mark.parametrize("language", ["ru", "en"])
def test_cite_uses_writer_name_without_notice_and_never_generates_fallback(bot_factory, monkeypatch, language):
    bot = bot_factory(language)
    bot.client = MagicMock()
    monkeypatch.setattr("writer_bot.random.choice", lambda items: items[0])
    quote, writer = bot.cite()
    assert quote
    assert "не проверены" not in writer and "unverified" not in writer
    monkeypatch.setattr("writer_bot.json.load", lambda file: [{"quote": "Цитата" if language == "ru" else "Quote", "writer": "Author"}])
    assert bot.cite()[1] == "Author"
    monkeypatch.setattr(Path, "exists", lambda self: False)
    monkeypatch.setitem(WriterBot.cite.__globals__, "t", lambda *args, **kwargs: [{"quote": "Quote", "writer": "Author"}])
    assert bot.cite() == ("Quote", "Author")
    monkeypatch.setitem(WriterBot.cite.__globals__, "t", lambda *args, **kwargs: None)
    quote, writer = bot.cite()
    assert quote and writer
    bot.client.chat.completions.create.assert_not_called()


@pytest.mark.parametrize("language", ["ru", "en"])
@pytest.mark.parametrize("with_context", [False, True])
def test_methodique_source_guard_with_and_without_rag(bot_factory, language, with_context):
    bot = bot_factory(language)
    bot.client = MagicMock()
    bot.client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="analysis"))]
    )
    if with_context:
        bot.use_rag = True
        bot.rag = MagicMock()
        bot.rag.get_context_for_query.return_value = "Relevant supplied book excerpt"
    assert bot.methodique("A short story") == "analysis"
    messages = bot.client.chat.completions.create.call_args.kwargs["messages"]
    assert messages[0]["content"] == bot.system_prompt
    instruction = messages[1]["content"].lower()
    assert ("не выдумывай" in instruction or "do not invent" in instruction)
    if with_context:
        assert "relevant supplied book excerpt" in instruction


def test_summary_is_not_restricted_to_writing(bot_factory):
    bot = bot_factory("en")
    bot.client = MagicMock()
    bot.client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="A conversation about food."))]
    )
    bot.history = [Message(role="user", content="Dinner plans")] * 12
    bot._maybe_summarize()
    messages = bot.client.chat.completions.create.call_args.kwargs["messages"]
    assert "writing" not in " ".join(m["content"] for m in messages).lower()
    assert bot.session_summary == "A conversation about food."


def test_secretary_keeps_own_prompt_on_language_change(monkeypatch):
    import telegram_bot as tb

    monkeypatch.setattr(WriterBot, "_init_llm", lambda self: None)
    owner = tb.TelegramWriterBot.__new__(tb.TelegramWriterBot)
    owner.secretary_sessions = {}
    owner.llm_model, owner.llm_api_key, owner.llm_api_base = "model", None, None
    session = owner._get_secretary_bot("connection", 123, "ru")
    assert session.system_prompt == tb.SECRETARY_SYSTEM_PROMPT
    assert owner._get_secretary_bot("connection", 123, "en") is session
    assert session.language == "en"
    assert session.system_prompt == tb.SECRETARY_SYSTEM_PROMPT


def test_scenario_set_has_both_languages_multiturn_and_coverage():
    scenarios = json.loads((ROOT / "tests" / "ash_persona_scenarios.json").read_text(encoding="utf-8"))
    assert len(scenarios) >= 20
    assert len({s["id"] for s in scenarios}) == len(scenarios)
    for language in ("ru", "en"):
        entries = [s for s in scenarios if s["language"] == language]
        assert len(entries) >= 10
        assert {"everyday", "help", "humor", "writing", "disagreement",
                "emotional", "topic-switch", "creative"} <= {s["category"] for s in entries}
        assert any(len(s["turns"]) > 1 for s in entries)
        assert all(s["expect"] and all(s["turns"]) for s in entries)
