"""Tests for bilingual admin changelog edit flow.

Run: py -m pytest tests/test_update_changelog_edit.py -v
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))


def _make_bot():
    """Build a lightweight stand-in with real update-edit methods bound on it."""
    import telegram_bot as tb

    bot = MagicMock()
    bot.admin_id = 42
    bot.user_langs = {42: "ru", 7: "en"}
    bot.all_users = {42, 7}
    bot.user_states = {}
    bot.pending_update_changelogs = {
        "ru": "Старый RU",
        "en": "Old EN",
    }
    bot.send_long_message = AsyncMock()

    cls = tb.TelegramWriterBot
    bot._clear_pending_update = cls._clear_pending_update.__get__(bot, cls)
    bot._format_update_preview = cls._format_update_preview.__get__(bot, cls)
    bot._send_update_preview = cls._send_update_preview.__get__(bot, cls)
    bot._start_update_edit = cls._start_update_edit.__get__(bot, cls)
    bot._handle_update_confirm_input = cls._handle_update_confirm_input.__get__(bot, cls)
    bot._handle_update_edit_input = cls._handle_update_edit_input.__get__(bot, cls)
    bot._process_update_broadcast = AsyncMock()
    return bot, tb


def _msg(user_id: int = 42):
    message = MagicMock()
    message.from_user = SimpleNamespace(id=user_id)
    message.answer = AsyncMock()
    return message


@pytest.mark.asyncio
async def test_preview_contains_edit_option():
    bot, _ = _make_bot()
    preview = bot._format_update_preview(bot.pending_update_changelogs)
    assert "править" in preview
    assert "Старый RU" in preview
    assert "Old EN" in preview


@pytest.mark.asyncio
async def test_missing_russian_changelog_is_translated_before_preview(monkeypatch):
    bot, tb = _make_bot()
    bot.llm_model = "test-model"
    bot.llm_api_key = "test-key"
    bot.llm_api_base = "https://example.invalid"
    writer = MagicMock()
    monkeypatch.setattr(tb, "WriterBot", lambda **kwargs: writer)
    generate = MagicMock(side_effect=[None, "• Fixed help"])
    translate = MagicMock(return_value="• Исправлена справка")
    save = MagicMock()
    monkeypatch.setattr(tb, "check_and_generate_changelog", generate)
    monkeypatch.setattr(tb, "translate_changelog_with_llm", translate)
    monkeypatch.setattr(tb, "save_current_hashes", save)

    async def run(func, *args, **kwargs):
        return func(*args, **kwargs)

    bot.execution.run = run
    await tb.TelegramWriterBot._check_and_notify_updates(bot)
    translate.assert_called_once_with(writer, "• Fixed help", "ru")
    assert bot.pending_update_changelogs == {"ru": "• Исправлена справка", "en": "• Fixed help"}
    save.assert_called_once()


def test_preview_warns_when_translation_is_missing():
    bot, _ = _make_bot()
    preview = bot._format_update_preview({"ru": "", "en": "• Fixed help"})
    assert "Не удалось подготовить обе версии" in preview
    assert "править" in preview


@pytest.mark.asyncio
async def test_broadcast_uses_each_users_language(monkeypatch):
    bot, tb = _make_bot()
    bot.pending_update_changelogs = {"ru": "• Исправлена справка", "en": "• Fixed help"}
    monkeypatch.setattr(tb, "init_cache", MagicMock())
    await tb.TelegramWriterBot._process_update_broadcast(bot)
    messages = {call.args[0]: call.args[1] for call in bot.send_long_message.await_args_list
                if len(call.args) > 1 and ("Исправлена справка" in call.args[1] or "Fixed help" in call.args[1])}
    assert "Исправлена справка" in messages[42]
    assert "Fixed help" in messages[7]
    assert not hasattr(bot, "pending_update_changelogs")


@pytest.mark.asyncio
async def test_edit_flow_saves_ru_then_en_and_returns_to_confirm():
    bot, _ = _make_bot()
    bot.user_states[42] = "update_confirm"
    msg = _msg(42)

    await bot._handle_update_confirm_input(msg, "править")
    assert bot.user_states[42] == "update_edit_ru"

    await bot._handle_update_edit_input(msg, "update_edit_ru", "Новый русский changelog")
    assert bot.pending_update_changelogs["ru"] == "Новый русский changelog"
    assert bot.user_states[42] == "update_edit_en"

    await bot._handle_update_edit_input(msg, "update_edit_en", "New English changelog")
    assert bot.pending_update_changelogs["en"] == "New English changelog"
    assert bot.user_states[42] == "update_confirm"

    preview_calls = [
        c for c in bot.send_long_message.await_args_list
        if c.args and "Предпросмотр обновления" in str(c.args[1])
    ]
    assert preview_calls, "expected updated preview after EN save"
    assert "Новый русский changelog" in preview_calls[-1].args[1]
    assert "New English changelog" in preview_calls[-1].args[1]


@pytest.mark.asyncio
async def test_confirm_yes_starts_broadcast():
    bot, _ = _make_bot()
    bot.user_states[42] = "update_confirm"
    msg = _msg(42)

    await bot._handle_update_confirm_input(msg, "да")
    bot._process_update_broadcast.assert_awaited_once()


@pytest.mark.asyncio
async def test_confirm_no_clears_pending():
    bot, _ = _make_bot()
    bot.user_states[42] = "update_confirm"
    msg = _msg(42)

    await bot._handle_update_confirm_input(msg, "нет")
    assert not hasattr(bot, "pending_update_changelogs")
    assert bot.user_states[42] == "chat"


@pytest.mark.asyncio
async def test_empty_edit_rejected():
    bot, _ = _make_bot()
    bot.user_states[42] = "update_edit_ru"
    msg = _msg(42)

    await bot._handle_update_edit_input(msg, "update_edit_ru", "   ")
    assert bot.pending_update_changelogs["ru"] == "Старый RU"
    assert bot.user_states[42] == "update_edit_ru"
    msg.answer.assert_awaited()
