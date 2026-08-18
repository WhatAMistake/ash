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
