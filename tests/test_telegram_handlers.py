import asyncio
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import telegram_bot as tb
from async_execution import AsyncExecution


def message(text="/block a story", user_id=7):
    return SimpleNamespace(
        text=text, caption=None, from_user=SimpleNamespace(id=user_id),
        chat=SimpleNamespace(id=user_id, type="private"), answer=AsyncMock(),
    )


def bot_for_block(reply):
    bot = tb.TelegramWriterBot.__new__(tb.TelegramWriterBot)
    bot.execution = AsyncExecution()
    bot.user_langs = {7: "en"}
    bot.user_states = {}
    bot.accumulated_text = {}
    bot.STATE_TO_DISCUSS = {"block_wait": "block_discuss"}
    bot.DISCUSSION_STATES = {"block_discuss": ("block_wait", "instr_block")}
    bot.bot = SimpleNamespace(send_chat_action=AsyncMock())
    bot._touch_user_activity = MagicMock()
    bot._get_writer_bot = MagicMock(return_value=SimpleNamespace(handle_block=reply))
    return bot


@pytest.mark.asyncio
async def test_block_command_offloads_and_transitions_to_discussion():
    bot = bot_for_block(lambda text: "advice")
    msg = message()
    await bot._handle_block(msg)
    assert bot.user_states[7] == "block_discuss"
    assert bot.accumulated_text[7] == "a story"
    assert "advice" in msg.answer.await_args.args[0]


@pytest.mark.asyncio
async def test_failed_tool_keeps_accumulated_text_and_state():
    bot = bot_for_block(lambda text: "Error: provider failure")
    bot.user_states[7] = "block_wait"
    bot.accumulated_text[7] = "draft"
    msg = message()
    await bot._handle_block(msg)
    assert bot.user_states[7] == "block_wait"
    assert bot.accumulated_text[7] == "draft"
    assert msg.answer.await_args.args[0] == tb.t("en", "error_llm")


@pytest.mark.asyncio
async def test_registered_text_and_reset_are_serialized_per_user():
    bot = tb.TelegramWriterBot.__new__(tb.TelegramWriterBot)
    bot.dp = tb.Dispatcher()
    bot.execution = AsyncExecution()
    bot._allow_in_chat = lambda msg: True
    bot.user_langs = {7: "en"}
    entered = asyncio.Event()
    release = asyncio.Event()
    events = []

    async def chat(msg):
        events.append("chat-start")
        entered.set()
        await release.wait()
        events.append("chat-end")

    async def reset(msg):
        events.append("reset")

    bot._handle_message = chat
    bot._handle_reset = reset
    bot._register_handlers()
    callbacks = {h.callback.__name__: h.callback for h in bot.dp.message.handlers}
    first = asyncio.create_task(callbacks["handle_text"](message("hello")))
    await entered.wait()
    second = asyncio.create_task(callbacks["cmd_reset"](message("/reset")))
    await asyncio.sleep(0.05)
    assert events == ["chat-start"]
    release.set()
    await asyncio.gather(first, second)
    assert events == ["chat-start", "chat-end", "reset"]


@pytest.mark.asyncio
async def test_blocking_writer_does_not_block_other_handlers():
    started = threading.Event()
    release = threading.Event()

    def reply(text):
        started.set()
        release.wait(5)
        return "advice"

    bot = bot_for_block(reply)
    task = asyncio.create_task(bot._handle_block(message()))
    try:
        assert await asyncio.to_thread(started.wait, 3)
        await asyncio.wait_for(asyncio.sleep(0.01), 1)
    finally:
        release.set()
        await task


@pytest.mark.asyncio
async def test_business_connection_disabled_during_request_does_not_reply(monkeypatch):
    monkeypatch.setattr(tb, "SECRETARY_ENABLED", True)
    monkeypatch.setattr(tb, "ADMIN_ID", 42)
    bot = tb.TelegramWriterBot.__new__(tb.TelegramWriterBot)
    bot.execution = AsyncExecution()
    bot.bot_user_id = 999
    connection = SimpleNamespace(
        user=SimpleNamespace(id=42), is_enabled=True,
        rights=SimpleNamespace(can_reply=True),
    )
    bot.business_connections = {"connection": connection}
    bot._secretary_language = lambda text: "en"
    bot._send_business_reply = AsyncMock()
    bot._notify_owner_about_urgent_message = AsyncMock()
    started = threading.Event()
    release = threading.Event()

    def chat(text):
        started.set()
        release.wait(5)
        return '{"reply":"Hello","urgency":"normal","owner_summary":""}'

    bot._get_secretary_bot = MagicMock(return_value=SimpleNamespace(chat=chat))
    msg = message("private message")
    msg.business_connection_id = "connection"
    task = asyncio.create_task(bot._handle_business_message(msg))
    try:
        assert await asyncio.to_thread(started.wait, 3)
        connection.is_enabled = False
    finally:
        release.set()
        await task
    bot._send_business_reply.assert_not_awaited()
    bot._notify_owner_about_urgent_message.assert_not_awaited()
