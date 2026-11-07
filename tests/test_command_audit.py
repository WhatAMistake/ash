import asyncio
import sys
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import telegram_bot as tb
from async_execution import AsyncExecution
from chat_scope import ChatScopedDict, current_group
from telegram_commands import command_help, descriptions


def msg(text, user=7, chat=7, chat_type="private", reply=None):
    return SimpleNamespace(
        text=text, caption=None, from_user=SimpleNamespace(id=user, username=None),
        chat=SimpleNamespace(id=chat, type=chat_type), answer=AsyncMock(),
        reply_to_message=reply, entities=None,
    )


def bot():
    b = tb.TelegramWriterBot.__new__(tb.TelegramWriterBot)
    b.dp = tb.Dispatcher()
    b.execution = AsyncExecution()
    b.user_langs = {7: "en", 8: "en"}
    b.user_states = ChatScopedDict()
    b.accumulated_text = ChatScopedDict()
    b.tool_discussions = ChatScopedDict()
    b.uploaded_documents = ChatScopedDict()
    b.sessions = ChatScopedDict()
    b.STATE_CONFIG = {"feedback": (20, "feedback_on_text"), "summary_wait": (10, None)}
    b.DISCUSSION_STATES = {"feedback_discuss": ("feedback", "instr_feedback")}
    b.STATE_TO_DISCUSS = {"feedback": "feedback_discuss"}
    b.WAITING_STATES = set(b.STATE_CONFIG) | set(b.DISCUSSION_STATES)
    b.SUMMARY_FORMAT_STATE = "summary_format"
    b.group_chats_enabled = True
    b.group_chats = set()
    b.last_auto_activation = {}
    b.group_last_user_message = OrderedDict()
    b.last_user_message = {}
    b.bot_user_id = 99
    b.bot_username = "ash"
    b.user_error_cooldown = {}
    b.user_error_count = {}
    b.bot = SimpleNamespace(send_chat_action=AsyncMock())
    b._touch_user_activity = MagicMock()
    b._check_and_send_daily_cite = AsyncMock()
    b._writer_call = AsyncMock(return_value="result")
    b._register_handlers()
    return b


def callbacks(b):
    return {h.callback.__name__: h.callback for h in b.dp.message.handlers}


def test_public_catalogue_matches_registered_handlers_and_keeps_three_tones():
    b = bot()
    registered = callbacks(b)
    for lang in ("en", "ru"):
        items = descriptions(lang)
        assert len(items) == len({name for name, _ in items})
        assert all("cmd_" + name in registered for name, _ in items)
        assert all(1 <= len(description) <= 256 for _, description in items)
        help_text = command_help(lang)
        for name, _ in items:
            assert "/" + name in help_text
        for name in ("feedback", "roast", "praise", "lobster", "xyx", "idea", "prompt"):
            assert name in dict(items)
        assert "/methodichque" in help_text


def test_command_filters_are_unique_and_alias_is_registered():
    b = bot()
    found = {}
    for handler in b.dp.message.handlers:
        for filter_object in handler.filters or ():
            filt = filter_object.callback
            if isinstance(filt, tb.Command):
                for name in filt.commands:
                    assert name not in found
                    found[name] = handler.callback.__name__
    assert all(name in found for name, _ in descriptions("en"))
    assert found["methodique"] == found["methodichque"] == "cmd_methodique"
    assert found["cancel"] == "cmd_cancel"


def test_idea_inline_buttons_do_not_overlap_reply_keyboard():
    for lang in ("ru", "en"):
        main = tb.get_main_keyboard(lang)
        main_labels = {button.text for row in main.keyboard for button in row}
        assert tb.t(lang, "idea_reference_yes") not in main_labels
        assert tb.t(lang, "idea_reference_no") not in main_labels


@pytest.mark.asyncio
async def test_group_workflows_are_isolated_from_private_other_groups_and_users():
    b = bot()
    handlers = callbacks(b)
    await handlers["cmd_feedback"](msg("/feedback"))
    await handlers["handle_text"](msg("private draft"))
    await handlers["cmd_feedback"](msg("/feedback", chat=-1, chat_type="supergroup"))
    await handlers["cmd_feedback"](msg("/feedback", chat=-2, chat_type="supergroup"))
    await handlers["cmd_feedback"](msg("/feedback", user=8, chat=-1, chat_type="supergroup"))
    reply = SimpleNamespace(from_user=SimpleNamespace(id=99))
    await handlers["handle_text"](msg("group one draft", chat=-1, chat_type="supergroup", reply=reply))
    await handlers["handle_text"](msg("group two draft", chat=-2, chat_type="supergroup", reply=reply))
    await handlers["handle_text"](msg("other user draft", user=8, chat=-1, chat_type="supergroup", reply=reply))
    assert b.accumulated_text[7] == "private draft"
    assert dict(b.accumulated_text)[(-1, 7)] == "group one draft"
    assert dict(b.accumulated_text)[(-2, 7)] == "group two draft"
    assert dict(b.accumulated_text)[(-1, 8)] == "other user draft"
    await handlers["cmd_cancel"](msg("/cancel", chat=-1, chat_type="supergroup"))
    assert b.user_states[7] == "feedback"
    assert dict(b.user_states)[(-1, 7)] == "chat"
    assert dict(b.accumulated_text)[(-2, 7)] == "group two draft"


@pytest.mark.asyncio
async def test_done_discussion_and_cancel_keep_other_contexts():
    b = bot()
    handlers = callbacks(b)
    await handlers["cmd_feedback"](msg("/feedback"))
    await handlers["handle_text"](msg("a draft long enough for feedback"))
    await handlers["cmd_done"](msg("/done"))
    assert b.user_states[7] == "feedback_discuss"
    b.tool_discussions[8] = {"state": "feedback_discuss"}
    await handlers["cmd_done"](msg("/done"))
    assert b.user_states[7] == "chat"
    assert 7 not in b.accumulated_text and 7 not in b.tool_discussions
    assert 8 in b.tool_discussions
    b.user_states[7] = "summary_format"
    b.accumulated_text[7] = "draft"
    b.uploaded_documents[7] = {"text": "draft"}
    await handlers["cmd_cancel"](msg("/cancel"))
    assert b.user_states[7] == "chat"
    assert 7 not in b.accumulated_text and 7 not in b.uploaded_documents


@pytest.mark.asyncio
async def test_summary_rejects_short_text_before_format_choice():
    b = bot()
    b.user_states[7] = "summary_wait"
    b.accumulated_text[7] = "short"
    message = msg("/done")
    await b._handle_done(message)
    assert b.user_states[7] == "summary_wait"
    assert message.answer.await_args.args[0] == tb.t("en", "text_too_short", min=10)


@pytest.mark.asyncio
@pytest.mark.parametrize("command,state", [
    ("feedback", "feedback"), ("style", "style"), ("roast", "roast"),
    ("praise", "praise"), ("corrector", "corrector_wait"), ("editor", "editor_wait"),
    ("methodique", "methodique_wait"), ("methodichque", "methodique_wait"),
    ("summary", "summary_wait"),
])
async def test_short_inline_text_is_retained_for_continuation(command, state):
    b = bot()
    handler = callbacks(b)["cmd_methodique" if command == "methodichque" else "cmd_" + command]
    await handler(msg("/" + command + " x"))
    assert b.user_states[7] == state
    assert b.accumulated_text[7] == "x"


@pytest.mark.asyncio
async def test_cancel_clears_only_admin_update_draft(monkeypatch):
    b = bot()
    monkeypatch.setattr(tb, "ADMIN_ID", 7)
    b.admin_id = 7
    b.pending_update_changelogs = {"en": "draft", "ru": "черновик"}
    b.user_states[7] = "update_edit_ru"
    await callbacks(b)["cmd_cancel"](msg("/cancel"))
    assert not hasattr(b, "pending_update_changelogs")
    assert b.user_states[7] == "chat"


@pytest.mark.asyncio
async def test_scoped_writer_session_context_propagates_to_worker():
    b = bot()
    b._writer_call = tb.TelegramWriterBot._writer_call.__get__(b)
    b.llm_model, b.llm_api_key, b.llm_api_base, b.use_rag = "test", None, None, False
    calls = []

    class Writer:
        def __init__(self, **kwargs):
            self.language = kwargs["language"]

        def identity(self):
            calls.append(self)
            return "ok"

    original = tb.WriterBot
    tb.WriterBot = Writer
    try:
        await b._writer_call(7, "identity")
        token = current_group.set(-1)
        try:
            await b._writer_call(7, "identity")
            await b._writer_call(7, "identity")
        finally:
            current_group.reset(token)
        await b._writer_call(7, "identity")
    finally:
        tb.WriterBot = original
    assert calls[0] is calls[3]
    assert calls[1] is calls[2]
    assert calls[0] is not calls[1]


def test_command_args_require_the_actual_command_and_support_bot_suffix():
    b = bot()
    assert b._command_args(msg("/methodichque@ash text"), "methodique") == ""
    assert b._command_args(msg("/methodichque@ash text"), "methodichque") == "text"
    assert b._command_args(msg("/dev_feedback@ash hello"), "dev_feedback") == "hello"
