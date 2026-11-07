import asyncio
import sys
import time
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import telegram_bot as tb
import writer_bot as wb
from async_execution import AsyncExecution


def make_bot():
    bot = tb.TelegramWriterBot.__new__(tb.TelegramWriterBot)
    bot.execution = AsyncExecution()
    bot.idea_references = OrderedDict()
    bot.user_langs = {1: "ru", 2: "en"}
    bot.group_chats_enabled = True
    bot._writer_call = AsyncMock()
    return bot


def make_message(user=1, chat=-100, message_id=101):
    sent = SimpleNamespace(message_id=message_id, chat=SimpleNamespace(id=chat, type="group"),
                           text="idea", answer=AsyncMock(), edit_reply_markup=AsyncMock())
    request = SimpleNamespace(from_user=SimpleNamespace(id=user), chat=sent.chat,
                              answer=AsyncMock(return_value=sent))
    return request, sent


def choice(request, sent, index=0, user=1):
    markup = request.answer.await_args.kwargs["reply_markup"]
    data = markup.inline_keyboard[index][0].callback_data
    return SimpleNamespace(data=data, from_user=SimpleNamespace(id=user),
                           message=sent, answer=AsyncMock())


def test_idea_original_random_algorithm(monkeypatch):
    bot = wb.WriterBot.__new__(wb.WriterBot)
    bot.language = "ru"
    data = {"idea_genre": ["a", "b"], "idea_setting": ["c", "d"], "idea_conflict": ["e", "f"]}
    bot._load_prompts_data = lambda: data
    selected = []

    def choose(options):
        selected.append(options)
        return options[-1]

    monkeypatch.setattr(wb.random, "choice", choose)
    assert bot.generate_idea() == "b + d + f"
    assert selected == [data["idea_genre"], data["idea_setting"], data["idea_conflict"]]


@pytest.mark.asyncio
async def test_idea_and_no_choice_only_removes_keyboard():
    bot = make_bot()
    bot._writer_call.return_value = "жанр + место + конфликт"
    request, sent = make_message()
    await bot._handle_idea(request)
    assert request.answer.await_args.args == ("Идея:\n\nжанр + место + конфликт",)
    markup = request.answer.await_args.kwargs["reply_markup"]
    assert [[button.text for button in row] for row in markup.inline_keyboard] == [
        ["Нужен референс"], ["Референс не нужен"]]
    bot._writer_call.assert_awaited_once_with(1, "generate_idea")
    callback = choice(request, sent, 1)
    await bot._handle_idea_reference(callback)
    callback.answer.assert_awaited_once()
    sent.edit_reply_markup.assert_awaited_once_with(reply_markup=None)
    sent.answer.assert_not_awaited()
    bot._writer_call.assert_awaited_once()


@pytest.mark.asyncio
async def test_repeated_ideas_are_independent_and_language_is_snapshotted():
    bot = make_bot()
    bot._writer_call.side_effect = ["first combination", "second combination", "second story", "first story"]
    first, first_sent = make_message(message_id=101)
    second, second_sent = make_message(message_id=102)
    await bot._handle_idea(first)
    await bot._handle_idea(second)
    bot.user_langs[1] = "en"
    await bot._handle_idea_reference(choice(second, second_sent))
    await bot._handle_idea_reference(choice(first, first_sent))
    assert bot._writer_call.await_args_list[2].args == (1, "generate_idea_reference", "second combination", "ru")
    assert bot._writer_call.await_args_list[3].args == (1, "generate_idea_reference", "first combination", "ru")
    assert second_sent.answer.await_args.args == ("second story",)
    assert second_sent.answer.await_args.kwargs["reply_parameters"].message_id == 102
    assert first_sent.answer.await_args.args == ("first story",)


@pytest.mark.asyncio
async def test_parallel_users_and_duplicate_callback():
    bot = make_bot()
    entered, release = asyncio.Event(), asyncio.Event()

    async def writer(user, method, *args):
        if method == "generate_idea":
            return f"idea {user}"
        if user == 1:
            entered.set()
            await release.wait()
        return f"story {user}"

    bot._writer_call.side_effect = writer
    one, sent_one = make_message(user=1, message_id=101)
    two, sent_two = make_message(user=2, message_id=102)
    await asyncio.gather(bot._handle_idea(one), bot._handle_idea(two))
    assert two.answer.await_args.args[0] == "Idea:\n\nidea 2"
    assert len(two.answer.await_args.kwargs["reply_markup"].inline_keyboard) == 2
    first = asyncio.create_task(bot._handle_idea_reference(choice(one, sent_one)))
    await entered.wait()
    try:
        duplicate = choice(one, sent_one)
        await bot._handle_idea_reference(duplicate)
        await asyncio.wait_for(bot._handle_idea_reference(choice(two, sent_two, user=2)), 1)
        assert duplicate.answer.await_args.kwargs["text"] == tb.t("ru", "idea_reference_expired")
        assert sent_two.answer.await_args.args == ("story 2",)
    finally:
        release.set()
        await first
    calls = [c.args for c in bot._writer_call.await_args_list if c.args[1] == "generate_idea_reference"]
    assert calls == [(1, "generate_idea_reference", "idea 1", "ru"),
                     (2, "generate_idea_reference", "idea 2", "en")]


@pytest.mark.asyncio
async def test_group_other_user_and_wrong_message_cannot_consume_choice():
    bot = make_bot()
    bot._writer_call.side_effect = ["idea", "story"]
    request, sent = make_message()
    await bot._handle_idea(request)
    intruder = choice(request, sent, user=2)
    await bot._handle_idea_reference(intruder)
    assert intruder.answer.await_args.kwargs["text"] == tb.t("en", "idea_reference_not_yours")
    _, wrong = make_message(message_id=999)
    await bot._handle_idea_reference(choice(request, wrong))
    sent.edit_reply_markup.assert_not_awaited()
    await bot._handle_idea_reference(choice(request, sent))
    assert sent.answer.await_args.args == ("story",)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["Error: provider failure", "", RuntimeError("secret")])
async def test_model_error_is_localized_and_never_retried(failure):
    bot = make_bot()
    bot._writer_call.side_effect = ["idea", failure]
    request, sent = make_message()
    await bot._handle_idea(request)
    await bot._handle_idea_reference(choice(request, sent))
    assert sent.answer.await_args.args == (tb.t("ru", "error_llm"),)
    await bot._handle_idea_reference(choice(request, sent))
    assert bot._writer_call.await_count == 2


@pytest.mark.asyncio
async def test_expired_and_telegram_api_errors():
    bot = make_bot()
    bot._writer_call.side_effect = ["idea", "another", "story"]
    request, sent = make_message()
    await bot._handle_idea(request)
    next(iter(bot.idea_references.values()))["expires"] = time.monotonic() - 1
    callback = choice(request, sent)
    callback.answer.side_effect = RuntimeError("expired callback")
    sent.edit_reply_markup.side_effect = RuntimeError("deleted")
    await bot._handle_idea_reference(callback)
    sent.answer.assert_not_awaited()
    request2, sent2 = make_message(message_id=102)
    await bot._handle_idea(request2)
    sent2.edit_reply_markup.side_effect = RuntimeError("deleted")
    sent2.answer.side_effect = RuntimeError("send failed")
    await bot._handle_idea_reference(choice(request2, sent2))
    await bot._handle_idea_reference(choice(request2, sent2))
    assert bot._writer_call.await_count == 3


@pytest.mark.asyncio
async def test_unknown_malformed_and_disabled_group_callbacks():
    bot = make_bot()
    bot._writer_call.return_value = "idea"
    request, sent = make_message()
    await bot._handle_idea(request)
    bot.group_chats_enabled = False
    disabled = choice(request, sent)
    await bot._handle_idea_reference(disabled)
    sent.edit_reply_markup.assert_not_awaited()
    assert bot._writer_call.await_count == 1
    bot.group_chats_enabled = True
    for data, message in [("idea_ref:yes:unknown", sent), ("idea_ref:bad", sent),
                          ("idea_ref:yes:unknown", None)]:
        callback = SimpleNamespace(data=data, message=message,
                                   from_user=SimpleNamespace(id=1), answer=AsyncMock())
        await bot._handle_idea_reference(callback)
        assert callback.answer.await_args.kwargs["text"] == tb.t("ru", "idea_reference_expired")


@pytest.mark.asyncio
async def test_command_menu_and_callback_handlers_registered():
    bot = make_bot()
    bot.dp = tb.Dispatcher()
    bot._handle_idea = AsyncMock()
    bot._handle_idea_reference = AsyncMock()
    bot._register_handlers()
    handlers = {handler.callback.__name__: handler.callback for handler in bot.dp.message.handlers}
    request, sent = make_message()
    await handlers["cmd_idea"](request)
    await handlers["btn_idea"](request)
    assert bot._handle_idea.await_count == 2
    callback_handlers = {handler.callback.__name__: handler.callback
                         for handler in bot.dp.callback_query.handlers}
    callback = SimpleNamespace(data="idea_ref:yes:token", message=sent,
                               from_user=SimpleNamespace(id=1), answer=AsyncMock())
    await callback_handlers["idea_reference_callback"](callback)
    bot._handle_idea_reference.assert_awaited_once_with(callback)


def test_reference_model_uses_only_given_combination_and_language(monkeypatch):
    bot = wb.WriterBot.__new__(wb.WriterBot)
    bot.client, bot.model = object(), "model"
    completion = MagicMock(return_value=object())
    monkeypatch.setattr(wb, "create_chat_completion", completion)
    monkeypatch.setattr(wb, "visible_completion_text", lambda response: "story")
    assert bot.generate_idea_reference("genre + setting + conflict", "ru") == "story"
    messages = completion.call_args.kwargs["messages"]
    assert messages[1] == {"role": "user", "content": "genre + setting + conflict"}
    assert "Russian" in messages[0]["content"]
    assert "existing book or film" in messages[0]["content"]


def test_reference_model_failure_is_safe(monkeypatch):
    bot = wb.WriterBot.__new__(wb.WriterBot)
    bot.client, bot.model = object(), "model"
    monkeypatch.setattr(wb, "create_chat_completion", MagicMock(side_effect=RuntimeError("secret")))
    assert bot.generate_idea_reference("combination", "en") == "Error: LLM request failed."
