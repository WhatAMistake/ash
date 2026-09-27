"""Exercise actual handler bodies without importing Telegram startup/configuration."""
import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from i18n import t
from intent_guard import Decision

SOURCE = Path(__file__).resolve().parents[1] / 'src' / 'telegram_bot.py'
TREE = ast.parse(SOURCE.read_text())
BOT = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == 'TelegramWriterBot')


def handler_class(*names):
    body = [n for n in BOT.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + body, type_ignores=[])
    namespace = {'asyncio': asyncio, 't': t, 'DEFAULT_LANG': 'ru'}
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), 'exec'), namespace)
    return type('Handlers', (), {name: namespace[name] for name in names})


COMMANDS = [('block', '_handle_block'), ('develop', '_handle_develop'),
    ('character', '_handle_character'), ('dialogue', '_handle_dialogue'),
    ('feedback', '_handle_feedback_cmd'), ('style', '_handle_style_cmd'),
    ('roast', '_handle_roast'), ('praise', '_handle_praise'),
    ('corrector', '_handle_corrector'), ('editor', '_handle_editor'),
    ('methodique', '_handle_methodique')]


@pytest.mark.asyncio
@pytest.mark.parametrize('operation,name', COMMANDS)
async def test_rejected_command_never_reaches_writer(operation, name):
    bot = handler_class(name, '_guard_allows')()
    bot.user_langs = {1: 'ru'}
    bot.user_states = {1: 'chat'}
    bot.accumulated_text = {1: 'keep this'}
    bot._touch_user_activity = lambda _: None
    bot._command_args = lambda *args: 'sample literary text ' * 4
    bot._get_writer_bot = lambda _: NS()
    bot._run_writer = AsyncMock(side_effect=AssertionError('writer called'))
    bot.bot = NS(send_chat_action=AsyncMock())
    bot.intent_guard = NS(check=AsyncMock(return_value=Decision(False, 'unavailable')))
    message = NS(from_user=NS(id=1), chat=NS(id=1), answer=AsyncMock(), text='/'+operation)
    await getattr(bot, name)(message)
    assert bot.intent_guard.check.call_args.kwargs['operation'] == operation
    assert bot.accumulated_text[1] == 'keep this'
    bot._run_writer.assert_not_called()
    message.answer.assert_awaited_once_with(t('ru', 'guard_retry'))


@pytest.mark.asyncio
async def test_writer_thread_does_not_block_loop_and_serializes():
    import threading
    bot = handler_class('_run_writer')()
    entered = threading.Event()
    release = threading.Event()
    class Writer:
        def work(self):
            entered.set()
            release.wait(2)
            return 42
    wb = Writer()
    task = asyncio.create_task(bot._run_writer(wb.work))
    try:
        for _ in range(100):
            if entered.is_set():
                break
            await asyncio.sleep(.005)
        assert entered.is_set()
        assert not task.done()  # Event loop progressed while the worker was blocked.
        task.cancel()
        await asyncio.sleep(.01)
        assert wb._async_lock.locked()  # Cancellation cannot release a running worker.
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not wb._async_lock.locked()


def test_guard_routing_boundaries():
    guarded = {n.name for n in BOT.body if isinstance(n, ast.AsyncFunctionDef)
               and any(isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                       and c.func.attr == '_guard_allows' for c in ast.walk(n))}
    assert {'_handle_message', '_handle_discussion', '_handle_done', '_handle_summary_choice'} <= guarded
    assert not any('secretary' in name or 'business' in name or 'admin' in name for name in guarded)


@pytest.mark.asyncio
@pytest.mark.parametrize('operation,name', COMMANDS[4:])
async def test_document_guard_preserves_pending_text(operation, name):
    bot = handler_class(name, '_guard_allows')()
    bot.user_langs = {1: 'en'}
    bot.user_states = {1: 'document_wait'}
    document = 'fiction manuscript ' * 8
    bot.accumulated_text = {1: document}
    bot._touch_user_activity = lambda _: None
    bot._command_args = lambda *args: ''
    bot._get_writer_bot = lambda _: pytest.fail('RAG initialized before guard')
    bot.bot = NS(send_chat_action=AsyncMock())
    bot.intent_guard = NS(check=AsyncMock(return_value=Decision(False, 'model')))
    message = NS(from_user=NS(id=1), chat=NS(id=1), answer=AsyncMock(), text='/'+operation)
    await getattr(bot, name)(message)
    bot.intent_guard.check.assert_awaited_once_with(document, operation=operation)
    assert bot.accumulated_text[1] == document
    assert bot.user_states[1] == 'document_wait'


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['done', 'summary'])
async def test_accumulated_guard_preserves_state(mode):
    name = '_handle_done' if mode == 'done' else '_handle_summary_choice'
    bot = handler_class(name, '_guard_allows')()
    bot.user_langs = {1: 'ru'}
    state = 'feedback' if mode == 'done' else 'summary_format'
    bot.user_states = {1: state}
    bot.SUMMARY_FORMAT_STATE = 'summary_format'
    bot.DISCUSSION_STATES = {}
    bot.STATE_CONFIG = {'feedback': (20, 'feedback_on_text')}
    bot.accumulated_text = {1: 'fiction manuscript ' * 8}
    bot._get_writer_bot = lambda _: pytest.fail('writer initialized before guard')
    bot.bot = NS(send_chat_action=AsyncMock())
    bot.intent_guard = NS(check=AsyncMock(return_value=Decision(False, 'timeout')))
    message = NS(from_user=NS(id=1), chat=NS(id=1), answer=AsyncMock())
    args = [] if mode == 'done' else [t('ru', 'btn_summary_sentence')]
    result = await getattr(bot, name)(message, *args)
    if mode == 'summary':
        assert result is True
    assert bot.user_states[1] == state
    assert bot.accumulated_text[1]
    bot.intent_guard.check.assert_awaited_once()
