"""No constructors, embeddings, credentials or network: test the wire contract."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from writer_bot import WriterBot, Message
from i18n import t


def writer(lang='ru', secretary=False):
    wb = WriterBot.__new__(WriterBot)
    wb.language = lang
    wb.system_prompt = 'Secretary JSON' if secretary else wb._load_system_prompt()
    wb.summarize_history = not secretary
    wb.rag = SimpleNamespace(get_context_for_query=lambda text, **kw: 'sources:' + text)
    wb.use_rag = not secretary
    wb.history = []
    wb.session_summary = None
    wb.history_max_chars = 24000
    wb.messages_since_summary = 0
    wb.client = object()
    wb.model = wb.chat_model = 'offline'
    return wb


@pytest.mark.parametrize('lang', ['ru', 'en'])
@pytest.mark.parametrize('command', [None, 'feedback', 'methodique', 'summary'])
def test_prefix_and_role_order(lang, command):
    wb = writer(lang)
    wb.session_summary = 'snapshot'
    wb.history = [Message('user', 'old'), Message('assistant', 'reply')]
    extra = 'Fixed command' if command else None
    a = wb._build_messages('first', extra, command)
    b = wb._build_messages('second', extra, command)
    assert a[:-1] == b[:-1]
    assert a[-1] != b[-1]
    assert 'sources:first' in a[-1]['content']
    assert all('sources:' not in m['content'] for m in a[:-1])
    expected = ['system'] + (['system'] if command else []) + ['user']
    if command != 'summary':
        expected += ['user', 'assistant']
    assert [m['role'] for m in a] == expected + ['user']
    assert a[0]['content'].count(t(lang, 'clean_text_instr')) == 1


def test_chat_keeps_exact_wire_prefix(monkeypatch):
    import writer_bot
    from llm_completion import visible_text
    calls = []
    def complete(*args, **kwargs):
        calls.append(kwargs['messages'])
        return visible_text({'choices': [{'message': {'content': 'reply'}}]})
    monkeypatch.setitem(WriterBot.chat.__globals__, 'create_chat_completion', complete)
    wb = writer()
    wb.chat('first')
    wb.chat('second')
    assert calls[1][:len(calls[0])] == calls[0]
    assert wb.history[0].content == calls[0][-1]['content']


def test_rollover_blocks_and_summary_failure(monkeypatch):
    wb = writer()
    wb.history = [Message('user' if i % 2 == 0 else 'assistant', str(i)) for i in range(42)]
    removed = []
    monkeypatch.setattr(wb, '_maybe_summarize', lambda turns: removed.extend(turns))
    wb._compact_history()
    assert len(removed) == 20
    assert len(wb.history) == 22
    assert wb.history[0].content == '20'
    wb._compact_history()
    assert len(removed) == 20  # No summary rewrite until next rollover.


def test_secretary_has_no_coach_summary_or_clean_instruction(monkeypatch):
    wb = writer(secretary=True)
    wb.history_max_chars = 4
    wb.history = [Message('user', 'long input'), Message('assistant', '{"reply":"ok"}')]
    monkeypatch.setattr(wb, '_maybe_summarize', lambda _: pytest.fail('coach summary'))
    wb._compact_history()
    assert wb.history == []
    assert wb._build_messages('hi')[0]['content'] == 'Secretary JSON'


def test_methodique_sources_are_user_data(monkeypatch):
    import writer_bot
    from llm_completion import visible_text
    calls = []
    def complete(*args, **kwargs):
        calls.append(kwargs['messages'])
        return visible_text({'choices': [{'message': {'content': 'reply'}}]})
    monkeypatch.setitem(WriterBot.chat.__globals__, 'create_chat_completion', complete)
    wb = writer()
    wb.methodique('a story')
    assert [m['role'] for m in calls[0]] == ['system', 'system', 'user']
    assert 'sources:Stephen' in calls[0][-1]['content']
    assert 'sources:Stephen' not in calls[0][1]['content']


def test_client_timeout_configuration(monkeypatch):
    import openai
    seen = {}
    monkeypatch.setattr(openai, 'OpenAI', lambda **kwargs: seen.update(kwargs) or object())
    wb = writer()
    wb.api_key = 'fake-offline-only'
    wb.api_base = 'https://offline.invalid/v1'
    monkeypatch.delenv('LLM_TIMEOUT_SECONDS', raising=False)
    monkeypatch.delenv('LLM_MAX_RETRIES', raising=False)
    wb._init_llm()
    assert seen['timeout'] == 60 and seen['max_retries'] == 2
    monkeypatch.setenv('LLM_TIMEOUT_SECONDS', '7.5')
    monkeypatch.setenv('LLM_MAX_RETRIES', '0')
    wb._init_llm()
    assert seen['timeout'] == 7.5 and seen['max_retries'] == 0
