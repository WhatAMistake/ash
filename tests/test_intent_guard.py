import asyncio
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from intent_guard import IntentGuard, parse_response, OPERATIONS


def response(content='Y', finish='stop', **fields):
    return NS(choices=[NS(finish_reason=finish, message=NS(content=content, **fields))])


class FakeClient:
    def __init__(self, result=None, error=None, delay=0):
        self.result = result if result is not None else response()
        self.error, self.delay, self.calls = error, delay, []
        self.chat = NS(completions=NS(create=self.create))

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return self.result


@pytest.fixture(autouse=True)
def forbid_real_sdk(monkeypatch):
    # Even an accidental SDK use cannot perform an HTTP completion.
    from openai.resources.chat.completions import AsyncCompletions, Completions
    def forbidden(*args, **kwargs):
        raise AssertionError('real SDK calls forbidden')
    monkeypatch.setattr(AsyncCompletions, 'create', forbidden)
    monkeypatch.setattr(Completions, 'create', forbidden)


@pytest.mark.parametrize('value,expected', [('Y', True), ('N', False), (' Y\n', True)])
def test_parse(value, expected):
    assert parse_response(response(value)) is expected


@pytest.mark.parametrize('value', ['', None, 'Yes', 'y', 'Y N', '```Y```', '{"ok":true}', '<think>x</think>Y', ['Y']])
def test_strict_parse(value):
    with pytest.raises(ValueError):
        parse_response(response(value))


@pytest.mark.parametrize('finish', ['length', 'content_filter', None])
def test_incomplete(finish):
    with pytest.raises(ValueError):
        parse_response(response('Y', finish))


@pytest.mark.asyncio
@pytest.mark.parametrize('provider', ['comet', 'openrouter'])
async def test_payload(provider):
    fake = FakeClient()
    d = await IntentGuard(fake, provider=provider).check('Мой рассказ', operation='feedback')
    assert d.ok and d.reason == 'model' and len(fake.calls) == 1
    p = fake.calls[0]
    assert p['max_tokens'] == 3 and p['temperature'] == 0 and not p['stream']
    assert [m['role'] for m in p['messages']] == ['system', 'user']
    assert json.loads(p['messages'][1]['content']) == {'operation': 'feedback', 'text': 'Мой рассказ'}
    if provider == 'comet':
        assert 'reasoning_effort' not in p
        assert p['extra_body'] == {'thinking': {'type': 'disabled'}}
    else:
        assert p['extra_body'] == {'reasoning': {'effort': 'none'}}
        assert 'reasoning_effort' not in p


@pytest.mark.asyncio
@pytest.mark.parametrize('fail_open', [False, True])
@pytest.mark.parametrize('kind', ['error', 'timeout', 'malformed', 'empty_choices', 'tool'])
async def test_fail_modes_not_cached(fail_open, kind):
    fake = FakeClient(error=RuntimeError('synthetic') if kind == 'error' else None,
        delay=.03 if kind == 'timeout' else 0,
        result={'malformed': response('Yes'), 'empty_choices': NS(choices=[]),
                'tool': response('Y', tool_calls=[{}])}.get(kind))
    guard = IntentGuard(fake, fail_open=fail_open, timeout=.005)
    for _ in range(2):
        d = await guard.check('text')
        assert d.ok is fail_open
        assert d.reason == ('timeout' if kind == 'timeout' else 'unavailable')
    assert len(fake.calls) == 2


@pytest.mark.asyncio
async def test_cancel_propagates():
    guard = IntentGuard(FakeClient(error=asyncio.CancelledError()))
    with pytest.raises(asyncio.CancelledError):
        await guard.check('text')


@pytest.mark.asyncio
async def test_cache_normalization_scope_lru():
    fake = FakeClient(response('N'))
    guard = IntentGuard(fake, cache_size=2)
    assert not (await guard.check(' e\u0301\r\nx ')).ok
    assert (await guard.check('é\nx')).reason == 'cache'
    await guard.check('b')
    await guard.check('é\nx')  # Promote A, so B is evicted next.
    await guard.check('c')
    assert (await guard.check('é\nx')).reason == 'cache'
    assert (await guard.check('b')).reason == 'model'
    assert (await guard.check('b', operation='feedback')).reason == 'model'
    assert all(isinstance(k, bytes) for k in guard._cache)


@pytest.mark.asyncio
async def test_expired_and_zero_cache():
    for kwargs in [{'cache_ttl': 0}, {'cache_size': 0}]:
        fake = FakeClient()
        g = IntentGuard(fake, **kwargs)
        await g.check('a')
        await g.check('a')
        assert len(fake.calls) == 2


@pytest.mark.asyncio
async def test_disabled_empty_oversize(monkeypatch):
    monkeypatch.setenv('INTENT_GUARD_ENABLED', '0')
    assert (await IntentGuard.from_env().check('anything')).reason == 'disabled'
    fake = FakeClient()
    g = IntentGuard(fake, max_chars=3)
    assert not (await g.check(' ')).ok
    assert (await g.check('1234')).reason == 'oversize'
    assert not fake.calls
    with pytest.raises(ValueError):
        await g.check('x', operation='injected')


@pytest.mark.asyncio
async def test_old_sdk_local_fallback():
    class OldSDK(FakeClient):
        async def create(self, **kwargs):
            self.calls.append(kwargs)
            if 'reasoning_effort' in kwargs:
                raise TypeError("unexpected keyword argument 'reasoning_effort'")
            return response()
    fake = OldSDK()
    assert (await IntentGuard(fake).check('text')).ok
    assert len(fake.calls) == 1  # Adapter never sends the unsupported SDK kwarg.
    assert fake.calls[0]['extra_body']['thinking'] == {'type': 'disabled'}


@pytest.mark.asyncio
async def test_queue_deadline():
    g = IntentGuard(FakeClient(delay=.1), timeout=.01, concurrency=1)
    decisions = await asyncio.gather(*(g.check(str(i)) for i in range(3)))
    assert all(d.reason == 'timeout' for d in decisions)


CASES = json.loads((ROOT / 'tests/intent_guard_cases.json').read_text())


def test_corpus_balance():
    assert len(CASES) == 40
    assert sum(c['expected'] == 'OK' for c in CASES) == 20
    assert len({c['id'] for c in CASES}) == 40
    assert {c['lang'] for c in CASES} == {'ru', 'en'}


@pytest.mark.asyncio
@pytest.mark.parametrize('case', CASES, ids=lambda c: c['id'])
async def test_corpus_transport_only(case):
    # The fake encodes the expected label: this is NOT classifier accuracy evidence.
    assert case['operation'] in OPERATIONS
    fake = FakeClient(response('Y' if case['expected'] == 'OK' else 'N'))
    d = await IntentGuard(fake).check(case['text'], operation=case['operation'])
    assert d.ok == (case['expected'] == 'OK')
    assert json.loads(fake.calls[0]['messages'][1]['content'])['text'] == case['text']


def test_env_explicit_provider_key(monkeypatch):
    import openai
    seen = {}
    def constructor(**kwargs):
        seen.update(kwargs)
        return FakeClient()
    monkeypatch.setattr(openai, 'AsyncOpenAI', constructor)
    monkeypatch.setenv('INTENT_GUARD_ENABLED', '1')
    monkeypatch.setenv('INTENT_GUARD_FAIL_MODE', 'closed')
    monkeypatch.setenv('INTENT_GUARD_API_BASE', 'https://openrouter.ai/api/v1')
    monkeypatch.setenv('INTENT_GUARD_API_KEY', 'fake-offline-only')
    monkeypatch.delenv('INTENT_GUARD_MODEL', raising=False)
    guard = IntentGuard.from_env()
    assert guard.provider == 'openrouter'
    assert guard.model == 'deepseek-flash'
    assert seen['max_retries'] == 0 and seen['timeout'] == 6
    assert seen['api_key'] == 'fake-offline-only'


def test_env_never_reuses_main_key(monkeypatch):
    monkeypatch.setenv('INTENT_GUARD_ENABLED', '1')
    monkeypatch.setenv('INTENT_GUARD_FAIL_MODE', 'closed')
    monkeypatch.setenv('INTENT_GUARD_API_BASE', 'https://openrouter.ai/api/v1')
    monkeypatch.delenv('INTENT_GUARD_API_KEY', raising=False)
    monkeypatch.setenv('OPENAI_API_KEY', 'fake-main-key-must-not-leak')
    with pytest.raises(ValueError, match='missing guard API key'):
        IntentGuard.from_env()


def test_invalid_fail_mode(monkeypatch):
    monkeypatch.setenv('INTENT_GUARD_FAIL_MODE', 'typo')
    with pytest.raises(ValueError, match='invalid fail mode'):
        IntentGuard.from_env()


def test_env_inherits_main_endpoint_and_key(monkeypatch):
    import openai
    seen = {}
    def constructor(**kwargs):
        seen.update(kwargs)
        return FakeClient()
    monkeypatch.setattr(openai, 'AsyncOpenAI', constructor)
    monkeypatch.setenv('INTENT_GUARD_ENABLED', '1')
    monkeypatch.setenv('INTENT_GUARD_FAIL_MODE', 'closed')
    monkeypatch.delenv('INTENT_GUARD_API_KEY', raising=False)
    monkeypatch.delenv('INTENT_GUARD_API_BASE', raising=False)
    IntentGuard.from_env(api_base='https://api.cometapi.com/v1', api_key='fake-main-only')
    assert seen['api_key'] == 'fake-main-only'
    assert seen['base_url'] == 'https://api.cometapi.com/v1'


def test_env_rejects_fail_open(monkeypatch):
    monkeypatch.setenv('INTENT_GUARD_FAIL_MODE', 'open')
    with pytest.raises(ValueError):
        IntentGuard.from_env()
