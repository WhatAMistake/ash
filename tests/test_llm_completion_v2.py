"""Offline only: fake clients and SDK MockTransport; never load Ash/.env."""
import copy
import importlib.util
import json
import sys
import socket
from pathlib import Path
from types import SimpleNamespace as NS

try:
    import httpx
except ImportError:  # openai 3.x uses httpx2; both are SDK dependencies.
    import httpx2 as httpx
import openai
import pytest

# Load the delivered module, not Ash's original, even under a combined test run.
MODULE = Path(__file__).resolve().parents[1] / "src" / "llm_completion.py"
SPEC = importlib.util.spec_from_file_location("ash_llm_completion_v2", MODULE)
llm = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = llm
SPEC.loader.exec_module(llm)

URLS = {
    "openrouter": "https://openrouter.ai/api/v1",
    "deepseek": "https://api.deepseek.com",
    "cometapi": "https://api.cometapi.com/v1",
    "openai": "https://api.openai.com/v1",
    "generic": "https://local.invalid/v1",
}


def response(message=None):
    return {"choices": [{"index": 0, "message": message or {"content": "OK"}}]}


def bad(message="Unknown parameter: 'reasoning'", param=None, code=None):
    body = {"error": {"message": message, "param": param, "code": code}}
    return openai.BadRequestError(message, response=httpx.Response(
        400, json=body, request=httpx.Request("POST", "https://offline.invalid/chat/completions")), body=body)


class FakeClient:
    def __init__(self, provider="openrouter", results=None):
        self.base_url = URLS[provider]
        self.calls = []
        self.results = iter(results or [response()])
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        assert "reasoning_effort" not in kwargs
        assert "thinking" not in kwargs
        assert "reasoning" not in kwargs
        result = next(self.results)
        if isinstance(result, Exception):
            raise result
        return result


def payload(provider, policy, **kwargs):
    return llm.completion_kwargs(base_url=URLS[provider], reasoning_policy=policy,
                                 model="deepseek-v4-pro", messages=[], **kwargs)


@pytest.mark.parametrize("provider,url", list(URLS.items()) + [
    ("deepseek", "https://api.deepseek.com/v1/"),
    ("openrouter", "HTTPS://OPENROUTER.AI:443/api/v1/"),
    ("cometapi", "https://api.cometapi.com./v1"),
    ("generic", "https://openrouter.ai.evil.test/v1"),
    ("generic", "https://evil.test/openrouter.ai"),
    ("generic", "https://openrouter.ai@evil.test/v1"),
    ("generic", "https://proxy.openrouter.ai/v1"),
    ("generic", "openrouter.ai"),
    ("generic", "https://[broken"),
    ("generic", None),
])
def test_detect_provider(provider, url):
    assert llm.detect_provider(url) == provider


@pytest.mark.parametrize("provider,policy,expected", [
    *[("openrouter", p, {"reasoning": {"effort": "none" if p == "off" else p}})
      for p in ("off", "low", "medium", "high")],
    *[("deepseek", p, {"reasoning_effort": v}) for p, v in
      [("off", "none"), ("low", "low"), ("medium", "high"), ("high", "high")]],
    ("cometapi", "off", {"thinking": {"type": "disabled"}}),
    *[("cometapi", p, {"thinking": {"type": "enabled"}, "reasoning_effort": v})
      for p, v in [("low", "low"), ("medium", "high"), ("high", "high")]],
    *[(provider, "passthrough", {}) for provider in URLS],
    *[("generic", p, {}) for p in ("off", "low", "medium", "high")],
    *[("openai", p, {}) for p in ("off", "low", "medium", "high")],
])
def test_provider_payload(provider, policy, expected):
    client = FakeClient(provider)
    result = llm.create_chat_completion(client, reasoning_policy=policy,
                                      model="deepseek-v4-pro", messages=[], temperature=0.7)
    assert result.text == "OK"
    assert result.provider == provider
    assert not result.retried_without_reasoning
    assert client.calls[0].get("extra_body", {}) == expected
    assert client.calls[0]["temperature"] == 0.7


@pytest.mark.parametrize("provider", ["openai", "cometapi"])
@pytest.mark.parametrize("policy", ["off", "low", "medium", "high"])
def test_openai_style_requires_capability(provider, policy):
    kwargs = dict(base_url=URLS[provider], model="some-model", reasoning_policy=policy)
    assert "extra_body" not in llm.completion_kwargs(**kwargs)
    expected = "none" if policy == "off" else policy
    assert llm.completion_kwargs(**kwargs, supports_reasoning_effort=True)["extra_body"] == {"reasoning_effort": expected}


def test_explicit_policy_overrides_stale_controls_without_mutation():
    extra = {"thinking": {"type": "disabled"}, "reasoning_effort": "none",
             "reasoning": {"exclude": True}, "enable_thinking": False,
             "chat_template_kwargs": {"enable_thinking": False, "reasoning_effort": "none", "other": 7},
             "provider": {"order": ["test"]}, "other": 1}
    before = copy.deepcopy(extra)
    result = payload("openrouter", "high", extra_body=extra, reasoning_effort="none")
    assert extra == before
    assert result["extra_body"] == {
        "reasoning": {"effort": "high"}, "chat_template_kwargs": {"other": 7},
        "provider": {"order": ["test"]}, "other": 1}
    assert "reasoning_effort" not in result


@pytest.mark.parametrize("policy", ["passthrough", "off", "high"])
def test_generic_never_sends_reasoning_controls(policy):
    out = payload("generic", policy, reasoning_effort="high", supports_reasoning_effort=True,
                  extra_body={"reasoning": {"effort": "high"}, "thinking": {"type": "enabled"},
                              "chat_template_kwargs": {"enable_thinking": True}, "other": 7})
    assert out["extra_body"] == {"other": 7}


@pytest.mark.parametrize("source,target", [("off", "none"), ("max", "xhigh"), ("adaptive", "medium"), ("low", "low")])
def test_passthrough_openrouter_normalization(source, target):
    out = payload("openrouter", "passthrough", reasoning_effort=source)
    assert out["extra_body"] == {"reasoning": {"effort": target}}


def test_passthrough_preserves_explicit_settings_and_sdk_safety():
    extra = {"reasoning": {"exclude": True}, "other": 9}
    assert payload("openrouter", "passthrough", extra_body=extra)["extra_body"] == extra
    assert payload("deepseek", "passthrough", reasoning_effort="max")["extra_body"] == {"reasoning_effort": "max"}


@pytest.mark.parametrize("extra", [
    {"thinking": {"type": "disabled"}, "reasoning_effort": "high"},
    {"thinking": {"type": "enabled"}, "reasoning_effort": "none"},
    {"reasoning": {"effort": "high"}, "thinking": {"type": "disabled"}},
    {"reasoning": {"enabled": False, "effort": "high"}},
    {"reasoning": {"enabled": False, "max_tokens": 100}},
    {"chat_template_kwargs": {"enable_thinking": False, "reasoning_effort": "high"}},
    {"reasoning": {"enabled": True, "effort": "none"}},
    {"reasoning": {"effort": "high", "max_tokens": 100}},
    {"reasoning_effort": "high", "chat_template_kwargs": {"enable_thinking": False}},
])
def test_passthrough_rejects_ambiguous_controls(extra):
    with pytest.raises(ValueError):
        payload("deepseek", "passthrough", extra_body=extra)


def test_conflicting_duplicate_effort_rejected():
    with pytest.raises(ValueError):
        payload("deepseek", "passthrough", reasoning_effort="high", extra_body={"reasoning_effort": "none"})
    with pytest.raises(ValueError):
        payload("openrouter", "passthrough", reasoning_effort="high", extra_body={"reasoning": {"effort": "none"}})


def test_policy_validation():
    with pytest.raises(ValueError):
        payload("openrouter", "max")
    with pytest.raises(TypeError):
        payload("deepseek", "off", extra_body=[])


@pytest.mark.parametrize("factory", [lambda d: d, lambda d: NS(**d),
                                      lambda d: NS(content=d["content"], model_extra={k: v for k, v in d.items() if k != "content"})])
def test_reasoning_dedup_and_details(factory):
    details = [{"type": "reasoning.encrypted", "id": "abc", "data": "opaque"},
               {"type": "reasoning.text", "text": "duplicate"}]
    msg = factory({"content": "<think>tag-only secret</think><final>Answer</final>",
                   "reasoning_content": "first", "reasoning": "duplicate",
                   "reasoning_text": "third", "reasoning_details": details})
    raw = response(msg)
    result = llm.visible_text(raw)
    assert result.text == "Answer"
    assert result.reasoning.text == "first"
    assert result.reasoning.details == details
    assert result.raw is raw
    assert llm.visible_completion_text(result) == "Answer"
    assert llm.extract_completion_text(result) == "Answer"
    assert llm.visible_text(result) is result


@pytest.mark.parametrize("fields,expected", [
    ({"reasoning_content": "", "reasoning": "second", "reasoning_text": "third"}, "second"),
    ({"reasoning_content": None, "reasoning": None, "reasoning_text": "third"}, "third"),
    ({"reasoning": "", "reasoning_text": ""}, ""),
    ({"reasoning_content": "same", "reasoning": "same", "reasoning_text": "same"}, "same"),
])
def test_first_nonempty_reasoning(fields, expected):
    assert llm.visible_text(response({"content": "ok", **fields})).reasoning.text == expected


def test_sdk_model_extras():
    from openai.types.chat import ChatCompletion
    raw = ChatCompletion(id="fake", object="chat.completion", created=0, model="test", choices=[
        {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "visible",
          "reasoning_content": "hidden", "reasoning": "hidden", "reasoning_details": [{"data": "opaque"}]}}
    ])
    result = llm.visible_text(raw)
    assert result.text == "visible"
    assert result.reasoning.text == "hidden"
    assert result.reasoning.details == [{"data": "opaque"}]


def test_content_allowlist_null_and_multiple_choices():
    content = [{"type": "thinking", "text": "SECRET"}, {"type": "reasoning", "text": "SECRET"},
               {"type": "unknown", "text": "SECRET"}, {"type": "text", "text": "A"},
               NS(type="output_text", text="B"), "C", {"type": "text", "text": {"hidden": 1}}]
    assert llm.visible_completion_text(response({"content": content})) == "ABC"
    assert llm.visible_completion_text(response({"content": None, "reasoning": "hidden"})) == ""
    assert llm.visible_completion_text({"choices": []}) == ""
    assert llm.visible_completion_text(None) == ""
    assert llm.visible_completion_text({"choices": [{"index": 1, "message": {"content": "other"}}]}) == ""


@pytest.mark.parametrize("text,expected", [
    ("before<think>secret</think>after", "beforeafter"),
    ("<THINKING a='b'>secret</THINKING><final>answer</final>", "answer"),
    ("<thought>secret</thought>ok", "ok"),
    ("<antthinking>secret</antthinking>ok", "ok"),
    ("<reasoning>secret</reasoning>ok", "ok"),
    ("[think]secret[/think]ok", "ok"),
    ("[thinking]secret[/thinking]ok", "ok"),
    ("◁think▷secret◁/think▷ok", "ok"),
    ("< think >secret< / think >ok", "ok"),
    ("</think>orphan close", "orphan close"),
    ("safe<think>unclosed secret", "safe"),
    ("safe[think]unclosed secret", "safe"),
    ("safe◁think▷unclosed secret", "safe"),
    ("<final>Visible</final>", "Visible"),
    ("<final>Visible", "Visible"),
    ("`<think>literal</think>` <think>secret</think>", "`<think>literal</think>`"),
    ("``<think>literal</think>``<think>secret</think>", "``<think>literal</think>``"),
    ("```py\n<think>literal</think>\n```\n<think>secret</think>ok", "```py\n<think>literal</think>\n```\nok"),
    ("~~~txt\n<think>literal</think>\n~~~\n<think>secret</think>ok", "~~~txt\n<think>literal</think>\n~~~\nok"),
    ("```\n<think>literal unclosed", "```\n<think>literal unclosed"),
    ("`<final>literal</final>` <final>answer</final>", "`<final>literal</final>` answer"),
    ("`[think]literal[/think]` [think]secret[/think]", "`[think]literal[/think]`"),
    ("`◁think▷literal◁/think▷` ◁think▷secret◁/think▷", "`◁think▷literal◁/think▷`"),
    ("<think>secret `</think>` still secret</think>ok", "ok"),
    ("plain <b>HTML</b> <thinkingcap> and a < b", "plain <b>HTML</b> <thinkingcap> and a < b"),
    ("  no tags  ", "  no tags  "),
])
def test_strip_tags(text, expected):
    assert llm.strip_reasoning_tags(text) == expected


def test_preserve_and_trim():
    assert llm.strip_reasoning_tags("before<think>tail", mode="preserve") == "beforetail"
    assert llm.strip_reasoning_tags("<think>hidden</think>ok", mode="preserve") == "ok"
    assert llm.strip_reasoning_tags(" <final>x</final> ", trim="none") == " x "
    assert llm.strip_reasoning_tags(" <final>x</final> ", trim="start") == "x "
    with pytest.raises(ValueError):
        llm.strip_reasoning_tags("", mode="invalid")


@pytest.mark.parametrize("message,param,code", [
    ("Unknown parameter: 'reasoning'", None, None),
    ("Unrecognized request argument supplied: reasoning_effort", None, None),
    ("Unsupported parameter: 'thinking'", "thinking", None),
    ("Parameter reasoning.effort is not supported", None, None),
    ("Rejected field", "reasoning", "unknown_parameter"),
    ("Extra inputs are not permitted: thinking", None, None),
])
def test_fallback_once(message, param, code, caplog):
    original = {"provider": {"order": ["test"]}, "other": 1}
    client = FakeClient(results=[bad(message, param, code), response()])
    result = llm.create_chat_completion(client, model="test", messages=[], reasoning_policy="high", extra_body=original)
    assert result.retried_without_reasoning is True
    assert result.provider == "openrouter"
    assert len(client.calls) == 2
    assert client.calls[0]["extra_body"]["reasoning"] == {"effort": "high"}
    assert client.calls[1]["extra_body"] == original
    assert original == {"provider": {"order": ["test"]}, "other": 1}
    assert "retrying once" in caplog.text
    assert message not in caplog.text


def test_fallback_removes_all_controls_not_other_data():
    client = FakeClient("cometapi", [bad("Unknown parameter: thinking"), response()])
    result = llm.create_chat_completion(client, model="deepseek-v4-pro", messages=[], reasoning_policy="high")
    assert result.retried_without_reasoning
    assert "extra_body" not in client.calls[1]


@pytest.mark.parametrize("exc", [
    bad("Unknown parameter: 'temperature'"),
    bad("Unknown parameter: 'temperature' for reasoning models"),
    bad("Unknown parameter: 'temperature' for reasoning models", param="temperature"),
    bad("Invalid value for reasoning_effort: medium"),
    bad("Unsupported value 'none' for parameter 'reasoning_effort'"),
    bad("reasoning_effort must be high or max"),
    bad("Insufficient credits"),
    TypeError("create() got an unexpected keyword argument 'reasoning_effort'"),
    RuntimeError("connection failed"),
])
def test_unrelated_errors_propagate(exc):
    client = FakeClient(results=[exc])
    with pytest.raises(type(exc)) as caught:
        llm.create_chat_completion(client, model="test", messages=[], reasoning_policy="high")
    assert caught.value is exc
    assert len(client.calls) == 1


def test_no_retry_if_no_controls_and_no_third_attempt():
    client = FakeClient(results=[bad()])
    with pytest.raises(openai.BadRequestError):
        llm.create_chat_completion(client, model="test", messages=[])
    assert len(client.calls) == 1
    client = FakeClient(results=[bad(), bad()])
    with pytest.raises(openai.BadRequestError):
        llm.create_chat_completion(client, model="test", messages=[], reasoning_policy="high")
    assert len(client.calls) == 2


class FakeStream:
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    def __iter__(self):
        for chunk in self.chunks:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk

    def close(self):
        self.closed = True


def chunk(content=None, **extra):
    return {"choices": [{"index": 0, "delta": {"content": content, **extra}}]}


@pytest.mark.parametrize("split", range(0, 39))
def test_buffered_stream_all_split_positions(split):
    text = "<think>SECRET</think>`<think>code</think>`"
    stream = FakeStream([chunk(text[:split], reasoning="r", reasoning_content="r"),
                         chunk(text[split:], reasoning_text="2", reasoning_details=[{"index": 0, "data": "a"}]),
                         {"choices": [], "usage": {"total_tokens": 12}}])
    client = FakeClient(results=[stream])
    results = list(llm.stream_chat_completion(client, model="test", messages=[]))
    assert len(results) == 1
    result = results[0]
    assert result.text == "`<think>code</think>`"
    assert result.reasoning.text == "r2"
    assert result.reasoning.details == [{"index": 0, "data": "a"}]
    assert len(result.raw) == 3
    assert stream.closed
    assert client.calls[0]["stream"] is True


def test_stream_single_char_deltas_and_unclosed_tail():
    text = "<final>Answer</final>\n```\n[think]literal[/think]\n```\n◁think▷SECRET"
    stream = FakeStream([chunk(c) for c in text])
    result, = llm.stream_chat_completion(FakeClient(results=[stream]), model="test", messages=[])
    assert result.text == "Answer\n```\n[think]literal[/think]\n```"


@pytest.mark.parametrize("lazy", [False, True])
def test_stream_retry_before_first_chunk(lazy):
    first = FakeStream([bad()]) if lazy else bad()
    second = FakeStream([chunk("OK")])
    client = FakeClient(results=[first, second])
    result, = llm.stream_chat_completion(client, model="test", messages=[], reasoning_policy="high")
    assert result.text == "OK" and result.retried_without_reasoning
    assert len(client.calls) == 2
    assert second.closed
    if lazy:
        assert first.closed


def test_stream_failure_after_chunk_never_replayed_or_yielded():
    stream = FakeStream([chunk("partial"), bad()])
    client = FakeClient(results=[stream])
    emitted = []
    with pytest.raises(openai.BadRequestError):
        for result in llm.stream_chat_completion(client, model="test", messages=[], reasoning_policy="high"):
            emitted.append(result)
    assert emitted == [] and stream.closed and len(client.calls) == 1


def test_stream_error_closes_and_second_failure_propagates():
    stream = FakeStream([RuntimeError("failed")])
    with pytest.raises(RuntimeError):
        list(llm.stream_chat_completion(FakeClient(results=[stream]), model="test", messages=[]))
    assert stream.closed
    client = FakeClient(results=[bad(), bad()])
    with pytest.raises(openai.BadRequestError):
        list(llm.stream_chat_completion(client, model="test", messages=[], reasoning_policy="high"))
    assert len(client.calls) == 2


def test_accidental_stream_use_rejected():
    client = FakeClient()
    with pytest.raises(ValueError):
        llm.create_chat_completion(client, stream=True)
    with pytest.raises(ValueError):
        llm.create_chat_completion(client, extra_body={"stream": True})
    with pytest.raises(ValueError):
        list(llm.stream_chat_completion(client, extra_body={"stream": False}))
    assert not client.calls


@pytest.mark.parametrize("provider,expected", [
    ("openrouter", {"reasoning": {"effort": "high"}}),
    ("deepseek", {"reasoning_effort": "high"}),
    ("cometapi", {"thinking": {"type": "enabled"}, "reasoning_effort": "high"}),
])
def test_real_sdk_serializes_extra_body_at_json_root(provider, expected):
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        assert "extra_body" not in body
        for key, value in expected.items():
            assert body[key] == value
        assert not any(k in body for k in {"reasoning", "thinking", "reasoning_effort"} - expected.keys())
        return httpx.Response(200, json={"id": "fake", "object": "chat.completion", "created": 0,
            "model": "deepseek-v4-pro", "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "OK", "reasoning_content": "private"}}]})

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        with openai.OpenAI(api_key="offline-test-not-a-real-key", base_url=URLS[provider],
                           http_client=http_client, max_retries=0) as client:
            result = llm.create_chat_completion(client, reasoning_policy="high", model="deepseek-v4-pro", messages=[])
    assert result.text == "OK" and result.reasoning.text == "private"
    assert len(seen) == 1


def test_legacy_sdk_signature_without_reasoning_kwarg():
    class OldClient:
        base_url = URLS["deepseek"]

        def __init__(self):
            self.chat = NS(completions=self)

        def create(self, *, model, messages, extra_body):
            assert extra_body == {"reasoning_effort": "none"}
            return response()

    assert llm.create_chat_completion(OldClient(), model="test", messages=[], reasoning_policy="off").text == "OK"


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Network access is forbidden in this test suite")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def test_empty_direct_reasoning_falls_back_to_model_extra():
    message = NS(content="ok", reasoning_content="", reasoning="second",
                 model_extra={"reasoning_content": "first"})
    assert llm.visible_text(response(message)).reasoning.text == "first"


def test_sdk_400_fallback_and_old_error_shape():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        if len(seen) == 1:
            return httpx.Response(400, json={"error": {"message": "Unknown parameter: reasoning",
                                                       "param": "reasoning", "code": "unknown_parameter"}})
        assert "reasoning" not in seen[-1]
        return httpx.Response(200, json={"id": "fake", "object": "chat.completion", "created": 0,
            "model": "test", "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "OK"}}]})

    with httpx.Client(transport=httpx.MockTransport(handler)) as transport:
        with openai.OpenAI(api_key="offline-test-not-a-real-key", base_url=URLS["openrouter"],
                           http_client=transport, max_retries=0) as client:
            result = llm.create_chat_completion(client, model="test", messages=[], reasoning_policy="high")
    assert result.text == "OK" and result.retried_without_reasoning
    assert len(seen) == 2


def test_sdk_sse_parse_and_close():
    frames = [chunk("<thi", reasoning_content="r", reasoning="r"),
              chunk("nk>hidden</think>Answer", reasoning_details=[{"id": "x", "data": "opaque"}])]
    frames = [{"id": "fake", "object": "chat.completion.chunk", "created": 0, "model": "test", **c} for c in frames]
    wire = "".join("data: " + json.dumps(c) + "\n\n" for c in frames) + "data: [DONE]\n\n"
    replies = []

    def handler(request):
        assert json.loads(request.content)["stream"] is True
        replies.append(httpx.Response(200, headers={"content-type": "text/event-stream"}, content=wire))
        return replies[-1]

    with httpx.Client(transport=httpx.MockTransport(handler)) as transport:
        with openai.OpenAI(api_key="offline-test-not-a-real-key", base_url=URLS["openrouter"],
                           http_client=transport, max_retries=0) as client:
            result, = llm.stream_chat_completion(client, model="test", messages=[])
    assert result.text == "Answer" and result.reasoning.text == "r"
    assert result.reasoning.details == [{"id": "x", "data": "opaque"}]
    assert replies[0].is_closed


def test_python_311_syntax():
    import ast
    ast.parse(MODULE.read_text(), feature_version=(3, 11))
