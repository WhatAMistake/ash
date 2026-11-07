import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import editorial_tools as et
import telegram_bot as tb
import writer_bot as wb


def completion(content, finish="stop"):
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason=finish, message=SimpleNamespace(content=content))])


def writer():
    bot = wb.WriterBot.__new__(wb.WriterBot)
    bot.client = object()
    bot.model = "test"
    bot.language = "ru"
    bot.system_prompt = "Ash"
    bot.history = []
    bot.session_summary = None
    bot.use_rag = False
    bot.rag = None
    return bot


def test_split_is_lossless_and_bounded_for_russian_english_and_long_lines():
    text = ("— Ну, ну! Не-а, я не я.\r\n\r\n"
            "“No, no,” she said. very very strange 🌙\n" * 200) + "я" * 5000
    parts = et.split_text(text, 300)
    assert len(parts) > 10
    assert "".join(parts) == text
    assert all(et.token_bound(p) <= 300 for p in parts)


def test_dictionary_normalization_and_contextual_candidates():
    entries = et.cliches()
    keys = [x.casefold().replace("ё", "е") for x in entries]
    assert len(keys) == len(set(keys))
    assert not any("[" in x or "(" in x or len(x.split()) < 2 for x in entries)
    assert "в мгновение ока" in et.cliche_candidates("Он исчез В МГНОВЕНИЕ ОКА.")
    assert "at the end of the day" in et.cliche_candidates("At the end of the day, she sighed.")
    assert "in order to" in et.cliche_candidates("in order to leave")
    assert "at the end of the day" not in et.cliche_candidates("the day was bright")


@pytest.mark.parametrize("bad", [
    [{"old": "no", "new": "yes"}],  # ambiguous anchor
    [{"old": "missing", "new": "word"}],
    [{"old": "no\nno", "new": "no no"}],
    [{"old": "no", "new": "no"}, {"old": "no", "new": "no"}],
    [{"old": "no", "new": ""}],
])
def test_edits_reject_missing_ambiguous_destructive_or_overlapping(bad):
    with pytest.raises(ValueError):
        et.apply_proofreading_edits("no\nno", bad)


def test_proofreader_processes_every_chunk_and_preserves_authorial_features(monkeypatch):
    text = ("— Не-а, ну ну, я так и сказал.\n\n"
            "“No, no,” she said. Ain't it strange?\n" * 180) + "Конец."
    calls = []
    def call(client, **kwargs):
        payload = json.loads(kwargs["messages"][-1]["content"])
        calls.append(payload)
        assert "style" in kwargs["messages"][0]["content"]
        return completion('{"edits":[]}')
    monkeypatch.setattr(wb, "create_chat_completion", call)
    assert writer().correct_text(text) == text
    assert len(calls) > 2
    assert "".join(x["target"] for x in calls) == text
    assert calls[1]["previous_context"]
    assert calls[0]["next_context"]


def test_proofreader_applies_only_unique_local_edits_and_fails_closed(monkeypatch):
    bot = writer()
    monkeypatch.setattr(wb, "create_chat_completion", lambda *a, **kw:
                        completion(json.dumps({"edits": [{"old": "teh cat", "new": "the cat"}]})))
    assert bot.correct_text("teh cat\n— No, no!") == "the cat\n— No, no!"
    monkeypatch.setattr(wb, "create_chat_completion", lambda *a, **kw:
                        completion('{"edits":[]}', "length"))
    assert bot.correct_text("something").startswith("Error:")
    monkeypatch.setattr(wb, "create_chat_completion", lambda *a, **kw:
                        completion('{"edits":[{"old":"missing","new":"x"}]}'))
    assert bot.correct_text("something").startswith("Error:")


def test_proofreader_retries_ambiguous_edits(monkeypatch):
    responses = iter([completion('{"edits":[{"old":"no","new":"yes"}]}'),
                      completion('{"edits":[]}')])
    monkeypatch.setattr(wb, "create_chat_completion", lambda *a, **kw: next(responses))
    assert writer().correct_text("no no") == "no no"


def test_editor_receives_dictionary_and_optional_rag_for_all_fragments(monkeypatch):
    bot = writer()
    bot.rag = SimpleNamespace(get_context_for_query=MagicMock(return_value="RAG advice"))
    bot.use_rag = True
    calls = []
    def call(client, **kwargs):
        calls.append(kwargs["messages"])
        return completion("No issue justified.")
    monkeypatch.setattr(wb, "create_chat_completion", call)
    text = ("At the end of the day, she said no. " * 200)
    result = bot.edit_text(text)
    payloads = [json.loads(messages[-1]["content"]) for messages in calls]
    assert "".join(p["target"] for p in payloads) == text
    assert all("at the end of the day" in p["dictionary_candidates"] for p in payloads)
    assert all("RAG advice" in str(messages) for messages in calls)
    assert "No issue justified." in result


def test_discussion_context_keeps_full_small_documents_and_identifies_partial():
    original = "begin " + "x" * 1000 + " specific ending"
    result = "Remark 2: specific ending needs attention."
    context, partial = et.discussion_context(original, result, "Remark 2?")
    assert not partial and original in context and result in context
    context, partial = et.discussion_context("irrelevant\n" * 6000 + "rareword", result,
                                            "rareword remark 2", budget=4000)
    assert partial and "rareword" in context and "Remark 2" in context


def fake_telegram():
    bot = tb.TelegramWriterBot.__new__(tb.TelegramWriterBot)
    bot.user_langs = {7: "en"}
    bot.user_states = {}
    bot.accumulated_text = {}
    bot.uploaded_documents = {}
    bot.tool_discussions = {}
    bot.STATE_TO_DISCUSS = {"editor_wait": "editor_discuss", "corrector_wait": "corrector_discuss"}
    bot.DISCUSSION_STATES = {"editor_discuss": ("editor_wait", "instr_editor"),
                             "corrector_discuss": ("corrector_wait", "instr_corrector")}
    bot.bot = SimpleNamespace(send_chat_action=AsyncMock())
    bot._writer_call = AsyncMock(return_value="Follow-up answer")
    return bot


def message():
    return SimpleNamespace(from_user=SimpleNamespace(id=7), chat=SimpleNamespace(id=7),
                           answer=AsyncMock(), answer_document=AsyncMock())


@pytest.mark.asyncio
async def test_discussion_stores_result_and_replaces_document_context():
    bot, msg = fake_telegram(), message()
    original = "A" * 800 + " end"
    await bot._finish_tool(msg, 7, "en", "Remark 1", "editor_wait", source_text=original)
    await bot._handle_discussion(msg, "Why remark 1?", "editor_discuss")
    args = bot._writer_call.await_args.args
    assert args[:5] == (7, "discuss_tool", original, "Remark 1", "Why remark 1?")
    await bot._finish_tool(msg, 7, "en", "New result", "editor_wait", source_text="New document")
    await bot._handle_discussion(msg, "Why?", "editor_discuss")
    args = bot._writer_call.await_args.args
    assert args[2:5] == ("New document", "New result", "Why?")
    assert args[5][0]["question"] == "Why?"


@pytest.mark.asyncio
async def test_txt_extract_limit_and_file_delivery(tmp_path):
    bot, msg = fake_telegram(), message()
    path = tmp_path / "text.txt"
    path.write_bytes("one\r\nдва".encode())
    text = await bot._extract_txt(str(path), 100)
    assert text == "one\r\nдва"
    with pytest.raises(ValueError):
        await bot._extract_txt(str(path), 2)
    bot.uploaded_documents[7] = {"name": "text.txt", "suffix": ".txt", "text": text}
    bot.user_states[7] = "document_wait"
    await bot._finish_tool(msg, 7, "en", text, "corrector_wait", source_text=text)
    upload = msg.answer_document.await_args.args[0]
    assert upload.filename == "text_corrected.txt"
    assert upload.data.decode() == text


@pytest.mark.asyncio
async def test_docx_table_and_paragraphs_roundtrip(tmp_path):
    from docx import Document
    doc = Document()
    doc.add_paragraph("A line\nwith break")
    table = doc.add_table(rows=1, cols=1)
    table.cell(0, 0).text = "teh cat"
    doc.add_paragraph("end")
    path = tmp_path / "story.docx"
    doc.save(path)
    bot, msg = fake_telegram(), message()
    text = await bot._extract_docx(str(path), 1000)
    assert text == "A line\nwith break\nteh cat\nend"
    document = {"name": "story.docx", "suffix": ".docx", "content": path.read_bytes()}
    await bot._send_corrected_document(msg, text.replace("teh cat", "the cat"), document, "en")
    upload = msg.answer_document.await_args.args[0]
    fixed = Document(io.BytesIO(upload.data))
    assert fixed.tables[0].cell(0, 0).text == "the cat"
    assert fixed.paragraphs[0].text == "A line\nwith break"
    assert fixed.paragraphs[-1].text == "end"


@pytest.mark.asyncio
async def test_docx_complex_changed_paragraph_falls_back_to_text(tmp_path):
    from docx import Document
    from docx.oxml import OxmlElement
    doc = Document()
    para = doc.add_paragraph("teh cat")
    para._p.append(OxmlElement("w:hyperlink"))
    path = tmp_path / "complex.docx"
    doc.save(path)
    bot, msg = fake_telegram(), message()
    document = {"name": "complex.docx", "suffix": ".docx", "content": path.read_bytes()}
    await bot._send_corrected_document(msg, "the cat", document, "en")
    assert msg.answer_document.await_args.args[0].filename == "complex_corrected.txt"
    assert "original is untouched" in msg.answer_document.await_args.kwargs["caption"]
