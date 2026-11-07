import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import writer_rag
from writer_bot import WriterBot


def write_category(path, category, chunks):
    (path / f"{category}.json").write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")


def item(id, text, title="Book", **extra):
    return {"id": id, "text": text, "author": "Author", "book_title": title, **extra}


def test_fallback_is_category_scoped_and_preserves_source(tmp_path):
    write_category(tmp_path, "craft", [item("a", "метафоры штампы", chapter="Глава 2")])
    write_category(tmp_path, "editorial", [item("b", "метафоры штампы", section="Раздел")])
    rag = writer_rag.WriterRAG(str(tmp_path), use_local_embeddings=False)
    results = rag.search_similar("метафоры штампы", category="editorial")
    assert len(results) == 1 and results[0].metadata == {
        "source": "book", "category": "editorial", "book_title": "Book",
        "author": "Author", "chapter": "", "section": "Раздел"}
    assert "curated summary; not verified verbatim" in rag.get_context_for_query("метафоры штампы", command="editor")
    assert rag.search_similar("метафоры штампы", category="unknown") == []
    assert rag.search_similar("книге «Missing» метафоры штампы", category="editorial") == []
    assert rag.search_similar("автора «Missing» метафоры штампы", category="editorial") == []
    assert rag.search_similar("zxqv wxyz", category="editorial") == []


def test_refresh_and_empty_category_do_not_resurrect_legacy(tmp_path):
    write_category(tmp_path, "craft", [item("a", "структура сюжета")])
    (tmp_path / "book_chunks.json").write_text(json.dumps([item("old", "legacy")]), encoding="utf-8")
    rag = writer_rag.WriterRAG(str(tmp_path), use_local_embeddings=False)
    assert rag.search_similar("структура сюжета", category="craft")
    assert rag.random_craft_chunk()["id"] == "a"
    write_category(tmp_path, "craft", [item("b", "характер персонажа конфликт")])
    assert not rag.search_similar("структура сюжета", category="craft")
    assert rag.search_similar("характер персонажа", category="craft")
    write_category(tmp_path, "craft", [])
    assert not rag.search_similar("legacy", category="craft")


def test_invalid_corpus_update_keeps_previous_snapshot(tmp_path):
    write_category(tmp_path, "craft", [item("a", "структура сюжета")])
    rag = writer_rag.WriterRAG(str(tmp_path), use_local_embeddings=False)
    (tmp_path / "craft.json").write_text('{"unexpected": []}', encoding="utf-8")
    assert rag.search_similar("структура сюжета", category="craft")
    (tmp_path / "craft.json").write_text('[{"id": "a", "text": null}]', encoding="utf-8")
    assert rag.search_similar("структура сюжета", category="craft")


def test_incremental_sync_updates_and_deletes_only_changes(tmp_path, monkeypatch):
    class Collection:
        def __init__(self):
            self.rows = {}
            self.upserts = []
            self.deletes = []
        def get(self, include):
            return {"ids": list(self.rows), "documents": [v[0] for v in self.rows.values()],
                    "metadatas": [v[1] for v in self.rows.values()]}
        def upsert(self, ids, documents, metadatas, embeddings):
            self.upserts.extend(ids)
            self.rows.update(zip(ids, zip(documents, metadatas)))
        def delete(self, ids):
            self.deletes.extend(ids)
            for id in ids:
                del self.rows[id]
    encoded = []
    monkeypatch.setattr(writer_rag, "_encode", lambda model, texts: encoded.extend(texts) or [[1.0]] * len(texts))
    write_category(tmp_path, "craft", [item("a", "first"), item("b", "second")])
    rag = writer_rag.WriterRAG(str(tmp_path), use_local_embeddings=False)
    collection = Collection()
    rag._sync_category("craft", collection, object())
    assert len(encoded) == 2
    rag._sync_category("craft", collection, object())
    assert len(encoded) == 2
    write_category(tmp_path, "craft", [item("b", "changed"), item("c", "third")])
    rag._refresh_corpus()
    rag._sync_category("craft", collection, object())
    assert encoded == ["first", "second", "changed", "third"]
    assert collection.deletes == ["craft_a"]
    assert set(collection.rows) == {"craft_b", "craft_c"}


def test_real_chroma_incremental_sync(tmp_path, monkeypatch):
    pytest.importorskip("chromadb")
    np = pytest.importorskip("numpy")
    class Embedder:
        def encode(self, texts, **kwargs):
            return np.array([[1.0, 0.0] for _ in texts])
    embedder = Embedder()
    encode = Mock(wraps=embedder.encode)
    embedder.encode = encode
    monkeypatch.setattr(writer_rag, "_get_model", lambda: embedder)
    write_category(tmp_path, "craft", [item("a", "first story"), item("b", "second story")])
    rag = writer_rag.WriterRAG(str(tmp_path))
    assert len(rag.search_similar("story", category="craft")) == 2
    assert len(encode.call_args_list) == 2  # index batch, query
    rag.search_similar("story", category="craft")
    assert len(encode.call_args_list) == 3  # no re-embedding on unchanged corpus
    write_category(tmp_path, "craft", [item("b", "changed story"), item("c", "new story")])
    assert len(rag.search_similar("story", category="craft")) == 2
    assert set(rag.collections["craft"].get()["ids"]) == {"craft_b", "craft_c"}
    assert encode.call_args_list[-2].args[0] == ["changed story", "new story"]


def test_shared_service_and_lazy_model(tmp_path, monkeypatch):
    monkeypatch.setattr(writer_rag, "_get_model", Mock(side_effect=RuntimeError("must be lazy")))
    assert writer_rag.get_writer_rag(str(tmp_path)) is writer_rag.get_writer_rag(str(tmp_path / "."))
    monkeypatch.setattr(WriterBot, "_init_llm", lambda self: None)
    a = WriterBot(data_dir=str(tmp_path))
    b = WriterBot(data_dir=str(tmp_path))
    assert a.rag is b.rag
    assert not writer_rag._get_model.called


def test_index_and_search_failures_fall_back(tmp_path, monkeypatch):
    write_category(tmp_path, "editorial", [item("a", "метафоры штампы")])
    rag = writer_rag.WriterRAG(str(tmp_path))
    monkeypatch.setattr(writer_rag, "_get_model", lambda: (_ for _ in ()).throw(RuntimeError()))
    assert rag.search_similar("метафоры штампы", category="editorial")
    assert rag._retry_at > 0


def test_query_exception_falls_back(tmp_path, monkeypatch):
    write_category(tmp_path, "editorial", [item("a", "метафоры штампы")])
    rag = writer_rag.WriterRAG(str(tmp_path))
    rag.embedder = object()
    collection = Mock()
    collection.count.side_effect = RuntimeError("database unavailable")
    rag.collections = {"editorial": collection}
    monkeypatch.setattr(rag, "_ensure_index", lambda: True)
    monkeypatch.setattr(writer_rag, "_encode", lambda model, texts: [[1.0]])
    result = rag.search_similar("метафоры штампы", category="editorial")
    assert len(result) == 1 and result[0].metadata["category"] == "editorial"
    assert rag._retry_at > 0


def test_low_semantic_score_uses_gated_fallback(tmp_path, monkeypatch):
    write_category(tmp_path, "style", [item("a", "метафоры штампы")])
    rag = writer_rag.WriterRAG(str(tmp_path))
    rag.embedder = object()
    collection = Mock()
    collection.count.return_value = 1
    collection.query.return_value = {"documents": [["unrelated"]], "metadatas": [[
        {"book_title": "Book", "category": "style"}]], "distances": [[1.0]]}
    rag.collections = {"style": collection}
    monkeypatch.setattr(rag, "_ensure_index", lambda: True)
    monkeypatch.setattr(writer_rag, "_encode", lambda model, texts: [[1.0]])
    assert rag.search_similar("метафоры штампы", category="style")[0].content == "метафоры штампы"
    assert not rag.search_similar("zxqv wxyz", category="style")


def test_embedding_model_is_singleton_across_threads(monkeypatch):
    import threading
    import types
    loads = []
    fake = types.ModuleType("sentence_transformers")
    fake.SentenceTransformer = lambda name: loads.append(name) or object()
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    monkeypatch.setattr(writer_rag, "_model", None)
    monkeypatch.setattr(writer_rag, "_model_retry_at", 0)
    results = []
    threads = [threading.Thread(target=lambda: results.append(writer_rag._get_model())) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(loads) == 1 and all(result is results[0] for result in results)


@pytest.mark.parametrize("command,category", [
    ("methodique", "craft"), ("feedback", "craft"), ("block", "craft"),
    ("develop", "craft"), ("character", "craft"), ("dialogue", "craft"),
    ("style", "style"), ("praise", "style"), ("editor", "editorial"), ("roast", "editorial")])
def test_command_routing(command, category, tmp_path):
    rag = writer_rag.WriterRAG(str(tmp_path), use_local_embeddings=False)
    rag.search_similar = Mock(return_value=[])
    rag.get_context_for_query("hello", command=command)
    assert rag.search_similar.call_args.kwargs["category"] == category


def test_methodique_uses_user_query_and_citation_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(WriterBot, "_init_llm", lambda self: None)
    bot = WriterBot(use_rag=False)
    bot.use_rag = True
    bot.rag = Mock()
    bot.rag.COMMAND_CATEGORIES = writer_rag.WriterRAG.COMMAND_CATEGORIES
    bot.rag.get_context_for_query.return_value = ""
    bot.client = Mock()
    monkeypatch.setitem(WriterBot.generate_response.__globals__, "create_chat_completion", lambda *a, **kw: Mock())
    monkeypatch.setitem(WriterBot.generate_response.__globals__, "visible_completion_text", lambda _: "answer")
    assert bot.methodique("my actual story") == "answer"
    bot.rag.get_context_for_query.assert_called_once_with("my actual story", max_chunks=3, command="methodique")
    messages = bot._build_messages("text", command="feedback")
    assert any("Never invent book references" in m["content"] for m in messages)


def test_benchmark_has_expected_cases():
    cases = json.loads((ROOT / "tests" / "rag_queries.json").read_text(encoding="utf-8"))
    assert len(cases) >= 25
    assert {c["category"] for c in cases} == {"craft", "style", "editorial"}
    assert {c["source_type"] for c in cases} == {"craft_book", "fiction", "curated_editorial", "none"}


def test_keyword_quality_benchmark():
    cases = json.loads((ROOT / "tests" / "rag_queries.json").read_text(encoding="utf-8"))
    rag = writer_rag.WriterRAG(str(ROOT / "data"), use_local_embeddings=False)
    positives = negatives = 0
    for case in cases:
        results = rag.search_similar(case["query"], 3, case["category"])
        assert all(r.metadata["category"] == case["category"] for r in results)
        if case["source_type"] == "none":
            negatives += not results
        else:
            positives += any(any(term.casefold() in r.content.casefold() for term in case["evidence"])
                             for r in results)
    assert positives >= 22
    assert negatives == 8


def test_chunking_does_not_invent_author_or_inline_chapter(monkeypatch, tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import chunk_books
    monkeypatch.setattr(chunk_books, "TIKTOKEN_AVAILABLE", False)
    assert chunk_books.parse_filename(Path("Multi Word Title.pdf")) == ("Unknown", "Multi Word Title")
    assert chunk_books.parse_filename(Path("Автор_Название_книги.pdf")) == ("Автор", "Название книги")
    with pytest.raises(ValueError):
        chunk_books.chunk_by_tokens("text", max_tokens=5, overlap=5)
    assert all(len(part.split()) <= 3 for part in chunk_books.chunk_by_paragraphs("one two three four five six", 3))
    source = tmp_path / "Author_Book.txt"
    source.write_text("A reference to Chapter II appears in this paragraph.", encoding="utf-8")
    chunks = chunk_books.process_book(source, "craft", max_tokens=20, overlap=0)
    assert chunks[0]["chapter"] == ""
