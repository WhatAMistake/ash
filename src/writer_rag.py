"""Shared, category-aware retrieval over the existing writing corpus."""

import json
import logging
import math
import os
import random
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CATEGORIES = ("craft", "style", "editorial")
# Calibrated against tests/rag_queries.json; see docs/rag-evaluation.md.
SEMANTIC_MIN_SIMILARITY = 0.60
KEYWORD_MIN_COVERAGE = 0.30
RETRY_SECONDS = 60
_model = None
_model_retry_at = 0.0
_model_lock = threading.RLock()
_services = {}
_services_lock = threading.Lock()


@dataclass
class RAGResult:
    content: str
    source: str
    relevance: float
    metadata: dict


def get_writer_rag(data_dir: Optional[str] = None) -> "WriterRAG":
    """One service per resolved corpus, and one model per process."""
    path = Path(data_dir or Path(__file__).resolve().parent.parent / "data").resolve()
    with _services_lock:
        if path not in _services:
            _services[path] = WriterRAG(str(path))
        return _services[path]


def _get_model():
    global _model, _model_retry_at
    with _model_lock:
        if _model is None and time.monotonic() >= _model_retry_at:
            try:
                from sentence_transformers import SentenceTransformer
                _model = SentenceTransformer(MODEL_NAME)
            except Exception as exc:
                _model_retry_at = time.monotonic() + RETRY_SECONDS
                logger.warning("rag_model_unavailable error_type=%s", type(exc).__name__)
        return _model


def _encode(model, texts):
    # SentenceTransformer and the shared index are used from Telegram worker threads.
    with _model_lock:
        return model.encode(texts, show_progress_bar=False, batch_size=8).tolist()


def _tokens(text):
    # Short common words add noise; prefixes allow basic Russian inflections.
    return {w[:6] for w in re.findall(r"[^\W\d_]{4,}", text.lower(), re.UNICODE)}


def _requested_title(query):
    match = re.search(r'(?:книг\w*|book)\s*[«"“]([^»"”]+)[»"”]', query, re.IGNORECASE)
    return match.group(1).casefold().strip() if match else None


def _requested_author(query):
    match = re.search(r'(?:автор\w*|author)\s*[«"“]([^»"”]+)[»"”]', query, re.IGNORECASE)
    return match.group(1).casefold().strip() if match else None


def _matches_source(chunk, title, author):
    return ((not title or title in str(chunk.get("book_title") or "").casefold()) and
            (not author or author in str(chunk.get("author") or "").casefold()))


def _index_is_corrupt(exc):
    message = str(exc).lower()
    return "hnsw" in message or "corrupt" in message


class WriterRAG:
    COMMAND_CATEGORIES = {
        "block": "craft", "develop": "craft", "character": "craft",
        "dialogue": "craft", "methodique": "craft", "feedback": "craft",
        "style": "style", "roast": "editorial", "praise": "style",
        "corrector": "editorial", "editor": "editorial",
    }

    def __init__(self, data_dir: Optional[str] = None, use_local_embeddings: bool = True):
        self.data_dir = Path(data_dir or Path(__file__).resolve().parent.parent / "data").resolve()
        self.use_local_embeddings = use_local_embeddings
        self.embedder = None
        self.collections = {}
        self._client = None
        self._lock = threading.RLock()
        self._signature = None
        self._indexed_signature = None
        self._retry_at = 0.0
        self._force_recovery = set()
        self._chunks = {category: [] for category in CATEGORIES}
        self._keyword_index = {}
        self._refresh_corpus()

    @property
    def craft_chunks(self):
        return self._chunks["craft"]

    @property
    def style_chunks(self):
        return self._chunks["style"]

    @property
    def editorial_chunks(self):
        return self._chunks["editorial"]

    @property
    def all_chunks(self):
        return sum((self._chunks[c] for c in CATEGORIES), [])

    def random_craft_chunk(self):
        with self._lock:
            try:
                self._refresh_corpus()
            except (OSError, ValueError, TypeError) as exc:
                logger.warning("rag_corpus_unavailable error_type=%s", type(exc).__name__)
            return random.choice(self.craft_chunks) if self.craft_chunks else None

    def _corpus_signature(self):
        paths = [self.data_dir / (c + ".json") for c in CATEGORIES]
        paths.append(self.data_dir / "book_chunks.json")
        return tuple((p.stat().st_mtime_ns, p.stat().st_size) if p.exists() else None for p in paths)

    def _refresh_corpus(self):
        signature = self._corpus_signature()
        if signature == self._signature:
            return
        def read_chunks(path):
            with path.open(encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                data = data.get("chunks")
            if not isinstance(data, list) or any(
                not isinstance(chunk, dict) or not isinstance(chunk.get("text", ""), str)
                for chunk in data
            ):
                raise ValueError(f"invalid RAG corpus: {path.name}")
            return data

        chunks = {}
        for cat in CATEGORIES:
            path = self.data_dir / (cat + ".json")
            if not path.exists():
                chunks[cat] = []
                continue
            chunks[cat] = read_chunks(path)
        if not any((self.data_dir / (cat + ".json")).exists() for cat in CATEGORIES) and (self.data_dir / "book_chunks.json").exists():
            chunks["craft"] = read_chunks(self.data_dir / "book_chunks.json")
        for cat in CATEGORIES:
            ids = [str(c.get("id", i)) for i, c in enumerate(chunks[cat])]
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate RAG chunk IDs in {cat}")
        self._chunks = chunks
        self._keyword_index = {}
        for cat in CATEGORIES:
            terms = [_tokens(c.get("text", "")) for c in chunks[cat]]
            frequencies = Counter(token for entry in terms for token in entry)
            self._keyword_index[cat] = (terms, frequencies)
        self._signature = signature

    @staticmethod
    def _metadata(chunk, category):
        # Editorial JSON is curated material: its attribution is not verification
        # that the text is a verbatim quotation from the named book.
        metadata = {
            "source": "book", "category": category,
            "book_title": str(chunk.get("book_title") or ""),
            "author": str(chunk.get("author") or ""),
            "chapter": str(chunk.get("chapter") or ""),
        }
        if chunk.get("section"):
            metadata["section"] = str(chunk["section"])
        return metadata

    def _sync_category(self, category, collection, model):
        chunks = self._chunks[category]
        expected = {
            f"{category}_{chunk.get('id', i)}": (chunk.get("text", ""), self._metadata(chunk, category))
            for i, chunk in enumerate(chunks)
            if chunk.get("text", "").strip()
        }
        old = collection.get(include=["documents", "metadatas"])
        existing = dict(zip(old["ids"], zip(old["documents"], old["metadatas"])))
        changed = [key for key, value in expected.items() if existing.get(key) != value]
        # Small bounded batches avoid encoding the entire corpus in memory.
        for start in range(0, len(changed), 32):
            ids = changed[start:start + 32]
            texts = [expected[key][0] for key in ids]
            collection.upsert(ids=ids, documents=texts, metadatas=[expected[key][1] for key in ids],
                              embeddings=_encode(model, texts))
        stale = list(existing.keys() - expected.keys())
        for start in range(0, len(stale), 500):
            collection.delete(ids=stale[start:start + 500])
        logger.info("rag_index_sync category=%s changed=%d removed=%d", category, len(changed), len(stale))

    def _ensure_index(self):
        if not self.use_local_embeddings or time.monotonic() < self._retry_at:
            return False
        try:
            if self.embedder is None:
                self.embedder = _get_model()
            if self.embedder is None:
                return False
            if self._client is None:
                import chromadb
                path = self.data_dir / "chromadb_writers"
                try:
                    # Chroma's Windows HNSW backend can fail to reopen an index
                    # through a Unicode absolute path, while the equivalent
                    # path relative to the service working directory works.
                    path = os.path.relpath(path, Path.cwd())
                except ValueError:  # Different Windows drives.
                    path = str(path)
                self._client = chromadb.PersistentClient(path=str(path))
            if self._indexed_signature != self._signature:
                for cat in CATEGORIES:
                    if cat not in self._force_recovery:
                        collection = self._client.get_or_create_collection(
                            name="writers_" + cat, metadata={"hnsw:space": "cosine", "category": cat})
                        try:
                            collection.count()
                            collection.get(limit=1, include=["documents"])
                        except Exception as exc:
                            if not _index_is_corrupt(exc):
                                raise
                            # Never delete or overwrite the broken legacy collection.
                            logger.warning("rag_legacy_index_unreadable category=%s", cat)
                            self._force_recovery.add(cat)
                    if cat in self._force_recovery:
                        for generation in range(1, 4):
                            suffix = "" if generation == 1 else str(generation)
                            collection = self._client.get_or_create_collection(
                                name=f"writers_recovered{suffix}_{cat}",
                                metadata={"hnsw:space": "cosine", "category": cat})
                            try:
                                collection.count()
                                collection.get(limit=1, include=["documents"])
                                break
                            except Exception as exc:
                                if not _index_is_corrupt(exc):
                                    raise
                        else:
                            raise RuntimeError("No readable recovery collection")
                    self._sync_category(cat, collection, self.embedder)
                    self.collections[cat] = collection
                self._indexed_signature = self._signature
            return True
        except Exception as exc:
            logger.warning("rag_index_unavailable error_type=%s", type(exc).__name__)
            self._retry_at = time.monotonic() + RETRY_SECONDS
            self._indexed_signature = None
            self.collections = {}
            self._client = None
            return False

    def search_similar(self, query: str, n_results: int = 5, category: Optional[str] = None) -> list[RAGResult]:
        if not query.strip() or n_results <= 0 or (category is not None and category not in CATEGORIES):
            return []
        with self._lock:
            try:
                self._refresh_corpus()
            except (OSError, ValueError, TypeError) as exc:
                # Keep the last complete snapshot, not a partially read corpus.
                logger.warning("rag_corpus_unavailable error_type=%s", type(exc).__name__)
            title = _requested_title(query)
            author = _requested_author(query)
            cats = [category] if category else CATEGORIES
            if (title or author) and not any(_matches_source(c, title, author)
                                             for cat in cats for c in self._chunks[cat]):
                return []
            if self._ensure_index():
                failed_cat = None
                try:
                    query_embedding = _encode(self.embedder, [query[:3000]])
                    found = []
                    for cat in ([category] if category else CATEGORIES):
                        failed_cat = cat
                        collection = self.collections[cat]
                        count = collection.count()
                        if not count:
                            continue
                        result = collection.query(query_embeddings=query_embedding,
                                                  n_results=min(n_results * (8 if title or author else 1), count))
                        for text, metadata, distance in zip(result["documents"][0], result["metadatas"][0],
                                                            result["distances"][0]):
                            score = 1 - distance
                            if score >= SEMANTIC_MIN_SIMILARITY and _matches_source(metadata, title, author):
                                found.append(RAGResult(text, metadata["book_title"], score, metadata))
                        failed_cat = None
                    if found:
                        return sorted(found, key=lambda item: item.relevance, reverse=True)[:n_results]
                    # Semantic recall is weak for some fiction excerpts; only
                    # fall back to independently confidence-gated keyword hits.
                    return self._keyword_fallback(query, n_results, category)
                except Exception as exc:
                    logger.warning("rag_search_unavailable error_type=%s", type(exc).__name__)
                    if failed_cat is not None and _index_is_corrupt(exc):
                        self._force_recovery.add(failed_cat)
                    self._retry_at = time.monotonic() + RETRY_SECONDS
                    self._indexed_signature = None
                    self.collections = {}
                    self._client = None
            return self._keyword_fallback(query, n_results, category)

    def _keyword_fallback(self, query: str, n_results: int, category: Optional[str] = None) -> list[RAGResult]:
        words = _tokens(query[:3000])
        title = _requested_title(query)
        author = _requested_author(query)
        if not words or (category is not None and category not in CATEGORIES):
            return []
        scored = []
        for cat in ([category] if category else CATEGORIES):
            terms, frequencies = self._keyword_index[cat]
            count = len(terms)
            weights = {word: math.log((count + 1) / (frequencies[word] + 1)) + 1 for word in words}
            total = sum(weights.values())
            for chunk, chunk_words in zip(self._chunks[cat], terms):
                if not _matches_source(chunk, title, author):
                    continue
                score = sum(weights[w] for w in words & chunk_words) / total
                if score >= KEYWORD_MIN_COVERAGE:
                    meta = self._metadata(chunk, cat)
                    scored.append(RAGResult(chunk["text"], meta["book_title"], score, meta))
        return sorted(scored, key=lambda item: item.relevance, reverse=True)[:n_results]

    def get_context_for_query(self, query: str, max_chunks: int = 3, command: Optional[str] = None) -> str:
        category = self.COMMAND_CATEGORIES.get(command)
        results = self.search_similar(query, max_chunks, category=category)
        if not results:
            return ""
        parts = ["Retrieved material (fiction examples, craft material, or curated editorial summaries as labelled):"]
        for i, result in enumerate(results, 1):
            meta = result.metadata
            source = " — ".join(filter(None, [meta.get("author"), meta.get("book_title")]))
            section = " / ".join(filter(None, [meta.get("chapter"), meta.get("section")]))
            label = "curated summary; not verified verbatim" if meta["category"] == "editorial" else "corpus excerpt; transcription not independently verified"
            parts.append(f"[{i}] [{meta['category']}; {label}] {source or 'source unknown'}"
                         + (f" ({section})" if section else ""))
            parts.append(result.content[:500])
        return "\n".join(parts)
