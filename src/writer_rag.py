"""
RAG pipeline for Ash bot.
Books-only: ChromaDB + sentence-transformers over writing-craft book chunks.
"""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False


@dataclass
class RAGResult:
    """Single search result."""
    content: str
    source: str
    relevance: float
    metadata: dict


class WriterRAG:
    """RAG over writing-craft books with category-aware search."""

    # Command to category mapping
    COMMAND_CATEGORIES = {
        "block": "craft",
        "develop": "craft", 
        "character": "craft",
        "dialogue": "craft",
        "methodique": "craft",
        "style": "style",
        "corrector": "style",
        "editor": "editorial",
    }

    def __init__(self, data_dir: Optional[str] = None, use_local_embeddings: bool = True):
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent.parent / "data"
        self.use_local_embeddings = use_local_embeddings
        
        # Load all category files
        self.craft_chunks = self._load_json("craft.json")
        self.style_chunks = self._load_json("style.json")
        self.editorial_chunks = self._load_json("editorial.json")
        
        # Fallback: if separate files don't exist, use book_chunks.json for craft
        if not self.craft_chunks and not self.style_chunks and not self.editorial_chunks:
            all_chunks = self._load_json("book_chunks.json")
            # Put everything in craft for now (can be refined later)
            self.craft_chunks = all_chunks
            self.style_chunks = []
            self.editorial_chunks = []
        
        self.all_chunks = self.craft_chunks + self.style_chunks + self.editorial_chunks
        
        self.embedder = None
        self.collections = {}  # category -> collection

        if EMBEDDINGS_AVAILABLE and use_local_embeddings:
            self._init_embeddings()

    def _load_json(self, filename: str) -> list:
        path = self.data_dir / filename
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else data.get("chunks", [])
        return []

    def _init_embeddings(self):
        try:
            self.embedder = SentenceTransformer(
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
        except Exception as e:
            print(f"Writer RAG: embedding model load failed: {e}")
            self.embedder = None
            return
        if CHROMADB_AVAILABLE:
            self._init_chroma()

    def _init_chroma(self):
        chroma_path = self.data_dir / "chromadb_writers"
        chroma_path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(chroma_path))
        
        # Create separate collections for each category
        for category in ["craft", "style", "editorial"]:
            self.collections[category] = client.get_or_create_collection(
                name=f"writers_{category}",
                metadata={"hnsw:space": "cosine", "category": category}
            )
        
        # Index chunks for each category
        if self.craft_chunks:
            self._index_category("craft", self.craft_chunks)
        if self.style_chunks:
            self._index_category("style", self.style_chunks)
        if self.editorial_chunks:
            self._index_category("editorial", self.editorial_chunks)

    def _index_category(self, category: str, chunks: list):
        """Index chunks for a specific category."""
        collection = self.collections.get(category)
        if not collection:
            return
            
        if collection.count() > 0:
            print(f"Writer RAG: {category} collection already indexed")
            return
            
        all_chunks = []
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "id": f"{category}_{chunk.get('id', i)}",
                "text": chunk.get("text", ""),
                "metadata": {
                    "source": "book",
                    "category": category,
                    "book_title": chunk.get("book_title", ""),
                    "author": chunk.get("author", ""),
                    "chapter": chunk.get("chapter", ""),
                }
            })
        
        if not all_chunks:
            print(f"Writer RAG: no {category} chunks to index")
            return
            
        texts = [c["text"] for c in all_chunks]
        embeddings = self.embedder.encode(texts, show_progress_bar=True)
        batch_size = 5000
        
        for i in range(0, len(all_chunks), batch_size):
            end_idx = min(i + batch_size, len(all_chunks))
            batch = all_chunks[i:end_idx]
            collection.add(
                ids=[c["id"] for c in batch],
                embeddings=embeddings[i:end_idx].tolist(),
                documents=[c["text"] for c in batch],
                metadatas=[c["metadata"] for c in batch],
            )
        print(f"Writer RAG: indexed {len(all_chunks)} {category} chunks")

    def search_similar(self, query: str, n_results: int = 5, category: Optional[str] = None) -> list[RAGResult]:
        """
        Search for similar chunks.
        If category is specified, search only in that category.
        Otherwise search in all available collections.
        """
        if not self.embedder:
            return self._keyword_fallback(query, n_results, category)
        
        # Determine which collections to search
        if category and category in self.collections:
            collections_to_search = {category: self.collections[category]}
        else:
            collections_to_search = self.collections
        
        all_results = []
        q_emb = self.embedder.encode([query])
        
        for cat, collection in collections_to_search.items():
            if collection.count() == 0:
                continue
                
            results = collection.query(
                query_embeddings=q_emb.tolist(),
                n_results=min(n_results, collection.count())
            )
            
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i]
                dist = results["distances"][0][i]
                all_results.append(RAGResult(
                    content=doc,
                    source=meta.get("book_title", "book"),
                    relevance=1 - dist,
                    metadata=meta
                ))
        
        # Sort by relevance and return top n_results
        all_results.sort(key=lambda x: x.relevance, reverse=True)
        return all_results[:n_results]

    def _keyword_fallback(self, query: str, n_results: int, category: Optional[str] = None) -> list[RAGResult]:
        """Fallback keyword search when embeddings not available."""
        words = set(query.lower().split())
        scored = []
        
        # Filter chunks by category if specified
        chunks_to_search = self.all_chunks
        if category == "craft":
            chunks_to_search = self.craft_chunks
        elif category == "style":
            chunks_to_search = self.style_chunks
        elif category == "editorial":
            chunks_to_search = self.editorial_chunks
        
        for chunk in chunks_to_search:
            text = chunk.get("text", "")
            chunk_words = set(text.lower().split())
            overlap = len(words & chunk_words)
            if overlap > 0:
                scored.append(RAGResult(
                    content=text,
                    source=chunk.get("book_title", "book"),
                    relevance=overlap / max(len(words), 1),
                    metadata=chunk
                ))
        scored.sort(key=lambda x: x.relevance, reverse=True)
        return scored[:n_results]

    def get_context_for_query(self, query: str, max_chunks: int = 3, command: Optional[str] = None) -> str:
        """
        Get context for a query, optionally filtered by command type.
        """
        # Determine category from command
        category = None
        if command:
            category = self.COMMAND_CATEGORIES.get(command)
        
        # For methodique, search all categories to get diverse authors (King, Truby, McKee, Vogler, Snyder)
        if command == "methodique":
            category = None  # Search all categories
            max_chunks = 6   # Get more chunks for diversity
        
        results = self.search_similar(query, max_chunks, category=category)
        if not results:
            return ""
        
        # Header based on category
        if category == "craft":
            header = "Relevant advice from plot & craft books:\n"
        elif category == "style":
            header = "Relevant advice from style & language books:\n"
        elif category == "editorial":
            header = "Relevant advice from editorial & language books (Nora Gal, etc.):\n"
        else:
            header = "Relevant advice from writing books:\n"
            
        parts = [header]
        for i, r in enumerate(results, 1):
            author = r.metadata.get("author", "")
            book = r.metadata.get("book_title", "")
            chapter = r.metadata.get("chapter", "")
            cat = r.metadata.get("category", "")
            prefix = f"[{i}]"
            if cat:
                prefix += f" [{cat}]"
            parts.append(f"{prefix} {author} — «{book}»" + (f" ({chapter})" if chapter else ""))
            parts.append(r.content[:500] + ("..." if len(r.content) > 500 else ""))
        return "\n".join(parts)
