#!/usr/bin/env python3
"""
Chunk books for RAG. PDF and TXT only.
Usage: python scripts/chunk_books.py --input books/craft/ --output data/chunks_craft.json --category craft
"""

import argparse
import json
import re
from pathlib import Path
from typing import Optional

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("Warning: tiktoken not installed. Using word-based chunking.")

try:
    import pypdf
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False



def extract_text_from_pdf(path: Path) -> str:
    """Extract text from PDF file. Opens in binary mode for better compatibility."""
    if not PYPDF_AVAILABLE:
        raise RuntimeError("pypdf not installed. pip install pypdf")
    
    text = ""
    
    # Open in binary mode - this is crucial for some PDFs
    with open(path, 'rb') as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
    
    return text




def chunk_by_tokens(text: str, max_tokens: int = 500, overlap: int = 100) -> list[str]:
    """Split text into chunks by token count."""
    if max_tokens <= 0 or overlap < 0 or overlap >= max_tokens:
        raise ValueError("max_tokens must be positive and 0 <= overlap < max_tokens")
    if TIKTOKEN_AVAILABLE:
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text)
        chunks = []
        i = 0
        while i < len(tokens):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk = enc.decode(chunk_tokens)
            chunks.append(chunk)
            i += max_tokens - overlap
        return chunks
    else:
        # Fallback: word-based chunking
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk_words = words[i:i + max_tokens]
            chunks.append(" ".join(chunk_words))
            i += max_tokens - overlap
        return chunks


def chunk_by_paragraphs(text: str, max_tokens: int = 500) -> list[str]:
    """Split text into chunks by paragraphs, respecting token limit."""
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    if TIKTOKEN_AVAILABLE:
        enc = tiktoken.get_encoding("cl100k_base")
        get_length = lambda x: len(enc.encode(x))
    else:
        get_length = lambda x: len(x.split())
    
    for para in paragraphs:
        para_tokens = get_length(para)
        if para_tokens > max_tokens:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk, current_tokens = [], 0
            chunks.extend(chunk_by_tokens(para, max_tokens, overlap=0))
            continue
        
        if current_tokens + para_tokens > max_tokens and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [para]
            current_tokens = para_tokens
        else:
            current_chunk.append(para)
            current_tokens += para_tokens
    
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
    
    return chunks


def parse_filename(file_path: Path) -> tuple[str, str]:
    """Parse author and book title from filename like 'Author_Title.pdf'."""
    stem = file_path.stem  # filename without extension
    # Only an explicit underscore separates author from title. A space may be
    # part of a multiword author or title and is not reliable evidence.
    parts = stem.split('_', 1)
    if len(parts) == 2:
        author, title = parts
        return author.strip(), title.replace('_', ' ').strip()
    # If only one part, use it as title and Unknown as author
    return "Unknown", stem.replace('_', ' ').strip()


def process_book(
    file_path: Path,
    category: str,
    author: Optional[str] = None,
    book_title: Optional[str] = None,
    max_tokens: int = 500,
    overlap: int = 100,
    by_paragraphs: bool = True,
) -> list[dict]:
    """Process a single book file into chunks."""
    # Auto-parse from filename if not provided
    if author is None or book_title is None:
        parsed_author, parsed_title = parse_filename(file_path)
        author = author or parsed_author
        book_title = book_title or parsed_title
    
    if max_tokens <= 0 or overlap < 0 or overlap >= max_tokens:
        raise ValueError("max_tokens must be positive and 0 <= overlap < max_tokens")
    suffix = file_path.suffix.lower()

    if suffix == '.pdf':
        text = extract_text_from_pdf(file_path)
    elif suffix == '.txt':
        text = file_path.read_text(encoding='utf-8')
    else:
        raise ValueError(f"Unsupported format: {suffix} (only PDF and TXT)")
    
    # Clean text: preserve paragraph structure, only normalize horizontal whitespace
    text = re.sub(r'[ \t]+', ' ', text)  # collapse spaces/tabs
    text = re.sub(r'\n{3,}', '\n\n', text)  # collapse 3+ newlines to 2
    text = text.strip()
    
    # Chunk: try paragraphs first, fallback to tokens if too few chunks
    if by_paragraphs:
        chunks = chunk_by_paragraphs(text, max_tokens)
        # If too few chunks (paragraph breaks are rare in this PDF), use token chunking
        if len(chunks) < 5:
            chunks = chunk_by_tokens(text, max_tokens, overlap)
    else:
        chunks = chunk_by_tokens(text, max_tokens, overlap)

    
    # Build output
    results = []
    base_id = file_path.stem.replace(' ', '_').lower()
    
    for i, chunk_text in enumerate(chunks):
        # Detect chapter if possible
        chapter = None
        # An inline reference to another chapter is not this chunk's chapter.
        chapter_match = re.match(r'^\s*((?:Chapter|Глава)\s+(?:\d+|[IVX]+)[^\n]{0,80})\n',
                                 chunk_text, re.IGNORECASE)
        if chapter_match:
            chapter = chapter_match.group(1).strip()
        
        chunk_data = {
            "id": f"{base_id}_{i+1}",
            "text": chunk_text,
            "category": category,
            "author": author or "Unknown",
            "book_title": book_title or file_path.stem,
            "chapter": chapter or "",
            "chunk_index": i + 1,
            "total_chunks": len(chunks),
        }
        results.append(chunk_data)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Chunk books for RAG")
    parser.add_argument("--input", "-i", required=True, help="Input file or directory")
    parser.add_argument("--output", "-o", required=True, help="Output JSON file")
    parser.add_argument("--category", "-c", required=True, help="Category: craft, style, plot, character")
    parser.add_argument("--author", "-a", help="Author name (for single file)")
    parser.add_argument("--book", "-b", help="Book title (for single file)")
    parser.add_argument("--max-tokens", type=int, default=500, help="Max tokens per chunk")
    parser.add_argument("--overlap", type=int, default=100, help="Token overlap between chunks")
    parser.add_argument("--by-paragraphs", action="store_true", default=True, help="Chunk by paragraphs")
    parser.add_argument("--by-tokens", action="store_true", help="Chunk by exact token count")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    all_chunks = []
    
    # Determine files to process
    if input_path.is_file():
        files = [input_path]
    else:
        # Debug: list all files in directory
        print(f"Looking in: {input_path.absolute()}")
        all_files = list(input_path.iterdir())
        print(f"All files in directory: {[f.name for f in all_files]}")
        
        # Case-insensitive glob for PDF and TXT
        files = []
        for pattern in ["*.pdf", "*.PDF", "*.Pdf", "*.txt", "*.TXT", "*.Txt"]:
            matched = list(input_path.glob(pattern))
            if matched:
                print(f"  Pattern '{pattern}' matched: {[f.name for f in matched]}")
            files.extend(matched)
        # Remove duplicates while preserving order
        seen = set()
        unique_files = []
        for f in files:
            if f not in seen:
                seen.add(f)
                unique_files.append(f)
        files = unique_files

    
    print(f"Found {len(files)} files to process")

    
    for file_path in files:
        print(f"Processing: {file_path.name}")
        try:
            chunks = process_book(
                file_path,
                category=args.category,
                author=args.author,
                book_title=args.book,
                max_tokens=args.max_tokens,
                overlap=args.overlap,
                by_paragraphs=not args.by_tokens,
            )
            all_chunks.extend(chunks)
            print(f"  → {len(chunks)} chunks")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    
    print(f"\nTotal: {len(all_chunks)} chunks saved to {output_path}")


if __name__ == "__main__":
    main()
