"""Bounded, lossless input partitioning and editorial context helpers."""

import difflib
import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path


def token_bound(text: str) -> int:
    """Conservative tokenizer-independent upper bound: at most one token per UTF-8 byte."""
    return len(text.encode("utf-8"))


def split_text(text: str, budget: int = 2400) -> list[str]:
    """Partition without dropping whitespace or splitting Unicode code points."""
    if budget < 4:
        raise ValueError("budget must fit a Unicode code point")
    parts = []
    start = 0
    while start < len(text):
        end, cost = start, 0
        while end < len(text):
            size = len(text[end].encode("utf-8"))
            if cost + size > budget:
                break
            cost += size
            end += 1
        if end < len(text):
            # Prefer paragraph/sentence/word boundaries without making tiny chunks.
            candidate = text[start:end]
            for pattern in (r"\n+", r"[.!?…]\s+", r"\s+"):
                matches = list(re.finditer(pattern, candidate))
                if matches and matches[-1].end() >= len(candidate) // 2:
                    end = start + matches[-1].end()
                    break
        parts.append(text[start:end])
        start = end
    assert "".join(parts) == text
    return parts


@lru_cache(maxsize=1)
def cliches() -> tuple[str, ...]:
    path = Path(__file__).resolve().parent.parent / "data" / "editor_cliches.json"
    entries = json.loads(path.read_text(encoding="utf-8"))
    seen, result = set(), []
    for raw in entries:
        if not isinstance(raw, str):
            continue
        entry = unicodedata.normalize("NFKC", raw).strip()
        entry = re.sub(r"\s*\([^)]*\)\s*$", "", entry).strip()
        # Templates and single words are too ambiguous for literal matching.
        if not entry or "[" in entry or "]" in entry or len(entry.split()) < 2:
            continue
        key = entry.casefold().replace("ё", "е")
        if key not in seen:
            seen.add(key)
            result.append(entry)
    return tuple(result)


def cliche_candidates(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold().replace("ё", "е")
    return [entry for entry in cliches()
            if re.search(r"(?<!\w)" + re.escape(entry.casefold().replace("ё", "е"))
                         + r"(?!\w)", normalized)]


def correction_changes(original: str, corrected: str, limit: int = 30) -> list[tuple[str, str]]:
    changes = []
    for tag, a, b, c, d in difflib.SequenceMatcher(None, original, corrected).get_opcodes():
        if tag != "equal" and (original[a:b].strip() or corrected[c:d].strip()):
            before, after = original[a:b], corrected[c:d]
            if len(before) > 150:
                before = before[:150] + "…"
            if len(after) > 150:
                after = after[:150] + "…"
            changes.append((before, after))
            if len(changes) >= limit:
                break
    return changes


def apply_proofreading_edits(chunk: str, edits: object) -> str:
    """Apply uniquely anchored, non-overlapping changes to the original fragment."""
    if not isinstance(edits, list):
        raise ValueError("Expected edit list")
    operations = []
    for edit in edits:
        if not isinstance(edit, dict) or set(edit) != {"old", "new"}:
            raise ValueError("Invalid edit")
        old, new = edit["old"], edit["new"]
        if not isinstance(old, str) or not isinstance(new, str) or not old:
            raise ValueError("Invalid edit text")
        start = chunk.find(old)
        if start < 0 or chunk.find(old, start + 1) >= 0:
            raise ValueError("Edit anchor is absent or ambiguous")
        if "\n" in old or "\n" in new or "\r" in old or "\r" in new:
            raise ValueError("Edit changes paragraph structure")
        if len(old) > 500 or difflib.SequenceMatcher(None, old, new, autojunk=False).ratio() < 0.5:
            raise ValueError("Edit is too extensive for proofreading")
        operations.append((start, start + len(old), new))
    operations.sort()
    if any(left[1] > right[0] for left, right in zip(operations, operations[1:])):
        raise ValueError("Overlapping edits")
    result = chunk
    for start, end, replacement in reversed(operations):
        result = result[:start] + replacement + result[end:]
    return result


def discussion_context(original: str, result: str, question: str, budget: int = 24000) -> tuple[str, bool]:
    """Use all content when it fits; otherwise select relevant, labelled passages."""
    full = f"ORIGINAL:\n{original}\n\nPREVIOUS RESULT:\n{result}"
    if token_bound(full) <= budget:
        return full, False
    words = set(re.findall(r"\w{3,}", question.casefold()))
    candidates = []
    for label, value in (("ORIGINAL", original), ("PREVIOUS RESULT", result)):
        for index, chunk in enumerate(split_text(value, 1600), 1):
            terms = set(re.findall(r"\w{3,}", chunk.casefold()))
            score = len(words & terms)
            # Preserve both sides and honour explicit passage numbers.
            if re.search(rf"(?<!\d){index}(?!\d)", question):
                score += 2
            candidates.append((score, label, index, chunk))
    selected, used = [], 0
    for score, label, index, chunk in sorted(candidates, key=lambda x: (-x[0], x[2], x[1])):
        item = f"{label}, part {index}:\n{chunk}\n\n"
        if used + token_bound(item) <= budget:
            selected.append(item)
            used += token_bound(item)
    return "".join(selected), True
