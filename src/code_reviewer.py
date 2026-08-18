"""
Code change reviewer for Ash bot.

Tracks file hashes + content snapshots, builds real diffs on startup,
and asks the LLM only about user-visible changes.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Files to track for user-facing changelog detection
TRACKED_FILES = [
    "src/telegram_bot.py",
    "src/writer_bot.py",
    "src/i18n.py",
    "src/writer_rag.py",
    "src/lang_utils.py",
    "src/word_stats.py",
    "src/code_reviewer.py",
]

# Known bot commands — never announce these as "new"
EXISTING_COMMANDS = [
    "start", "help", "lang", "switchlang", "reset", "block", "develop",
    "character", "dialogue", "prompt", "idea", "feedback", "style", "roast",
    "praise", "corrector", "editor", "count_me", "stats", "lobster", "pun",
    "xyx", "methodique", "methodichque", "cite", "cite_off", "cite_on",
    "cite_when", "summary", "cry_baby", "dev_feedback", "done",
    "confo_enable37", "upload", "admin", "debug",
]

# Caps so prompts stay focused
MAX_DIFF_CHARS_PER_FILE = 4500
MAX_TOTAL_DIFF_CHARS = 14000
MAX_NEW_FILE_CHARS = 2500

WITTY_COMMENTS_RU = [
    "Код, как вино — с каждым обновлением становится лучше.",
    "Баги исправлены, новые баги ещё не обнаружены.",
    "Обновление установлено. Надеюсь, ничего не сломал.",
    "Код переписан. Старые костыли заменены на новые, более элегантные.",
    "Код обновлён. Если что-то сломалось — это специально.",
]

WITTY_COMMENTS_EN = [
    "Code, like wine — better with every update.",
    "Bugs fixed. New bugs not yet discovered.",
    "Update installed. Hopefully I didn't break anything.",
    "Code rewritten. Old crutches replaced with newer, more elegant ones.",
    "Code updated. If something broke — that was on purpose.",
]

# Back-compat alias used by older tests/scripts
WITTY_COMMENTS = WITTY_COMMENTS_RU


def calculate_file_hash(file_path: Path) -> str:
    """Calculate short SHA256 hash of file contents."""
    try:
        with open(file_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except Exception:
        return ""


def _backup_dir(data_dir: Path) -> Path:
    return data_dir / "code_backup"


def _backup_path(data_dir: Path, rel_path: str) -> Path:
    # Keep nested paths flat but stable: src/telegram_bot.py -> src_telegram_bot.py
    return _backup_dir(data_dir) / rel_path.replace("\\", "/").replace("/", "_")


def load_stored_hashes(data_dir: Path) -> Dict[str, str]:
    """Load previously stored file hashes."""
    hash_file = data_dir / "code_hashes.json"
    if not hash_file.exists():
        return {}
    try:
        with open(hash_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except Exception:
        pass
    return {}


def save_hashes(data_dir: Path, hashes: Dict[str, str]) -> None:
    """Save current file hashes."""
    hash_file = data_dir / "code_hashes.json"
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        with open(hash_file, "w", encoding="utf-8") as f:
            json.dump(hashes, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Failed to save hashes: {e}")


def load_file_backup(data_dir: Path, rel_path: str) -> Optional[str]:
    """Load previous snapshot for a tracked file, if any."""
    backup_file = _backup_path(data_dir, rel_path)
    if not backup_file.exists():
        return None
    try:
        return backup_file.read_text(encoding="utf-8")
    except Exception:
        return None


def save_file_backup(data_dir: Path, rel_path: str, content: str) -> None:
    """Save content snapshot used for the next real diff."""
    backup_file = _backup_path(data_dir, rel_path)
    try:
        backup_file.parent.mkdir(parents=True, exist_ok=True)
        backup_file.write_text(content, encoding="utf-8")
    except Exception as e:
        print(f"Failed to save backup for {rel_path}: {e}")


def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def get_changed_files(project_root: Path, stored_hashes: Dict[str, str]) -> List[Tuple[str, str, str]]:
    """
    Compare current files with stored hashes.
    Returns list of (filename, old_hash, new_hash) for changed files.
    """
    changed: List[Tuple[str, str, str]] = []
    for rel_path in TRACKED_FILES:
        file_path = project_root / rel_path
        if not file_path.exists():
            continue
        current_hash = calculate_file_hash(file_path)
        old_hash = stored_hashes.get(rel_path, "")
        if current_hash and current_hash != old_hash:
            changed.append((rel_path, old_hash, current_hash))
    return changed


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 20)] + "\n...[truncated]..."


def build_unified_diff(rel_path: str, old_content: str, new_content: str) -> str:
    """Build a compact unified diff for one file."""
    old_lines = old_content.splitlines()
    new_lines = new_content.splitlines()

    if not old_content and new_content:
        body = _truncate(new_content, MAX_NEW_FILE_CHARS)
        return f"--- /dev/null\n+++ b/{rel_path}\n@@ new file @@\n{body}"

    if old_content and not new_content:
        return f"--- a/{rel_path}\n+++ /dev/null\n@@ deleted file @@"

    diff_lines = list(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{rel_path}",
            tofile=f"b/{rel_path}",
            lineterm="",
            n=3,
        )
    )
    if not diff_lines:
        return ""

    diff_text = "\n".join(diff_lines)
    return _truncate(diff_text, MAX_DIFF_CHARS_PER_FILE)


def extract_commands_from_text(content: str) -> List[str]:
    """Extract command names from code/diff text."""
    commands: List[str] = []
    patterns = [
        r'Command\(\s*["\']/?(\w+)["\']',
        r'@self\.dp\.message\(\s*Command\(\s*["\']/?(\w+)["\']',
        r'async def cmd_(\w+)\(',
        r'Command\(["\'](\w+)["\']',
    ]
    for pattern in patterns:
        for match in re.findall(pattern, content):
            cmd = match.lower().lstrip("/")
            if cmd and cmd not in commands:
                commands.append(cmd)
    return commands


def extract_new_commands_from_diff(diff_text: str) -> List[str]:
    """
    Commands that appear only on added lines of a diff and are not already known.
    """
    added_chunks: List[str] = []
    for line in diff_text.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            added_chunks.append(line[1:])
    added_text = "\n".join(added_chunks)
    found = extract_commands_from_text(added_text)
    known = {c.lower().lstrip("/") for c in EXISTING_COMMANDS}
    return [c for c in found if c not in known]


def collect_change_set(
    project_root: Path,
    data_dir: Path,
    changed_files: List[Tuple[str, str, str]],
) -> Dict:
    """
    Build structured change info with real diffs and new-command hints.
    """
    file_diffs: List[Dict[str, str]] = []
    new_commands: List[str] = []
    total_chars = 0

    for rel_path, old_hash, new_hash in changed_files:
        file_path = project_root / rel_path
        new_content = read_text_file(file_path) if file_path.exists() else ""
        old_content = load_file_backup(data_dir, rel_path)
        if old_content is None:
            # No snapshot yet for this path: avoid dumping whole file as "changes".
            if not old_hash:
                diff_text = (
                    f"File newly tracked: {rel_path}\n"
                    f"(no previous snapshot; hash {new_hash})"
                )
            else:
                diff_text = (
                    f"File changed: {rel_path}\n"
                    f"(previous snapshot missing; old_hash={old_hash}, new_hash={new_hash})\n"
                    f"Current head (truncated):\n{_truncate(new_content, 1200)}"
                )
        else:
            diff_text = build_unified_diff(rel_path, old_content, new_content)

        if not diff_text.strip():
            continue

        remaining = MAX_TOTAL_DIFF_CHARS - total_chars
        if remaining <= 200:
            file_diffs.append(
                {
                    "path": rel_path,
                    "old_hash": old_hash,
                    "new_hash": new_hash,
                    "diff": f"File changed: {rel_path} (diff omitted — budget exhausted)",
                }
            )
            break

        clipped = _truncate(diff_text, min(MAX_DIFF_CHARS_PER_FILE, remaining))
        total_chars += len(clipped)
        file_diffs.append(
            {
                "path": rel_path,
                "old_hash": old_hash,
                "new_hash": new_hash,
                "diff": clipped,
            }
        )
        for cmd in extract_new_commands_from_diff(clipped):
            if cmd not in new_commands:
                new_commands.append(cmd)

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "changed_files_count": len(file_diffs),
        "new_commands": new_commands,
        "file_diffs": file_diffs,
        "has_user_signal": bool(new_commands) or any(
            _diff_looks_user_facing(item["diff"]) for item in file_diffs
        ),
    }


def _diff_looks_user_facing(diff_text: str) -> bool:
    """
    Cheap heuristic: ignore pure import/comment/whitespace-only diffs.
    Still lets the LLM decide final user-facing wording.
    """
    meaningful = 0
    for line in diff_text.splitlines():
        if not (line.startswith("+") or line.startswith("-")):
            continue
        if line.startswith("+++") or line.startswith("---"):
            continue
        body = line[1:].strip()
        if not body:
            continue
        if body.startswith("#") or body.startswith('"""') or body.startswith("'''"):
            continue
        if body.startswith("import ") or body.startswith("from "):
            continue
        if body.startswith("print(") or body.startswith("logger."):
            continue
        meaningful += 1
        if meaningful >= 2:
            return True
    lower = diff_text.lower()
    if "command(" in lower or "async def cmd_" in lower or "t(lang," in lower:
        return True
    return meaningful > 0


def get_witty_comment(num_changes: int = 1, lang: str = "ru") -> str:
    """Get a witty comment based on language."""
    _ = num_changes  # kept for call-site compatibility
    if lang == "ru":
        return random.choice(WITTY_COMMENTS_RU)
    return random.choice(WITTY_COMMENTS_EN)


def _is_empty_changelog(text: str) -> bool:
    if not text:
        return True
    cleaned = text.strip().lower()
    # Exact no-op answers from the model
    if cleaned in {"нет", "none", "n/a", "na", "empty", "no", "-"}:
        return True
    if len(cleaned) < 8:
        return True
    empty_markers = [
        "нет пользовательских",
        "нет видимых",
        "no user-facing",
        "no user facing",
        "no visible changes",
        "no changes",
        "внутренние улучшения",
        "internal improvements",
        "under the hood",
        "технические изменения",
        "technical only",
    ]
    return any(marker in cleaned for marker in empty_markers)


_THINK_TAG_RE = re.compile(
    r"<(think|thinking|thought|reasoning)\b[^>]*>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)
_THINK_OPEN_RE = re.compile(
    r"<(think|thinking|thought|reasoning)\b[^>]*>[\s\S]*",
    re.IGNORECASE,
)
_THINK_ALT_RE = re.compile(
    r"(?:\[/?think\]|\[/thinking\]|◁/?think▷)",
    re.IGNORECASE,
)
_ANSWER_MARKER_RE = re.compile(
    r"(?is)(?:^|\n)\s*(?:"
    r"final\s+answer|answer|changelog|итоговый\s+changelog|итог|итого|ответ"
    r")\s*[:\-]\s*"
)
_TECHNICAL_RE = re.compile(
    r"(?ix)"
    r"(\b(src/|def\s+|class\s+|import\s+|from\s+\S+\s+import|async\s+def|"
    r"unified\s+diff|file\s+hash|code\s+hash|refactor(?:ing)?\b|traceback\b|"
    r"max_tokens|api[_\s]?key)|"
    r"\w+\.py\b|`[^`]+`|"
    r"[a-z_][a-z0-9_]{2,}\.[a-z_][a-z0-9_]+\()"
)
_ANALYSIS_PREFIX_RE = re.compile(
    r"(?ix)^(?:"
    r"looking at|let me|i (?:will|need to|see|notice|think)|"
    r"the (?:diff|change|file|code)|this (?:diff|change|file|update)|"
    r"смотрю|давайте|анализ|рассужд|в файле|в диффе"
    r")"
)
MAX_CHANGELOG_BULLETS = 3
MAX_BULLET_CHARS = 140
MAX_CHANGELOG_CHARS = 420


def _extract_completion_text(response) -> str:
    """Take only the visible assistant text. Never include reasoning fields."""
    try:
        message = response.choices[0].message
    except Exception:
        return ""

    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")

    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                part_type = str(part.get("type") or "").lower()
                if part_type in {"thinking", "reasoning", "thought"}:
                    continue
                parts.append(str(part.get("text") or part.get("content") or ""))
        content = "".join(parts)

    return (content or "").strip()


def _strip_model_thoughts(text: str) -> str:
    """Hide chain-of-thought / analysis and keep only the final answer."""
    if not text:
        return ""

    cleaned = text.replace("\r\n", "\n")
    cleaned = _THINK_TAG_RE.sub("", cleaned)
    cleaned = _THINK_OPEN_RE.sub("", cleaned)
    cleaned = _THINK_ALT_RE.sub("", cleaned)

    marker = _ANSWER_MARKER_RE.search(cleaned)
    if marker:
        cleaned = cleaned[marker.end():]

    return cleaned.strip()


def _looks_technical(line: str) -> bool:
    if _TECHNICAL_RE.search(line):
        return True
    if _ANALYSIS_PREFIX_RE.search(line.strip()):
        return True
    return False


def _sanitize_changelog(text: str) -> str:
    """
    Turn a raw model reply into a short user-facing changelog.
    Drops hidden thoughts, analysis, and technical leftovers.
    """
    cleaned = _strip_model_thoughts(text)
    cleaned = re.sub(r"^```(?:\w+)?\s*|\s*```$", "", cleaned).strip()
    if not cleaned:
        return ""
    # Whole-reply empty answers like NONE / НЕТ should stay silent.
    if "\n" not in cleaned and _is_empty_changelog(cleaned):
        return ""

    raw_lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    has_explicit_bullets = any(
        re.match(r"^(?:[-*•]|—|\d+[.)])\s+", raw) for raw in raw_lines
    )
    # Several unmarked lines are leftover reasoning, not a release note.
    if not has_explicit_bullets and len(raw_lines) > 1:
        return ""

    bullets: List[str] = []
    for raw in raw_lines:
        was_bullet = bool(re.match(r"^(?:[-*•]|—|\d+[.)])\s+", raw))
        if has_explicit_bullets and not was_bullet:
            continue
        line = re.sub(r"^(?:[-*•]|—|\d+[.)])\s+", "", raw).strip()
        line = line.strip("`\"'")
        if not line or _is_empty_changelog(line):
            continue
        if _looks_technical(line):
            continue
        # Long prose is almost always leftover reasoning, not a release note.
        if len(line) > MAX_BULLET_CHARS:
            if not was_bullet:
                continue
            line = line[: MAX_BULLET_CHARS - 1].rstrip(" ,;:") + "…"
        if line not in bullets:
            bullets.append(line)
        if len(bullets) >= MAX_CHANGELOG_BULLETS:
            break

    if not bullets:
        return ""

    result = "\n".join(f"• {item}" for item in bullets)
    if len(result) > MAX_CHANGELOG_CHARS:
        result = result[: MAX_CHANGELOG_CHARS - 1].rstrip() + "…"
    return result


def generate_changelog_with_llm(
    writer_bot,
    changed_files: List[Tuple[str, str, str]],
    project_root: Path,
    lang: str = "ru",
    change_set: Optional[Dict] = None,
) -> str:
    """
    Use LLM to generate a human-readable changelog from real diffs.
    Returns empty string when there is nothing user-visible to announce.
    """
    data_dir = project_root / "data"
    if change_set is None:
        change_set = collect_change_set(project_root, data_dir, changed_files)

    file_diffs = change_set.get("file_diffs") or []
    if not file_diffs:
        return ""

    # If the diff is pure noise and no new commands, skip LLM entirely.
    if not change_set.get("has_user_signal") and not change_set.get("new_commands"):
        print("[CHANGELOG] Diff looks non-user-facing; skipping LLM")
        return ""

    diff_sections = []
    for item in file_diffs:
        diff_sections.append(
            f"### {item['path']} ({item.get('old_hash') or '∅'} → {item.get('new_hash')})\n"
            f"{item['diff']}"
        )
    diff_content = "\n\n".join(diff_sections)
    new_cmds = ", ".join(f"/{c}" for c in change_set.get("new_commands") or []) or "—"
    known_cmds = ", ".join(f"/{c}" for c in EXISTING_COMMANDS)

    if lang == "ru":
        prompt = f"""Напиши короткий changelog для обычных пользователей Telegram-бота Ash — писательского редактора.

Ниже — РЕАЛЬНЫЙ diff. Пиши ТОЛЬКО по нему. Не выдумывай.

Правила:
1) Только то, что человек заметит в боте: новые/починенные команды, ответы, тексты, режимы.
2) Игнорируй код, рефакторинг, импорты, хеши, логи, внутренние функции.
3) Если видимых изменений нет — верни ровно: НЕТ
4) 1–3 очень коротких пункта. Без вступлений, без анализа, без markdown.
5) Не называй «новой» команду из списка известных, если она лишь правилась.
6) Не пиши ход мыслей, не цитируй diff, не называй файлы и функции.
7) Тон: конкретный и едкий, как у старого редактора.

Известные команды (не новые сами по себе): {known_cmds}
Новые команды по diff (подсказка, может быть пусто): {new_cmds}

DIFF:
{diff_content}

Верни только готовые пункты или «НЕТ». Никакого рассуждения."""
        system = (
            "Ты редактор коротких пользовательских release notes. "
            "Отвечай только финальными пунктами. Никакого анализа и скрытых мыслей. "
            "Если сомневаешься — ответь НЕТ."
        )
    else:
        prompt = f"""Write a short changelog for ordinary users of Ash, a writing-coach Telegram bot.

Below is a REAL diff. Write ONLY from it. Do not invent features.

Rules:
1) Only what a person will notice in the bot: new/fixed commands, replies, copy, modes.
2) Ignore code, refactoring, imports, hashes, logs, internal helpers.
3) If there are no user-visible changes — return exactly: NONE
4) 1–3 very short bullets. No preamble, no analysis, no markdown.
5) Do not call a command "new" if it only appears in the known list and was merely edited.
6) Do not show your reasoning, quote the diff, or name files and functions.
7) Tone: specific and cynical, like an old editor.

Known commands (not new by themselves): {known_cmds}
New commands hinted by diff (may be empty): {new_cmds}

DIFF:
{diff_content}

Return only the finished bullets or NONE. No reasoning."""
        system = (
            "You write short user-facing release notes. "
            "Reply with the final bullets only. No analysis and no hidden thoughts. "
            "If unsure, answer NONE."
        )

    try:
        if writer_bot and getattr(writer_bot, "client", None):
            response = writer_bot.client.chat.completions.create(
                model=writer_bot.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.15,
                max_tokens=180,
            )
            changelog = _sanitize_changelog(_extract_completion_text(response))
            if not changelog:
                return ""

            witty = get_witty_comment(change_set.get("changed_files_count", 1), lang)
            return f"{witty}\n\n{changelog}"
    except Exception as e:
        print(f"LLM changelog generation failed: {e}")

    # Honest non-LLM fallback: only if we clearly saw new commands
    new_commands = change_set.get("new_commands") or []
    if new_commands:
        witty = get_witty_comment(change_set.get("changed_files_count", 1), lang)
        if lang == "ru":
            lines = [witty, ""] + [f"• Новая команда /{c}" for c in new_commands[:5]]
        else:
            lines = [witty, ""] + [f"• New command /{c}" for c in new_commands[:5]]
        return "\n".join(lines)

    # No reliable user-facing signal — stay silent instead of inventing noise
    return ""


def generate_user_friendly_fallback(analysis: Dict, lang: str = "ru") -> str:
    """
    Minimal fallback. Prefer silence over fake 'improvements'.
    Kept for backward compatibility with older callers/tests.
    """
    new_commands = analysis.get("new_commands") or []
    if not new_commands:
        return ""

    witty = get_witty_comment(analysis.get("changed_files_count", 1), lang)
    if lang == "ru":
        lines = [witty] + [f"• Новая команда /{c}" for c in new_commands[:5]]
    else:
        lines = [witty] + [f"• New command /{c}" for c in new_commands[:5]]
    return "\n".join(lines)


def analyze_code_changes(
    project_root: Path,
    changed_files: List[Tuple[str, str, str]],
) -> Dict[str, List[str]]:
    """
    Lightweight analysis used by older tooling.
    Now based on diffs/snapshots when available.
    """
    data_dir = project_root / "data"
    change_set = collect_change_set(project_root, data_dir, changed_files)
    return {
        "new_commands": change_set.get("new_commands", []),
        "all_commands": change_set.get("new_commands", []),
        "has_changes": bool(change_set.get("file_diffs")),
        "changed_files_count": change_set.get("changed_files_count", 0),
    }


def collect_current_hashes(project_root: Path) -> Dict[str, str]:
    current_hashes: Dict[str, str] = {}
    for rel_path in TRACKED_FILES:
        file_path = project_root / rel_path
        if file_path.exists():
            current_hashes[rel_path] = calculate_file_hash(file_path)
    return current_hashes


def save_current_snapshots(project_root: Path) -> None:
    """Save content snapshots for all tracked files."""
    data_dir = project_root / "data"
    for rel_path in TRACKED_FILES:
        file_path = project_root / rel_path
        if not file_path.exists():
            continue
        content = read_text_file(file_path)
        if content:
            save_file_backup(data_dir, rel_path, content)


def save_current_hashes(project_root: Path) -> None:
    """
    Calculate and save current file hashes + content snapshots.
    Call after changelog generation / successful notify path.
    """
    data_dir = project_root / "data"
    current_hashes = collect_current_hashes(project_root)
    save_hashes(data_dir, current_hashes)
    save_current_snapshots(project_root)
    print(
        f"[HASHES] Saved {len(current_hashes)} file hashes + snapshots to "
        f"{data_dir / 'code_hashes.json'}"
    )


def check_and_generate_changelog(
    project_root: Path,
    writer_bot,
    admin_id: int,
    lang: str = "ru",
    should_save_hashes: bool = True,
) -> Optional[str]:
    """
    Main entry point: check for changes and generate changelog.
    Returns changelog text if user-visible changes detected, None otherwise.

    Hashes/snapshots are saved only when should_save_hashes=True.
    Startup path should pass False, generate RU+EN, then call save_current_hashes()
    once if either language produced a changelog (or after baselining).
    """
    _ = admin_id  # reserved for callers / future per-admin behavior

    try:
        data_dir = project_root / "data"
        stored_hashes = load_stored_hashes(data_dir)

        # First run / empty cache: baseline silently, do not invent a release.
        if not stored_hashes:
            print("[CHANGELOG] No baseline hashes — saving baseline, no changelog")
            save_current_hashes(project_root)
            return None

        changed_files = get_changed_files(project_root, stored_hashes)
        print(f"[CHANGELOG] Stored hashes: {len(stored_hashes)} files")
        print(f"[CHANGELOG] Changed files: {[c[0] for c in changed_files]}")

        if not changed_files:
            print("[CHANGELOG] No changes detected")
            return None

        change_set = collect_change_set(project_root, data_dir, changed_files)
        print(
            f"[CHANGELOG] Diff set: {change_set.get('changed_files_count')} files, "
            f"new_commands={change_set.get('new_commands')}, "
            f"user_signal={change_set.get('has_user_signal')}"
        )

        print(f"[CHANGELOG] Generating changelog for {len(changed_files)} changed files...")
        changelog = generate_changelog_with_llm(
            writer_bot,
            changed_files,
            project_root,
            lang,
            change_set=change_set,
        )

        if should_save_hashes:
            save_current_hashes(project_root)
            print("[CHANGELOG] Hashes+snapshots saved (should_save_hashes=True)")

        if not changelog or not changelog.strip():
            return None
        return changelog

    except Exception as e:
        print(f"Changelog generation error: {e}")
        raise
