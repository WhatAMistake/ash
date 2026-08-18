"""Smoke tests for hash/snapshot changelog detection."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from code_reviewer import (
    _extract_completion_text,
    _is_empty_changelog,
    _sanitize_changelog,
    build_unified_diff,
    check_and_generate_changelog,
    extract_new_commands_from_diff,
    generate_changelog_with_llm,
    generate_user_friendly_fallback,
    get_changed_files,
    load_stored_hashes,
    save_current_hashes,
)


def test_empty_changelog_markers():
    assert _is_empty_changelog("")
    assert _is_empty_changelog("НЕТ")
    assert _is_empty_changelog("NONE")
    assert _is_empty_changelog("Internal improvements")
    assert not _is_empty_changelog("• Fixed /pun so it sees context")


def test_unified_diff_contains_added_line():
    diff = build_unified_diff("src/i18n.py", "old line\n", "old line\nnew line\n")
    assert "src/i18n.py" in diff
    assert "+new line" in diff


def test_extract_new_commands_ignores_known():
    diff = (
        "--- a/src/telegram_bot.py\n"
        "+++ b/src/telegram_bot.py\n"
        "+        @self.dp.message(Command(\"start\"))\n"
        "+        @self.dp.message(Command(\"brandnew\"))\n"
    )
    assert extract_new_commands_from_diff(diff) == ["brandnew"]


def test_first_run_baselines_silently(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    tracked = src / "telegram_bot.py"
    tracked.write_text("print('hello')\n", encoding="utf-8")

    monkeypatch.setattr("code_reviewer.TRACKED_FILES", ["src/telegram_bot.py"])
    result = check_and_generate_changelog(tmp_path, writer_bot=None, admin_id=0, lang="ru")
    assert result is None
    hashes = load_stored_hashes(tmp_path / "data")
    assert "src/telegram_bot.py" in hashes


def test_fallback_stays_silent_without_new_commands():
    assert generate_user_friendly_fallback({"new_commands": [], "changed_files_count": 2}) == ""


def test_changed_files_detects_hash_mismatch(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    tracked = src / "i18n.py"
    tracked.write_text("old\n", encoding="utf-8")
    monkeypatch.setattr("code_reviewer.TRACKED_FILES", ["src/i18n.py"])
    save_current_hashes(tmp_path)
    tracked.write_text("new\n", encoding="utf-8")
    changed = get_changed_files(tmp_path, load_stored_hashes(tmp_path / "data"))
    assert changed and changed[0][0] == "src/i18n.py"


def test_strips_think_tags_and_keeps_short_bullets():
    raw = (
        "<think>\n"
        "The diff in src/telegram_bot.py adds a helper and changes hashes.\n"
        "I should mention the internal refactor of _format_update_preview.\n"
        "</think>\n"
        "- Теперь можно править текст обновления перед рассылкой\n"
        "- Исправлен сбой при отправке длинного ответа\n"
    )
    cleaned = _sanitize_changelog(raw)
    assert "think" not in cleaned.lower()
    assert "telegram_bot.py" not in cleaned
    assert "_format_update_preview" not in cleaned
    assert "править текст обновления" in cleaned
    assert "длинного ответа" in cleaned
    assert cleaned.count("•") == 2


def test_uses_final_answer_after_reasoning_dump():
    raw = (
        "Looking at the diff I notice src/code_reviewer.py now tracks snapshots.\n"
        "This is a large internal change with hashes and backups.\n\n"
        "Changelog:\n"
        "- Daily prompts arrive more reliably\n"
    )
    cleaned = _sanitize_changelog(raw)
    assert cleaned == "• Daily prompts arrive more reliably"
    assert "code_reviewer.py" not in cleaned
    assert "hashes" not in cleaned


def test_drops_technical_or_empty_output():
    assert _sanitize_changelog("NONE") == ""
    assert _sanitize_changelog("НЕТ") == ""
    assert _sanitize_changelog("Internal improvements to hashing") == ""
    assert _sanitize_changelog("Updated src/telegram_bot.py helper `_clip()`") == ""


def test_drops_unmarked_reasoning_dump():
    raw = (
        "Looking at the files I see many internal helpers changed.\n"
        "The hash snapshot logic is different now.\n"
        "I should not invent user-facing features from this."
    )
    assert _sanitize_changelog(raw) == ""


def test_extract_completion_ignores_reasoning_fields():
    message = SimpleNamespace(
        content="• Fixed the help text",
        reasoning="long hidden chain of thought about diffs",
        reasoning_content="more hidden thoughts",
    )
    response = SimpleNamespace(choices=[SimpleNamespace(message=message)])
    assert _extract_completion_text(response) == "• Fixed the help text"

    multipart = SimpleNamespace(
        content=[
            {"type": "thinking", "text": "I will inspect the unified diff now."},
            {"type": "text", "text": "- Snapshot button is easier to find"},
        ]
    )
    response = SimpleNamespace(choices=[SimpleNamespace(message=multipart)])
    assert _extract_completion_text(response) == "- Snapshot button is easier to find"


def test_generate_changelog_sanitizes_model_output():
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=(
                        "<think>Inspect src/i18n.py and the hash snapshot.</think>\n"
                        "Answer:\n"
                        "- Help text is clearer\n"
                        "- Writing prompts are easier to start\n"
                    )
                )
            )
        ]
    )
    writer = SimpleNamespace(client=client, model="test-model")
    change_set = {
        "file_diffs": [{"path": "src/i18n.py", "old_hash": "a", "new_hash": "b", "diff": "+ help"}],
        "new_commands": [],
        "has_user_signal": True,
        "changed_files_count": 1,
    }

    result = generate_changelog_with_llm(
        writer,
        [("src/i18n.py", "a", "b")],
        project_root,
        lang="en",
        change_set=change_set,
    )

    assert result
    assert "i18n.py" not in result
    assert "<think>" not in result
    assert "Help text is clearer" in result
    assert "Writing prompts are easier to start" in result
    assert result.count("•") == 2
