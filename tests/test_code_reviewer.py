"""Smoke tests for hash/snapshot changelog detection."""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))

from code_reviewer import (
    _is_empty_changelog,
    build_unified_diff,
    check_and_generate_changelog,
    extract_new_commands_from_diff,
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
