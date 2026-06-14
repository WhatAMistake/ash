"""
Code change reviewer for Writer's Tears bot.
Tracks file hashes and generates changelogs on startup.
"""

import os
import json
import hashlib
import re
import random
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
# Files to track for changes
TRACKED_FILES = [
    "src/telegram_bot.py",
    "src/writer_bot.py", 
    "src/i18n.py",
    "src/writer_rag.py",
    "src/lang_utils.py",
    "src/word_stats.py",
]

# Existing commands that should NOT be mentioned as new in changelogs
EXISTING_COMMANDS = [
    "start", "help", "lang", "switchlang", "reset", "block", "develop",
    "character", "dialogue", "prompt", "idea", "feedback", "style", "roast",
    "praise", "corrector", "editor", "count_me", "stats", "lobster", "pun",
    "xyx", "methodique", "methodichque", "cite", "cite_off", "cite_on",
    "cite_when", "summary", "cry_baby", "dev_feedback", "done", "confo_enable37", "upload",
    # Internal functions that should NOT appear in changelog
    "get_writing_prompt", "generate_idea", "develop_idea", "character_help",
    "dialogue_help", "methodique_random", "cry_baby_reply", "lobster_love"
]


# Witty comments for changelog
WITTY_COMMENTS = [
    "Код, как вино — с каждым обновлением становится лучше.",
    "Баги исправлены, новые баги ещё не обнаружены.",
    "Обновление установлено. Надеюсь, ничего не сломал.",
    "Код переписан. Старые костыли заменены на новые, более элегантные.",
    "Код обновлён. Если что-то сломалось — это специально.",
]




def calculate_file_hash(file_path: Path) -> str:
    """Calculate SHA256 hash of file contents."""
    try:
        with open(file_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except Exception:
        return ""


def load_stored_hashes(data_dir: Path) -> Dict[str, str]:
    """Load previously stored file hashes."""
    hash_file = data_dir / "code_hashes.json"
    if hash_file.exists():
        try:
            with open(hash_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_hashes(data_dir: Path, hashes: Dict[str, str]) -> None:
    """Save current file hashes."""
    hash_file = data_dir / "code_hashes.json"
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        with open(hash_file, 'w', encoding='utf-8') as f:
            json.dump(hashes, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Failed to save hashes: {e}")


def get_last_deploy_tag(project_root: Path) -> Optional[str]:
    """Get the most recent deploy tag, or None if none exists."""
    try:
        result = subprocess.run(
            ["git", "tag", "--list", "--sort=-creatordate", "deploy-*"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        )
        tags = [t.strip() for t in result.stdout.strip().split("\n") if t.strip()]
        return tags[0] if tags else None
    except Exception:
        return None


def get_git_diff_files(project_root: Path, old_tag: Optional[str]) -> List[Tuple[str, str, str]]:
    """
    Get list of changed tracked files between old_tag and HEAD.
    Returns list of (rel_path, old_hash, new_hash) for changed files.
    If old_tag is None, treats all tracked files as changed.
    """
    changed = []
    for rel_path in TRACKED_FILES:
        file_path = project_root / rel_path
        if not file_path.exists():
            continue

        current_hash = calculate_file_hash(file_path)

        if old_tag is None:
            changed.append((rel_path, "", current_hash))
            continue

        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", f"{old_tag}..HEAD", "--", rel_path],
                cwd=project_root,
                capture_output=True,
                text=True,
                check=True,
            )
            if result.stdout.strip():
                changed.append((rel_path, old_tag, current_hash))
        except Exception:
            pass

    return changed


def create_deploy_tag(project_root: Path) -> str:
    """Create a new deploy tag and return it."""
    tag_name = f"deploy-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    subprocess.run(
        ["git", "tag", tag_name],
        cwd=project_root,
        capture_output=True,
        check=True,
    )
    return tag_name


def get_changed_files(project_root: Path, stored_hashes: Dict[str, str]) -> List[Tuple[str, str, str]]:
    """
    Compare current files with stored hashes.
    Returns list of (filename, old_hash, new_hash) for changed files.
    """
    changed = []
    for rel_path in TRACKED_FILES:
        file_path = project_root / rel_path
        if file_path.exists():
            current_hash = calculate_file_hash(file_path)
            old_hash = stored_hashes.get(rel_path, "")
            if current_hash != old_hash:
                changed.append((rel_path, old_hash, current_hash))
    return changed

def extract_commands_from_code(content: str) -> List[str]:
    """Extract command names from code content."""
    commands = []
    patterns = [
        r'Command\(["\'](/\w+)["\']',
        r'@self\.dp\.message\(Command\(["\'](\w+)["\']',
        r'async def cmd_(\w+)\(',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, content)
        for match in matches:
            cmd = match if match.startswith('/') else f"/{match}"
            if cmd not in commands:
                commands.append(cmd)
    return commands


def analyze_code_changes(project_root: Path, changed_files: List[Tuple[str, str, str]]) -> Dict[str, List[str]]:
    """Analyze what actually changed in the code."""
    result = {
        "new_commands": [],
        "all_commands": [],
        "has_changes": False,
        "changed_files_count": len(changed_files)
    }
    
    for rel_path, old_hash, new_hash in changed_files:
        file_path = project_root / rel_path
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            commands = extract_commands_from_code(content)
            if commands:
                result["all_commands"].extend(commands)
            
            result["has_changes"] = True
        except Exception:
            result["has_changes"] = True
    
    # Remove duplicates
    result["all_commands"] = list(set(result["all_commands"]))
    return result


def get_witty_comment(num_changes: int, lang: str = "ru") -> str:
    """Get a witty comment."""
    return random.choice(WITTY_COMMENTS)




def generate_changelog_with_llm(writer_bot, changed_files: List[Tuple[str, str, str]], project_root: Path, lang: str = "ru") -> str:
    """
    Use LLM to generate a human-readable changelog based on file changes.
    """
    if not changed_files:
        return ""
    
    num_changes = len(changed_files)
    
    # Generate content for each changed file
    diff_sections = []
    for rel_path, old_hash, new_hash in changed_files:
        file_path = project_root / rel_path
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            diff_sections.append(f"File: {rel_path}\n{content[:5000]}")
        except Exception as e:
            diff_sections.append(f"File: {rel_path}\n[Error: {e}]")
    
    diff_content = "\n\n".join(diff_sections)
    
    # Known existing commands to prevent hallucination
    known_cmds = ", ".join(EXISTING_COMMANDS[:12])
    
    if lang == "ru":
        prompt = f"""Ты пишешь changelog для пользователей.
Пиши ТОЛЬКО о том, что изменилось для ПОЛЬЗОВАТЕЛЯ (новые команды, исправления в ответах, новые темы, исправление багов).
Игнорируй чисто техническую чепуху: структуру папок, импорты, мелкий рефакторинг.

СТРОГИЕ ПРАВИЛА:
1) Если есть исправления в логике команд (например, /pun теперь видит контекст, а /summary соблюдает формат) — ОБЯЗАТЕЛЬНО напиши об этом.
2) Пиши максимально кратко: 1-3 пункта.
3) НИКАКИХ "внутренних улучшений" и "оптимизаций".
4) Не используй markdown (звездочки, жирный шрифт).

УЖЕ СУЩЕСТВУЮЩИЕ: {known_cmds}...

Изменённые файлы ({num_changes} шт.):

{diff_content}

Напиши changelog на русском. Будь конкретным и едким, как старый редактор."""

    else:
        prompt = f"""You are writing release notes for users.
Write ONLY about changes that are VISIBLE to the user (new commands, fixed responses, new themes).
Ignore technical fluff: .env loading, folder structure, refactoring, imports, etc.

RULES:
1. If there are no user-facing changes — return an empty string.
2. Be extremely brief: 1-2 items max.
3. NO "internal improvements" or "under the hood" talk.
4. If a command was fixed (e.g., /pun) — say so: "Fixed /pun command, it now sees context correctly."
5. No markdown formatting.

Changed files:
{diff_content}

Write changelog in English. Be specific and cynical."""

    
    try:
        if writer_bot and writer_bot.client:
            response = writer_bot.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[                    {"role": "system", "content": "Ты пишешь краткие notes к релизам. Не придумывай. Будь честным. НЕ используй markdown (звёздочки, жирный шрифт)." if lang == "ru" else "You write brief release notes. Don't hallucinate. Be honest. NO markdown formatting (no asterisks, no bold)."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=800
            )            
            changelog = response.choices[0].message.content.strip()
            
            # Add witty comment if changelog is meaningful
            if changelog and len(changelog) > 10 and "Internal" not in changelog:
                witty = get_witty_comment(num_changes, lang)
                return f"{witty}\n\n{changelog}"
            else:
                return changelog


    except Exception as e:
        print(f"LLM changelog generation failed: {e}")
    
    # Fallback: simple message
    if lang == "ru":
        return "Внутренние улучшения и исправления."
    else:
        return "Internal improvements and fixes."


def generate_user_friendly_fallback(analysis: Dict, lang: str = "ru") -> str:
    """Generate a user-friendly changelog without technical details."""
    num_changes = analysis.get("changed_files_count", 1)
    
    if lang == "ru":
        witty = get_witty_comment(num_changes, lang)
        lines = [witty]
        
        if analysis["all_commands"]:
            for cmd in analysis["all_commands"][:5]:
                lines.append(f"- Доступна команда {cmd}")
        
        if len(lines) == 1:
            lines.append("- Улучшена работа и исправлены ошибки")
    else:
        witty = get_witty_comment(num_changes, lang)
        lines = [witty]
        
        if analysis["all_commands"]:
            for cmd in analysis["all_commands"][:5]:
                lines.append(f"- Command {cmd} is available")
        
        if len(lines) == 1:
            lines.append("- Improvements and bug fixes")



    
    return "\n".join(lines)


def check_and_generate_changelog(project_root: Path, writer_bot, admin_id: int, lang: str = "ru", should_save_tag: bool = True) -> Optional[str]:
    """
    Main entry point: check for changes and generate changelog.
    Returns changelog text if changes detected, None otherwise.

    Uses git deploy tags to track changes between deployments.
    Creates a new deploy-* tag after successful changelog generation.

    Args:
        project_root: Path to project root
        writer_bot: WriterBot instance for LLM calls
        admin_id: Admin user ID (for reference, not used in this function)
        lang: Language code (ru/en)
        should_save_tag: If True, creates a new deploy tag after generating changelog.
                         Set to False when testing or when you want to create tag
                         manually after confirming notifications were sent.
    """

    try:
        old_tag = get_last_deploy_tag(project_root)
        changed_files = get_git_diff_files(project_root, old_tag)

        if not changed_files:
            return None

        changelog = generate_changelog_with_llm(writer_bot, changed_files, project_root, lang)

        if should_save_tag:
            create_deploy_tag(project_root)

        return changelog

    except Exception as e:
        print(f"Changelog generation error: {e}")
        raise
