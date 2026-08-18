#!/usr/bin/env python3
"""
Test changelog generation without restarting the bot.
Run from project root: python3 scripts/test_changelog.py [--lang ru|en] [--save-hashes]

Examples:
    python3 scripts/test_changelog.py              # Test in Russian (default)
    python3 scripts/test_changelog.py --lang en    # Test in English
    python3 scripts/test_changelog.py --save-hashes # Test and update cache
"""

import sys
import os
import argparse
import importlib.util
from pathlib import Path

script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent.resolve()
src_dir = project_root / "src"

sys.path.insert(0, str(src_dir))
os.chdir(project_root)

from dotenv import load_dotenv
load_dotenv()

from code_reviewer import (
    TRACKED_FILES,
    check_and_generate_changelog,
    get_changed_files,
    load_stored_hashes,
    save_current_hashes,
)


def load_writer_bot():
    """Load WriterBot class directly from file."""
    spec = importlib.util.spec_from_file_location("writer_bot", src_dir / "writer_bot.py")
    wb_module = importlib.util.module_from_spec(spec)
    sys.modules["writer_bot"] = wb_module
    spec.loader.exec_module(wb_module)
    return wb_module.WriterBot


def print_header(text, width=70):
    print("\n" + "=" * width)
    print(f" {text}")
    print("=" * width)


def print_section(text):
    print(f"\n▶ {text}")


def print_success(text):
    print(f"  ✓ {text}")


def print_error(text):
    print(f"  ✗ {text}")


def print_info(text):
    print(f"  • {text}")


def check_env():
    required = ["OPENAI_API_KEY"]
    missing = [var for var in required if not os.getenv(var)]
    if missing:
        print_error(f"Missing environment variables: {', '.join(missing)}")
        print_info("Make sure .env file exists in project root")
        return False
    return True


def test_changelog(lang="ru", save_hashes=False):
    print_header(f"CHANGELOG TEST [{lang.upper()}]")

    print_section("Checking environment...")
    if not check_env():
        return 1
    print_success("Environment OK")

    print_section("Loading WriterBot...")
    try:
        WriterBot = load_writer_bot()
        print_success("WriterBot class loaded")
    except Exception as e:
        print_error(f"Failed to load WriterBot: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print_section("Initializing LLM client...")
    try:
        wb = WriterBot(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
            api_base=os.getenv("OPENAI_API_BASE"),
            use_rag=False,
        )
        print_success(f"LLM client ready (model: {wb.model})")
    except Exception as e:
        print_error(f"Failed to initialize: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print_section("Scanning for code changes...")
    data_dir = project_root / "data"
    stored_hashes = load_stored_hashes(data_dir)
    if not stored_hashes:
        print_info("No baseline hashes — first run would save a silent baseline")
        print_info("Run with --save-hashes or: python3 scripts/init_code_cache.py")
        if save_hashes:
            save_current_hashes(project_root)
            print_success("Baseline saved")
        return 0

    changed_files = get_changed_files(project_root, stored_hashes)
    if not changed_files:
        print_info("No code changes detected")
        print_info("Next restart will NOT show 'bot updated' message")
        print_info("To force changelog, modify any tracked file or delete data/code_hashes.json")
        print_info(f"Tracked files: {', '.join(TRACKED_FILES)}")
        return 0

    print_success(f"Found {len(changed_files)} changed file(s):")
    for rel_path, old_hash, new_hash in changed_files:
        status = "NEW" if not old_hash else "MODIFIED"
        print_info(f"{rel_path:30} [{status}]")

    print_section("Generating changelog with LLM...")
    print_info("This may take a few seconds...")

    try:
        changelog = check_and_generate_changelog(
            project_root,
            wb,
            admin_id=0,
            lang=lang,
            should_save_hashes=save_hashes,
        )

        if changelog:
            print_header("CHANGELOG PREVIEW", width=70)
            print(changelog)
            print("=" * 70)

            print_section("Summary")
            print_success("Changelog generated successfully")
            print_info(f"Language: {lang}")
            print_info(f"Changed files: {len(changed_files)}")
            if save_hashes:
                print_success("Cache updated - next restart won't trigger notifications")
            else:
                print_info("Run with --save-hashes to update cache")
                print_info("Or run: python3 scripts/init_code_cache.py")
        else:
            print_info("No user-facing changelog (internal-only or empty LLM answer)")
            if save_hashes:
                print_success("Cache updated")
            else:
                print_info("Run with --save-hashes to update cache")

    except Exception as e:
        print_error(f"Error generating changelog: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Test changelog generation without restarting the bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 scripts/test_changelog.py              # Test in Russian (default)
  python3 scripts/test_changelog.py --lang en    # Test in English
  python3 scripts/test_changelog.py --save-hashes # Test and update cache
        """,
    )
    parser.add_argument(
        "--lang",
        choices=["ru", "en"],
        default="ru",
        help="Language for changelog (default: ru)",
    )
    parser.add_argument(
        "--save-hashes",
        action="store_true",
        help="Update hash cache after testing (prevents duplicate notifications)",
    )

    args = parser.parse_args()
    raise SystemExit(test_changelog(lang=args.lang, save_hashes=args.save_hashes))


if __name__ == "__main__":
    main()
