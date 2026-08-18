#!/usr/bin/env python3
"""
Initialize code hash cache to prevent "bot updated" message on next restart.
Run this after deploying new code to mark current state as "baseline".

Usage:
    python3 scripts/init_code_cache.py
    # or from scripts directory:
    cd scripts && python3 init_code_cache.py
"""

import sys
from pathlib import Path

script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent.resolve()
sys.path.insert(0, str(project_root / "src"))

from code_reviewer import TRACKED_FILES, collect_current_hashes, save_current_hashes


def init_cache():
    """Initialize code hash cache with current file states."""
    current_hashes = collect_current_hashes(project_root)
    for rel_path in TRACKED_FILES:
        if rel_path in current_hashes:
            print(f"✓ {rel_path}: {current_hashes[rel_path]}")
        else:
            print(f"✗ {rel_path}: file not found")

    save_current_hashes(project_root)
    data_dir = project_root / "data"
    print(f"\n✅ Cache initialized: {data_dir / 'code_hashes.json'}")
    print(f"   Backups saved to: {data_dir / 'code_backup'}")
    print("   Next restart will NOT show 'bot updated' message")
    print("   (unless you actually change the code)")
    return 0


if __name__ == "__main__":
    sys.exit(init_cache())
