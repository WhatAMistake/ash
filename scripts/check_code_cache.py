#!/usr/bin/env python3
"""
Check current code hash cache status.
Shows which files are tracked and whether cache exists.

Usage:
    python3 scripts/check_code_cache.py
    # or from scripts directory:
    cd scripts && python3 check_code_cache.py
"""

import sys
from pathlib import Path

script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent.resolve()
sys.path.insert(0, str(project_root / "src"))

from code_reviewer import TRACKED_FILES, calculate_file_hash, load_stored_hashes


def check_cache():
    """Check current cache status."""
    data_dir = project_root / "data"
    hash_file = data_dir / "code_hashes.json"
    backup_dir = data_dir / "code_backup"

    print("=" * 60)
    print("СТАТУС КЭША КОДА")
    print("=" * 60)
    print(f"Корень проекта: {project_root}")
    print(f"Директория данных: {data_dir}")

    has_backups = backup_dir.exists() and any(backup_dir.iterdir())
    stored_hashes = load_stored_hashes(data_dir)

    if not hash_file.exists() or not stored_hashes:
        print(f"\nОШИБКА: Файл кэша НЕ НАЙДЕН: {hash_file}")
        if has_backups:
            print(f"OK: Но резервные копии существуют в: {backup_dir}")
            print("   (Запустите init для обновления кэша хешей)")
        else:
            print("\nВНИМАНИЕ: Кэш и резервные копии не найдены!")
            print("   Запустите: python3 scripts/init_code_cache.py")
        return 1

    print(f"\nФайл кэша: {hash_file}")
    print(f"   Отслеживаемые файлы: {len(TRACKED_FILES)}")
    print(f"   Файлов в кэше: {len(stored_hashes)}")
    if has_backups:
        print(f"   Резервные копии: {backup_dir}")

    print("\n" + "-" * 60)
    print("СТАТУС ФАЙЛОВ:")
    print("-" * 60)

    changes_detected = False
    for rel_path in TRACKED_FILES:
        file_path = project_root / rel_path
        current_hash = calculate_file_hash(file_path) if file_path.exists() else "N/A"
        stored_hash = stored_hashes.get(rel_path, "НЕТ В КЭШЕ")

        if current_hash == stored_hash:
            status = "OK"
        elif stored_hash == "НЕТ В КЭШЕ":
            status = "НОВЫЙ"
            changes_detected = True
        else:
            status = "ИЗМЕНЕН"
            changes_detected = True

        print(f"{status} {rel_path}")
        print(f"      Текущий: {current_hash}")
        print(f"      В кэше:  {stored_hash}")
        print()

    print("-" * 60)
    if changes_detected:
        print("ВНИМАНИЕ: Обнаружены изменения! При следующем запуске БУДЕТ показано сообщение 'бот обновлен'")
    else:
        print("OK: Все файлы совпадают с кэшем. При следующем запуске сообщение НЕ будет показано")
    print("-" * 60)

    return 0 if not changes_detected else 1


if __name__ == "__main__":
    sys.exit(check_cache())
