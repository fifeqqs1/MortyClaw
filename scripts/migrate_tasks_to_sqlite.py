#!/usr/bin/env python3

import argparse

from mortyclaw.core.config import TASKS_FILE
from mortyclaw.core.storage.runtime import get_task_repository


def main() -> None:
    parser = argparse.ArgumentParser(description="Import legacy tasks.json into MortyClaw runtime.sqlite3")
    parser.add_argument("--source", default=TASKS_FILE, help="Path to legacy tasks.json")
    parser.add_argument(
        "--default-thread-id",
        default="local_geek_master",
        help="Fallback thread_id for legacy tasks without a session owner",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing tasks with the same ID")
    args = parser.parse_args()

    result = get_task_repository().import_legacy_tasks(
        file_path=args.source,
        default_thread_id=args.default_thread_id,
        overwrite=args.force,
    )
    print(f"migrate_tasks_to_sqlite: imported={result['imported']} skipped={result['skipped']}")


if __name__ == "__main__":
    main()
