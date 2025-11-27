#!/usr/bin/env python3
"""Bulk update instruction/task strings stored in meta/info.json.

Example:
    python tools/update_dataset_instruction.py datasets/coffee \
        --instruction "put the coffee pod into the coffee maker"
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable


def _load_meta(meta_path: Path) -> dict:
    if not meta_path.exists():
        raise FileNotFoundError(f"meta/info.json not found under {meta_path.parent}")
    return json.loads(meta_path.read_text())


def _write_meta(meta_path: Path, payload: dict) -> None:
    meta_path.write_text(json.dumps(payload, indent=2) + "\n")


def _normalize_tasks(tasks: Iterable[str] | None, fallback: str) -> list[str]:
    task_list = list(tasks) if tasks else [fallback]
    if not all(isinstance(item, str) and item.strip() for item in task_list):
        raise ValueError("All task entries must be non-empty strings")
    return task_list


def _apply_updates(meta: dict, new_instruction: str, new_tasks: list[str], update_tasks: bool) -> tuple[int, int]:
    datalist = meta.get("datalist")
    if not isinstance(datalist, list):
        raise ValueError("meta/info.json does not contain a valid 'datalist' list")

    instruction_updates = 0
    task_updates = 0
    for entry in datalist:
        if entry.get("instruction") != new_instruction:
            entry["instruction"] = new_instruction
            instruction_updates += 1
        if update_tasks and entry.get("tasks") != new_tasks:
            entry["tasks"] = list(new_tasks)
            task_updates += 1
    return instruction_updates, task_updates


def _backup_file(path: Path, suffix: str) -> Path:
    backup_path = path.with_suffix(path.suffix + suffix)
    shutil.copy2(path, backup_path)
    return backup_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Update dataset instructions/tasks across all episodes.")
    parser.add_argument("dataset_path", type=Path, help="Path to dataset directory (contains meta/info.json)")
    parser.add_argument("--instruction", required=True, help="Instruction string to store for every episode")
    parser.add_argument(
        "--task",
        dest="tasks",
        action="append",
        help="Optional task entry to store (repeatable). Defaults to the instruction value.",
    )
    parser.add_argument(
        "--no-task-update",
        dest="update_tasks",
        action="store_false",
        help="Do not touch the 'tasks' list; only update the instruction field.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show the planned updates without writing to disk")
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create a side-by-side backup meta/info.json.bak before writing",
    )
    parser.add_argument(
        "--backup-suffix",
        default=".bak",
        help="Custom suffix appended to meta/info.json when --backup is set (default: .bak)",
    )
    args = parser.parse_args()

    dataset_path = args.dataset_path.expanduser().resolve()
    meta_path = dataset_path / "meta" / "info.json"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory {dataset_path} does not exist")

    meta = _load_meta(meta_path)
    new_instruction = args.instruction.strip()
    if not new_instruction:
        raise ValueError("Instruction string cannot be empty")
    new_tasks = _normalize_tasks(args.tasks, new_instruction)

    instruction_updates, task_updates = _apply_updates(
        meta=meta,
        new_instruction=new_instruction,
        new_tasks=new_tasks,
        update_tasks=args.update_tasks,
    )
    total_entries = len(meta.get("datalist", []))

    print(
        f"[update] instruction changes: {instruction_updates}/{total_entries}, "
        f"task changes: {task_updates if args.update_tasks else 0}/{total_entries}"
    )

    if args.dry_run:
        print("[update] Dry-run enabled; no files were modified.")
        return

    if args.backup:
        backup_path = _backup_file(meta_path, args.backup_suffix)
        print(f"[update] Wrote backup to {backup_path}")

    _write_meta(meta_path, meta)
    print(f"[update] Updated {meta_path}")


if __name__ == "__main__":
    main()
