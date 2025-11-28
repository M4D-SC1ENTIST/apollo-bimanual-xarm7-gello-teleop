#!/usr/bin/env python3
"""Combine multiple teleop datasets into one destination dataset.

Example:
    python tools/combine_datasets.py \
        apollo-bimanual-xarm7-gello-teleop/datasets/unified_coffee_cup_engine \
        apollo-bimanual-xarm7-gello-teleop/datasets/coffee_v2 \
        apollo-bimanual-xarm7-gello-teleop/datasets/cup_v2 \
        apollo-bimanual-xarm7-gello-teleop/datasets/engine \
        --copy-meta --dest-start-index 0
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable, List


def _list_episode_dirs(dataset_root: Path) -> List[Path]:
    return sorted(
        (p for p in dataset_root.iterdir() if p.is_dir() and p.name.startswith("episode_")),
        key=lambda p: p.name,
    )


def _episode_index_from_name(name: str) -> int:
    try:
        return int(name.split("_")[-1])
    except ValueError as exc:  # pragma: no cover - guardrail
        raise ValueError(f"Invalid episode directory name: {name}") from exc


def _format_episode_name(idx: int) -> str:
    return f"episode_{idx:06d}"


def _determine_destination_start(dest_root: Path, override_start: int | None) -> int:
    if override_start is not None:
        return override_start
    existing = _list_episode_dirs(dest_root) if dest_root.exists() else []
    if not existing:
        return 0
    last_idx = _episode_index_from_name(existing[-1].name)
    return last_idx + 1


def _copy_meta_if_needed(src: Path, dest: Path, enabled: bool) -> None:
    if not enabled:
        return
    dest_meta = dest / "meta"
    if dest_meta.exists():
        return
    src_meta = src / "meta"
    if not src_meta.exists():
        return
    print(f"[combine] Copying meta from {src_meta} -> {dest_meta}")
    shutil.copytree(src_meta, dest_meta)


def combine_datasets(
    dest_dataset: Path,
    src_datasets: Iterable[Path],
    dest_start_index: int | None,
    dry_run: bool,
    copy_meta: bool,
) -> None:
    if not src_datasets:
        raise ValueError("At least one source dataset must be provided")

    dest_dataset.mkdir(parents=True, exist_ok=True)
    dest_index = _determine_destination_start(dest_dataset, dest_start_index)
    copied_count = 0

    for src in src_datasets:
        if not src.exists():
            raise FileNotFoundError(f"Source dataset not found: {src}")
        _copy_meta_if_needed(src, dest_dataset, copy_meta)

        episodes = _list_episode_dirs(src)
        if not episodes:
            print(f"[combine] No episodes found in {src}, skipping")
            continue

        print(f"[combine] Merging {len(episodes)} episodes from {src} starting at index {dest_index}")
        for episode_dir in episodes:
            dest_name = _format_episode_name(dest_index)
            dest_path = dest_dataset / dest_name
            if dest_path.exists():
                raise FileExistsError(f"Destination episode already exists: {dest_path}")
            print(f"  -> {episode_dir} -> {dest_path}")
            if not dry_run:
                shutil.copytree(episode_dir, dest_path)
            dest_index += 1
            copied_count += 1

    if dry_run:
        print(f"[combine] Dry run complete. {copied_count} episodes would be copied.")
    else:
        print(f"[combine] Done. Copied {copied_count} episodes into {dest_dataset}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine multiple dataset folders into one.")
    parser.add_argument("dest_dataset", type=Path, help="Destination dataset directory")
    parser.add_argument(
        "src_datasets",
        nargs="+",
        type=Path,
        help="One or more source dataset directories to merge (order matters)",
    )
    parser.add_argument(
        "--dest-start-index",
        type=int,
        default=None,
        help="Destination episode index to start writing at (default: append after existing).",
    )
    parser.add_argument(
        "--copy-meta",
        action="store_true",
        help="Copy the first available src/meta directory into the destination if dest/meta is absent.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print actions without copying data.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    combine_datasets(
        dest_dataset=args.dest_dataset.resolve(),
        src_datasets=[p.resolve() for p in args.src_datasets],
        dest_start_index=args.dest_start_index,
        dry_run=args.dry_run,
        copy_meta=args.copy_meta,
    )


if __name__ == "__main__":
    main()
