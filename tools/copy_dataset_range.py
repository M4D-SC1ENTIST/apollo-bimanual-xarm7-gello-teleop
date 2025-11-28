#!/usr/bin/env python3
"""Copy a range of dataset episodes into another dataset with re-indexing.

Example:
    # Copy episodes 200-251 from coffee to coffee_v2, appended after the
    # highest existing episode in the destination (episode_000000).
    python tools/copy_dataset_range.py \
        /home/xiatao/Projects/osa/apollo-bimanual-xarm7-gello-teleop/datasets/coffee \
        /home/xiatao/Projects/osa/apollo-bimanual-xarm7-gello-teleop/datasets/coffee_v2 \
        --start-episode 200 --end-episode 251
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path
from typing import Iterable, List, Tuple


def _list_episode_dirs(dataset_root: Path) -> List[Path]:
    """Return sorted episode directories under dataset_root."""
    if not dataset_root.exists():
        return []
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


def _resolve_episode_range(paths: Iterable[Path], start: int, end: int) -> List[Path]:
    selected: List[Path] = []
    for episode_dir in paths:
        idx = _episode_index_from_name(episode_dir.name)
        if start <= idx <= end:
            selected.append(episode_dir)
    if not selected:
        raise RuntimeError(f"No episodes found in range [{start}, {end}].")
    expected_count = end - start + 1
    if len(selected) != expected_count:
        missing = expected_count - len(selected)
        raise RuntimeError(
            f"Expected {expected_count} episodes in range [{start}, {end}] but found {len(selected)} (missing {missing})."
        )
    return selected


def _determine_destination_start(dest_root: Path, override_start: int | None) -> int:
    episode_dirs = _list_episode_dirs(dest_root)
    if override_start is not None:
        new_name = _format_episode_name(override_start)
        target_path = dest_root / new_name
        if target_path.exists():
            raise FileExistsError(f"Destination episode {new_name} already exists.")
        return override_start
    if not episode_dirs:
        return 0
    last_idx = _episode_index_from_name(episode_dirs[-1].name)
    return last_idx + 1


def _meta_info_path(dataset_root: Path) -> Path:
    return dataset_root / "meta" / "info.json"


def _load_info(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"info.json not found: {path}")
    return json.loads(path.read_text())


def _write_info(path: Path, info: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2) + "\n")


def _ensure_dest_info(path: Path, template: dict) -> None:
    if path.exists():
        return
    fresh = copy.deepcopy(template)
    fresh["datalist"] = []
    _write_info(path, fresh)


def _build_episode_entry_map(info: dict) -> dict[str, dict]:
    mapping: dict[str, dict] = {}
    for entry in info.get("datalist", []):
        episode_name = Path(entry.get("top_path", "")).name
        if episode_name:
            mapping[episode_name] = entry
    return mapping


def _update_destination_info(
    src_info: dict,
    dest_info_path: Path,
    copied_pairs: List[Tuple[str, Path]],
) -> None:
    if not copied_pairs:
        return
    dest_info = _load_info(dest_info_path)
    src_map = _build_episode_entry_map(src_info)
    datalist = dest_info.setdefault("datalist", [])
    for episode_name, dest_dir in copied_pairs:
        if episode_name not in src_map:
            raise KeyError(f"Episode {episode_name} missing from source meta info")
        new_entry = copy.deepcopy(src_map[episode_name])
        new_entry["top_path"] = str(dest_dir.resolve())
        datalist.append(new_entry)
    _write_info(dest_info_path, dest_info)


def copy_episode_range(
    src_dataset: Path,
    dst_dataset: Path,
    start_episode: int,
    end_episode: int,
    dest_start_index: int | None,
    dry_run: bool,
) -> None:
    if end_episode < start_episode:
        raise ValueError("end_episode must be >= start_episode")
    if not src_dataset.exists():
        raise FileNotFoundError(f"Source dataset not found: {src_dataset}")
    if not dst_dataset.exists():
        if dry_run:
            print(f"[dry-run] Destination dataset directory {dst_dataset} would be created.")
        else:
            dst_dataset.mkdir(parents=True, exist_ok=True)
    elif not dry_run:
        dst_dataset.mkdir(parents=True, exist_ok=True)

    src_episodes = _list_episode_dirs(src_dataset)
    selected = _resolve_episode_range(src_episodes, start_episode, end_episode)

    src_info = _load_info(_meta_info_path(src_dataset))
    dest_info_path = _meta_info_path(dst_dataset)
    if not dry_run:
        _ensure_dest_info(dest_info_path, src_info)

    dest_index = _determine_destination_start(dst_dataset, dest_start_index)
    print(
        f"Copying episodes {start_episode}-{end_episode} "
        f"({len(selected)} episodes) to {dst_dataset} starting at index {dest_index}."
    )

    copied_pairs: List[Tuple[str, Path]] = []
    for offset, src_dir in enumerate(selected):
        dest_idx = dest_index + offset
        dest_dir = dst_dataset / _format_episode_name(dest_idx)
        if dest_dir.exists():
            raise FileExistsError(f"Destination episode already exists: {dest_dir}")
        print(f"  -> {src_dir.name} -> {dest_dir.name}")
        if not dry_run:
            shutil.copytree(src_dir, dest_dir)
            copied_pairs.append((src_dir.name, dest_dir))

    if dry_run:
        print("Dry run complete. No files were copied.")
    else:
        _update_destination_info(src_info, dest_info_path, copied_pairs)
        print("Copy complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy and re-index dataset episodes.")
    parser.add_argument("src_dataset", type=Path, help="Source dataset directory")
    parser.add_argument("dst_dataset", type=Path, help="Destination dataset directory")
    parser.add_argument("--start-episode", type=int, required=True, help="First episode index to copy")
    parser.add_argument("--end-episode", type=int, required=True, help="Last episode index to copy (inclusive)")
    parser.add_argument(
        "--dest-start-index",
        type=int,
        default=None,
        help="Destination episode index to assign to the first copied episode (default: 1 + highest existing)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print what would be copied without copying")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    copy_episode_range(
        src_dataset=args.src_dataset.resolve(),
        dst_dataset=args.dst_dataset.resolve(),
        start_episode=args.start_episode,
        end_episode=args.end_episode,
        dest_start_index=args.dest_start_index,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
