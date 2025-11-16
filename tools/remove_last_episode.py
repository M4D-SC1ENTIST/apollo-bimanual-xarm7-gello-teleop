#!/usr/bin/env python3
"""
Remove the most recent episode(s) from a local dataset.

Usage:
    python tools/remove_last_episode.py /path/to/dataset --count 1
"""

import argparse
import json
import shutil
from pathlib import Path


def remove_episodes(dataset_path: Path, count: int, dry_run: bool):
    meta_path = dataset_path / "meta" / "info.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta/info.json not found under {dataset_path}")
    meta = json.loads(meta_path.read_text())

    episodes = sorted(dataset_path.glob("episode_*"))
    if len(episodes) < count:
        raise RuntimeError(f"Dataset only has {len(episodes)} episodes; cannot remove {count}.")

    to_remove = episodes[-count:]
    print("[cleanup] Episodes to remove:")
    for ep in to_remove:
        print(f"  - {ep}")
    if dry_run:
        print("[cleanup] Dry run enabled; nothing deleted.")
        return

    for ep in to_remove:
        shutil.rmtree(ep)

    if "datalist" in meta:
        meta["datalist"] = meta["datalist"][:-count]
    meta_path.write_text(json.dumps(meta, indent=2))
    print("[cleanup] Removal complete.")


def main():
    parser = argparse.ArgumentParser(description="Remove the latest episode(s) from a dataset.")
    parser.add_argument("dataset_path", type=Path, help="Path to dataset directory")
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of most recent episodes to remove (default: 1)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without modifying anything.",
    )
    args = parser.parse_args()

    dataset_path = args.dataset_path.resolve()
    remove_episodes(dataset_path, args.count, args.dry_run)


if __name__ == "__main__":
    main()

