#!/usr/bin/env python3
"""Repair dataset meta info by rebuilding datalist entries from disk."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import pyarrow.parquet as pq  # type: ignore[import]
except ImportError as exc:  # pragma: no cover - dependency hint
    raise SystemExit(
        "pyarrow is required for repair_dataset_meta.py. Install it via `pip install pyarrow`."
    ) from exc

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi"}


def _load_info(meta_path: Path) -> dict:
    if not meta_path.exists():
        raise FileNotFoundError(f"info.json not found: {meta_path}")
    return json.loads(meta_path.read_text())


def _write_info(meta_path: Path, info: dict) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(info, indent=2) + "\n")


def _list_episode_dirs(dataset_path: Path) -> List[Path]:
    return sorted(
        (p for p in dataset_path.iterdir() if p.is_dir() and p.name.startswith("episode_")),
        key=lambda p: p.name,
    )


def _entry_map(info: dict) -> Dict[str, dict]:
    mapping: Dict[str, dict] = {}
    for entry in info.get("datalist", []):
        top_path = entry.get("top_path", "")
        episode_name = Path(top_path).name if top_path else entry.get("episode", "")
        if episode_name:
            mapping[episode_name] = entry
    return mapping


def _infer_frame_count(episode_dir: Path, data_file: str) -> int:
    parquet_path = episode_dir / data_file
    if parquet_path.exists():
        parquet_file = pq.ParquetFile(parquet_path)
        return parquet_file.metadata.num_rows
    depth_dir = episode_dir / "depth"
    if depth_dir.exists():
        depth_frames = [p for p in depth_dir.glob("*.png")]
        if depth_frames:
            return len(depth_frames)
    raise FileNotFoundError(f"Unable to infer frame count for {episode_dir}; missing parquet/depth data")


def _discover_videos(video_dir: Path) -> Dict[str, str]:
    videos: Dict[str, str] = {}
    for file in sorted(video_dir.glob("*")):
        if file.is_file() and file.suffix.lower() in VIDEO_EXTENSIONS:
            videos[file.stem] = file.name
    return videos


def _update_audio_fields(entry: dict, episode_dir: Path) -> None:
    audio_dir = episode_dir / "audio"
    if not audio_dir.exists():
        entry.pop("audio_dir", None)
        entry.pop("audio", None)
        return
    entry["audio_dir"] = audio_dir.name
    audio_info = entry.get("audio", {})
    raw_file = audio_dir / "raw_audio.wav"
    if raw_file.exists():
        audio_info["raw_file"] = f"{audio_dir.name}/raw_audio.wav"
    index_file = audio_dir / "audio_index.parquet"
    if index_file.exists():
        audio_info["index_file"] = f"{audio_dir.name}/audio_index.parquet"
    if audio_info:
        entry["audio"] = audio_info
    else:
        entry.pop("audio", None)


def _rebuild_entry(episode_dir: Path, base_entry: dict) -> dict:
    entry = copy.deepcopy(base_entry)
    entry["top_path"] = str(episode_dir.resolve())
    entry["data_file"] = entry.get("data_file", "data.parquet")

    frame_count = _infer_frame_count(episode_dir, entry["data_file"])
    entry["start_frame"] = 0
    entry["end_frame"] = max(frame_count - 1, 0)

    video_dir = episode_dir / "videos"
    if video_dir.exists():
        entry["video_dir"] = video_dir.name
        videos = _discover_videos(video_dir)
        if videos:
            entry["videos"] = videos
    else:
        entry.pop("video_dir", None)
        entry.pop("videos", None)

    depth_dir = episode_dir / "depth"
    if depth_dir.exists():
        entry["depth_dir"] = depth_dir.name
    else:
        entry.pop("depth_dir", None)

    _update_audio_fields(entry, episode_dir)

    return entry


def repair_dataset_meta(dataset_path: Path, dry_run: bool) -> Tuple[int, int, List[str]]:
    dataset_path = dataset_path.resolve()
    meta_path = dataset_path / "meta" / "info.json"
    info = _load_info(meta_path)

    episode_dirs = _list_episode_dirs(dataset_path)
    if not episode_dirs:
        raise RuntimeError(f"No episode_* directories found under {dataset_path}")

    existing_entries = _entry_map(info)
    updated_entries: List[dict] = []
    updated_count = 0

    for episode_dir in episode_dirs:
        name = episode_dir.name
        original = existing_entries.get(name, {})
        new_entry = _rebuild_entry(episode_dir, original)
        if original != new_entry:
            updated_count += 1
        updated_entries.append(new_entry)

    missing = sorted(set(existing_entries.keys()) - {ep.name for ep in episode_dirs})
    info["datalist"] = updated_entries

    if dry_run:
        print(f"[dry-run] Would update {updated_count} entries and write {len(updated_entries)} total episodes.")
    else:
        _write_info(meta_path, info)
        print(f"[repair] Updated {updated_count} entries. info.json now tracks {len(updated_entries)} episodes.")

    return updated_count, len(updated_entries), missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair broken datalist entries inside meta/info.json.")
    parser.add_argument("dataset", type=Path, help="Path to the dataset directory (containing episode_* folders)")
    parser.add_argument("--dry-run", action="store_true", help="Only report intended changes without writing info.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    updated_count, total, missing = repair_dataset_meta(args.dataset, args.dry_run)
    if missing:
        print(f"[repair] Warning: {len(missing)} entries in info.json referenced missing episodes: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if args.dry_run:
        print("[dry-run] No files were modified.")


if __name__ == "__main__":
    main()
