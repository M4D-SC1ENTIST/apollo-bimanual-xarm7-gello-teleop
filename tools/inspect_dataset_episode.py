#!/usr/bin/env python3
"""
Visualize a recorded teleoperation episode (RGB, depth, audio, trajectories).

Usage:
    python tools/inspect_dataset_episode.py /path/to/dataset --episode-index 0 --play-audio
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import torchaudio
import torch

try:
    import matplotlib.pyplot as plt  # type: ignore
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit("matplotlib is required for visualization. pip install matplotlib") from exc


def _load_meta(dataset_path: Path) -> dict:
    meta_path = dataset_path / "meta" / "info.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta/info.json not found under {dataset_path}")
    return json.loads(meta_path.read_text())


def _resolve_episode_dir(dataset_path: Path, episode_idx: int) -> Path:
    episode_dir = dataset_path / f"episode_{episode_idx:06d}"
    if not episode_dir.exists():
        raise FileNotFoundError(f"Episode directory {episode_dir} does not exist.")
    return episode_dir


def _show_image(ax, title: str, img: np.ndarray):
    ax.imshow(img)
    ax.set_title(title)
    ax.axis("off")


def _load_first_depth(depth_dir: Path) -> np.ndarray | None:
    pngs = sorted(depth_dir.glob("*.png"))
    if not pngs:
        return None
    depth = cv2.imread(str(pngs[0]), cv2.IMREAD_UNCHANGED)
    if depth is None:
        return None
    return depth.astype(np.float32)


def _plot_trajectories(ax, data: np.ndarray, title: str):
    ax.plot(data)
    ax.set_title(title)
    ax.set_xlabel("Frame")


def _play_audio(waveform: torch.Tensor, sample_rate: int):
    try:
        import sounddevice as sd  # type: ignore
    except ImportError:
        print("[inspect] sounddevice not installed; skipping playback.")
        return
    sd.play(waveform.numpy(), sample_rate)
    sd.wait()


def visualize_episode(dataset_path: Path, episode_idx: int, play_audio: bool):
    episode_dir = _resolve_episode_dir(dataset_path, episode_idx)

    # RGB frame
    video_path = episode_dir / "videos" / "cam_front.mp4"
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    if ret and frame is not None:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        rgb_frame = None

    depth = _load_first_depth(episode_dir / "depth")

    # Load trajectories
    data_table = pq.read_table(episode_dir / "data.parquet")
    states = np.stack(data_table["observation.state"].to_pylist())
    actions = np.stack(data_table["action"].to_pylist())
    torque = (
        np.stack(data_table["observation.torque"].to_pylist())
        if "observation.torque" in data_table.column_names
        else None
    )

    # Audio
    audio_path = episode_dir / "audio" / "raw_audio.wav"
    waveform, sample_rate = torchaudio.load(str(audio_path))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    if rgb_frame is not None:
        _show_image(axes[0, 0], "RGB (frame 0)", rgb_frame)
    else:
        axes[0, 0].text(0.5, 0.5, "No RGB frame", ha="center")
        axes[0, 0].axis("off")

    if depth is not None:
        axes[0, 1].imshow(depth, cmap="viridis")
        axes[0, 1].set_title("Depth (frame 0)")
        axes[0, 1].axis("off")
    else:
        axes[0, 1].text(0.5, 0.5, "No depth frame", ha="center")
        axes[0, 1].axis("off")

    _plot_trajectories(axes[1, 0], states, "Joint Positions")
    action_plot = actions if actions.ndim == 2 else actions[:, 0, :]
    _plot_trajectories(axes[1, 1], action_plot, "Actions")
    if torque is not None:
        plt.figure()
        _plot_trajectories(plt.gca(), torque, "Torques")

    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(waveform.squeeze(0).numpy())
    plt.title("Audio waveform")
    plt.xlabel("Sample")
    plt.show()

    if play_audio:
        _play_audio(waveform.squeeze(0), sample_rate)


def main():
    parser = argparse.ArgumentParser(description="Inspect a recorded dataset episode.")
    parser.add_argument("dataset_path", type=Path, help="Path to dataset directory")
    parser.add_argument(
        "--episode-index",
        type=int,
        default=None,
        help="Episode index to visualize (default: latest)",
    )
    parser.add_argument(
        "--play-audio",
        action="store_true",
        help="Play the episode audio using sounddevice (if installed).",
    )
    args = parser.parse_args()

    dataset_path = args.dataset_path.resolve()
    meta = _load_meta(dataset_path)
    available = sorted(p.name for p in dataset_path.glob("episode_*"))
    if not available:
        raise RuntimeError(f"No episodes found under {dataset_path}")
    if args.episode_index is None:
        target_idx = int(available[-1].split("_")[-1])
    else:
        target_idx = args.episode_index
    print(f"[inspect] Visualizing episode {target_idx}")
    visualize_episode(dataset_path, target_idx, args.play_audio)


if __name__ == "__main__":
    main()

