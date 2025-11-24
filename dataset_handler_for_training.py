from __future__ import annotations

import io
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import pyarrow.parquet as pq
from mmengine import fileio  # type: ignore
from PIL import Image

from ..utils import read_parquet, read_video_to_frames  # type: ignore
from .base import DomainHandler  # type: ignore


def _maybe_join(root: str, maybe_rel: str | None) -> str | None:
    if maybe_rel is None:
        return None
    p = Path(maybe_rel)
    if p.is_absolute() or str(p).startswith(("s3://", "oss://")):
        return str(p)
    return fileio.join_path(root, str(p))


def _load_bytes(path: str) -> bytes:
    if fileio.exists(path):
        with fileio.get_local_path(path) as local_path:
            return Path(local_path).read_bytes()
    return fileio.get(path)


class MultiModalLeRobotHandler(DomainHandler):
    """
    Dataset handler for MultiModal LeRobot episodes.

    Expected `meta` structure (per dataset JSON):
        {
            "dataset_name": "multimodal-lerobot",
            "fps": 10,
            "state_key": "observation.state",
            "action_key": "action",
            "torque_key": "observation.torque",            # optional
            "torque_time_key": "observation.torque_time", # optional
            "cameras": {
                "cam_front": {"depth": true},
                "cam_left": {"depth": false}
            },
            "audio": {"enabled": true, "sample_rate": 48000, "buffer_frames": 16},
            "datalist": [
                {
                    "top_path": "/path/to/episode_000123",
                    "data_file": "data.parquet",
                    "video_dir": "videos",
                    "depth_dir": "depth",
                    "audio_dir": "audio",
                    "videos": {"cam_front": "cam_front.mp4"},
                    "tasks": ["pick up the block"],
                    "instruction": "pick up the block",
                    "start_frame": 0,
                    "end_frame": 900
                }
            ]
        }
    """

    dataset_name = "multimodal-lerobot"

    def __init__(self, meta: dict, num_views: int, audio_buffer_frames: Optional[int] = None) -> None:
        super().__init__(meta, num_views)
        cfg = meta.get("modality_config", {})
        self.fps = meta.get("fps", cfg.get("fps", 10))
        self.state_key = meta.get("state_key", "observation.state")
        self.action_key = meta.get("action_key", "action")
        self.torque_key = meta.get("torque_key")
        self.torque_time_key = meta.get("torque_time_key")
        self.depth_scale = float(meta.get("depth_scale", cfg.get("depth_scale", 1e-3)))
        self.depth_vis_range = cfg.get("depth_vis_range_m", meta.get("depth_vis_range_m", [0.2, 2.0]))
        if isinstance(self.depth_vis_range, (int, float)):
            self.depth_vis_range = [0.0, float(self.depth_vis_range)]
        if len(self.depth_vis_range) != 2:
            self.depth_vis_range = [0.0, 2.0]
        self.audio_cfg = meta.get("audio", cfg.get("audio", {}))
        self.audio_enabled = bool(self.audio_cfg.get("enabled", False))
        default_buffer = int(self.audio_cfg.get("buffer_frames", 1)) if self.audio_cfg else 1
        override_buffer = audio_buffer_frames if audio_buffer_frames is not None else default_buffer
        self.audio_buffer = max(1, int(override_buffer))
        self.audio_sample_rate = int(self.audio_cfg.get("sample_rate", 48000))
        self.audio_stride = max(1, int(self.audio_cfg.get("stride", 1)))
        self.samples_per_audio_frame = max(
            1, int(round(self.audio_sample_rate / max(1, self.fps)))
        )
        self.train_stride = max(1, int(meta.get("train_stride", 1)))
        self.eval_stride = max(1, int(meta.get("eval_stride", max(1, self.fps))))
        self.language_fallback = meta.get("default_instruction", "")
        self.camera_cfg = meta.get("cameras", {})

    def iter_episode(
        self,
        traj_idx: int,
        *,
        num_actions: int,
        training: bool,
        image_aug,
        lang_aug_map: dict | None,
        action_mode: str,
        **kwargs,
    ) -> Iterable[dict]:
        episode_meta = self.meta["datalist"][traj_idx]
        episode = self._prepare_episode_payload(episode_meta)

        skip_images = kwargs.get("skip_images", False)
        rgb_buffers = {}
        if not skip_images:
            rgb_buffers = {
                cam: read_video_to_frames(path) for cam, path in episode["videos"].items()
            }
            
        if rgb_buffers:
            num_frames = len(next(iter(rgb_buffers.values())))
        else:
            # Fallback to state length if images are skipped or missing
            num_frames = len(episode["state"])

        if num_frames == 0:
            return

        depth_cache: Dict[Tuple[str, int], Image.Image] = {}

        states = episode["state"]
        actions = episode["action"]
        timestamps = episode["timestamps"]
        torque = episode["torque"]

        start_frame = int(episode_meta.get("start_frame", 0))
        stop_frame = int(
            min(
                episode_meta.get("end_frame", num_frames - 1),
                num_frames - num_actions - 1,
            )
        )
        stride = (
            episode_meta.get("train_stride", self.train_stride)
            if training
            else episode_meta.get("eval_stride", self.eval_stride)
        )
        if stop_frame < start_frame:
            return
        idxs = list(range(start_frame, stop_frame + 1, max(1, stride)))
        if training:
            random.shuffle(idxs)

        base_instruction = episode_meta.get("instruction") or self._fallback_instruction(
            episode_meta
        )

        for idx in idxs:
            proprio = torch.as_tensor(states[idx], dtype=torch.float32)
            action_seq = torch.as_tensor(
                actions[idx : idx + num_actions], dtype=torch.float32
            )
            if action_seq.shape[0] != num_actions:
                continue

            if not skip_images:
                imgs = self._build_visual_stack(
                    idx, rgb_buffers, episode["depth_templates"], depth_cache, image_aug
                )
                if imgs is None:
                    continue
            else:
                # Dummy images for stats calculation
                # Shape: [V, C, H, W] -> [1, 3, 224, 224] dummy
                imgs = {
                    "tensor": torch.empty(1, 3, 224, 224), 
                    "mask": torch.ones(1, dtype=torch.bool)
                }

            sample = {
                "language_instruction": self._augment_instruction(
                    base_instruction, lang_aug_map, training
                ),
                "image_input": imgs["tensor"],
                "image_mask": imgs["mask"],
                "proprio": proprio,
                "action": action_seq,
            }

            if torque is not None:
                torque_seq = torch.as_tensor(
                    torque[idx : idx + num_actions], dtype=torch.float32
                )
                if torque_seq.shape[0] == num_actions:
                    sample["ft_input"] = torque_seq

            if self.audio_enabled:
                audio_tensor = None
                if episode["audio_info"] is not None:
                    audio_tensor = self._build_audio_receding_window(idx, episode["audio_info"])
                elif episode["audio_template"] is not None:
                    audio_tensor = self._build_audio_receding_window_from_template(
                        idx, episode["audio_template"]
                    )
                if audio_tensor is not None:
                    sample["audio_input"] = audio_tensor

            yield sample

    # -------------------------------------------------------------------------
    # Episode-level helpers
    # -------------------------------------------------------------------------
    def _prepare_episode_payload(self, episode_meta: dict) -> dict:
        root = episode_meta.get("top_path") or episode_meta.get("root")
        if root is None:
            raise ValueError("Episode meta must contain 'top_path' or 'root'")

        data_rel = episode_meta.get("data_path")
        if data_rel is None:
            data_dir = episode_meta.get("data_dir", "")
            data_file = episode_meta.get("data_file", "data.parquet")
            rel = fileio.join_path(data_dir, data_file) if data_dir else data_file
            data_path = _maybe_join(root, rel)
        else:
            data_path = _maybe_join(root, data_rel)
        table = read_parquet(data_path)

        states = np.asarray(table[self.state_key])
        actions = np.asarray(table[self.action_key])
        timestamps = (
            np.asarray(table["timestamp"])
            if "timestamp" in table
            else np.arange(len(states)) / float(self.fps)
        )

        torque = None
        if self.torque_key and self.torque_key in table:
            torque_values = np.asarray(table[self.torque_key])
            if torque_values.ndim == 1:
                torque_values = torque_values[:, None]
            if torque_values.shape[0] != states.shape[0]:
                torque_times = (
                    np.asarray(table[self.torque_time_key])
                    if self.torque_time_key and self.torque_time_key in table
                    else np.linspace(0, timestamps[-1], torque_values.shape[0])
                )
                torque = self._resample_sequence(
                    torque_values, torque_times, timestamps
                )
            else:
                torque = torque_values

        videos = self._resolve_video_paths(root, episode_meta)
        depth_templates = self._resolve_depth_templates(root, episode_meta)
        audio_info = self._load_audio_artifacts(root, episode_meta)
        audio_template = None
        if audio_info is None:
            audio_template = self._resolve_audio_template(root, episode_meta)

        return {
            "state": states,
            "action": actions,
            "timestamps": timestamps,
            "torque": torque,
            "videos": videos,
            "depth_templates": depth_templates,
            "audio_info": audio_info,
            "audio_template": audio_template,
        }

    def _resolve_video_paths(self, root: str, episode_meta: dict) -> Dict[str, str]:
        result: Dict[str, str] = {}
        video_dir = episode_meta.get("video_dir", "videos")
        declared = episode_meta.get("videos", {})
        cameras = declared.keys() if declared else self.camera_cfg.keys()
        for cam in cameras:
            if cam in declared:
                rel = declared[cam]
            else:
                rel = fileio.join_path(video_dir, f"{cam}.mp4")
            resolved = _maybe_join(root, rel)
            if resolved is not None:
                result[cam] = resolved
        if not result:
            raise ValueError("No video paths resolved for episode")
        return result

    def _resolve_depth_templates(self, root: str, episode_meta: dict) -> Dict[str, str]:
        templates: Dict[str, str] = {}
        if not self.camera_cfg:
            return templates
        declared = episode_meta.get("depth_templates", {})
        depth_dir = episode_meta.get("depth_dir", "depth")
        pattern = episode_meta.get("depth_pattern", "{cam}_depth_{frame:06d}.png")
        for cam, spec in self.camera_cfg.items():
            if not spec.get("depth", False):
                continue
            if cam in declared:
                rel = declared[cam]
                templates[cam] = _maybe_join(root, rel)
            else:
                rel = fileio.join_path(depth_dir, pattern)
                templates[cam] = _maybe_join(root, rel)
        return templates

    def _resolve_audio_template(self, root: str, episode_meta: dict) -> str | None:
        if not self.audio_enabled:
            return None
        audio_dir = episode_meta.get("audio_dir", "audio")
        pattern = episode_meta.get("audio_pattern", "audio_{frame:06d}.pt")
        rel = fileio.join_path(audio_dir, pattern)
        return _maybe_join(root, rel)

    def _load_audio_artifacts(self, root: str, episode_meta: dict) -> Dict[str, Any] | None:
        if not self.audio_enabled:
            return None
        audio_meta = episode_meta.get("audio")
        if not audio_meta:
            return None
        raw_rel = audio_meta.get("raw_file")
        index_rel = audio_meta.get("index_file")
        if raw_rel is None or index_rel is None:
            return None
        raw_path = _maybe_join(root, raw_rel)
        index_path = _maybe_join(root, index_rel)
        if raw_path is None or index_path is None:
            return None
        waveform, sr = torchaudio.load(raw_path)
        index_table = pq.read_table(index_path)
        sample_start = np.asarray(index_table["sample_start"].to_pylist(), dtype=np.int64)
        sample_end = np.asarray(index_table["sample_end"].to_pylist(), dtype=np.int64)
        return {
            "waveform": waveform.squeeze(0),
            "sample_start": sample_start,
            "sample_end": sample_end,
            "sample_rate": sr,
        }

    # -------------------------------------------------------------------------
    # Visual helpers
    # -------------------------------------------------------------------------
    def _build_visual_stack(
        self,
        frame_idx: int,
        rgb_buffers: Dict[str, np.ndarray],
        depth_templates: Dict[str, str],
        depth_cache: Dict[Tuple[str, int], torch.Tensor],
        image_aug,
    ) -> dict | None:
        views: List[torch.Tensor] = []

        for cam, frames in rgb_buffers.items():
            if frame_idx >= len(frames):
                continue
            rgb = Image.fromarray(frames[frame_idx])
            views.append(image_aug(rgb))
            if len(views) == self.num_views:
                break

        if len(views) < self.num_views and depth_templates:
            for cam, template in depth_templates.items():
                rgb_depth = self._get_depth_rgb(cam, frame_idx, template, depth_cache)
                if rgb_depth is None:
                    continue
                views.append(image_aug(rgb_depth))
                if len(views) == self.num_views:
                    break

        if not views:
            return None

        valid_views = len(views)
        while len(views) < self.num_views:
            views.append(torch.zeros_like(views[0]))

        image_mask = torch.zeros(self.num_views, dtype=torch.bool)
        image_mask[:valid_views] = True

        return {"tensor": torch.stack(views, dim=0), "mask": image_mask}

    def _get_depth_rgb(
        self,
        cam: str,
        frame_idx: int,
        template: str,
        cache: Dict[Tuple[str, int], torch.Tensor],
    ) -> Image.Image | None:
        key = (cam, frame_idx)
        if key in cache:
            return cache[key]
        path = template.format(cam=cam, frame=frame_idx)
        try:
            depth_m = self._load_depth_meters(path)
        except FileNotFoundError:
            return None
        rgb = self._depth_to_rgb(depth_m)
        pil_img = Image.fromarray(rgb)
        cache[key] = pil_img
        return pil_img

    def _load_depth_meters(self, path: str) -> np.ndarray:
        if fileio.exists(path):
            with fileio.get_local_path(path) as lp:
                with open(lp, "rb") as f:
                    depth_uint16 = np.array(Image.open(f), dtype=np.uint16)
        else:
            depth_uint16 = np.array(Image.open(io.BytesIO(_load_bytes(path))), dtype=np.uint16)
        depth = depth_uint16.astype(np.float32) * self.depth_scale
        return depth

    def _depth_to_rgb(self, depth: np.ndarray) -> np.ndarray:
        d_min, d_max = self.depth_vis_range
        depth_norm = np.clip((depth - d_min) / (d_max - d_min + 1e-6), 0.0, 1.0)
        depth_uint8 = (depth_norm * 255.0).astype(np.uint8)
        if depth_uint8.ndim == 2:
            depth_uint8 = depth_uint8[..., None]
        return np.repeat(depth_uint8, 3, axis=-1)

    # -------------------------------------------------------------------------
    # Audio helpers
    # -------------------------------------------------------------------------
    def _build_audio_receding_window(
        self, frame_idx: int, audio_info: Dict[str, Any]
    ) -> torch.Tensor | None:
        waveform = audio_info["waveform"]
        sample_start = audio_info["sample_start"]
        sample_end = audio_info["sample_end"]
        max_frame = len(sample_start) - 1
        if max_frame < 0:
            return None

        chunks: List[torch.Tensor] = []
        for offset in range(-(self.audio_buffer - 1), 1):
            idx = max(0, frame_idx + offset * self.audio_stride)
            idx = min(idx, max_frame)
            start = int(sample_start[idx])
            end = int(sample_end[idx])
            start = max(0, min(start, waveform.shape[-1]))
            end = max(start, min(end, waveform.shape[-1]))
            chunk = waveform[start:end]
            chunk = self._normalize_audio_chunk(chunk)
            chunks.append(chunk)
        if not chunks:
            return None
        return torch.cat(chunks, dim=0)

    def _build_audio_receding_window_from_template(
        self, frame_idx: int, template: str
    ) -> torch.Tensor | None:
        chunks: List[torch.Tensor] = []
        for offset in range(-(self.audio_buffer - 1), 1):
            idx = max(0, frame_idx + offset * self.audio_stride)
            path = template.format(frame=idx)
            try:
                chunk = self._load_audio_chunk(path)
            except FileNotFoundError:
                return None
            chunks.append(chunk)
        if not chunks:
            return None
        return torch.cat(chunks, dim=0)

    def _normalize_audio_chunk(self, chunk: torch.Tensor) -> torch.Tensor:
        chunk = chunk.float().view(-1)
        if chunk.numel() > self.samples_per_audio_frame:
            chunk = chunk[-self.samples_per_audio_frame :]
        elif chunk.numel() < self.samples_per_audio_frame:
            pad = self.samples_per_audio_frame - chunk.numel()
            chunk = F.pad(chunk, (0, pad))
        return chunk

    def _load_audio_chunk(self, path: str) -> torch.Tensor:
        if fileio.exists(path):
            with fileio.get_local_path(path) as lp:
                tensor = torch.load(lp, map_location="cpu")
        else:
            tensor = torch.load(io.BytesIO(_load_bytes(path)), map_location="cpu")
        if not torch.is_tensor(tensor):
            tensor = torch.as_tensor(tensor)
        return self._normalize_audio_chunk(tensor)

    # -------------------------------------------------------------------------
    # Misc helpers
    # -------------------------------------------------------------------------
    def _resample_sequence(
        self, values: np.ndarray, src_times: np.ndarray, dst_times: np.ndarray
    ) -> np.ndarray:
        dst = np.empty((len(dst_times), values.shape[-1]), dtype=np.float32)
        for d in range(values.shape[-1]):
            dst[:, d] = np.interp(dst_times, src_times, values[:, d])
        return dst

    def _fallback_instruction(self, episode_meta: dict) -> str:
        if "instruction" in episode_meta and episode_meta["instruction"]:
            return episode_meta["instruction"]
        tasks = episode_meta.get("tasks") or self.meta.get("tasks") or []
        if tasks:
            return tasks[0]
        return self.language_fallback or "perform the demonstrated task"

    def _augment_instruction(self, base: str, lang_aug_map: dict | None, training: bool) -> str:
        if training and lang_aug_map and base in lang_aug_map:
            return random.choice(lang_aug_map[base])
        return base
