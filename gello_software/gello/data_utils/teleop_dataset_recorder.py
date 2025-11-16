from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torchaudio
import zmq

from gello.cameras.realsense_camera import RealSenseCamera
from gello.data_utils.keyboard_interface import KBReset

try:
    from multimodal_dataset.audio_processor import PyAudioRecorder
except ImportError:  # pragma: no cover - optional dependency
    PyAudioRecorder = None  # type: ignore

try:
    import alsaaudio  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    alsaaudio = None


CAMERA_NAME = "cam_front"
VIDEO_DIR = "videos"
DEPTH_DIR = "depth"
AUDIO_DIR = "audio"
META_DIR = "meta"


@dataclass
class DatasetRecordingConfig:
    enabled: bool
    dataset_name: Optional[str]
    instruction: Optional[str]
    root: Path
    enable_rgb: bool = True
    enable_depth: bool = False
    enable_audio: bool = False
    enable_torque: bool = False
    fps: float = 12.0
    resolution: int = 224
    audio_sample_rate: int = 48_000
    audio_buffer_frames: int = 16
    stream_address: Optional[str] = None
    stream_timeout_s: float = 5.0
    realsense_device_id: Optional[str] = None
    realsense_flip: bool = False
    audio_device_name: Optional[str] = None
    audio_device_index: Optional[int] = None
    audio_backend: str = "alsaaudio"
    audio_alsa_device: Optional[str] = None


class ViewerStreamSubscriber:
    """Subscribe to the RealSense viewer ZeroMQ stream."""

    def __init__(self, address: str):
        self._address = address
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.SUBSCRIBE, b"")
        # Poll periodically so we can exit cleanly.
        self._socket.setsockopt(zmq.RCVTIMEO, 1000)
        self._socket.connect(address)
        self._latest: Optional[Dict[str, Any]] = None
        self._ready = threading.Event()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            try:
                payload = self._socket.recv_pyobj()
            except zmq.Again:
                continue
            except zmq.ZMQError:
                break
            if isinstance(payload, dict) and "rgb" in payload:
                self._latest = payload
                self._ready.set()

    def wait_for_frame(self, timeout: float) -> bool:
        return self._ready.wait(timeout)

    def latest(self) -> Optional[Dict[str, Any]]:
        return self._latest

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        try:
            self._socket.close(0)
        finally:
            self._context.term()


class LocalRealSenseStream:
    """Fallback RealSense capture when viewer stream is unavailable."""

    def __init__(self, device_id: Optional[str], flip: bool, resolution: int):
        self._camera = RealSenseCamera(device_id=device_id, flip=flip)
        self._resolution = resolution
        self._lock = threading.Lock()
        self._latest: Optional[Dict[str, Any]] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            try:
                rgb, depth = self._camera.read()
            except Exception:
                continue

            rgb_resized = cv2.resize(
                rgb, (self._resolution, self._resolution), interpolation=cv2.INTER_AREA
            )
            depth_img = depth.squeeze(-1)
            depth_resized = cv2.resize(
                depth_img, (self._resolution, self._resolution), interpolation=cv2.INTER_NEAREST
            ).astype(np.uint16)
            payload = {
                "timestamp": time.time(),
                "rgb": rgb_resized,
                "depth": depth_resized,
            }
            with self._lock:
                self._latest = payload

    def wait_for_frame(self, timeout: float) -> bool:
        waited = 0.0
        interval = 0.1
        while waited < timeout:
            with self._lock:
                if self._latest is not None:
                    return True
            time.sleep(interval)
            waited += interval
        return False

    def latest(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            if self._latest is None:
                return None
            return dict(self._latest)

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._camera.stop()


class CameraFrameSource:
    """Handles camera frames either from the viewer stream or directly from the RealSense."""

    def __init__(self, config: DatasetRecordingConfig):
        self._config = config
        self._mode = "stream"
        self._stream: Optional[ViewerStreamSubscriber] = None
        self._local_stream: Optional[LocalRealSenseStream] = None

        if config.stream_address:
            self._stream = ViewerStreamSubscriber(config.stream_address)
            self._stream.start()
            if not self._stream.wait_for_frame(config.stream_timeout_s):
                print(
                    "[dataset] Viewer stream not detected within "
                    f"{config.stream_timeout_s}s, falling back to direct RealSense capture."
                )
                self._stream.stop()
                self._stream = None
                self._mode = "local"
            else:
                print(f"[dataset] Subscribed to viewer stream at {config.stream_address}.")
        else:
            self._mode = "local"

        if self._mode == "local":
            self._local_stream = LocalRealSenseStream(
                config.realsense_device_id, config.realsense_flip, config.resolution
            )
            self._local_stream.start()
            if not self._local_stream.wait_for_frame(config.stream_timeout_s):
                print("[dataset] Unable to read frames from RealSense camera.")

    def latest(self) -> Optional[Tuple[float, np.ndarray, Optional[np.ndarray]]]:
        payload: Optional[Dict[str, Any]] = None
        if self._mode == "stream" and self._stream:
            payload = self._stream.latest()
        elif self._local_stream:
            payload = self._local_stream.latest()
        if payload is None:
            return None
        depth = payload.get("depth")
        if depth is not None and depth.ndim == 2:
            depth = depth[:, :, None]
        return payload["timestamp"], payload["rgb"], depth

    def close(self):
        if self._stream:
            self._stream.stop()
        if self._local_stream:
            self._local_stream.stop()


class ALSAAudioRecorder:
    """Recorder powered by the pyalsaaudio bindings."""

    def __init__(
        self,
        device: str,
        sample_rate: int,
        channels: int,
        period_size: int,
    ):
        if alsaaudio is None:
            raise RuntimeError("pyalsaaudio is not installed. Install with pip install pyalsaaudio.")
        self.device = device
        self.sample_rate = sample_rate
        self.channels = channels
        self.period_size = period_size
        self.pcm = None

    def start(self):
        pcm = alsaaudio.PCM(
            type=alsaaudio.PCM_CAPTURE,
            mode=alsaaudio.PCM_NORMAL,
            device=self.device,
        )
        pcm.setchannels(self.channels)
        pcm.setrate(self.sample_rate)
        pcm.setformat(alsaaudio.PCM_FORMAT_S16_LE)
        pcm.setperiodsize(self.period_size)
        self.pcm = pcm

    def read_chunk(self, num_samples: int) -> torch.Tensor:
        if self.pcm is None:
            raise RuntimeError("ALSA device not started.")
        bytes_per_sample = 2 * self.channels
        required_bytes = num_samples * bytes_per_sample
        buffer = bytearray()
        while len(buffer) < required_bytes:
            length, data = self.pcm.read()
            if length <= 0 and not buffer:
                continue
            buffer.extend(data)
        array = np.frombuffer(buffer[:required_bytes], dtype=np.int16)
        if self.channels > 1:
            array = array.reshape(-1, self.channels).mean(axis=1)
        tensor = torch.from_numpy(array.astype(np.float32) / 32768.0)
        return tensor

    def stop(self):
        if self.pcm is not None:
            self.pcm.close()
        self.pcm = None


class ArecordRecorder:
    """Audio recorder powered by the `arecord` CLI."""

    def __init__(
        self,
        device: str,
        sample_rate: int,
        channels: int,
        chunk_size: int,
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.process: Optional[subprocess.Popen] = None
        self.stdout = None
        self.stderr = None

    def start(self):
        cmd = [
            "arecord",
            "-D",
            self.device,
            "-f",
            "S16_LE",
            "-c",
            str(self.channels),
            "-r",
            str(self.sample_rate),
            "-t",
            "raw",
            "-q",
            "-F",
            str(self.chunk_size),
            "-B",
            str(self.chunk_size * 4),
        ]
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("arecord executable not found. Install alsa-utils.") from exc
        if self.process.stdout is None:
            raise RuntimeError("Failed to open arecord stdout stream.")
        self.stdout = self.process.stdout
        self.stderr = self.process.stderr

        time.sleep(0.2)
        if self.process.poll() is not None:
            err = ""
            if self.stderr is not None:
                err = self.stderr.read().decode(errors="ignore")
            raise RuntimeError(f"arecord failed to start: {err.strip() or 'unknown error'}")

    def read_chunk(self, num_samples: int) -> torch.Tensor:
        if self.stdout is None:
            raise RuntimeError("arecord stream not started.")
        bytes_per_sample = 2 * self.channels
        total_bytes = num_samples * bytes_per_sample
        buffer = bytearray()
        while len(buffer) < total_bytes:
            chunk = self.stdout.read(total_bytes - len(buffer))
            if not chunk:
                raise RuntimeError("arecord stream ended unexpectedly.")
            buffer.extend(chunk)
        array = np.frombuffer(buffer, dtype=np.int16)
        if self.channels > 1:
            array = array.reshape(-1, self.channels).mean(axis=1)
        tensor = torch.from_numpy(array.astype(np.float32) / 32768.0)
        return tensor

    def stop(self):
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process = None
        self.stdout = None
        if self.stderr:
            self.stderr.close()
            self.stderr = None


class AudioCaptureManager:
    """Continuously captures microphone audio and provides fixed-size windows per frame."""

    def __init__(
        self,
        sample_rate: int,
        fps: float,
        *,
        backend: str = "pyaudio",
        device_index: Optional[int] = None,
        alsa_device: Optional[str] = None,
    ):
        self._sample_rate = sample_rate
        self._samples_per_frame = int(round(sample_rate / fps))
        self._backend = backend
        self._device_index = device_index
        self._alsa_device = alsa_device
        self._buffer: deque[np.ndarray] = deque()
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._chunk_size = max(self._samples_per_frame, 1024)
        self._recorder = self._build_recorder()

    def _build_recorder(self):
        if self._backend == "pyaudio":
            if PyAudioRecorder is None:
                raise RuntimeError("PyAudio backend requested but PyAudio is not installed.")
            return PyAudioRecorder(
                sample_rate=self._sample_rate,
                channels=1,
                chunk_size=1024,
                device_index=self._device_index,
            )
        if self._backend == "alsaaudio":
            if self._alsa_device is None:
                raise RuntimeError("ALSA device string required for alsaaudio backend (e.g., plughw:1,0).")
            return ALSAAudioRecorder(
                device=self._alsa_device,
                sample_rate=self._sample_rate,
                channels=1,
                period_size=self._chunk_size,
            )
        if self._backend == "arecord":
            if shutil.which("arecord") is None:
                raise RuntimeError("arecord executable not found. Install alsa-utils or switch backend.")
            device = self._alsa_device
            if device is None:
                raise RuntimeError(
                    "ALSA device string is required for arecord backend (e.g., plughw:1,0)."
                )
            return ArecordRecorder(
                device=device,
                sample_rate=self._sample_rate,
                channels=1,
                chunk_size=self._chunk_size,
            )
        raise ValueError(f"Unsupported audio backend '{self._backend}'")

    def start(self):
        if self._running:
            return
        self._recorder.start()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            try:
                chunk = self._recorder.read_chunk(self._chunk_size)
            except Exception as exc:
                print(f"[dataset] Audio recorder error: {exc}")
                self._running = False
                break
            data = chunk.squeeze(-1).numpy()
            with self._lock:
                self._buffer.append(data)

    def consume(self) -> torch.Tensor:
        needed = self._samples_per_frame
        collected: List[np.ndarray] = []
        with self._lock:
            while needed > 0 and self._buffer:
                chunk = self._buffer.popleft()
                if len(chunk) <= needed:
                    collected.append(chunk)
                    needed -= len(chunk)
                else:
                    collected.append(chunk[:needed])
                    self._buffer.appendleft(chunk[needed:])
                    needed = 0

        if needed > 0:
            collected.append(np.zeros(needed, dtype=np.float32))

        data = np.concatenate(collected).astype(np.float32)
        return torch.from_numpy(data)

    def stop(self):
        if not self._running:
            return
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._recorder.stop()
        with self._lock:
            self._buffer.clear()

    @property
    def running(self) -> bool:
        return self._running


def _list_audio_devices() -> List[Dict[str, Any]]:
    try:
        import pyaudio
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("PyAudio is required for audio recording.") from exc

    pa = pyaudio.PyAudio()
    devices: List[Dict[str, Any]] = []
    try:
        for idx in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(idx)
            devices.append(info)
    finally:
        pa.terminate()
    return devices


def _resolve_audio_device(
    device_index: Optional[int],
    device_name: Optional[str],
) -> Tuple[int, str, Optional[str]]:
    devices = _list_audio_devices()
    input_devices = [
        info for info in devices if info.get("maxInputChannels", 0) > 0
    ]

    if not input_devices:
        raise RuntimeError("No audio input devices detected. Check ALSA/PortAudio configuration.")

    def device_summary() -> str:
        lines = []
        for info in input_devices:
            lines.append(f"[{info['index']}] {info['name']}")
        return "\n  ".join(lines)

    if device_index is not None:
        for info in input_devices:
            if info["index"] == device_index:
                return device_index, info["name"], _infer_alsa_device(info["name"])
        raise RuntimeError(
            f"No audio input device with index {device_index}. Available devices:\n  {device_summary()}"
        )

    if device_name:
        needle = device_name.lower()
        for info in input_devices:
            if needle in info["name"].lower():
                return info["index"], info["name"], _infer_alsa_device(info["name"])
        raise RuntimeError(
            f"No audio input device matching '{device_name}'. Available devices:\n  {device_summary()}"
        )

    default = next((info for info in input_devices if info.get("defaultSampleRate")), input_devices[0])
    return default["index"], default["name"], _infer_alsa_device(default["name"])


def _infer_alsa_device(device_name: str) -> Optional[str]:
    if "(" in device_name and ")" in device_name:
        inside = device_name.split("(")[-1].split(")")[0]
        if inside.startswith("hw:") or inside.startswith("plughw:"):
            if inside.startswith("plughw:"):
                return inside
            return "plughw:" + inside.split("hw:")[-1]
    return None


class EpisodeWriter:
    """Handles per-episode artifact creation and parquet logging."""

    def __init__(
        self,
        dataset_dir: Path,
        dataset_name: str,
        episode_index: int,
        config: DatasetRecordingConfig,
        instruction: Optional[str],
    ):
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.episode_index = episode_index
        self.config = config
        self.instruction = instruction or ""
        self.episode_dir = dataset_dir / f"episode_{episode_index:06d}"
        self.episode_dir.mkdir(parents=True, exist_ok=True)

        self.video_dir = self.episode_dir / VIDEO_DIR
        self.depth_dir = self.episode_dir / DEPTH_DIR
        self.audio_dir = self.episode_dir / AUDIO_DIR

        self.video_dir.mkdir(exist_ok=True)
        if config.enable_depth:
            self.depth_dir.mkdir(exist_ok=True)
        if config.enable_audio:
            self.audio_dir.mkdir(exist_ok=True)

        self.data_path = self.episode_dir / "data.parquet"

        self._frame_count = 0
        self._timestamps: List[float] = []
        self._frame_indices: List[int] = []
        self._states: List[List[float]] = []
        self._actions: List[List[float]] = []
        self._torques: List[Optional[List[float]]] = []
        self._audio_chunks: List[torch.Tensor] = []
        self._audio_index_rows: List[Dict[str, int]] = []
        self._audio_sample_cursor: int = 0

        self.state_dim: Optional[int] = None
        self.action_dim: Optional[int] = None
        self.torque_dim: Optional[int] = None

        self._video_writer: Optional[cv2.VideoWriter] = None
        if config.enable_rgb:
            video_path = self.video_dir / f"{CAMERA_NAME}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._video_writer = cv2.VideoWriter(
                str(video_path), fourcc, config.fps, (config.resolution, config.resolution)
            )

    def log_step(
        self,
        timestamp: float,
        state: List[float],
        action: List[float],
        torque: Optional[List[float]],
        rgb_frame: Optional[np.ndarray],
        depth_frame: Optional[np.ndarray],
        audio_chunk: Optional[torch.Tensor],
    ):
        if self.state_dim is None:
            self.state_dim = len(state)
        if self.action_dim is None:
            self.action_dim = len(action)
        if torque is not None and self.torque_dim is None:
            self.torque_dim = len(torque)

        self._timestamps.append(timestamp)
        self._frame_indices.append(self._frame_count)
        self._states.append(state)
        self._actions.append(action)
        self._torques.append(torque)

        if self._video_writer is not None:
            frame = rgb_frame
            if frame is None:
                frame = np.zeros(
                    (self.config.resolution, self.config.resolution, 3), dtype=np.uint8
                )
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self._video_writer.write(bgr)

        if self.config.enable_depth and depth_frame is not None:
            depth_img = depth_frame.squeeze(-1).astype(np.uint16)
            depth_path = self.depth_dir / f"{CAMERA_NAME}_depth_{self._frame_count:06d}.png"
            cv2.imwrite(str(depth_path), depth_img)

        if self.config.enable_audio:
            if audio_chunk is None:
                samples = 0
            else:
                samples = int(audio_chunk.numel())
                if samples > 0:
                    self._audio_chunks.append(audio_chunk.detach().cpu())
            start = self._audio_sample_cursor
            end = start + samples
            self._audio_sample_cursor = end
            self._audio_index_rows.append(
                {
                    "frame_index": self._frame_count,
                    "sample_start": start,
                    "sample_end": end,
                }
            )

        self._frame_count += 1

    def close(self):
        if self._video_writer is not None:
            self._video_writer.release()
        self._write_audio_artifacts()
        self._write_parquet()

    def _write_parquet(self):
        if not self._timestamps:
            return

        data: Dict[str, pa.Array] = {
            "timestamp": pa.array(self._timestamps, type=pa.float64()),
            "frame_index": pa.array(self._frame_indices, type=pa.int32()),
            "observation.state": pa.array(
                self._states, type=pa.list_(pa.float32())
            ),
            "action": pa.array(self._actions, type=pa.list_(pa.float32())),
        }
        if self.config.enable_torque:
            torque_values = [
                t if t is not None else [0.0] * (self.torque_dim or 0)
                for t in self._torques
            ]
            data["observation.torque"] = pa.array(
                torque_values, type=pa.list_(pa.float32())
            )

        table = pa.Table.from_pydict(data)
        pq.write_table(table, self.data_path)

    def _write_audio_artifacts(self):
        if not self.config.enable_audio or not self._audio_index_rows:
            return

        audio_path = self.audio_dir / "raw_audio.wav"
        index_path = self.audio_dir / "audio_index.parquet"

        if self._audio_chunks:
            waveform = torch.cat(self._audio_chunks, dim=0).unsqueeze(0)
        else:
            waveform = torch.zeros(1, 1, dtype=torch.float32)
        torchaudio.save(
            str(audio_path),
            waveform,
            self.config.audio_sample_rate,
        )

        table = pa.table(
            {
                "frame_index": [row["frame_index"] for row in self._audio_index_rows],
                "sample_start": [row["sample_start"] for row in self._audio_index_rows],
                "sample_end": [row["sample_end"] for row in self._audio_index_rows],
            }
        )
        pq.write_table(table, index_path)

    def build_meta_entry(self) -> Dict[str, Any]:
        entry: Dict[str, Any] = {
            "top_path": str(self.episode_dir.resolve()),
            "data_file": "data.parquet",
            "videos": {},
            "tasks": [self.instruction] if self.instruction else [],
            "instruction": self.instruction,
            "start_frame": 0,
            "end_frame": max(0, self._frame_count - 1),
        }
        if self.config.enable_rgb:
            entry["video_dir"] = VIDEO_DIR
            entry["videos"][CAMERA_NAME] = f"{CAMERA_NAME}.mp4"
        if self.config.enable_depth:
            entry["depth_dir"] = DEPTH_DIR
        if self.config.enable_audio:
            entry["audio_dir"] = AUDIO_DIR
            entry["audio"] = {
                "raw_file": str(Path(AUDIO_DIR) / "raw_audio.wav"),
                "index_file": str(Path(AUDIO_DIR) / "audio_index.parquet"),
            }
        return entry

    @property
    def frame_count(self) -> int:
        return self._frame_count


class DatasetRecorder:
    """Manages dataset-level metadata and episode indexing."""

    def __init__(self, config: DatasetRecordingConfig):
        self.config = config
        self.root = config.root
        self.root.mkdir(parents=True, exist_ok=True)
        self._metadata_cache: Dict[str, Dict[str, Any]] = {}

    def normalize_name(self, name: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip().lower())
        return slug or "untitled_dataset"

    def generate_auto_dataset_name(self) -> str:
        idx = 0
        while (self.root / f"untitled_dataset_{idx}").exists():
            idx += 1
        return f"untitled_dataset_{idx}"

    def start_episode(self, dataset_name: str, instruction: Optional[str]) -> EpisodeWriter:
        dataset_name = self.normalize_name(dataset_name)
        dataset_dir = self.root / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        episode_index = self._next_episode_index(dataset_dir)
        writer = EpisodeWriter(dataset_dir, dataset_name, episode_index, self.config, instruction)
        return writer

    def finalize_episode(self, dataset_name: str, writer: EpisodeWriter):
        metadata = self._load_metadata(dataset_name)
        metadata.setdefault("datalist", []).append(writer.build_meta_entry())
        metadata["fps"] = self.config.fps
        metadata["state_key"] = "observation.state"
        metadata["action_key"] = "action"
        if metadata.get("torque_key") is None and self.config.enable_torque:
            metadata["torque_key"] = "observation.torque"
        metadata.setdefault("torque_time_key", None)

        cameras = metadata.setdefault("cameras", {})
        prev_camera = cameras.get(CAMERA_NAME, {})
        cameras[CAMERA_NAME] = {
            "depth": prev_camera.get("depth", False) or self.config.enable_depth,
            "rgb": True,
            "resolution": prev_camera.get("resolution", [self.config.resolution, self.config.resolution]),
        }

        prev_audio = metadata.get("audio", {"enabled": False})
        audio_enabled = prev_audio.get("enabled", False) or self.config.enable_audio
        metadata["audio"] = (
            {
                "enabled": audio_enabled,
                "sample_rate": self.config.audio_sample_rate,
                "buffer_frames": self.config.audio_buffer_frames,
            }
            if audio_enabled
            else {"enabled": False}
        )

        self._save_metadata(dataset_name, metadata)

    def _meta_path(self, dataset_name: str) -> Path:
        dataset_dir = self.root / dataset_name
        meta_dir = dataset_dir / META_DIR
        meta_dir.mkdir(parents=True, exist_ok=True)
        return meta_dir / "info.json"

    def _load_metadata(self, dataset_name: str) -> Dict[str, Any]:
        if dataset_name in self._metadata_cache:
            return self._metadata_cache[dataset_name]
        meta_path = self._meta_path(dataset_name)
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text())
        else:
            metadata = {
                "dataset_name": dataset_name,
                "fps": self.config.fps,
                "state_key": "observation.state",
                "action_key": "action",
                "torque_key": "observation.torque" if self.config.enable_torque else None,
                "torque_time_key": None,
                "cameras": {
                    CAMERA_NAME: {
                        "depth": self.config.enable_depth,
                        "rgb": self.config.enable_rgb,
                        "resolution": [self.config.resolution, self.config.resolution],
                    }
                },
                "audio": (
                    {
                        "enabled": True,
                        "sample_rate": self.config.audio_sample_rate,
                        "buffer_frames": self.config.audio_buffer_frames,
                    }
                    if self.config.enable_audio
                    else {"enabled": False}
                ),
                "datalist": [],
            }
        self._metadata_cache[dataset_name] = metadata
        return metadata

    def _save_metadata(self, dataset_name: str, metadata: Dict[str, Any]):
        meta_path = self._meta_path(dataset_name)
        meta_path.write_text(json.dumps(metadata, indent=2))
        self._metadata_cache[dataset_name] = metadata

    @staticmethod
    def _next_episode_index(dataset_dir: Path) -> int:
        existing = sorted(dataset_dir.glob("episode_*"))
        if not existing:
            return 0
        last = existing[-1].name.split("_")[-1]
        try:
            return int(last) + 1
        except ValueError:
            return len(existing)


class TeleopDatasetController:
    """Handles keyboard-driven dataset recording during teleoperation."""

    def __init__(
        self,
        config: DatasetRecordingConfig,
        arm_to_use: str,
        realsense_fallback_device: Optional[str] = None,
    ):
        self.config = config
        self.arm_to_use = arm_to_use
        self.recorder = DatasetRecorder(config)
        self.camera_source = (
            CameraFrameSource(config) if (config.enable_rgb or config.enable_depth) else None
        )
        self.audio_manager: Optional[AudioCaptureManager] = None
        self.audio_backend = config.audio_backend
        self.audio_device_index: Optional[int] = None
        self.audio_device_name: Optional[str] = None
        self.audio_alsa_device: Optional[str] = config.audio_alsa_device
        self._audio_retry_delay = 0.3
        self._audio_max_retries = 3
        self._audio_samples_per_frame = int(round(config.audio_sample_rate / max(config.fps, 1.0)))
        if config.enable_audio:
            idx, name, inferred = _resolve_audio_device(
                config.audio_device_index,
                config.audio_device_name,
            )
            self.audio_device_index = idx
            self.audio_device_name = name
            if self.audio_alsa_device is None:
                self.audio_alsa_device = inferred
        self.keyboard = KBReset() if config.enabled else None

        self.active_dataset_name: Optional[str] = (
            self.recorder.normalize_name(config.dataset_name) if config.dataset_name else None
        )
        self.current_writer: Optional[EpisodeWriter] = None
        self.recording = False
        self.last_log_time = 0.0
        self.interval = 1.0 / max(config.fps, 1.0)
        self._warned_camera = False
        self._warned_torque = False

    def update(self, obs: Dict[str, Any], action: np.ndarray):
        if not self.config.enabled or self.keyboard is None:
            return

        state = self.keyboard.update()

        if state == "start":
            self._start_episode()
        elif state == "save" and self.recording:
            now = time.time()
            if now - self.last_log_time >= self.interval:
                self.last_log_time = now
                audio_chunk = self._next_audio_chunk()
                self._log_step(obs, action, now, audio_chunk)
        elif state == "normal" and self.recording:
            self._stop_episode()

    def _start_episode(self):
        if self.recording:
            return
        dataset_name = self.active_dataset_name
        auto_created = False
        if dataset_name is None:
            dataset_name = self.recorder.generate_auto_dataset_name()
            auto_created = True
        dataset_name = self.recorder.normalize_name(dataset_name)
        if auto_created:
            print(f"[dataset] Recording into auto-created dataset '{dataset_name}'.")
        else:
            print(f"[dataset] Recording into dataset '{dataset_name}'.")
        self.active_dataset_name = dataset_name
        instruction = self.config.instruction or ""
        self.current_writer = self.recorder.start_episode(dataset_name, instruction)
        self.last_log_time = 0.0
        self.recording = True
        print(f"[dataset] Started new episode in dataset '{dataset_name}'.")

    def _log_step(
        self,
        obs: Dict[str, Any],
        action: np.ndarray,
        timestamp: float,
        audio_chunk: Optional[torch.Tensor],
    ):
        if self.current_writer is None:
            return
        joint_positions = obs.get("joint_positions")
        if joint_positions is None:
            return
        joint_positions = np.asarray(joint_positions, dtype=np.float32)
        action_vec = np.asarray(action, dtype=np.float32)

        torque_vec = None
        if self.config.enable_torque:
            joint_torques = obs.get("joint_torques")
            if joint_torques is None and not self._warned_torque:
                print("[dataset] joint_torques missing from observations; torque logging disabled.")
                self._warned_torque = True
            if joint_torques is not None:
                torque_vec = np.asarray(joint_torques, dtype=np.float32).tolist()

        rgb = None
        depth = None
        if self.camera_source is not None:
            frame = self.camera_source.latest()
            if frame is None and not self._warned_camera:
                print("[dataset] Waiting for RealSense frames...")
                self._warned_camera = True
            elif frame is not None:
                _, rgb, depth = frame
                if rgb is not None:
                    rgb = np.asarray(rgb, dtype=np.uint8)
                if depth is not None:
                    depth = depth.astype(np.uint16)

        self.current_writer.log_step(
            timestamp=timestamp,
            state=joint_positions.tolist(),
            action=action_vec.tolist(),
            torque=torque_vec,
            rgb_frame=rgb,
            depth_frame=depth,
            audio_chunk=audio_chunk,
        )

    def _stop_episode(self):
        if not self.recording or self.current_writer is None:
            return
        self.current_writer.close()
        if self.active_dataset_name is not None:
            self.recorder.finalize_episode(self.active_dataset_name, self.current_writer)
        print(
            f"[dataset] Episode finished with {self.current_writer.frame_count} frames "
            f"in dataset '{self.active_dataset_name}'."
        )
        self.current_writer = None
        self.recording = False
        self._warned_camera = False
        self._warned_torque = False

    def close(self):
        if self.recording:
            self._stop_episode()
        if self.camera_source:
            self.camera_source.close()
        if self.audio_manager:
            self.audio_manager.stop()
        self.audio_manager = None

    def _backend_candidates(self) -> List[str]:
        order = []
        preferred = self.config.audio_backend or "alsaaudio"
        for backend in (preferred, "alsaaudio", "arecord", "pyaudio"):
            if backend not in order:
                order.append(backend)
        return order

    def _start_audio_pipeline(self):
        if not self.config.enable_audio:
            return
        if self.audio_manager is not None:
            try:
                self.audio_manager.stop()
            except Exception:
                pass
            self.audio_manager = None
        last_exc: Optional[Exception] = None
        for backend in self._backend_candidates():
            for attempt in range(1, self._audio_max_retries + 1):
                try:
                    self._initialize_audio_manager(backend)
                    self.audio_manager.start()
                    self.audio_backend = backend
                    return
                except Exception as exc:
                    last_exc = exc
                    print(
                        f"[dataset] {backend} backend attempt {attempt}/"
                        f"{self._audio_max_retries} failed: {exc}"
                    )
                    if self.audio_manager is not None:
                        try:
                            self.audio_manager.stop()
                        except Exception:
                            pass
                        self.audio_manager = None
                    time.sleep(self._audio_retry_delay)
        raise RuntimeError(
            f"All audio backends failed to start ({last_exc}). "
            "Check microphone connection and ALSA configuration."
        )

    def _next_audio_chunk(self) -> torch.Tensor:
        if not self.config.enable_audio:
            return torch.zeros(0)
        chunk = self._consume_audio_chunk()
        if chunk is None:
            return torch.zeros(self._audio_samples_per_frame)
        return chunk

    def _ensure_audio_stream(self):
        if not self.config.enable_audio:
            return
        if self.audio_manager is None or not self.audio_manager.running:
            self._start_audio_pipeline()

    def _initialize_audio_manager(self, backend: str):
        self.audio_manager = AudioCaptureManager(
            self.config.audio_sample_rate,
            self.config.fps,
            backend=backend,
            device_index=self.audio_device_index,
            alsa_device=self.audio_alsa_device,
        )
        self.config.audio_backend = backend
        self.audio_backend = backend
        if self.audio_device_name:
            device_desc = (
                f"[{self.audio_device_index}] {self.audio_device_name}"
                if self.audio_device_index is not None
                else self.audio_device_name
            )
            backend_desc = (
                f"{device_desc}: {self.audio_alsa_device}"
                if backend in {"alsaaudio", "arecord"} and self.audio_alsa_device
                else device_desc
            )
            print(f"[dataset] Using audio device {backend_desc}")

    def _start_audio_drain(self):
        self._ensure_audio_stream()

    def _stop_audio_drain(self):
        if self.audio_manager:
            self.audio_manager.stop()
            self.audio_manager = None
    def _consume_audio_chunk(self) -> Optional[torch.Tensor]:
        if not self.config.enable_audio:
            return None
        self._ensure_audio_stream()
        if self.audio_manager is None:
            return None
        try:
            return self.audio_manager.consume()
        except Exception as exc:
            print(f"[dataset] Audio recorder error: {exc}")
            try:
                self.audio_manager.stop()
            except Exception:
                pass
            self.audio_manager = None
            return None

