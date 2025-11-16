#!/usr/bin/env python3
"""
Utility script to visualize RGB and depth streams from an Intel RealSense D435i.

Intended for teleoperation sessions where the operator needs live perception
feedback alongside the gello control loop. Supports both OpenCV (when GUI
backends like GTK are available) and a pygame-based fallback for headless
systems or minimal desktop setups.
"""

import argparse
import signal
import sys
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from gello.cameras.realsense_camera import RealSenseCamera


COLORMAP_LOOKUP = {
    "jet": cv2.COLORMAP_JET,
    "turbo": cv2.COLORMAP_TURBO,
    "inferno": cv2.COLORMAP_INFERNO,
    "gray": cv2.COLORMAP_BONE,
}


@dataclass
class ViewerConfig:
    device_id: Optional[str]
    flip: bool
    depth_scale: float
    max_depth_m: float
    show_depth: bool
    colormap: str
    window_name: str
    depth_window_name: str
    overlay_info: bool
    fullscreen: bool
    backend: str


class OpenCVDisplayBackend:
    """Display implementation that uses cv2 windows."""

    def __init__(self, config: ViewerConfig):
        import cv2

        self.cv2 = cv2
        self.config = config
        self.window_flags = cv2.WINDOW_FULLSCREEN if config.fullscreen else cv2.WINDOW_NORMAL
        try:
            cv2.namedWindow(config.window_name, self.window_flags)
            if config.show_depth:
                cv2.namedWindow(config.depth_window_name, self.window_flags)
        except cv2.error as exc:
            raise RuntimeError(
                "OpenCV GUI backend is unavailable. "
                "Install GTK/Qt support or rerun with --viewer-backend pygame."
            ) from exc

    def show(self, rgb_frame: np.ndarray, depth_frame: Optional[np.ndarray]) -> bool:
        cv2 = self.cv2
        cv2.imshow(self.config.window_name, cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
        if self.config.show_depth and depth_frame is not None:
            cv2.imshow(self.config.depth_window_name, cv2.cvtColor(depth_frame, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            return False
        return True

    def close(self):
        self.cv2.destroyAllWindows()


class PygameDisplayBackend:
    """Display implementation that uses pygame (SDL) surfaces."""

    def __init__(self, config: ViewerConfig):
        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError(
                "pygame is not installed. Install it (pip install pygame) or use the OpenCV backend."
            ) from exc

        self.pygame = pygame
        pygame.init()
        pygame.display.set_caption(
            config.window_name
            if not config.show_depth
            else f"{config.window_name} / {config.depth_window_name}"
        )
        self.config = config
        self.screen = None

    def _ensure_screen(self, width: int, height: int, has_depth: bool):
        if self.screen is not None:
            return
        total_width = width * (2 if self.config.show_depth and has_depth else 1)
        self.screen = self.pygame.display.set_mode(
            (total_width, height),
            self.pygame.FULLSCREEN if self.config.fullscreen else 0,
        )

    def _make_surface(self, frame: np.ndarray):
        frame_swapped = np.swapaxes(frame, 0, 1)
        frame_contiguous = np.ascontiguousarray(frame_swapped)
        return self.pygame.surfarray.make_surface(frame_contiguous)

    def show(self, rgb_frame: np.ndarray, depth_frame: Optional[np.ndarray]) -> bool:
        pygame = self.pygame
        height, width = rgb_frame.shape[:2]
        self._ensure_screen(width, height, depth_frame is not None)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                return False

        rgb_surface = self._make_surface(rgb_frame)
        self.screen.blit(rgb_surface, (0, 0))

        if self.config.show_depth and depth_frame is not None:
            depth_surface = self._make_surface(depth_frame)
            self.screen.blit(depth_surface, (width, 0))

        pygame.display.flip()
        return True

    def close(self):
        if self.screen is not None:
            self.pygame.display.quit()
        self.pygame.quit()


def _make_display_backend(config: ViewerConfig):
    if config.backend == "pygame":
        return PygameDisplayBackend(config)
    elif config.backend == "opencv":
        return OpenCVDisplayBackend(config)
    else:
        raise ValueError(f"Unknown viewer backend: {config.backend}")


class RealSenseViewer:
    """Continuously pulls RGB/depth frames and visualizes them."""

    def __init__(self, config: ViewerConfig):
        self.config = config
        self.camera = RealSenseCamera(device_id=config.device_id, flip=config.flip)
        self._shutdown = False
        self.backend = self._init_backend(config)

    def close(self):
        if self.backend:
            self.backend.close()
        self.camera.stop()

    def _init_backend(self, config: ViewerConfig):
        try:
            return _make_display_backend(config)
        except RuntimeError as exc:
            if config.backend == "opencv":
                print(
                    "[RealSenseViewer] OpenCV backend unavailable, attempting pygame fallback...",
                    file=sys.stderr,
                )
                fallback_config = ViewerConfig(
                    device_id=config.device_id,
                    flip=config.flip,
                    depth_scale=config.depth_scale,
                    max_depth_m=config.max_depth_m,
                    show_depth=config.show_depth,
                    colormap=config.colormap,
                    window_name=config.window_name,
                    depth_window_name=config.depth_window_name,
                    overlay_info=config.overlay_info,
                    fullscreen=config.fullscreen,
                    backend="pygame",
                )
                self.config = fallback_config
                return _make_display_backend(fallback_config)
            raise

    def _render_depth(self, depth: np.ndarray) -> np.ndarray:
        """Convert raw depth (uint16, meters via scale) into a color visualization."""
        depth_metric = depth.astype(np.float32) * self.config.depth_scale
        max_depth = self.config.max_depth_m
        if max_depth <= 0:
            # Auto-scale: use 99th percentile to avoid outliers
            valid = depth_metric[depth_metric > 0]
            max_depth = float(np.percentile(valid, 99)) if valid.size > 0 else 1.0
        depth_clipped = np.clip(depth_metric, 0.0, max_depth)
        if max_depth == 0:
            normalized = depth_clipped
        else:
            normalized = depth_clipped / max_depth
        depth_u8 = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
        colormap = COLORMAP_LOOKUP.get(self.config.colormap, cv2.COLORMAP_JET)
        depth_color_bgr = cv2.applyColorMap(depth_u8, colormap)

        if self.config.overlay_info:
            center_value = depth_metric[depth_metric.shape[0] // 2, depth_metric.shape[1] // 2]
            cv2.putText(
                depth_color_bgr,
                f"Center: {center_value:.3f} m",
                (10, depth_color_bgr.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        return cv2.cvtColor(depth_color_bgr, cv2.COLOR_BGR2RGB)

    def run(self):
        last_time = time.time()
        frames_rendered = 0
        current_fps = 0.0

        try:
            while not self._shutdown:
                rgb, depth = self.camera.read()
                rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                frames_rendered += 1
                now = time.time()
                if now - last_time >= 1.0:
                    current_fps = frames_rendered / (now - last_time)
                    frames_rendered = 0
                    last_time = now

                if self.config.overlay_info:
                    cv2.putText(
                        rgb_bgr,
                        f"{current_fps:4.1f} FPS",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                rgb_display = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

                depth_display = None
                if self.config.show_depth:
                    depth_img = depth.squeeze(-1)
                    depth_display = self._render_depth(depth_img)

                if not self.backend.show(rgb_display, depth_display):
                    break
        finally:
            self.close()


def parse_args() -> ViewerConfig:
    parser = argparse.ArgumentParser(
        description="Visualize RGB + depth streams from an Intel RealSense D435i."
    )
    parser.add_argument("--device-id", type=str, default=None, help="Optional serial number")
    parser.add_argument("--flip", action="store_true", help="Rotate frames by 180 degrees")
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=0.001,
        help="Meters per depth unit reported by the camera (default: 0.001 for D435i)",
    )
    parser.add_argument(
        "--max-depth-m",
        type=float,
        default=2.5,
        help="Depth range (meters) mapped to color. Set <=0 for auto.",
    )
    parser.add_argument(
        "--hide-depth",
        action="store_true",
        help="Disable the depth visualization window.",
    )
    parser.add_argument(
        "--colormap",
        choices=list(COLORMAP_LOOKUP.keys()),
        default="jet",
        help="Colormap used for depth visualization.",
    )
    parser.add_argument(
        "--window-name",
        type=str,
        default="RealSense RGB",
        help="OpenCV window name for RGB feed.",
    )
    parser.add_argument(
        "--depth-window-name",
        type=str,
        default="RealSense Depth",
        help="OpenCV window name for depth feed.",
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Disable overlay text (FPS, center depth).",
    )
    parser.add_argument(
        "--fullscreen",
        action="store_true",
        help="Open viewer windows in fullscreen mode.",
    )
    parser.add_argument(
        "--viewer-backend",
        choices=["opencv", "pygame"],
        default="opencv",
        help="GUI backend for visualization (default: opencv).",
    )

    args = parser.parse_args()
    config = ViewerConfig(
        device_id=args.device_id,
        flip=args.flip,
        depth_scale=args.depth_scale,
        max_depth_m=args.max_depth_m,
        show_depth=not args.hide_depth,
        colormap=args.colormap,
        window_name=args.window_name,
        depth_window_name=args.depth_window_name,
        overlay_info=not args.no_overlay,
        fullscreen=args.fullscreen,
        backend=args.viewer_backend,
    )
    return config


def main():
    config = parse_args()
    viewer = RealSenseViewer(config)

    def _signal_handler(signum, _frame):
        viewer.close()
        sys.exit(0)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _signal_handler)

    viewer.run()


if __name__ == "__main__":
    main()

