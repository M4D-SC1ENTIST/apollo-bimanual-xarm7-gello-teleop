#!/usr/bin/env python3

import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _build_run_env_command(
    python_executable: str,
    arm_to_use: str,
    extra_args: List[str],
) -> Tuple[List[str], Path]:
    """Create the command that launches the teleoperation environment."""
    gello_root = Path(__file__).resolve().parent / "gello_software"
    script_path = Path("experiments") / "run_env.py"
    cmd = [
        python_executable,
        str(script_path),
        "--agent",
        "gello",
        "--arm-to-use",
        arm_to_use,
    ]
    cmd.extend(extra_args)
    return cmd, gello_root


def _build_xarm_command(
    python_executable: str,
    arm_to_use: str,
    viewpoint_arm_start_joints_state: list,
) -> Tuple[List[str], Path]:
    """Create the command that launches the xArm7 robot server."""
    gello_root = Path(__file__).resolve().parent / "gello_software"
    # Use a small Python snippet so we can propagate arm_to_use.
    python_snippet = (
        "from experiments.launch_xarm7_robots import xArm7RobotsManager\n"
        f"xArm7RobotsManager(arm_to_use={arm_to_use!r}, viewpoint_arm_start_joints_state={viewpoint_arm_start_joints_state})"
    )
    cmd = [python_executable, "-u", "-c", python_snippet]
    return cmd, gello_root


def _build_realsense_viewer_command(
    python_executable: str,
    device_id: Optional[str],
    flip: bool,
    depth_scale: float,
    max_depth_m: float,
    show_depth: bool,
    fullscreen: bool,
    backend: str,
) -> Tuple[List[str], Path]:
    """Create the command that launches the RealSense perception viewer."""
    gello_root = Path(__file__).resolve().parent / "gello_software"
    script_path = Path("experiments") / "view_realsense_camera.py"
    cmd: List[str] = [python_executable, str(script_path)]
    if device_id:
        cmd.extend(["--device-id", device_id])
    if flip:
        cmd.append("--flip")
    if depth_scale != 0.001:
        cmd.extend(["--depth-scale", str(depth_scale)])
    if max_depth_m != 2.5:
        cmd.extend(["--max-depth-m", str(max_depth_m)])
    if not show_depth:
        cmd.append("--hide-depth")
    if fullscreen:
        cmd.append("--fullscreen")
    if backend != "opencv":
        cmd.extend(["--viewer-backend", backend])
    return cmd, gello_root


def _launch_process(name: str, command: List[str], cwd: Path) -> subprocess.Popen:
    """Start a subprocess and return the handle."""
    print(f"[launch_teleop] Starting {name}: {' '.join(command)} (cwd={cwd})")
    return subprocess.Popen(command, cwd=str(cwd))


def _terminate_process(name: str, process: subprocess.Popen, timeout: float = 5.0) -> None:
    """Terminate a subprocess gracefully, then force kill if needed."""
    if process.poll() is not None:
        return
    print(f"[launch_teleop] Terminating {name} (pid={process.pid})")
    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"[launch_teleop] {name} did not exit in {timeout}s, killing.")
        process.kill()
        process.wait()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Launch the teleoperation environment and xArm7 robot server together. "
            "Any additional arguments are forwarded to run_env.py."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--arm-to-use",
        dest="arm_to_use",
        choices=["left", "right", "both"],
        default="right",
        help="Which arm(s) to teleoperate (default: right).",
    )
    parser.add_argument(
        "--viewpoint-option",
        dest="viewpoint_option",
        choices=["coffee", "engine", "cup", "whiteboard", "active", "none"],
        default="none",
        help="Which viewpoint option to use (default: none).",
    )
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python interpreter to use when launching child processes.",
    )
    parser.add_argument(
        "--enable-camera-viewer",
        action="store_true",
        help="Launch the RealSense perception viewer alongside teleop.",
    )
    parser.add_argument(
        "--realsense-device-id",
        default=None,
        help="Optional serial number for the RealSense D435i used for visualization.",
    )
    parser.add_argument(
        "--realsense-depth-scale",
        type=float,
        default=0.001,
        help="Meters per depth unit reported by the RealSense depth image (default 0.001).",
    )
    parser.add_argument(
        "--realsense-max-depth",
        type=float,
        default=2.5,
        help="Maximum depth (meters) mapped to the viewer color map.",
    )
    parser.add_argument(
        "--realsense-flip",
        action="store_true",
        help="Rotate RealSense frames 180 degrees before visualization.",
    )
    parser.add_argument(
        "--viewer-hide-depth",
        action="store_true",
        help="Disable the depth window when launching the viewer.",
    )
    parser.add_argument(
        "--viewer-fullscreen",
        action="store_true",
        help="Open the viewer windows in fullscreen mode.",
    )
    parser.add_argument(
        "--viewer-backend",
        choices=["opencv", "pygame"],
        default="pygame",
        help="Visualization backend to use for the RealSense viewer.",
    )
    args, forward_args = parser.parse_known_args()
    
    print(f"Viewpoint option: {args.viewpoint_option}")
    if args.viewpoint_option == "none":
        viewpoint_arm_start_joints_state = None
    elif args.viewpoint_option == "coffee":
        print("Setting coffee viewpoint")
        viewpoint_arm_start_joints_state = [0.63, 2.19562420, -0.39444441, 0.05410521, 1.19380521, 0.87091930, 1.91288086, -1.83259571]
    elif args.viewpoint_option == "active":
        print("Active viewpoint has not been implemented yet")
        viewpoint_arm_start_joints_state = None
    else:
        print("Invalid viewpoint option")
        return

    commands: Dict[str, Tuple[List[str], Path]] = {}
    commands["run_env"] = _build_run_env_command(
        python_executable=args.python_executable,
        arm_to_use=args.arm_to_use,
        extra_args=forward_args,
    )
    commands["xarm"] = _build_xarm_command(
        python_executable=args.python_executable,
        arm_to_use=args.arm_to_use,
        viewpoint_arm_start_joints_state=viewpoint_arm_start_joints_state,
    )
    if args.enable_camera_viewer:
        commands["realsense_viewer"] = _build_realsense_viewer_command(
            python_executable=args.python_executable,
            device_id=args.realsense_device_id,
            flip=args.realsense_flip,
            depth_scale=args.realsense_depth_scale,
            max_depth_m=args.realsense_max_depth,
            show_depth=not args.viewer_hide_depth,
            fullscreen=args.viewer_fullscreen,
            backend=args.viewer_backend,
        )

    processes: Dict[str, subprocess.Popen] = {}
    try:
        for name, (cmd, cwd) in commands.items():
            processes[name] = _launch_process(name, cmd, cwd)

        # Handle signals gracefully.
        shutdown = False

        def _signal_handler(signum, frame):
            nonlocal shutdown
            print(f"[launch_teleop] Received signal {signum}, shutting down.")
            shutdown = True

        previous_handlers = {
            sig: signal.signal(sig, _signal_handler)
            for sig in (signal.SIGINT, signal.SIGTERM)
        }

        exit_code = 0
        try:
            while not shutdown:
                for name, process in processes.items():
                    ret = process.poll()
                    if ret is not None:
                        print(f"[launch_teleop] {name} exited with code {ret}.")
                        exit_code = ret
                        shutdown = True
                        break
                if shutdown:
                    break
                time.sleep(0.5)
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)

    finally:
        for name, process in processes.items():
            _terminate_process(name, process)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()

