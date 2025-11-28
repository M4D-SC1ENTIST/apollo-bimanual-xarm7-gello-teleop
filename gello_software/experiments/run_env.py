import glob
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import tyro

from gello.env import RobotEnv
from gello.robots.robot import PrintRobot
from gello.utils.launch_utils import instantiate_from_dict
from gello.zmq_core.robot_node import ZMQClientRobot
from gello.data_utils.teleop_dataset_recorder import (
    DatasetRecordingConfig,
    TeleopDatasetController,
)


def print_color(*args, color=None, attrs=(), **kwargs):
    import termcolor

    if len(args) > 0:
        args = tuple(termcolor.colored(arg, color=color, attrs=attrs) for arg in args)
    print(*args, **kwargs)


@dataclass
class Args:
    agent: str = "none"
    robot_port: int = 6001
    wrist_camera_port: int = 5000
    base_camera_port: int = 5001
    hostname: str = "127.0.0.1"
    robot_type: str = None  # only needed for quest agent or spacemouse agent
    hz: int = 100
    start_joints: Optional[Tuple[float, ...]] = None

    gello_port: Optional[str] = None
    mock: bool = False
    use_save_interface: bool = False
    data_dir: str = "~/bc_data"
    arm_to_use: str = "right" # "left", "right", "both"
    verbose: bool = False
    enable_dataset_recorder: bool = False
    dataset_name: Optional[str] = None
    dataset_instruction: Optional[str] = None
    dataset_root: str = "datasets"
    dataset_enable_depth: bool = False
    dataset_enable_audio: bool = False
    dataset_enable_torque: bool = False
    dataset_fps: float = 12.0
    dataset_resolution: int = 224
    dataset_audio_buffer_frames: int = 16
    dataset_stream_address: Optional[str] = None
    dataset_stream_timeout: float = 5.0
    dataset_camera_device_id: Optional[str] = None
    dataset_camera_flip: bool = False
    dataset_audio_device_name: Optional[str] = None
    dataset_audio_device_index: Optional[int] = None
    dataset_audio_backend: str = "pyaudio"
    dataset_audio_alsa_device: Optional[str] = None
    dataset_registry_key: str = "multimodal-lerobot"
    gripper_control_mode: Literal["continuous", "discrete"] = "continuous"

    def __post_init__(self):
        if self.start_joints is not None:
            self.start_joints = np.array(self.start_joints)


GRIPPER_BINARY_THRESHOLD = 0.5


def _infer_gripper_indices(length: int, arm_to_use: str) -> Tuple[int, ...]:
    if length <= 0:
        return tuple()
    if arm_to_use == "both" and length % 2 == 0:
        per_arm = length // 2
        return (per_arm - 1, length - 1)
    return (length - 1,)


def _build_gripper_transform(mode: str, arm_to_use: str):
    if mode != "discrete":
        return None

    def _transform(action: np.ndarray) -> np.ndarray:
        arr = np.asarray(action, dtype=np.float32).copy()
        indices = _infer_gripper_indices(arr.shape[-1], arm_to_use)
        for idx in indices:
            if 0 <= idx < arr.shape[-1]:
                arr[..., idx] = 1.0 if arr[..., idx] >= GRIPPER_BINARY_THRESHOLD else 0.0
        return arr

    return _transform


def main(args):
    if args.mock:
        robot_client = PrintRobot(8, dont_print=True)
        camera_clients = {}
    else:
        camera_clients = {
            # you can optionally add camera nodes here for imitation learning purposes
            # "wrist": ZMQClientCamera(port=args.wrist_camera_port, host=args.hostname),
            # "base": ZMQClientCamera(port=args.base_camera_port, host=args.hostname),
        }
        robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
    env = RobotEnv(robot_client, control_rate_hz=args.hz, camera_dict=camera_clients)

    agent_cfg = {}
    if args.arm_to_use == "both":
        if args.agent == "gello":
            # dynamixel control box port map (to distinguish left and right gello)
            right = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAKRM7P-if00-port0"
            left = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAKROCJ-if00-port0"
            agent_cfg = {
                "_target_": "gello.agents.agent.BimanualAgent",
                "agent_left": {
                    "_target_": "gello.agents.gello_agent.GelloAgent",
                    "port": left,
                },
                "agent_right": {
                    "_target_": "gello.agents.gello_agent.GelloAgent",
                    "port": right,
                },
            }
        elif args.agent == "quest":
            agent_cfg = {
                "_target_": "gello.agents.agent.BimanualAgent",
                "agent_left": {
                    "_target_": "gello.agents.quest_agent.SingleArmQuestAgent",
                    "robot_type": args.robot_type,
                    "which_hand": "l",
                },
                "agent_right": {
                    "_target_": "gello.agents.quest_agent.SingleArmQuestAgent",
                    "robot_type": args.robot_type,
                    "which_hand": "r",
                },
            }
        elif args.agent == "spacemouse":
            left_path = "/dev/hidraw0"
            right_path = "/dev/hidraw1"
            agent_cfg = {
                "_target_": "gello.agents.agent.BimanualAgent",
                "agent_left": {
                    "_target_": "gello.agents.spacemouse_agent.SpacemouseAgent",
                    "robot_type": args.robot_type,
                    "device_path": left_path,
                    "verbose": args.verbose,
                },
                "agent_right": {
                    "_target_": "gello.agents.spacemouse_agent.SpacemouseAgent",
                    "robot_type": args.robot_type,
                    "device_path": right_path,
                    "verbose": args.verbose,
                    "invert_button": True,
                },
            }
        else:
            raise ValueError(f"Invalid agent name for bimanual: {args.agent}")

        # System setup specific. This reset configuration works well on our setup. If you are mounting the robot
        # differently, you need a separate reset joint configuration.
        #reset_joints_left = np.array([1.62448565, -0.84829138, -2.4190877, 1.40052446, -0.63813601, 1.26400017, -0.69796126, 0.0])
        #reset_joints_right = np.array([1.009359, 0.704097, 0.208621, 1.537049, 6.211088, 1.013946, 1.191903, 0.00375])
        #reset_joints = np.concatenate([reset_joints_left, reset_joints_right])
        #curr_joints = env.get_obs()["joint_positions"]
        #max_delta = (np.abs(curr_joints - reset_joints)).max()
        #steps = min(int(max_delta / 0.01), 100)

        #for jnt in np.linspace(curr_joints, reset_joints, steps):
        #    env.step(jnt)
    elif args.arm_to_use == "left":
        if args.agent == "gello":
            gello_port = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAKROCJ-if00-port0"
            agent_cfg = {
                "_target_": "gello.agents.gello_agent.GelloAgent",
                "port": gello_port,
                "start_joints": args.start_joints,
            }
    elif args.arm_to_use == "right":
        if args.agent == "gello":
            gello_port = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAKRM7P-if00-port0"
            agent_cfg = {
                "_target_": "gello.agents.gello_agent.GelloAgent",
                "port": gello_port,
                "start_joints": args.start_joints,
            }

    else:
        if args.agent == "gello":
            gello_port = args.gello_port
            if gello_port is None:
                usb_ports = glob.glob("/dev/serial/by-id/*")
                print(f"Found {len(usb_ports)} ports")
                if len(usb_ports) > 0:
                    gello_port = usb_ports[0]
                    print(f"using port {gello_port}")
                else:
                    raise ValueError(
                        "No gello port found, please specify one or plug in gello"
                    )
            agent_cfg = {
                "_target_": "gello.agents.gello_agent.GelloAgent",
                "port": gello_port,
                "start_joints": args.start_joints,
            }
            if args.start_joints is None:
                reset_joints = np.deg2rad(
                    [0, -90, 90, -90, -90, 0, 0]
                )  # Change this to your own reset joints
            else:
                reset_joints = np.array(args.start_joints)

            curr_joints = env.get_obs()["joint_positions"]
            if reset_joints.shape == curr_joints.shape:
                max_delta = (np.abs(curr_joints - reset_joints)).max()
                steps = min(int(max_delta / 0.01), 100)

                for jnt in np.linspace(curr_joints, reset_joints, steps):
                    env.step(jnt)
                    time.sleep(0.001)
        elif args.agent == "quest":
            agent_cfg = {
                "_target_": "gello.agents.quest_agent.SingleArmQuestAgent",
                "robot_type": args.robot_type,
                "which_hand": "l",
            }
        elif args.agent == "spacemouse":
            agent_cfg = {
                "_target_": "gello.agents.spacemouse_agent.SpacemouseAgent",
                "robot_type": args.robot_type,
                "verbose": args.verbose,
            }
        elif args.agent == "dummy" or args.agent == "none":
            agent_cfg = {
                "_target_": "gello.agents.agent.DummyAgent",
                "num_dofs": robot_client.num_dofs(),
            }
        elif args.agent == "policy":
            raise NotImplementedError("add your imitation policy here if there is one")
        else:
            raise ValueError("Invalid agent name")

    agent = instantiate_from_dict(agent_cfg)
    # going to start position
    print("Going to start position")
    start_pos = agent.act(env.get_obs())
    obs = env.get_obs()
    joints = obs["joint_positions"]

    abs_deltas = np.abs(start_pos - joints)
    id_max_joint_delta = np.argmax(abs_deltas)

    max_joint_delta = 0.8
    if abs_deltas[id_max_joint_delta] > max_joint_delta:
        id_mask = abs_deltas > max_joint_delta
        print()
        ids = np.arange(len(id_mask))[id_mask]
        for i, delta, joint, current_j in zip(
            ids,
            abs_deltas[id_mask],
            start_pos[id_mask],
            joints[id_mask],
        ):
            print(
                f"joint[{i}]: \t delta: {delta:4.3f} , leader: \t{joint:4.3f} , follower: \t{current_j:4.3f}"
            )
        return

    print(f"Start pos: {len(start_pos)}", f"Joints: {len(joints)}")
    assert len(start_pos) == len(
        joints
    ), f"agent output dim = {len(start_pos)}, but env dim = {len(joints)}"

    max_delta = 0.05
    for _ in range(25):
        obs = env.get_obs()
        command_joints = agent.act(obs)
        current_joints = obs["joint_positions"]
        delta = command_joints - current_joints
        max_joint_delta = np.abs(delta).max()
        if max_joint_delta > max_delta:
            delta = delta / max_joint_delta * max_delta
        env.step(current_joints + delta)

    obs = env.get_obs()
    joints = obs["joint_positions"]
    print("joints: ", joints)
    action = agent.act(obs)
    if (action - joints > 0.5).any():
        print("Action is too big")

        # print which joints are too big
        joint_index = np.where(action - joints > 0.8)
        for j in joint_index:
            print(
                f"Joint [{j}], leader: {action[j]}, follower: {joints[j]}, diff: {action[j] - joints[j]}"
            )
        exit()

    from gello.utils.control_utils import SaveInterface, run_control_loop

    dataset_controller = None
    if args.enable_dataset_recorder:
        ds_config = DatasetRecordingConfig(
            enabled=True,
            dataset_name=args.dataset_name,
            instruction=args.dataset_instruction,
            root=Path(args.dataset_root).expanduser(),
            enable_rgb=True,
            enable_depth=args.dataset_enable_depth,
            enable_audio=args.dataset_enable_audio,
            enable_torque=args.dataset_enable_torque,
            fps=args.dataset_fps,
            resolution=args.dataset_resolution,
            audio_sample_rate=48_000,
            audio_buffer_frames=args.dataset_audio_buffer_frames,
            stream_address=args.dataset_stream_address,
            stream_timeout_s=args.dataset_stream_timeout,
            realsense_device_id=args.dataset_camera_device_id,
            realsense_flip=args.dataset_camera_flip,
            audio_device_name=args.dataset_audio_device_name,
            audio_device_index=args.dataset_audio_device_index,
            audio_backend=args.dataset_audio_backend,
            audio_alsa_device=args.dataset_audio_alsa_device,
            dataset_registry_key=args.dataset_registry_key,
            gripper_control_mode=args.gripper_control_mode,
        )
        dataset_controller = TeleopDatasetController(ds_config, args.arm_to_use)

    save_interface = None
    if args.use_save_interface and not args.enable_dataset_recorder:
        save_interface = SaveInterface(
            data_dir=args.data_dir, agent_name=args.agent, expand_user=True
        )

    run_control_loop(
        env,
        agent,
        save_interface,
        use_colors=True,
        dataset_controller=dataset_controller,
        action_transform=_build_gripper_transform(args.gripper_control_mode, args.arm_to_use),
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
