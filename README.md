# APOLLO Bimanual xArm7 Teleop with Gello

## Setup

1. Clone the repo and submodules
    ```bash
    git clone https://github.com/M4D-SC1ENTIST/apollo-bimanual-xarm7-gello-teleop.git

    cd apollo-bimanual-xarm7-gello-teleop

    git submodule update --init --recursive
    ```

2. Create and activate a virtual environment. If you don't have [uv](https://docs.astral.sh/uv/) installed, you need to first install it.
    ```bash
    uv venv --python 3.11
    source .venv/bin/activate  # Run this every time you open a new shell

    cd gello_software/
    uv pip install -r requirements.txt
    uv pip install -e .
    uv pip install -e third_party/DynamixelSDK/python

    cd ..
    uv pip install -e multimodal-lerobot-dataset
    ```

## Usage

### Launch teleop control

Run the orchestrator script to bring up the Gello teleop agent and the xArm7 server:

```bash
python launch_teleop.py --arm-to-use <left|right|both> --viewpoint-option <coffee|active|none>
```

Example invocations:

```bash
python launch_teleop.py --arm-to-use left --viewpoint-option none
python launch_teleop.py --arm-to-use right --viewpoint-option none
python launch_teleop.py --arm-to-use both --viewpoint-option coffee
```

### Add RealSense perception viewer

To visualize the D435i RGB/depth feed while teleoperating, enable the viewer process:

```bash
python launch_teleop.py \
    --arm-to-use right \
    --viewpoint-option none \
    --enable-camera-viewer \
    --realsense-device-id <SERIAL> \
    --viewer-backend pygame  # default backend
```

Key viewer options (all optional):

- `--realsense-device-id`: serial number if multiple cameras are connected.
- `--realsense-depth-scale`: meters per depth unit (default `0.001`).
- `--realsense-max-depth`: clamp depth visualization range (meters).
- `--realsense-flip`: rotate the stream 180° if the camera is mounted upside-down.
- `--viewer-hide-depth`: show RGB only.
- `--viewer-fullscreen`: fullscreen display for control rooms.
- `--viewer-backend`: `pygame` (default) or `opencv` if GTK/Qt is available.

Omit `--enable-camera-viewer` if you only need arm teleoperation.

## Troubleshoot
- If during installing `multimodal-lerobot-dataset`, there is error on installing `pyaudio`, please run
    ```
    sudo apt install portaudio19-dev
    ```
