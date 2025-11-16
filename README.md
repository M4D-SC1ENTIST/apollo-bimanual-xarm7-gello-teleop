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

### Record multimodal datasets

Use the multimodal-lerobot recorder to log synchronized RGB/depth, proprioception, and optional audio/torque:

```bash
python launch_teleop.py \
    --arm-to-use right \
    --viewpoint-option coffee \
    --enable-dataset-recorder \
    --dataset-name coffee \
    --dataset-instruction "put the coffee pod into the coffee maker" \
    --dataset-enable-depth \
    --dataset-enable-audio \
    --dataset-enable-torque
```

Key details:

- Recordings are stored under `datasets/<dataset_name>/episode_<idx>/`. This directory is git-ignored.
- If `--dataset-name` is omitted, the recorder auto-creates `untitled_dataset_<n>` the first time you press `S`.
- Press `S` (start/save) to begin logging an episode, `Q` to stop. Press `S` again to start the next episode; all episodes append to the same dataset.
- `--dataset-instruction` populates both `tasks` and `instruction` fields in the metadata. Restart `launch_teleop.py` with the same dataset name but a different instruction to mix tasks within one dataset.
- Depth, audio, and torque logging are optional (`--dataset-enable-depth`, `--dataset-enable-audio`, `--dataset-enable-torque`). RGB + joint state/action streams are always captured at 12 Hz, audio at 48 kHz mono.
- For microphones plugged in via ALSA/PortAudio, point the recorder to the correct device using `--dataset-audio-device-name "RØDE NT-USB Mini"` (substring match) or `--dataset-audio-device-index <idx>`; you can also force a specific ALSA PCM via `--dataset-audio-alsa-device plughw:1,0`.
- The recorder defaults to the PyAudio backend but can be forced (or will automatically fall back) to `arecord` by passing `--dataset-audio-backend arecord` if PortAudio has trouble configuring the device.
- The dataset metadata lives in `datasets/<dataset_name>/meta/info.json` and follows the schema consumed by the provided `dataset_handler_for_training.py`.
- The recorder consumes the RealSense viewer feed. `--enable-dataset-recorder` automatically launches the viewer with a ZeroMQ stream so you still get live perception while saving demonstrations.

## Troubleshoot
- If during installing `multimodal-lerobot-dataset`, there is error on installing `pyaudio`, please run
    ```
    sudo apt install portaudio19-dev
    ```
