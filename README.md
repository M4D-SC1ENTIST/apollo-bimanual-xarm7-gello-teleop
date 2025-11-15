# APOLLO Bimanual xArm7 Teleop with Gello

## Setup

1. Clone the repo and submodules
    ```bash
    git clone https://github.com/M4D-SC1ENTIST/apollo-bimanual-xarm7-gello-teleop.git

    cd apollo-bimanual-xarm7-gello-teleop
    ```

2. Create and activate a virtual environment. If you don't have [uv](https://docs.astral.sh/uv/) installed, you need to first install it.
    ```bash
    uv venv --python 3.11
    source .venv/bin/activate  # Run this every time you open a new shell

    cd gello_software/
    uv pip install -r requirements.txt
    uv pip install -e .
    uv pip install -e third_party/DynamixelSDK/python
    ```

## Usage

- Run teleop
    ```
    python launch_teleop.py --arm-to-use <arm_to_use>
    ```
    for example
    ```
    python launch_teleop.py --arm-to-use left --viewpoint-option none
    python launch_teleop.py --arm-to-use right --viewpoint-option none
    python launch_teleop.py --arm-to-use both --viewpoint-option coffee
    ```
