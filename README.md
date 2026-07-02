# block_piper

Vision-guided block picking and stacking with an AgileX Piper arm, Intel RealSense depth camera, and a YOLO OBB detector.

## What It Does

This project detects blocks on a desktop, estimates their position and angle in the robot base frame, plans a layered block structure, and commands the Piper arm to pick and place each block.

Core pipeline:

1. `detection_system.py` captures RealSense color/depth frames and runs YOLO detection.
2. `task_scheduler.py` defines block types, build order, placement positions, and layer strategy.
3. `command_executor.py` converts planned tasks into Piper arm and gripper commands.
4. `main.py` coordinates camera observation, target selection, pose refinement, grasping, and placement.

## Project Layout

```text
.
|-- main.py                 # Main pick-and-place build loop
|-- detection_system.py     # RealSense + YOLO detection and coordinate conversion
|-- task_scheduler.py       # Block definitions, build plan, placement strategy
|-- command_executor.py     # Piper motion and gripper execution
|-- visual_block.py         # Optional build-plan visualization helpers
|-- config/
|   `-- camera.yaml         # Camera, YOLO, and hand-eye parameters
|-- activate_hardware.sh    # One-command hardware activation and checks
|-- enable_can.sh           # Minimal CAN interface helper for Linux
`-- README.md
```

YOLO weights are intentionally not tracked. Put the trained model at the path configured in `config/camera.yaml`, for example:

```text
weights/best_1210_2.pt
```

## Requirements

Hardware:

- AgileX Piper arm
- USB-CAN adapter
- Intel RealSense depth camera

Python/runtime dependencies:

- `piper_sdk`
- `pyrealsense2`
- `ultralytics`
- `opencv-python`
- `numpy`
- `scipy`
- `pyyaml`
- `matplotlib`

Install the Piper SDK from the official repository and make sure it is importable by Python:

```bash
git clone https://github.com/agilexrobotics/piper_sdk.git
```

## Configuration

Edit `config/camera.yaml` before running:

- `detection.model_path`: path to the YOLO model file
- `detection.confidence_threshold`: detector confidence threshold
- `camera_extrinsics.matrix`: hand-eye calibration transform
- `camera_streams`: RealSense color/depth stream settings

## Run

Activate hardware and run checks:

```bash
bash activate_hardware.sh
```

Activate hardware and start the demo:

```bash
bash activate_hardware.sh --run
```

Useful options:

```bash
bash activate_hardware.sh --can can1
bash activate_hardware.sh --conda-env piper --run
bash activate_hardware.sh --check-only
bash activate_hardware.sh --viewer
```

If you only need to enable the CAN interface:

```bash
sudo bash enable_can.sh
```

Then run the main program manually:

```bash
python main.py
```

## Warmup Check

Before the full demo, you can send the Piper arm to the zero pose and show the RealSense color image:

```bash
python warmup_check.py
```

Useful options:

```bash
python warmup_check.py --can can1
python warmup_check.py --skip-arm
python warmup_check.py --skip-camera
python warmup_check.py --seconds 10
```

## Notes

- The main program expects the Piper arm and RealSense camera to be connected.
- The repository keeps only source code and configuration. Generated images, numpy calibration outputs, caches, and model weights are ignored.
- Review all workspace positions and gripper parameters before running on real hardware.
