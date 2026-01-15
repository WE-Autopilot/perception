# Lane Detection — ONNX Runtime (ROS2)

This package provides a **ROS2-compatible Ultra-Fast Lane Detection (UFLD) pipeline**
using **ONNX Runtime**. It replaces the legacy TensorRT-based implementation and is
designed to run **portably on any machine** without GPU or TensorRT dependencies.

The pipeline supports:
- Offline image-based testing
- ONNX model inference
- ROS2 topic-based data flow
- Visualization via RViz2

---

## Package Overview

The pipeline consists of **three ROS2 nodes**:

| Node | Description |
|-----|------------|
| `camera_image_publisher` | Publishes images from a local folder as `/camera/image_raw` |
| `ufld_onnx_node` | Runs ONNX Runtime inference and publishes raw model outputs |
| `ufld_visualizer` | Visualizes camera input and (optionally) lane annotations |

---

## Prerequisites

- ROS2 (Humble or compatible)
- Python 3
- `onnxruntime`
- `opencv-python`
- `cv_bridge`

Build the workspace:

```bash
cd ~/perception_ws
colcon build --symlink-install
source install/setup.bash

Running the Pipeline

You must run the nodes in the following order, each in a separate terminal.

Terminal 1 — Camera Image Publisher

This node simulates a camera feed by publishing images from a local directory.

⚠️ Images are not included in the repository and must remain local.
Supported formats: .png, .jpg, .jpeg.

source ~/perception_ws/install/setup.bash
ros2 run lane_detection camera_image_publisher \
  --ros-args -p image_folder:=/absolute/path/to/your/images

Verify publishing
ros2 topic hz /camera/image_raw

Terminal 2 — ONNX UFLD Inference Node

This node:

Loads the ONNX UFLD model

Applies preprocessing

Runs inference

Publishes raw ONNX outputs

source ~/perception_ws/install/setup.bash
ros2 run lane_detection ufld_onnx_node

Terminal 3 — Visualization Node

Subscribes to:

/camera/image_raw

/ufld/raw_output

Publishes:

/ufld/debug_image

source ~/perception_ws/install/setup.bash
ros2 run lane_detection ufld_visualizer

Terminal 4 — RViz2
rviz2

RViz Configuration

Add → Image

Topic: /ufld/debug_image

Reliability Policy: Best Effort

History Policy: Keep Last

Expected Topics
ros2 topic list

/camera/image_raw
/ufld/raw_output
/ufld/debug_image

Lane Annotation Status (Important)

Lane annotations are intentionally disabled for the provided ONNX model.

Why

ONNX output size: 57600

Classic UFLD grid expects: 4 × 101 × 20 = 8080

Output semantics do not match grid-based decoding

Applying classic decoding would be incorrect.

Current State

✔ ONNX inference works

✔ ROS2 pipeline works

✔ Visualization works

⚠ Lane decoding pending correct output specification

Design Notes

ONNX Runtime used for portability

Input shape (1 × 3 × H × W) preserved

Raw outputs published for downstream use

Decoding deferred by design

Summary

This package delivers a clean ONNX-based UFLD pipeline for ROS2 with correct inference
and visualization. Lane decoding is intentionally deferred until model output
semantics are confirmed.
