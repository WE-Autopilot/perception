# UFLD ONNX Lane Detection (ROS2 Jazzy)

This package integrates **Ultra-Fast Lane Detection (UFLD)** into the WE-Autopilot perception stack using **ONNX Runtime**.  
It replaces the original TensorRT backend with a clean, portable ONNX inference pipeline.

---

## 🚗 Overview

This package provides:

- A ROS2 node (`ufld_onnx_node`) that:
  - Loads an ONNX lane-detection model  
  - Subscribes to `/camera/image_raw`
  - Runs preprocessing + ONNX inference  
  - Publishes model outputs on `/ufld/raw_output`

- An image publisher (`image_publisher`) for testing with offline images.

---

## 📦 Package Structure

```
perception_onnx/
│
├── models/
│   └── culane_res34.onnx           # UFLD ONNX model (tracked via Git LFS)
│
├── perception_onnx/
│   ├── ufld_onnx_inference.py      # ONNX preprocessing + inference class
│   ├── postprocessing.py           # (optional future decoding logic)
│   └── nodes/
│       ├── ufld_onnx_node.py       # Main ROS2 node
│       └── image_publisher.py      # Test image publisher
│
├── setup.py
├── setup.cfg
├── package.xml
└── README.md
```

---

## 🧰 Dependencies

### Install ONNX Runtime + OpenCV

Use a virtual environment to avoid numpy version conflicts:

```bash
python3 -m venv onnx_venv
source onnx_venv/bin/activate

pip install onnxruntime opencv-python numpy pyyaml
```

### ROS2 Dependencies

Installed automatically with:

```bash
sudo apt install ros-jazzy-cv-bridge ros-jazzy-image-transport
```

---

## 🔧 Build Instructions (ROS2 Workspace)

Assuming your workspace is:

```
~/perception_ws/
```

Build the package:

```bash
cd ~/perception_ws
colcon build --symlink-install
source install/setup.bash
```

---

## ▶️ Running the Lane Detection Node

### 1. Activate your ONNX virtual environment

```bash
source onnx_venv/bin/activate
```

### 2. Source ROS2 workspace

```bash
source ~/perception_ws/install/setup.bash
```

### 3. Run the ONNX node

```bash
ros2 run perception_onnx ufld_onnx_node
```

You should see:

```
[INFO] Using ONNX model: .../models/culane_res34.onnx
[INFO] Subscribed to /camera/image_raw
[INFO] Publishing UFLD outputs on /ufld/raw_output
```

---

## 📤 Running with Offline Test Images

Use the image publisher:

```bash
ros2 run perception_onnx image_publisher --ros-args -p image_folder:=/path/to/images
```

This publishes images to:

```
/camera/image_raw
```

The ONNX node will automatically process them.

---

## 📡 ROS2 Topics

| Topic                  | Type                     | Description                        |
|------------------------|--------------------------|------------------------------------|
| `/camera/image_raw`    | sensor_msgs/Image        | Input images                       |
| `/ufld/raw_output`     | Float32MultiArray*       | Raw model output tensors           |

(*) A custom lane message will be added in the future.

---

## 🔄 Pipeline Diagram

```
+---------------------+      +------------------------+
|  Image Publisher    | ---> |  ufld_onnx_node        |
| (/camera/image_raw) |      |  - Preprocess          |
|                     |      |  - ONNX Inference      |
+---------------------+      |  - Publish Outputs     |
                             +-----------+------------+
                                         |
                                         v
                                /ufld/raw_output
```

---

## 🛠 Future Work

- Implement full lane decoding (polylines)
- Publish a custom `LaneArray` ROS2 message
- Integrate into perception stack visualization

---

## 👤 Author

Implemented by **Zayan Khan**  
WE-Autopilot Perception Team

---

## ✔ License

This package follows the main repository license.
