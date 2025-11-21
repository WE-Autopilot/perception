<p align="center"> <img src="assets/red.jpeg" alt="Project Banner" width="100%"> </p> <h1 align="center">Perception ONNX – ROS2 UFLD Lane Detection</h1> <p align="center"> <strong>ONNX-powered lane detection inference inside ROS2 Jazzy</strong><br> UTM • ARM64 • UFLD • Computer Vision • Autonomous Vehicles </p>

🚀 Overview

This repository integrates the Ultra-Fast Lane Detection (UFLD) ONNX model into a ROS2 Jazzy environment running on ARM64 (UTM). It includes:

A ROS2 Python package (perception_onnx)

An ONNX inference node

A test image publisher node

A dataset extraction tool

A clean, modular ROS2 workspace layout

This project demonstrates how to deploy ONNX models inside ROS2 with high performance and modular design for autonomous vehicle perception.

🧱 Project Structure

perception_ws/
├── src/
│ └── perception_onnx/
│ ├── nodes/
│ │ ├── image_publisher.py
│ │ └── ufld_onnx_node.py
│ ├── models/
│ ├── resource/perception_onnx
│ ├── setup.py
│ ├── package.xml
│ └── test/
├── sample_images/
│ ├── archive.zip
│ └── extracted/
│ ├── img_0.png … img_14.png
├── install/
├── build/
└── log/

🛠️ Installation

Clone into ROS2 workspace

mkdir -p ~/perception_ws/src
cd ~/perception_ws/src
git clone https://github.com/YOUR_USERNAME/perception_onnx.git


Create + activate virtual environment

python3 -m venv ~/ros_venv
source ~/ros_venv/bin/activate
pip install --upgrade pip
pip install opencv-python onnxruntime numpy cv_bridge


Build the ROS2 package

cd ~/perception_ws
colcon build --packages-select perception_onnx
source install/setup.bash


📸 Extract sample images (optional)

If your dataset is huge, extract only 5 images:

python3 sample_images/extract_five.py


Images will appear in:

sample_images/extracted/


🔄 Running the Nodes

Publish test images

ros2 run perception_onnx image_publisher


Run the ONNX UFLD inference node

ros2 run perception_onnx ufld_onnx_node


📦 ONNX Model Location

Place your ufld.onnx model here:

perception_ws/src/perception_onnx/models/ufld.onnx


🧭 Roadmap

✔️ Workspace + package created
✔️ Image publisher working
✔️ Dataset extractor script
⬜ ONNX UFLD inference implementation
⬜ Lane overlay visualization
⬜ Integration with WE-Autopilot full perception stack

🏷️ Badges

<p align="center"> <img src="https://img.shields.io/badge/ROS2-Jazzy-purple?style=for-the-badge"> <img src="https://img.shields.io/badge/ONNX-Model-blue?style=for-the-badge"> <img src="https://img.shields.io/badge/UTM-ARM64-green?style=for-the-badge"> <img src="https://img.shields.io/badge/Python-3.12-yellow?style=for-the-badge"> <img src="https://img.shields.io/badge/License-MIT-red?style=for-the-badge"> </p>

📄 License

MIT License © 2025