# Setup Instructions for `mapping_interface` ROS 2 Package

This guide helps collaborators set up their environment for the `mapping_interface` ROS 2 package, including the external `RANSAC` package.

---

## Prerequisites

Before you begin, make sure you have:

- ROS 2 Jazzy installed
- Python 3.12 installed
- Python virtual environment support:

```bash
sudo apt update
sudo apt install python3-venv python3-pip -y
```

---

## Virtual Environment

Create a virtual environment:

```bash
cd ~/perception/ros2
python3 -m venv ./venv
```

Activate the virtual environment:

```bash
source ~/perception/ros2/venv/bin/activate
```

Upgrade pip and build tools:

```bash
pip install --upgrade pip setuptools wheel
```

## Install RANSAC Package

Navigate to the RANSAC directory and install:

```bash
cd ~/perception/RANSAC
pip install -e .
```

<!-- Verify installation:

```bash
python3 -c "from RANSAC.plane_utils import estimate; print('OK')"
``` -->

## Build ROS2 Workspace

Build the workspace:

```bash
cd ~/perception/ros2
source /opt/ros/jazzy/setup.bash
colcon build
source install/setup.bash
```

## Run Ground Plane Node

If not already activated, activate the virtual environment:

```bash
source ~/perception/ros2/venv/bin/activate
```

Source ROS2 and workspace setup:

```bash
source /opt/ros/jazzy/setup.bash
source ~/perception/ros2/install/setup.bash
```

Run the node:

```bash
ros2 run mapping_interface groundPlaneNode
```
