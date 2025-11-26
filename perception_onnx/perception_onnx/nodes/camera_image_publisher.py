#!/usr/bin/env python3
"""
camera_image_publisher.py

ROS2 Node that publishes test images to /camera/image_raw.

This node is used to simulate a camera feed for debugging and
testing the ONNX Ultra-Fast Lane Detection (UFLD) inference node.

Features:
    - Loads all images (PNG/JPG/JPEG) from a given directory
    - Publishes one image per second
    - Automatically stops after all images are published
    - File path is configurable via ROS2 parameter:
          image_folder:=/path/to/images
"""

import os
import glob
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class CameraImagePublisher(Node):
    """
    A ROS2 Node that publishes images at a fixed rate.

    This node:
        - Reads PNG/JPG images from a directory
        - Publishes one image every N seconds
        - Outputs to topic `/camera/image_raw`
        - Logs each published image
    """

    def __init__(self):
        # -------------------------------------------
        # Initialize ROS2 node with name: camera_image_publisher
        # -------------------------------------------
        super().__init__("camera_image_publisher")

        # -------------------------------------------
        # Declare parameters (configurable via launch file or CLI)
        #
        # image_folder : directory containing .jpg/.png images
        # publish_rate : seconds between publishing images
        # -------------------------------------------
        self.declare_parameter("image_folder", "")
        self.declare_parameter("publish_rate", 1.0)

        # Retrieve parameter values
        image_folder = self.get_parameter("image_folder").value
        publish_rate = float(self.get_parameter("publish_rate").value)

        # -------------------------------------------
        # Validate image folder parameter
        # -------------------------------------------
        if not image_folder:
            self.get_logger().error(
                "Parameter 'image_folder' is not set.\n"
                "Usage example:\n"
                "  ros2 run perception_onnx camera_image_publisher "
                "--ros-args -p image_folder:=/path/to/images"
            )
            raise SystemExit(1)

        if not os.path.isdir(image_folder):
            self.get_logger().error(f"Directory does not exist: {image_folder}")
            raise SystemExit(1)

        # -------------------------------------------
        # Load all image files (JPG, PNG, JPEG)
        #
        # glob.glob finds files matching path patterns
        # -------------------------------------------
        image_extensions = ["*.png", "*.jpg", "*.jpeg"]
        self.images = []

        for ext in image_extensions:
            self.images.extend(glob.glob(os.path.join(image_folder, ext)))

        # Sort ensures images are published in consistent order
        self.images = sorted(self.images)

        # -------------------------------------------
        # Ensure at least one image exists
        # -------------------------------------------
        if not self.images:
            self.get_logger().error(
                f"No images found inside folder: {image_folder}\n"
                "Supported formats: .png, .jpg, .jpeg"
            )
            raise SystemExit(1)

        # Log how many images were found
        self.get_logger().info(f"Loaded {len(self.images)} images from: {image_folder}")

        # -------------------------------------------
        # Create ROS publisher
        #
        # Publishes sensor_msgs/Image messages
        # -------------------------------------------
        self.publisher = self.create_publisher(Image, "/camera/image_raw", 10)

        # CvBridge converts OpenCV images <-> ROS Image messages
        self.bridge = CvBridge()

        # Keeps track of which image we are publishing
        self.index = 0

        # -------------------------------------------
        # Timer to call publish_image() repeatedly
        #
        # Runs every <publish_rate> seconds
        # -------------------------------------------
        self.timer = self.create_timer(publish_rate, self.publish_image)

    # ============================================================
    # PUBLISH IMAGE CALLBACK
    # Called automatically by the timer every <publish_rate> seconds
    # ============================================================
    def publish_image(self):
        # Stop if we reached the last image
        if self.index >= len(self.images):
            self.get_logger().info("All images have been published. Stopping timer.")
            self.timer.cancel()
            return

        # Get the file path of the next image
        img_path = self.images[self.index]

        # Read image using OpenCV
        img = cv2.imread(img_path)

        # If OpenCV fails to read the file
        if img is None:
            self.get_logger().warn(f"Could not read image: {img_path}")
            self.index += 1
            return

        # Convert OpenCV BGR image -> ROS Image message
        msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")

        # Publish the message on /camera/image_raw
        self.publisher.publish(msg)

        # Log successful publish
        self.get_logger().info(f"Published image: {img_path}")

        # Move to next image
        self.index += 1


# ============================================================
# MAIN FUNCTION
# Standard ROS2 startup + shutdown boilerplate
# ============================================================
def main(args=None):
    rclpy.init(args=args)             # initialize ROS2 system
    node = CameraImagePublisher()     # create node instance
    rclpy.spin(node)                  # keep node alive until shutdown
    node.destroy_node()               # cleanup
    rclpy.shutdown()                  # shutdown ROS2


if __name__ == "__main__":
    main()