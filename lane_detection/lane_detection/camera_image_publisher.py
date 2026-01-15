#!/usr/bin/env python3
"""
camera_image_publisher.py

ROS2 node that publishes test images to /camera/image_raw.

Designed for offline testing of perception pipelines.
"""

import os
import glob
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class CameraImagePublisher(Node):
    def __init__(self):
        super().__init__("camera_image_publisher")

        self.declare_parameter("image_folder", "")
        self.declare_parameter("publish_rate", 1.0)

        image_folder = self.get_parameter("image_folder").value
        publish_rate = float(self.get_parameter("publish_rate").value)

        self.publisher = self.create_publisher(Image, "/camera/image_raw", 10)
        self.bridge = CvBridge()
        self.images = []
        self.index = 0

        if not image_folder:
            self.get_logger().warn(
                "No image_folder provided. Node will remain idle."
            )
            return

        if not os.path.isdir(image_folder):
            raise RuntimeError(f"Directory does not exist: {image_folder}")

        for ext in ("*.png", "*.jpg", "*.jpeg"):
            self.images.extend(glob.glob(os.path.join(image_folder, ext)))

        self.images = sorted(self.images)

        if not self.images:
            raise RuntimeError(f"No images found in {image_folder}")

        self.get_logger().info(
            f"Loaded {len(self.images)} images from {image_folder}"
        )

        self.timer = self.create_timer(publish_rate, self.publish_image)

    def publish_image(self):
        img_path = self.images[self.index]
        img = cv2.imread(img_path)

        if img is None:
            self.get_logger().warn(f"Failed to read image: {img_path}")
            return

        msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera"

        self.publisher.publish(msg)

        self.get_logger().info(
            f"Published image [{self.index + 1}/{len(self.images)}]: {img_path}"
        )

        # 🔁 LOOP forever
        self.index = (self.index + 1) % len(self.images)


def main(args=None):
    rclpy.init(args=args)
    node = CameraImagePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
