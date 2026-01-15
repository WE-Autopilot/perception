#!/usr/bin/env python3
"""
ufld_visualizer.py

ROS2 node that visualizes the camera stream and confirms receipt of
ONNX UFLD inference output.

NOTE:
This visualizer does NOT attempt to decode lane geometry, because the
exported ONNX model output format does not match the classic UFLD
(grid-based) decoder assumptions.

This node is intended for debugging and pipeline validation.
"""

import json
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge


class UFLDVisualizer(Node):
    def __init__(self):
        super().__init__("ufld_visualizer")

        self.bridge = CvBridge()

        # ------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("output_topic", "/ufld/raw_output")
        self.declare_parameter("debug_topic", "/ufld/debug_image")

        image_topic = self.get_parameter("image_topic").value
        output_topic = self.get_parameter("output_topic").value
        debug_topic = self.get_parameter("debug_topic").value

        # ------------------------------------------------------------
        # ROS interfaces
        # ------------------------------------------------------------
        self.sub_image = self.create_subscription(
            Image, image_topic, self.image_callback, 10
        )

        self.sub_output = self.create_subscription(
            String, output_topic, self.output_callback, 10
        )

        self.pub_debug = self.create_publisher(
            Image, debug_topic, 10
        )

        # ------------------------------------------------------------
        # Internal state
        # ------------------------------------------------------------
        self.last_output_size = None
        self.logged_once = False

        self.get_logger().info("UFLD visualizer started (pass-through mode).")

    # ------------------------------------------------------------
    # Output callback
    # ------------------------------------------------------------
    def output_callback(self, msg: String):
        try:
            payload = json.loads(msg.data)
            tensors = payload.get("outputs", [])

            if not tensors:
                return

            data = np.array(tensors[0], dtype=np.float32)
            self.last_output_size = data.size

            if not self.logged_once:
                self.get_logger().info(
                    f"Received ONNX output tensor with {data.size} values "
                    "(decoder not applied)."
                )
                self.logged_once = True

        except Exception as e:
            self.get_logger().warn(f"Failed to parse ONNX output: {e}")

    # ------------------------------------------------------------
    # Image callback
    # ------------------------------------------------------------
    def image_callback(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # Overlay debug text if ONNX output is present
        if self.last_output_size is not None:
            cv2.putText(
                frame,
                f"ONNX output received ({self.last_output_size} values)",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )
        else:
            cv2.putText(
                frame,
                "Waiting for ONNX output...",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
            )

        self.pub_debug.publish(
            self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        )


def main(args=None):
    rclpy.init(args=args)
    node = UFLDVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
