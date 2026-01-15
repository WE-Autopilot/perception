#!/usr/bin/env python3
"""
ufld_onnx_node.py

ROS2 node that runs Ultra-Fast Lane Detection (UFLD) using an ONNX model.

This node:
    - Subscribes to a camera image topic
    - Runs ONNX inference via UFldOnnx
    - Publishes raw ONNX outputs as JSON

This satisfies:
    Team 2 – Package ONNX model (ROS node that publishes a ROS type)
"""

import json

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory

from lane_detection.ufld_onnx_inference import UFldOnnx


class UFldOnnxNode(Node):
    """
    ROS2 node wrapping UFldOnnx inference.
    """

    def __init__(self):
        super().__init__("ufld_onnx_node")

        # ------------------------------------------------------------
        # Declare parameters
        # ------------------------------------------------------------
        self.declare_parameter("onnx_path", "")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("output_topic", "/ufld/raw_output")
        self.declare_parameter("input_height", 320)
        self.declare_parameter("input_width", 1600)

        image_topic = self.get_parameter("image_topic").value
        output_topic = self.get_parameter("output_topic").value
        input_height = int(self.get_parameter("input_height").value)
        input_width = int(self.get_parameter("input_width").value)
        onnx_path_param = self.get_parameter("onnx_path").value

        # ------------------------------------------------------------
        # Resolve ONNX model path (ROS-correct way)
        # ------------------------------------------------------------
        if onnx_path_param:
            onnx_path = onnx_path_param
        else:
            pkg_share = get_package_share_directory("lane_detection")
            onnx_path = f"{pkg_share}/models/ufld_resnet34.onnx"

        # ------------------------------------------------------------
        # Initialize inference engine
        # ------------------------------------------------------------
        self.ufld = UFldOnnx(
            engine_path=onnx_path,
            config_path=None,
            input_size=(input_height, input_width),
            providers=["CPUExecutionProvider"],
            use_imagenet_norm=True,
        )

        # ------------------------------------------------------------
        # ROS I/O
        # ------------------------------------------------------------
        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            10,
        )

        self.publisher = self.create_publisher(
            String,
            output_topic,
            10,
        )

        # ------------------------------------------------------------
        # Log configuration once
        # ------------------------------------------------------------
        self.get_logger().info("UFLD ONNX node initialized")
        self.get_logger().info(f"ONNX model path: {onnx_path}")
        self.get_logger().info(f"Subscribed to: {image_topic}")
        self.get_logger().info(f"Publishing to: {output_topic}")
        self.get_logger().info(
            f"Input size: (H={input_height}, W={input_width})"
        )

    # ------------------------------------------------------------
    # Image callback
    # ------------------------------------------------------------
    def image_callback(self, msg: Image):
        # Convert ROS Image -> OpenCV BGR
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # Run inference
        outputs = self.ufld.infer_raw(frame)

        # Serialize outputs to JSON
        output_json = json.dumps(
            {"outputs": [o.tolist() for o in outputs]}
        )

        out_msg = String()
        out_msg.data = output_json

        self.publisher.publish(out_msg)

        self.get_logger().debug(
            f"Published UFLD output ({len(outputs)} tensors)"
        )


def main(args=None):
    rclpy.init(args=args)
    node = UFldOnnxNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
