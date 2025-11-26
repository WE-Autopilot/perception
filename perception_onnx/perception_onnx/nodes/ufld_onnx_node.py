#!/usr/bin/env python3
"""
ufld_onnx_inference_node.py

ROS2 node that runs Ultra-Fast Lane Detection (UFLD) using an ONNX model.

It:
    - Subscribes to a camera image topic
    - Uses UFldOnnx to run inference
    - Publishes raw ONNX outputs as JSON on a ROS topic

This node is designed as a drop-in "Team 2: Package ONNX model" component
that Mapping & Localization can build on top of.
"""

import json
from pathlib import Path

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

from perception_onnx.ufld_onnx_inference import UFldOnnx


class UFLDOnnxInferenceNode(Node):
    """
    ROS2 node that wraps UFldOnnx and exposes it as a ROS2 pipeline.
    """

    def __init__(self):
        super().__init__("ufld_onnx_inference_node")

        # ------------------------------------------------------------------
        # Declare parameters
        #
        # onnx_path    : Path to the ONNX model. If empty, a default is used.
        # image_topic  : Input image topic.
        # output_topic : Output topic for model results.
        # input_height : Model input height (default 320).
        # input_width  : Model input width (default 1600).
        # ------------------------------------------------------------------
        self.declare_parameter("onnx_path", "")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("output_topic", "/ufld/raw_output")
        self.declare_parameter("input_height", 320)
        self.declare_parameter("input_width", 1600)

        onnx_path_param = self.get_parameter("onnx_path").value
        image_topic = self.get_parameter("image_topic").value
        output_topic = self.get_parameter("output_topic").value
        input_height = int(self.get_parameter("input_height").value)
        input_width = int(self.get_parameter("input_width").value)

        # ------------------------------------------------------------------
        # Resolve default ONNX path if none is provided
        #
        # We assume the model is stored in this package at:
        #   <package_root>/models/culane_res34.onnx
        # This is a repo-relative, non-user-specific path, safe for GitHub.
        # ------------------------------------------------------------------
        if onnx_path_param:
            onnx_path = Path(onnx_path_param)
        else:
            # __file__ = nodes/ufld_onnx_inference_node.py
            # parents[1] = perception_onnx (package root)
            package_root = Path(__file__).resolve().parents[1]
            onnx_path = package_root / "models" / "culane_res34.onnx"

        if not onnx_path.is_file():
            self.get_logger().error(
                f"ONNX model not found at: {onnx_path}\n"
                "Please provide a valid path via the 'onnx_path' parameter."
            )
            raise SystemExit(1)

        # ------------------------------------------------------------------
        # Initialize UFLD ONNX inference engine
        # ------------------------------------------------------------------
        self.ufld = UFldOnnx(
            engine_path=onnx_path,
            config_path=None,  # kept for API parity
            input_size=(input_height, input_width),
            providers=["CPUExecutionProvider"],  # change if GPU available
            use_imagenet_norm=True,
        )

        # ROS utilities
        self.bridge = CvBridge()

        # Subscriber to camera images
        self.subscription = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            10,
        )

        # Publisher for raw JSON output
        self.publisher = self.create_publisher(
            String,
            output_topic,
            10,
        )

        # Log configuration
        self.get_logger().info(f"Using ONNX model: {onnx_path}")
        self.get_logger().info(f"Subscribed to image topic: {image_topic}")
        self.get_logger().info(f"Publishing UFLD outputs on: {output_topic}")
        self.get_logger().info(
            f"Model input size: (H={input_height}, W={input_width})"
        )

    # ----------------------------------------------------------------------
    # IMAGE CALLBACK
    # ----------------------------------------------------------------------
    def image_callback(self, msg: Image):
        """
        Called each time an image is received.

        Steps:
            1. Convert ROS Image -> OpenCV BGR image
            2. Run UFLD ONNX inference
            3. Convert outputs to JSON
            4. Publish on output topic
        """
        # Convert ROS Image message to OpenCV BGR image
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # Run model inference and get raw outputs
        outputs = self.ufld.infer_raw(frame)

        # Convert numpy arrays -> Python lists -> JSON
        output_json = json.dumps(
            {"outputs": [o.tolist() for o in outputs]}
        )

        # Wrap JSON in a String ROS message
        out_msg = String()
        out_msg.data = output_json

        # Publish
        self.publisher.publish(out_msg)

        self.get_logger().info(
            f"Published UFLD ONNX output with {len(outputs)} tensors"
        )


def main(args=None):
    rclpy.init(args=args)
    node = UFLDOnnxInferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
