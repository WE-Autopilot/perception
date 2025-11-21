#!/usr/bin/env python3
import os, sys, json
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import onnxruntime as ort

# Add your perception repo to Python path
PERCEPTION_PATH = os.path.expanduser("~/perception_ws/src/perception")
sys.path.append(PERCEPTION_PATH)

# TODO: CHANGE THIS import to the real file/class of UFLD
# Example: from lane_detection.ufld import UFLD
# For now, placeholder. You'll update later.
# from <your_UFLD_file> import UFLD


class UFLDOnnxNode(Node):
    def __init__(self):
        super().__init__("ufld_onnx_node")

        # Parameters
        self.declare_parameter("onnx_path", f"{PERCEPTION_PATH}/weights/ufld.onnx")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("output_topic", "/perception/lanes")

        onnx_path = self.get_parameter("onnx_path").value
        img_topic = self.get_parameter("image_topic").value
        out_topic = self.get_parameter("output_topic").value

        # ONNX Runtime session
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.output_names = [o.name for o in self.sess.get_outputs()]

        self.bridge = CvBridge()

        self.sub = self.create_subscription(Image, img_topic, self.callback, 10)
        self.pub = self.create_publisher(String, out_topic, 10)

        self.get_logger().info(f"Loaded ONNX model: {onnx_path}")

    def preprocess(self, img):
        img = cv2.resize(img, (800, 288))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2,0,1))[None, ...]
        return img

    def callback(self, msg: Image):
        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        inp = self.preprocess(img)

        outs = self.sess.run(self.output_names, {self.input_name: inp})

        out_msg = String()
        out_msg.data = json.dumps({ "outputs": [o.tolist() for o in outs] })
        self.pub.publish(out_msg)


def main():
    rclpy.init()
    node = UFLDOnnxNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

