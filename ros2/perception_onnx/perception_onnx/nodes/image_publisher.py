#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import glob
import time

class ImagePublisher(Node):
    def __init__(self):
        super().__init__("image_publisher")
        self.pub = self.create_publisher(Image, "/camera/image_raw", 10)
        self.bridge = CvBridge()

        # load jpg/png files from a folder
        # load jpg/png files from extracted folder
        self.images = sorted(
            glob.glob("/home/zayan/perception_ws/sample_images/extracted/*.png") +
            glob.glob("/home/zayan/perception_ws/sample_images/extracted/*.jpg") +
            glob.glob("/home/zayan/perception_ws/sample_images/extracted/*.jpeg")
        )

        if not self.images:
            self.get_logger().error("No images found in extracted sample_images folder!")
            exit(1)


        self.index = 0
        self.timer = self.create_timer(1.0, self.publish_image)

        self.get_logger().info(f"Publishing {len(self.images)} test images...")

    def publish_image(self):
        if self.index >= len(self.images):
            self.get_logger().info("Done publishing images.")
            return

        img = cv2.imread(self.images[self.index])
        msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        self.pub.publish(msg)
        self.get_logger().info(f"Published {self.images[self.index]}")
        self.index += 1


def main():
    rclpy.init()
    node = ImagePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

