#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from geometry_msgs.msg import Vector3


import sys

print("Python exec:", sys.executable)
print("First few sys.path entries:", sys.path[:5])

from plane_utils import estimate
from plane_utils import test
from algo_utils import ransac
import numpy as np


class groundPlaneNode(Node):
    def __init__(self):
        super().__init__("my_node")
        self.get_logger().info("Node started!")

        # subscription to pointcloud topic
        pointCloudSub = self.create_subscription(
            PointCloud2, "pointcloud", self.callback, 10
        )

    def callback(self, msg):
        points = []
        for p in pc2.read_points(msg, skip_nans=True):
            points.append([p[0], p[1], p[2]])
        array = np.array(points, dtype=np.float32)

        # just using a static np array rn for testing
        test1 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])

        ransacEstimation = ransac.RANSAC_noInit(
            data=array,
            estimate_fn=estimate.estimate_plane,
            test_fn=test.test_plane,
            thresh=5,
            max_retry=10,
        )
        # print(ransacEstimation)
        self.normal_pub = self.create_publisher(Vector3, "normal", 10)
        self.point_pub = self.create_publisher(Vector3, "point", 10)

        normal_msg = Vector3()
        normal_msg.x = float(ransacEstimation["normal"][0])
        normal_msg.y = float(ransacEstimation["normal"][1])
        normal_msg.z = float(ransacEstimation["normal"][2])
        self.normal_pub.publish(normal_msg)
        self.get_logger().info(
            f"Published normal: {normal_msg.x}, {normal_msg.y}, {normal_msg.z}"
        )

        point_msg = Vector3()
        point_msg.x = float(ransacEstimation["point"][0])
        point_msg.y = float(ransacEstimation["point"][1])
        point_msg.z = float(ransacEstimation["point"][2])
        self.point_pub.publish(point_msg)
        self.get_logger().info(
            f"Published point: {point_msg.x}, {point_msg.y}, {point_msg.z}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = groundPlaneNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
