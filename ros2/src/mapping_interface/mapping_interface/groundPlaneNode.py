#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2

# from .RANSAC.algo_utils import ransac
from .RANSAC.plane_utils import estimate
import numpy as np


class groundPlaneNode(Node):
    def __init__(self):
        super().__init__("my_node")
        self.get_logger().info("Node started!")

        test1 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        result = estimate.estimate_plane(test1)
        print(result)

        # subscription to pointcloud topic
        pointCloudSub = self.create_subscription(
            PointCloud2, "pointcloud", self.callback, 10
        )

    def callback(self, msg):
        # Convert PointCloud2 to Python list of points
        points_list = list(
            pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        )
        print("Points:", points_list)


def main(args=None):
    rclpy.init(args=args)
    node = groundPlaneNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
