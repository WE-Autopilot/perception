#!/usr/bin/env python3
import rclpy
from rclpy.node import Node


from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2

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
        print(ransacEstimation)


def main(args=None):
    rclpy.init(args=args)
    node = groundPlaneNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
