#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

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


def main(args=None):
    rclpy.init(args=args)
    node = groundPlaneNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
