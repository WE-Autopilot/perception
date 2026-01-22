#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import numpy as np
import struct
from std_msgs.msg import Header


class PointCloudPublisher(Node):
    def __init__(self):
        super().__init__("pointcloud_publisher")
        self.publisher_ = self.create_publisher(PointCloud2, "pointcloud", 10)
        self.timer = self.create_timer(1.0, self.timer_callback)

    def timer_callback(self):
        # Example: create 5 points

        points = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)]

        # Convert to bytes
        cloud_data = b"".join([struct.pack("fff", *p) for p in points])

        msg = PointCloud2()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        msg.height = 1
        msg.width = len(points)

        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        msg.is_bigendian = False
        msg.point_step = 12  # 3 * 4 bytes
        msg.row_step = msg.point_step * len(points)
        msg.is_dense = True
        msg.data = cloud_data

        self.publisher_.publish(msg)
        self.get_logger().info("Published PointCloud2!")


def main(args=None):
    rclpy.init(args=args)
    node = PointCloudPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
