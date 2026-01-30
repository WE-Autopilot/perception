#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from plyfile import PlyData
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

        file_path = "src/mapping_interface/mapping_interface/0000000599_0000000846.ply"

        ply_data = PlyData.read(file_path)
        vertex_data = ply_data["vertex"]

        data_dict = {}

        # Extract XYZ coordinates
        x = np.array(vertex_data["x"])
        y = np.array(vertex_data["y"])
        z = np.array(vertex_data["z"])
        data_dict["points"] = np.column_stack((x, y, z))
        points = data_dict["points"]

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
