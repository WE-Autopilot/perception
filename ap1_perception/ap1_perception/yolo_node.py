import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from geometry_msgs.msg import 
from message_filters import Subscriber, ApproximateTimeSynchronizer


DEPTH_TOPIC = "camera/camera/aligned_depth_to_color/image_raw"
COLOR_TOPIC = "camera/camera/color/image_raw"
PC_TOPIC = "/camera/camera/depth/color/points"
INFO_TOPIC = "camera/camera/aligned_depth_to_color/camera_info"

class yolo_node(Node):
    def __init__(self):
        super().__init__('ap1_perception')
