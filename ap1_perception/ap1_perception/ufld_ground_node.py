import numpy as np
import rclpy
from rclpy.node import Node
from sensors_msgs.msg import CameraInfo, Image, PointCloud, PointCloud2
from shape_msgs.msg import Plane
from message_filters import Subscriber, ApproximateTimeSynchronizer
from cv_bridge import CvBridge
import sensor_msgs_py.point_cloud2 as pc

from .ufld import UFLDONNX
from .ransac import GroundRANSAC
from .ransac._plane import Plane

"""
MAJOR CHANGE FROM FINAL INTERFACE DOC:
Going to use: sensor_msgs/PointCloud since PointStampedArray doesn't exist.
Basically will be a poinclout representation of the lane boundaries.

The advantage this gives us:
1) Built in message type that ROS understands
2) Usage of channels field in the message type that gives us a built in way to
assign lane IDs, rgb, and confidence.

Additionally, the node will publish the estimated plane as shape_msgs/Plane which
represents a plane using [a, b, c, d] from the plane equation: ax + by + cz = d
"""

DEPTH_TOPIC = "camera/camera/aligned_depth_to_color/image_raw"
COLOR_TOPIC = "camera/camera/color/image_raw"
PC_TOPIC = "/camera/camera/depth/color/points"
INFO_TOPIC = "camera/camera/aligned_depth_to_color/camera_info"

PUBLISH_TOPIC = "perception/lane_boundaries"

class UfldNode(Node):
    def __init__(self):
        super().__init__('ap1_perception')

        self._bridge = CvBridge()
        self._K: np.ndarray | None = None
        self._model: UFLDONNX | None = None

        self._pub = self.create_publisher(PointCloud, PUBLISH_TOPIC, 10)

        self.create_subscription(CameraInfo, INFO_TOPIC, self._camera_info_callback, 10)

        self.color_sub = self.create_subscription(self, Image, COLOR_TOPIC)

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        if self._K is not None:
            return
        self._K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self._model = UFLDONNX(
        ori_size=)
        self.get_logger().info("Camera intrinsics received — YOLO node ready.")


class RansacNode(Node):
    def __init__(self):
        super().__init__('ap1_perception')

        self._bridge = CvBridge()
        self._K: np.ndarray | None = None
