import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from geometry_msgs.msg import PoseArray, PoseStamped
from message_filters import Subscriber, ApproximateTimeSynchronizer
from cv_bridge import CvBridge
import sensor_msgs_py.point_cloud2 as pc2

from .yolo import YOLO
from .projection import pointcloud_to_pixel, check_box
from .ransac import GroundRANSAC
from .ransac._plane import Plane


DEPTH_TOPIC = "camera/camera/aligned_depth_to_color/image_raw"
COLOR_TOPIC = "camera/camera/color/image_raw"
PC_TOPIC = "/camera/camera/depth/color/points"
INFO_TOPIC = "camera/camera/aligned_depth_to_color/camera_info"

PUBLISH_TOPIC = "perception/entities"


def normal_to_quaternion(normal: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a unit normal vector to a quaternion with the z-axis aligned to the normal.

    Returns (qx, qy, qz, qw).
    """
    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)

    z = np.array([0.0, 0.0, 1.0])
    dot = np.dot(z, normal)

    if dot > 1.0 - 1e-6:
        return (0.0, 0.0, 0.0, 1.0)
    if dot < -1.0 + 1e-6:
        return (1.0, 0.0, 0.0, 0.0)

    axis = np.cross(z, normal)
    axis /= np.linalg.norm(axis)
    angle = np.arccos(np.clip(dot, -1.0, 1.0))

    s = np.sin(angle / 2.0)
    return (axis[0] * s, axis[1] * s, axis[2] * s, np.cos(angle / 2.0))


def ransac_plane_normal(points: np.ndarray) -> np.ndarray:
    """Fit a plane to 3D points using GroundRANSAC and return the unit normal.

    The normal is oriented to face the camera (negative z in camera frame).
    """
    centroid = np.median(points, axis=0)
    # Initial estimate: plane at the centroid with normal pointing toward the camera.
    initial = Plane([0.0, 0.0, -1.0], centroid)
    ransac = GroundRANSAC(estimate=initial, max_retry=20, p_thresh=1, r_thresh=0.7, l_thresh=0.01)
    plane = ransac(points)

    normal = plane.normal
    if normal[2] > 0:
        normal = -normal
    return normal


class YoloNode(Node):
    def __init__(self):
        super().__init__('ap1_perception_yolo')

        self._bridge = CvBridge()
        self._K: np.ndarray | None = None
        self._model: YOLO | None = None

        self._pub = self.create_publisher(PoseArray, PUBLISH_TOPIC, 10)

        self.create_subscription(CameraInfo, INFO_TOPIC, self._camera_info_callback, 10)

        color_sub = Subscriber(self, Image, COLOR_TOPIC)
        pc_sub = Subscriber(self, PointCloud2, PC_TOPIC)
        self._sync = ApproximateTimeSynchronizer(
            [color_sub, pc_sub], queue_size=5, slop=0.05
        )
        self._sync.registerCallback(self._sync_callback)

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        if self._K is not None:
            return
        self._K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self._model = YOLO(K=self._K)
        self.get_logger().info("Camera intrinsics received — YOLO node ready.")

    def _sync_callback(self, color_msg: Image, pc_msg: PointCloud2) -> None:
        if self._K is None or self._model is None:
            return

        color = self._bridge.imgmsg_to_cv2(color_msg, desired_encoding='rgb8')

        # read_points_numpy returns a plain (N, 3) float array directly
        xyz = pc2.read_points_numpy(pc_msg, field_names=('x', 'y', 'z'), skip_nans=True).astype(np.float32)

        if xyz.shape[0] == 0:
            return

        boxes, _ = self._model.forward(color)

        if boxes.shape[0] == 0:
            return

        pixel_coords = pointcloud_to_pixel(self._K, xyz)  # (N, 2)

        pose_array = PoseArray()
        pose_array.header = color_msg.header

        for box in boxes:
            mask = check_box(pixel_coords, box)
            box_xyz = xyz[mask]

            if len(box_xyz) < 3:
                continue

            centroid = np.median(box_xyz, axis=0)
            normal = ransac_plane_normal(box_xyz)
            qx, qy, qz, qw = normal_to_quaternion(normal)

            pose = PoseStamped().pose
            pose.position.x = float(centroid[0])
            pose.position.y = float(centroid[1])
            pose.position.z = float(centroid[2])
            pose.orientation.x = float(qx)
            pose.orientation.y = float(qy)
            pose.orientation.z = float(qz)
            pose.orientation.w = float(qw)

            pose_array.poses.append(pose)

        self._pub.publish(pose_array)


def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
