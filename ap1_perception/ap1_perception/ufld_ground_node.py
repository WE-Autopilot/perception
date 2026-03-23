import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, PointCloud, PointCloud2, ChannelFloat32
from geometry_msgs.msg import Point32
from shape_msgs.msg import Plane as PlaneMsg
from message_filters import Subscriber, ApproximateTimeSynchronizer
from cv_bridge import CvBridge
import sensor_msgs_py.point_cloud2 as pc2

from .ufld import UFLDONNX
from .ransac import GroundRANSAC
from .projection import ground_proj

"""
MAJOR CHANGE FROM FINAL INTERFACE DOC:
Going to use: sensor_msgs/PointCloud since PointStampedArray doesn't exist.
Basically will be a pointcloud representation of the lane boundaries.

The advantage this gives us:
1) Built in message type that ROS understands
2) Usage of channels field in the message type that gives us a built in way to
assign lane IDs, rgb, and confidence.

Additionally, the node will publish the estimated ground plane as shape_msgs/Plane which represents a plane using [a, b, c, d] from the plane equation: 
ax + by + cz + d = 0
"""

DEPTH_TOPIC = "camera/camera/aligned_depth_to_color/image_raw"
COLOR_TOPIC = "camera/camera/color/image_raw"
PC_TOPIC    = "/camera/camera/depth/color/points"
INFO_TOPIC  = "camera/camera/aligned_depth_to_color/camera_info"

LANE_TOPIC  = "perception/lane_boundaries"
PLANE_TOPIC = "perception/ground_plane"


class UfldGroundNode(Node):
    def __init__(self):
        super().__init__('ap1_perception_ufld_ground')

        self._bridge = CvBridge()
        self._K: np.ndarray | None = None
        self._model: UFLDONNX | None = None
        self._ransac = GroundRANSAC()

        self._lane_pub = self.create_publisher(PointCloud, LANE_TOPIC, 10)
        self._plane_pub = self.create_publisher(PlaneMsg, PLANE_TOPIC, 10)

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
        self._model = UFLDONNX(ori_size=(msg.width, msg.height))

        self.get_logger().info(f"UFLD model initialized with image size ({msg.width}, {msg.height}).")

    def _sync_callback(self, color_msg: Image, pc_msg: PointCloud2) -> None:
        if self._K is None or self._model is None:
            return

        # --- Ground plane estimation via RANSAC ---
        xyz = pc2.read_points_numpy(
            pc_msg, field_names=('x', 'y', 'z'), skip_nans=True
        ).astype(np.float32)

        if xyz.shape[0] == 0:
            return

        plane = self._ransac(xyz)

        # Publish ground plane: ax + by + cz + d = 0 → d = -(n · p)
        plane_msg = PlaneMsg()
        plane_msg.coef[0] = float(plane.normal[0])
        plane_msg.coef[1] = float(plane.normal[1])
        plane_msg.coef[2] = float(plane.normal[2])
        plane_msg.coef[3] = float(-np.dot(plane.normal, plane.point))
        self._plane_pub.publish(plane_msg)

        # --- UFLD lane detection ---
        frame = self._bridge.imgmsg_to_cv2(color_msg, desired_encoding='rgb8')
        smooth_coords, lane_exists = self._model(frame)

        cloud = PointCloud()
        cloud.header = color_msg.header
        lane_id_channel = ChannelFloat32(name='lane_id', values=[])

        for lane_idx, (lane_coords, exists) in enumerate(zip(smooth_coords, lane_exists)):
            if not exists:
                continue

            # lanes in pixel coords
            lane_px = np.array(lane_coords, dtype=np.float64)
            # lanes in 3d coords - post projection
            pts_3d = ground_proj(self._K, lane_px, plane)

            for pt in pts_3d:
                cloud.points.append(Point32(
                    x=float(pt[0]), y=float(pt[1]), z=float(pt[2])
                ))
                lane_id_channel.values.append(float(lane_idx))

        cloud.channels.append(lane_id_channel)
        self._lane_pub.publish(cloud)


def main(args=None):
    rclpy.init(args=args)
    node = UfldGroundNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
