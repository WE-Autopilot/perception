from ._ufld_onnx import UFLDONNX
import numpy as np
from ..projection import ground_proj, get_horizon, pointcloud_to_pixel

class UFLD:
    def __init__(self, ori_size, K=None, onnx_path=None, config_path=None, num_wps=16, wp_thresh=16, horizon_dist=16):
        self.K = K
        self.horizon_dist = horizon_dist
        self.ufld_onnx = UFLDONNX(ori_size, onnx_path, config_path, num_wps, wp_thresh, horizon_dist)

    def __call__(self, img, plane, K=None, ray_disp=0, smooth=True):
        if self.K is None and K is None:
            raise Exception("Missing K.")

        if K is None:
            K = self.K

        # Calculate horizon slope (m) and intercept (b)
        horizon_3d = get_horizon(plane, length=self.horizon_dist)
        m, b = 0, 0
        if (horizon_3d[:, 2] > 0.1).all():
            horizon_2d = pointcloud_to_pixel(K, horizon_3d)
            x1, y1 = horizon_2d[0]
            x2, y2 = horizon_2d[1]
            if abs(x2 - x1) > 1e-6:
                m = (y2 - y1) / (x2 - x1)
                b = y1 - m * x1
            else:
                m = 0
                b = y1

        wp, lane_exists = self.ufld_onnx(img, smooth, m=m, b=b)
        points = ground_proj(K, wp, plane, ray_disp)

        return points, lane_exists
