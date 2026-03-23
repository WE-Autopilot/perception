from ._ufld_onnx import UFLDONNX
from ..projection import ground_proj

class UFLD:
    def __init__(self, ori_size, K=None, onnx_path=None, config_path=None, num_wps=16, wp_thresh=16):
        self.K = K
        self.ufld_onnx = UFLDONNX(ori_size, onnx_path, config_path, num_wps, wp_thresh)

    def __call__(self, img, plane, K=None, ray_disp=0, smooth=True):
        if self.K is None and K is None:
            raise Exception("Missing K.")

        if K is None:
            K = self.K

        wp = self.ufld_onnx(img, smooth)
        points = ground_proj(K, wp, plane, ray_disp)

        return points
