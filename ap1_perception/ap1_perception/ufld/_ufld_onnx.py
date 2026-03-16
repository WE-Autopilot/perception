import cv2
import numpy as np
import onnxruntime as ort
import torch
import os
import sys
from pathlib import Path
import importlib.util


class UFLDONNX:
    def __init__(self, ori_size, onnx_path=None, config_path=None, num_wps=16, wp_thresh=16):
        if onnx_path == None or config_path == None:
            curr_path = Path(__file__).resolve().parent

        if onnx_path == None:
            onnx_path = f"{curr_path}/model.onnx"
        if config_path == None:
            config_path = f"{curr_path}/config.py"

        providers=['CPUExecutionProvider']

        self.session = ort.InferenceSession(onnx_path, providers=providers)

        # Input / output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        self.num_wps = num_wps
        self.wp_thresh = wp_thresh

        cfg = _load_config(config_path)
        self.ori_img_w, self.ori_img_h = ori_size
        self.cut_height = int(self.ori_img_h * (1 - cfg.crop_ratio))
        self.input_width = cfg.train_width
        self.input_height = cfg.train_height
        self.num_row = cfg.num_row
        self.num_col = cfg.num_col

        # Anchors
        self.row_anchor = np.linspace(1 - cfg.crop_ratio, 1, self.num_row)
        self.col_anchor = np.linspace(0, 1, self.num_col)

        # Normalization (standard ImageNet)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


    def pred2coords(self, pred):
        batch_size, num_grid_row, num_cls_row, num_lane_row = pred['loc_row'].shape
        batch_size, num_grid_col, num_cls_col, num_lane_col = pred['loc_col'].shape

        max_indices_row = pred['loc_row'].argmax(1)
        valid_row = pred['exist_row'].argmax(1)

        max_indices_col = pred['loc_col'].argmax(1)
        valid_col = pred['exist_col'].argmax(1)

        coords = []
        row_lane_idx = [1, 2]
        col_lane_idx = [0, 3]

        # ROW
        for lane_index in row_lane_idx:
            lane_coords = []
            if valid_row[0, :, lane_index].sum() > num_cls_row / 2:
                for k in range(valid_row.shape[1]):
                    if valid_row[0, k, lane_index]:
                        start = max(0, max_indices_row[0, k, lane_index] - self.input_width)
                        end = min(num_grid_row - 1,
                                  max_indices_row[0, k, lane_index] + self.input_width)
                        ind_list = torch.arange(start, end + 1).long()

                        weights = pred['loc_row'][0, ind_list, k, lane_index].softmax(0)
                        out_tmp = (weights * ind_list.float()).sum() + 0.5
                        out_tmp = out_tmp / (num_grid_row - 1) * self.ori_img_w

                        lane_coords.append((int(out_tmp), int(self.row_anchor[k] * self.ori_img_h)))
            coords.append(lane_coords)

        # COL
        for lane_index in col_lane_idx:
            lane_coords = []
            if valid_col[0, :, lane_index].sum() > num_cls_col / 4:
                for k in range(valid_col.shape[1]):
                    if valid_col[0, k, lane_index]:
                        start = max(0, max_indices_col[0, k, lane_index] - self.input_width)
                        end = min(num_grid_col - 1,
                                  max_indices_col[0, k, lane_index] + self.input_width)
                        ind_list = torch.arange(start, end + 1).long()

                        weights = pred['loc_col'][0, ind_list, k, lane_index].softmax(0)
                        out_tmp = (weights * ind_list.float()).sum() + 0.5
                        out_tmp = out_tmp / (num_grid_col - 1) * self.ori_img_h

                        lane_coords.append((int(self.col_anchor[k] * self.ori_img_w), int(out_tmp)))
            coords.append(lane_coords)

        # kinda cooked you gotta do this
        return [coords[2], coords[0][::-1], coords[1], coords[3]]

    def lane_lerp(self, coords):
        print("\nstarting")
        coords = np.array(coords)
        xs, ys = coords.T
        smooth_ys = np.linspace(ys.min(), ys.max(), self.num_wps)

        smooth_coords = []
        for smooth_y in smooth_ys:
            weights = 1 / ((ys - smooth_y) ** 2 + 1)
            weights /= weights.sum()
            smooth_x = (xs * weights).sum()
            smooth_coords.append((smooth_x, smooth_y))

        return smooth_coords

    def smooth_anchors(self, coords):
        smooth_coords = []
        lane_exists = []

        for lane_coords in coords:
            lane_exists.append(len(lane_coords) >= self.wp_thresh)
            if len(lane_coords) < self.wp_thresh:
                smooth_coords.append([[0, 0] for _ in range(self.num_wps)])
                continue

            smooth_coords.append(self.lane_lerp(lane_coords))

        return np.array(smooth_coords, dtype=np.int64), lane_exists

    def __call__(self, img, smooth=True):
        im0 = img.copy()

        # ---- PREPROCESSING ----
        im0 = im0[self.cut_height:, :, :]
        im0 = cv2.resize(im0, (self.input_width, self.input_height))

        im0 = im0.astype(np.float32) / 255.0
        im0 = im0.transpose(2, 0, 1)
        im0 = (im0 - self.mean) / self.std
        im0 = im0[np.newaxis, :, :, :]

        # ---- ONNX INFERENCE ----
        im0 = im0.astype(np.float32)
        outputs = self.session.run(self.output_names, {self.input_name: im0})

        # Convert ONNX output list
        preds = {name: torch.tensor(out) 
                 for name, out in zip(self.output_names, outputs)}

        coords = self.pred2coords(preds)

        if not smooth:
            return coords, [1, 1, 1, 1]

        smooth_coords, lane_exists = self.smooth_anchors(coords)
        return smooth_coords, lane_exists


def _load_config(config_path):
    config_path = os.path.abspath(config_path)
    module_name = os.path.splitext(os.path.basename(config_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, config_path)
    cfg_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_module)
    return cfg_module

