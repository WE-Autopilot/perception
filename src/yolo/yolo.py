from ultralytics import YOLO as _YOLO
import numpy as np
from pathlib import Path

from ..projection import box_select_points


class YOLO:
    def __init__(self, classes=[11], K=None, model_path=None):
        if model_path == None:
            model_path = f"{Path(__file__).resolve().parent}/yolo11n.pt"

        self.K = K
        self.model = _YOLO(model_path, verbose=False, task="detect")
        self.classes = classes

    
    def forward(self, img):
        results = self.model(img, classes=[11], verbose=False)[0]
        img_cls = results.boxes.cls.cpu().numpy()
        boxes = results.boxes.xyxy.cpu().numpy()
        return boxes, img_cls

    def __call__(self, img, points, K=None):
        if K is None and self.K is None:
            raise Exception("No camera intrensics (K) provided during call or init.")

        if K is None:
            K = self.K

        boxes, img_cls = self.forward(img)
        centroids = box_select_points(K, points, boxes)

        return centroids, img_cls
