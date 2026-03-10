from ultralytics import YOLO
from pathlib import Path

from ../projections import box_select_points


class YOLO:
    def __init__(self, classes=[11], K=None, model_path=None):
        if model_path == None:
            model_path = Path(__file__).resolve().parent

        self.K = K
        self.model = YOLO(f"{model_path}/yolo11n.pt", verbose=False, task="detect")
        self.classes = classes

    
    def forward(self, img):
        results = self.model(img, classes=[11], verbose=False)[0]
        img_cls = results.boxes.cls
        boxes = results.boxes.xyxy
        return boxes, img_cls

    def __call__(self, img, points, K=None):
        if K == None and self.K == None:
            raise Exception("No camera intrensics (K) provided during call or init.")

        if K == None:
            K = self.K

        boxes, img_cls = self.forward(img)
        centroids = box_select_points(K, points, boxes)

        return centroids, img_cls
