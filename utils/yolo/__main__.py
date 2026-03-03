from PIL import Image
from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np


def vis_boxes(results, img):
    img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for result in results:
        # Extract bounding boxes in [x1, y1, x2, y2] format
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)

            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Plot the top-left and bottom-right corners
            cv2.circle(img_cv, (x1, y1), 5, (0, 0, 255), -1)
            cv2.circle(img_cv, (x2, y2), 5, (255, 0, 0), -1)

    cv2.imshow("Output", img_cv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = np.array(Image.open("stop.jpeg"))
curr_path = Path(__file__).resolve().parent
model = YOLO(f"{curr_path}/yolo11n.pt", verbose=False, task="detect")
results = model(img, classes=[11], verbose=True)

vis_boxes(results, img)
