from PIL import Image
import cv2
import numpy as np
from pathlib import Path

from ._yolo import YOLO


img_path = f"{Path(__file__).resolve().parent}/stop.jpeg"
img = np.array(Image.open(img_path))
model = YOLO()
boxes, img_cls = model.forward(img)
points = np.random.randn(100, 3)
K = np.random.randn(3, 3)
centroids, img_cls = model(img, points, K)
print(f"Centroids: {centroids.shape}")


img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

for box in boxes:
    x1, y1, x2, y2 = map(int, box)

    cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Plot the top-left and bottom-right corners
    cv2.circle(img_cv, (x1, y1), 5, (0, 0, 255), -1)
    cv2.circle(img_cv, (x2, y2), 5, (255, 0, 0), -1)

cv2.imshow("Output", img_cv)
while cv2.waitKey(1) & 0xFF != ord('q'):
    pass
cv2.destroyAllWindows()

