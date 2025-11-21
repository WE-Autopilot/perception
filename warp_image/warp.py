import cv2
import numpy as np
import sys
import os

# Add parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import onnx_model_old.testModel_execute as onnx

file_name = "warp_image/sddefault"
ext = ".jpg"
# run onnx model to get lanes
img, lanes = onnx.main(file_name + ext)

# currently just warping to min-max rectangle, could be improved to a trapezoid more in line with lane positions
boundaries = {
    "minx": min(x for x, y in lanes[0]),
    "maxx": max(x for x, y in lanes[0]),
    "miny": min(y for x, y in lanes[0]),
    "maxy": max(y for x, y in lanes[0]),
}
pts_src = np.float32(
    [
        [boundaries["minx"], boundaries["miny"]],  # top-left
        [boundaries["maxx"], boundaries["miny"]],  # top-right
        [boundaries["maxx"], boundaries["maxy"]],  # bottom-right
        [boundaries["minx"], boundaries["maxy"]],  # bottom-left
    ]
)


# Desired output rectangle size
width, height = 800, 800
pts_dst = np.float32([[0, 0], [width, 0], [width, height], [0, height]])


# Compute homography
H = cv2.getPerspectiveTransform(pts_src, pts_dst)

# Warp into bird’s eye view
warped = cv2.warpPerspective(img, H, (width, height))

# Save or display
cv2.imwrite(file_name + "_warped" + ext, warped)

# transforming lane points (data not the image)
warped_lanes = []
for lane in lanes:
    # Convert lane points to homogeneous coordinates
    lane_points = np.array(lane, dtype=np.float32).reshape(-1, 1, 2)

    # Apply perspective transformation to the points
    warped_lane_points = cv2.perspectiveTransform(lane_points, H)

    # Convert back to list of tuples
    warped_lane = [(int(pt[0][0]), int(pt[0][1])) for pt in warped_lane_points]
    warped_lanes.append(warped_lane)

# display warped image with warped lanes just to check that lane values were transformed correctly
for lane in warped_lanes:
    for i in range(len(lane) - 1):
        pt1 = (int(lane[i][0]), int(lane[i][1]))
        pt2 = (int(lane[i + 1][0]), int(lane[i + 1][1]))
        cv2.line(warped, pt1, pt2, (0, 0, 255), thickness=2)
cv2.imshow("Warped Image with Warped Lanes", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()
