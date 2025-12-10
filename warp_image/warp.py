import cv2
import numpy as np
import sys
import os

# Add parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import onnx_model_old.testModel_execute as onnx

file_name = "warp_image/0000000114"
ext = ".png"
# run onnx model to get lanes
img, lanes = onnx.main(file_name + ext)

all_points = []
for lane in lanes:
    all_points.extend(lane)

print("number of points", len(all_points))
# currently just warping to min-max rectangle, could be improved to a trapezoid more in line with lane positions
boundaries = {
    "minx": min(x for x, y in all_points),
    "maxx": max(x for x, y in all_points),
    "miny": min(y for x, y in all_points),
    "maxy": max(y for x, y in all_points),
}
# PIXEL_MARGIN = 50
# topBits = []
# for x, y in all_points:
#     if y in range(boundaries["miny"] - PIXEL_MARGIN, boundaries["miny"] + PIXEL_MARGIN):
#         topBits.append(x)

# bottomBits = []
# for x, y in all_points:
#     if y in range(boundaries["maxy"] - PIXEL_MARGIN, boundaries["maxy"] + PIXEL_MARGIN):
#         bottomBits.append(x)

# print(len(topBits), topBits)
# topBitsBoundaries = {
#     "minx": min(topBits),
#     "maxx": max(topBits),
# }

# print(len(bottomBits), bottomBits)
# bottomBitsBoundaries = {
#     "minx": min(bottomBits),
#     "maxx": max(bottomBits),
# }

# pts_src = np.float32(
#     [
#         [topBitsBoundaries["minx"], boundaries["miny"]],  # top-left
#         [topBitsBoundaries["maxx"], boundaries["miny"]],  # top-right
#         [bottomBitsBoundaries["minx"], boundaries["maxy"]],  # bottom-left
#         [bottomBitsBoundaries["maxx"], boundaries["maxy"]],  # bottom-right
#     ]
# )

# Sort points by Y coordinate
sorted_points = sorted(all_points, key=lambda p: p[1])

# Get top 10% and bottom 10% of points
num_points = len(sorted_points)
top_count = max(2, int(num_points * 0.1))  # at least 2 points
bottom_count = max(2, int(num_points * 0.1))

top_points = sorted_points[:top_count]
bottom_points = sorted_points[-bottom_count:]

# Get boundaries
top_boundaries = {
    "minx": min(x for x, y in top_points),
    "maxx": max(x for x, y in top_points),
    "y": min(y for x, y in top_points),  # use the minimum Y
}

bottom_boundaries = {
    "minx": min(x for x, y in bottom_points),
    "maxx": max(x for x, y in bottom_points),
    "y": max(y for x, y in bottom_points),  # use the maximum Y
}

pts_src = np.float32(
    [
        [top_boundaries["minx"], top_boundaries["y"]],  # top-left
        [top_boundaries["maxx"], top_boundaries["y"]],  # top-right
        [bottom_boundaries["minx"], bottom_boundaries["y"]],  # bottom-left
        [bottom_boundaries["maxx"], bottom_boundaries["y"]],  # bottom-right
    ]
)


print("Source points for homography:\n", pts_src)


# Desired output rectangle size
width, height = 800, 800
pts_dst = np.float32([[0, 0], [width, 0], [0, height], [width, height]])


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
