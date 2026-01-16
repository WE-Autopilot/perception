import numpy as np
from plyfile import PlyData, PlyElement
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys
import os

# Add parent directory to path to import from RANSAC module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from RANSAC.plane_utils.estimate import estimate_plane


def plot_plane_and_points(data, normal, point_on_plane):
    # Extract points and colors from data dictionary
    points = data["points"]
    colors = data.get("colors", None)

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Scatter plot of the point cloud with colors if available
    if colors is not None:
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=colors,
            marker="o",
            s=1,
            alpha=0.6,
        )
    else:
        ax.scatter(
            points[:, 0], points[:, 1], points[:, 2], c="b", marker="o", s=1, alpha=0.6
        )

    # Create a grid of points to represent the plane
    # Plane equation: normal[0]*x + normal[1]*y + normal[2]*z + d = 0
    d = -point_on_plane.dot(normal)
    xx, yy = np.meshgrid(
        range(int(np.min(points[:, 0])), int(np.max(points[:, 0]))),
        range(int(np.min(points[:, 1])), int(np.max(points[:, 1]))),
    )
    # Rearrange to solve for z
    zz = (-normal[0] * xx - normal[1] * yy - d) / normal[2]

    # Plot the plane
    ax.plot_surface(xx, yy, zz, alpha=0.3, color="r")

    # Set labels
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    ax.set_title("Point Cloud with Estimated Plane")

    plt.show()


# file_path = "../RANSAC/test_data_cleaned/test104_cleaned.npy"
# data = np.load(file_path, allow_pickle=True).item()
# result = estimate_plane(data["points"])


# plot_plane_and_points(data, result["normal"], result["point"])
