import numpy as np
from plyfile import PlyData, PlyElement
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Assuming data is Nx3 numpy array
# Function chooses 3 points randomly from the data and finds the plane normal vector
def estimate_plane(data):

    indicies = np.random.choice(len(data), size=3, replace=False)
    chosen_points = data[indicies]
    # print("Chosen points:\n", chosen_points)

    # Get the three points
    p1 = chosen_points[0]
    p2 = chosen_points[1]
    p3 = chosen_points[2]

    # Create two vectors from the three points
    v1 = p2 - p1  # Vector from p1 to p2
    v2 = p3 - p1  # Vector from p1 to p3

    # Calculate normal vector using cross product
    normal = np.cross(v1, v2)

    # Normalize the normal vector
    normal_normalized = normal / np.linalg.norm(normal)

    # print("Vector 1 (p2-p1):", v1)
    # print("Vector 2 (p3-p1):", v2)
    # print("Normal vector:", normal)
    # print("Normalized normal:", normal_normalized)

    # # plotting is just for visualization purposes and is probably unnecessary
    # if plot:
    #     # Plot the points and normal vector
    #     fig = plt.figure(figsize=(10, 8))
    #     ax = fig.add_subplot(111, projection="3d")

    #     # Plot the three chosen points
    #     ax.scatter(
    #         chosen_points[:, 0],
    #         chosen_points[:, 1],
    #         chosen_points[:, 2],
    #         c="red",
    #         s=100,
    #         marker="o",
    #         label="Selected Points",
    #     )

    #     # Label each point
    #     for i, point in enumerate(chosen_points):
    #         ax.text(point[0], point[1], point[2], f"  P{i+1}", fontsize=10)

    #     # Calculate centroid of the three points (for normal vector origin)
    #     centroid = np.mean(chosen_points, axis=0)

    #     # Plot the normal vector from the centroid
    #     scale = np.linalg.norm(p2 - p1) * 0.5  # Scale normal vector for visibility
    #     ax.quiver(
    #         centroid[0],
    #         centroid[1],
    #         centroid[2],
    #         normal_normalized[0],
    #         normal_normalized[1],
    #         normal_normalized[2],
    #         length=scale,
    #         color="blue",
    #         arrow_length_ratio=0.3,
    #         linewidth=2,
    #         label="Normal Vector",
    #     )

    #     # Draw lines connecting the three points to show the triangle
    #     triangle = np.vstack([chosen_points, chosen_points[0]])
    #     ax.plot(
    #         triangle[:, 0],
    #         triangle[:, 1],
    #         triangle[:, 2],
    #         "g--",
    #         alpha=0.6,
    #         label="Triangle",
    #     )

    #     ax.set_xlabel("X")
    #     ax.set_ylabel("Y")
    #     ax.set_zlabel("Z")
    #     ax.set_title("Plane Estimation: 3 Random Points and Normal Vector")
    #     ax.legend()
    #     plt.show()

    return normal_normalized


# reading a ply file and call the estimate_plane function
file_path = "0000000599_0000000846.ply"

ply_data = PlyData.read(file_path)
vertex_data = ply_data["vertex"]

data_dict = {}

# Extract XYZ coordinates
x = np.array(vertex_data["x"])
y = np.array(vertex_data["y"])
z = np.array(vertex_data["z"])
data_dict["points"] = np.column_stack((x, y, z))

estimate_plane(data_dict["points"])
