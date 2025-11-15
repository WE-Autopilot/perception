import numpy as np
from plyfile import PlyData, PlyElement

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D


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

    normal_mag = np.linalg.norm(normal)

    if normal_mag < 1e-6:
        return {"normal": np.zeros(3), "point": np.zeros(3), "fail": True}
    normal_normalized = normal / normal_mag
    
    
    return {"normal": normal_normalized, "point": p1, "fail": False}

    # print("Vector 1 (p2-p1):", v1)
    # print("Vector 2 (p3-p1):", v2)
    # print("Normal vector:", normal)
    # print("Normalized normal:", normal_normalized)


# reading a ply file and call the estimate_plane function
file_path = "000000599_0000000846.ply"

ply_data = PlyData.read(file_path)
vertex_data = ply_data["vertex"]

data_dict = {}

# Extract XYZ coordinates
x = np.array(vertex_data["x"])
y = np.array(vertex_data["y"])
z = np.array(vertex_data["z"])
data_dict["points"] = np.column_stack((x, y, z))

estimate_plane(data_dict["points"])

# test = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])

# estimate_plane(test)