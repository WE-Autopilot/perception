import numpy as np


def coords_to_directions(coords, Cx, Cy, Fx, Fy):
    # coords shape: (N, 2)
    # subtract principal point
    xy = coords - np.array([Cx, Cy])

    # divide by focal length
    norm_xy = xy / np.array([Fx, Fy])

    # append the Z=1 component
    dirs = np.hstack([norm_xy, np.ones((coords.shape[0], 1))])

    # normalize each row
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs = dirs / norms

    return dirs  # shape (N, 3)


# def coords_to_directions_unvectorized(coords, Cx, Cy, Fx, Fy):
#     # coords shape: (N, 2)
#     N = coords.shape[0]
#     directions = np.zeros((N, 3))

#     for i in range(N):
#         u, v = coords[i]
#         d = np.array([(u - Cx) / Fx, (v - Cy) / Fy, 1])
#         d = d / np.linalg.norm(d)
#         directions[i] = d
#     # directons shape: (N, 3)
#     return directions
