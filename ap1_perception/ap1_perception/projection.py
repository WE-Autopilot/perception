import numpy as np
import time


def pointcloud_to_pixel(K, points):
    f, c = K[[[0, 1], [0, 1]], [[0, 1], [2, 2]]]
    xy = points[:, :2]
    z = points[:, 2:3]
    coords = f * (xy / z) + c
    return coords


def get_dir_vec(K, coords):
    K_inv = np.linalg.inv(K)
    pad = np.ones(coords.shape[:-1] + (1,))
    pixels = np.concatenate((coords, pad), axis=-1)
    d = pixels @ K_inv.T
    norm_d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    return norm_d


def get_plane_inter(rays, plane, ray_disp=0):
    normal = plane.normal
    point = plane.point - ray_disp
    factor = (point @ normal) / (rays @ normal)
    inter = rays * factor[..., None] + ray_disp
    return inter


def get_point_proj(points, plane):
    normal = plane.normal
    point = plane.point
    dist = (points - point) @ normal / (normal @ normal)
    proj = points - dist[..., None] * normal
    return proj


def get_horizon(plane, length=16, dir_vec=np.array([0, 0, 1])):
    p0 = get_point_proj(np.zeros(3), plane)
    n = plane.normal

    # Project dir_vec onto plane
    dir_proj = dir_vec - (dir_vec @ n / (n @ n)) * n
    dir_unit = dir_proj / np.linalg.norm(dir_proj)

    p_mid = p0 + length * dir_unit

    # Perpendicular direction on plane
    perp_dir = np.cross(n, dir_unit)
    perp_dir = perp_dir / np.linalg.norm(perp_dir)

    # Return two points defining the segment
    p1 = p_mid - length * perp_dir
    p2 = p_mid + length * perp_dir

    return np.stack([p1, p2])


def check_box(coords, box):
    BOX_COMP = np.array([0, 1])
    bounds = box.reshape(2, 2).T
    comps = coords[..., None] < bounds
    mask = (comps == BOX_COMP).all(axis=(-2, -1))
    return mask


def ground_proj(K, wp, plane, ray_disp=0):
    rays = get_dir_vec(K, wp)
    proj_wp = get_plane_inter(rays, plane, ray_disp)
    return proj_wp


def get_centroid(points):
    points = points.reshape(-1, 3)
    centroid = np.median(points, axis=0)
    return centroid


def box_select_points(K, points, boxes):
    if len(boxes) == 0:
        return np.zeros((0, 3))

    coords = pointcloud_to_pixel(K, points)

    item_centroids = []
    for box in boxes:
        mask = check_box(coords, box)
        if mask.sum() == 0:
            item_centroids.append(np.zeros(3))
            continue

        item_points = points[mask]
        item_centroid = get_centroid(item_points)
        item_centroids.append(item_centroid)

    return np.vstack(item_centroids)
