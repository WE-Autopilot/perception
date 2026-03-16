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
    pixels = np.concat((coords, pad), axis=-1)
    d = pixels @ K_inv.T
    norm_d = d / np.linalg.norm(d)
    return norm_d


def get_plane_inter(rays, plane, ray_disp=0):
    normal = plane.normal
    point = plane.point - ray_disp
    factor = (point @ normal) / (rays @ normal)
    inter = rays * factor[..., None] + ray_disp
    return inter


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
    coords = pointcloud_to_pixel(K, points)

    item_centroids = []
    for box in boxes:
        mask = check_box(coords, box)
        item_points = points[mask]
        item_centroid = get_centroid(item_points)
        item_centroids.append(item_centroid)

    return np.vstack(item_centroids)
