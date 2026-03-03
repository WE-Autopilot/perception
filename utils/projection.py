import numpy as np
import time


def pointcloud_to_pixel(K, points):
    f, c = K[[[0, 1], [0, 1]], [[0, 1], [2, 2]]]
    xy = points[:, :2]
    z = points[:, 2:3]
    coords = f * (xy / z) + c
    return 


def get_dir_vec(K, coords):
    K_inv = np.linalg.inv(K)
    pad = np.ones(coords.shape[:-1] + (1,))
    pixels = np.concat((coords, pad), axis=-1)
    d = pixels @ K_inv.T
    norm_d = d / np.linalg.norm(d)
    return norm_d


def get_plane_inter(rays, plane, ray_disp=0):
    normal = plane["normal"]
    point = plane["point"] - ray_disp
    factor = (point @ normal) / (rays @ normal)
    inter = rays * factor + ray_disp
    return inter


def ground_proj(K, wp, plane, ray_disp=0):
    rays = get_dir_vec(K, wp)
    proj_wp = get_plane_inter(rays, plane, ray_disp)
    return proj_wp
