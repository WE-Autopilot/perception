import numpy as np
from ._plane import Plane


def get_plane_dist(data, estimate):
    diffs = data - estimate.point
    distances = np.abs(diffs @ estimate.normal)

    return distances


def prune_distal(data, estimate, thresh=1):
    distances = get_plane_dist(data, estimate)
    return data[distances < thresh]


def estimate_plane(data):
    indicies = np.random.choice(len(data), size=3, replace=False)
    chosen_points = data[indicies]

    p1, p2, p3 = chosen_points

    v1 = p2 - p1
    v2 = p3 - p1

    normal = np.cross(v1, v2)

    return Plane(normal, p1)


def test_plane(data, estimate, thresh=1):
    if estimate.failed:
        return 0

    distances = get_plane_dist(data, estimate)
    loss = np.sum(distances < thresh) / len(data)
    return loss


def generic_ransac(data, initial_estimate, estimate_fn, test_fn, max_retry=10, thresh=0.8):
    best_estimate = initial_estimate
    best_score = test_fn(data, initial_estimate)

    if best_score > thresh:
        return initial_estimate, best_score

    for _ in range(max_retry):
        estimate = estimate_fn(data)
        if estimate.failed:
            continue

        score = test_fn(data, estimate)

        if score > thresh:
            return estimate, score

        if score > best_score:
            best_estimate = estimate
            best_score = score

    return best_estimate, best_score


def ground_ransac_factory(max_retry=10, r_thresh=0.8, l_thresh=0.001):
    return lambda data, initial_estimate: generic_ransac(
            data,
            initial_estimate,
            estimate_plane,
            lambda data, estimate: test_plane(data, estimate, l_thresh),
            max_retry,
            r_thresh
            )
