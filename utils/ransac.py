import numpy as np


def get_plane_dist(data, estimate):
    diffs = data - estimate["point"]
    distances = np.abs(diffs @ estimate["normal"])

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

    normal_mag = np.linalg.norm(normal)
    if normal_mag < 1e-6:
        return {"normal": normal, "point": p1, "failed": True}

    return {"normal": normal / normal_mag, "point": p1, "failed": False}


def test_plane(data, estimate, thresh=1):
    if estimate.get("failed"):
        return float("inf")

    distances = get_plane_dist(data, estimate)
    loss = np.sum(distances < thresh) / len(data)
    return loss


def generic_ransac(data, initial_estimate, estimate_fn, test_fn, max_retry=10, thresh=0.8):
    best_estimate = initial_estimate
    best_score = test_fn(data, initial_estimate)

    if best_score < thresh:
        return initial_estimate

    for _ in range(max_retry):
        estimate = estimate_fn(data)
        if estimate.get("failed"):
            continue

        score = test_fn(data, estimate)

        if score > thresh:
            return estimate, score

        if score > best_score:
            best_estimate = estimate
            best_score = score

    return best_estimate, score


def ransac_factory(max_retry=10, p_thresh=0.8, l_thresh=0.001):
    return lambda data, initial_estimate: generic_ransac(
            data,
            initial_estimate,
            estimate_plane,
            lambda data, estimate: test_plane(data, estimate, l_thresh),
            max_retry,
            p_thresh
            )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def gen_plane(directions=np.array([[1, 0, 0], [0, 1, 0]]), disp=0, size=(10, 10), dir_len=20, noise=0.2, noise_min=-1, noise_max=1):
        i_range = np.linspace(-100, 100, dir_len)
        s = np.array(np.meshgrid(i_range, i_range), dtype=np.float64).reshape(2, -1).T
        points = s @ directions + disp

        num_points = len(points)
        num_noise = int(num_points * noise)

        inds = np.random.choice(num_points, size=num_noise, replace=False)
        normal = np.cross(*directions)
        points[inds] += np.random.uniform(noise_min, noise_max, num_noise)[:, None] * normal

        return points

    points = gen_plane(np.array([[1, 0, 0], [0, 1, 0]]), noise=0.5, noise_min=-10, noise_max=10)

    ransac = ransac_factory()
    initial_estimate = {"point": np.array([0, 0, 0]), "normal": np.array([0, 0.05, 1]), "failed": True}
    pruned_points = prune_distal(points, initial_estimate)
    estimate, score = ransac(pruned_points, initial_estimate)
    print(estimate, score)
    point = estimate["point"]
    normal = estimate["normal"]
    
    ax = plt.axes(projection='3d')
    ax.scatter(*points.T, color="blue")
    ax.scatter(*pruned_points.T, color="red", marker="o", s=100)
    ax.quiver(*point, *(normal * 10), color="red")
    plt.show()
