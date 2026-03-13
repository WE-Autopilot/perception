import matplotlib.pyplot as plt
import numpy as np

from .ground_ransac import GroundRANSAC
from ._plane import Plane


def gen_plane(directions=np.array([[1, 0, 0], [0, 1, 0]]), disp=0, size=(10, 10), dir_len=20, noise=0.2, noise_min=-1, noise_max=1):
    i_range = np.linspace(-size[0], size[0], dir_len)
    j_range = np.linspace(-size[1], size[1], dir_len)
    s = np.array(np.meshgrid(i_range, j_range), dtype=np.float64).reshape(2, -1).T
    points = s @ directions + disp

    num_points = len(points)
    num_noise = int(num_points * noise)

    inds = np.random.choice(num_points, size=num_noise, replace=False)
    normal = np.cross(*directions)
    points[inds] += np.random.uniform(noise_min, noise_max, num_noise)[:, None] * normal

    return points

points = gen_plane(np.array([[1, 0, 0], [0, 1, 0]]), noise=0.8, noise_min=-10, noise_max=10)
initial_estimate = Plane([0, 0.3, 1], [0, 0, 0], True)

ransac = GroundRANSAC(initial_estimate)
pruned_points = ransac.prune(points)
estimate = ransac(points)

print(f"Estimate: {estimate}\nScore: {ransac.get_score() * 100:.2f}%")
point = estimate.point
normal = estimate.normal

ax = plt.axes(projection='3d')
ax.scatter(*points.T, color="blue")
ax.scatter(*pruned_points.T, color="red", marker="o", s=100)
ax.quiver(*point, *(normal * 10), color="orange", linewidths=5)
plt.show()
