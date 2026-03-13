import matplotlib.pyplot as plt
import numpy as np

from .ground_ransac import GroundRANSAC


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

points = gen_plane(np.array([[1, 0, 0], [0, 1, 0]]), noise=0.9, noise_min=-10, noise_max=10)
initial_estimate = {"point": np.array([0, 0, 0]), "normal": np.array([0, 0.05, 1]), "failed": True}

ransac = GroundRANSAC(initial_estimate)
estimate = ransac(points)

print(f"Estimate: {estimate}\nScore: {ransac.get_score() * 100:.2f}%")
point = ransac.get_estimate()["point"]
normal = ransac.get_estimate()["normal"]

ax = plt.axes(projection='3d')
ax.scatter(*points.T, color="blue")
ax.scatter(*ransac.prune(points).T, color="red", marker="o", s=100)
ax.quiver(*point, *(normal * 10), color="red")
plt.show()
