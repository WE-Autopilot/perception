import numpy as np


class Plane:
    def __init__(self, normal, point, failed=False):
        self.point = np.array(point, dtype=np.float64)
        self.normal = np.array(normal, dtype=np.float64)

        normal_mag = np.linalg.norm(self.normal)
        self.failed = normal_mag < 1e-6 or failed 
        if normal_mag >= 1e-6:
            self.normal /= normal_mag



    def __str__(self):
        failed = "FAILED" if self.failed else "VALID"
        return f"[normal: {self.normal}, point: {self.point}] ({failed})"
