import numpy as np

from .utils import prune_distal, ground_ransac_factory
from ._plane import Plane


_DEFAULT_ESTIMATE = Plane([0, 0, 0], [0, 1, 0], True)


class GroundRANSAC:
    def __init__(self, estimate=_DEFAULT_ESTIMATE, max_retry=10, p_thresh=1, r_thresh=0.8, l_thresh=0.001):
        self.estimate = estimate
        self.score = 0
        self.prune = lambda points: prune_distal(points, self.estimate, p_thresh)
        self.ransac = ground_ransac_factory(max_retry, r_thresh, l_thresh)

    def __call__(self, raw_points):
        points = self.prune(raw_points)
        self.estimate, self.score = self.ransac(points, self.estimate)
        return self.estimate

    def set_estimate(self, estimate):
        self.estimate = estimate

    def get_estimate(self):
        return self.estimate

    def get_score(self):
        return self.score
