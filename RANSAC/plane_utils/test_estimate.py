import numpy as np
from estimate import estimate_plane


# Three collinear points (should be degenerate)
data = np.array([
    [0, 0, 0],
    [1, 1, 1],
    [2, 2, 2],  # This point is on the same line
])

est = estimate_plane(data)

if est is None:
    print(" Passed: Degenerate sample correctly returned None")
else:
    print(" Failed: Degenerate sample should return None, got:", est)
