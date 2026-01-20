import numpy as np

def test_plane(data, estimate):
    """
    Evaluate how well the given plane estimate fits the data points.

    Args:
        data (np.ndarray): Nx3 array of 3D points
        estimate (dict): {"point": np.ndarray(3,), "normal": np.ndarray(3,)} 
                         representing the plane

    Returns:
        float: Mean Squared Error (MSE) of point-to-plane distances
    """
    # Extract plane parameters
    p0 = np.asarray(estimate["point"])
    n = np.asarray(estimate["normal"])

    # Ensure the normal vector is valid
    norm = np.linalg.norm(n)
    if norm < 1e-8:  # Degenerate case (collinear points)
        return float("inf")
    n = n / norm  # Normalize to unit length

    # Vector from plane point to all points
    diffs = data - p0  # Shape (N, 3)

    # Signed distances (dot product of diffs with normal)
    distances = np.dot(diffs, n)  # Shape (N,)

    # Mean squared error (MSE) — lower means plane fits better
    mse = np.mean(distances ** 2)
    return float(mse)