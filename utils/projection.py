import numpy as np
import time


def coords_to_directions(coords, Cx, Cy, Fx, Fy):
    # coords shape: (N, 2)
    # subtract principal point
    xy = coords - np.array([Cx, Cy])

    # divide by focal length
    norm_xy = xy / np.array([Fx, Fy])

    # append the Z=1 component
    dirs = np.hstack([norm_xy, np.ones((coords.shape[0], 1))])

    # normalize each row
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs = dirs / norms

    return dirs  # shape (N, 3)


def projectLane(dir_vecs, cam_displacement, plane):
    
    """
    Maps lanes to the ground plane

    Args:
        dir_vecs (np.ndarray): Nx3 array of vectors from camera to lane
        cam_displacement (np.ndarray(3,)): the camera's position as a point in 3d space. Will probably just be the origin 
        plane (dict): {"point": np.ndarray(3,), "normal": np.ndarray(3,)} 
                         representing the plane

    Returns:
        np.ndarray(3,): point on ground plane where a lane is
    """

    normal = np.asarray(plane["normal"])
    plane_point = np.asarray(plane["point"])
    ray_origin = np.asarray(cam_displacement)
    intersections = []

    for dir_vec in dir_vecs:
        ray_dir = np.asarray(dir_vec)
        
        # normalize direction vector
        ray_dir_norm = np.linalg.norm(ray_dir)
        if ray_dir_norm < 1e-10:
            return None
        ray_dir = ray_dir / ray_dir_norm
        
        # compute denominator (ray direction · plane normal)
        denom = np.dot(ray_dir, normal)
        
        # check if ray is parallel to plane
        if abs(denom) < 1e-10:
            return None
        
        # compute distance t along ray
        t = np.dot(plane_point - ray_origin, normal) / denom
        
        intersection = ray_origin + t * ray_dir
        intersections.append(intersection)
    
    return np.array(intersections)


if __name__ == "__main__":
    # Example usage
    coords = np.random.randint(0, 1280, (1000000, 2))
    # princpal point
    Cx, Cy = 640, 320
    # focal length
    Fx, Fy = 800, 800

    # start_time = time.time()
    # directions = coords_to_directions_unvectorized(coords, Cx, Cy, Fx, Fy)
    # print("total time", time.time() - start_time)

    start_time = time.time()
    directions = coords_to_directions(coords, Cx, Cy, Fx, Fy)
    print("total time", time.time() - start_time)

