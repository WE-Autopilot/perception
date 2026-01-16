import numpy as np
import time
from coords_to_directions import coords_to_directions, coords_to_directions_unvectorized


def main():
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


if __name__ == "__main__":
    main()
