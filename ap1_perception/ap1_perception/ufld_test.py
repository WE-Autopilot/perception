import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time

from .bag_reader import BagReader
from .ufld import UFLD
from .ransac import GroundRANSAC, Plane
from .projection import pointcloud_to_pixel, get_horizon

def visualize_results(img, points, colors, lane_points, lane_exists, lane_2d, plane, K, file_name, stride=20, max_d=16):
    fig = plt.figure(figsize=(18, 8))
    
    # 3D Plot
    ax1 = fig.add_subplot(121, projection='3d')
    
    sampled_points = points[::stride]
    sampled_colors = colors[::stride]
    x, y, z = sampled_points.T
    
    # Flip Y for visualization (so "up" is positive)
    sc = ax1.scatter(x, -y, z, s=1, c=sampled_colors, alpha=1)
    
    # Visualize Ground Plane
    if not plane.failed:
        grid_x, grid_z = np.meshgrid(np.linspace(-max_d, max_d, 10), np.linspace(0, max_d, 10))
        n = plane.normal
        p = plane.point
        d = np.dot(n, p)
        
        if abs(n[1]) > 1e-6:
            grid_y = (d - n[0]*grid_x - n[2]*grid_z) / n[1]
            # Flip Y for visualization
            ax1.plot_surface(grid_x, -grid_y, grid_z, alpha=0.2, color='gray')

    # Visualize Lanes in 3D
    lane_colors = ['red', 'green', 'blue', 'yellow']
    for i, (lane, exists) in enumerate(zip(lane_points, lane_exists)):
        if exists:
            lx, ly, lz = lane.T
            # Flip Y for visualization
            ax1.plot(lx, -ly, lz, color=lane_colors[i % len(lane_colors)], linewidth=3, label=f'Lane {i}')

    ax1.set_xlabel('X (Horizontal)')
    ax1.set_ylabel('Y (Vertical)')
    ax1.set_zlabel('Z (Depth)')
    ax1.set_title('3D Lanes & Ground Plane')
    
    # Match yolo_test view and limits
    ax1.view_init(elev=60, azim=15, vertical_axis='y')
    ax1.invert_zaxis()
    ax1.set_xlim(-max_d, max_d)
    ax1.set_ylim(-max_d, max_d)
    ax1.set_zlim(max_d, 0)

    # 2D Plot
    ax2 = fig.add_subplot(122)
    ax2.imshow(img)
    
    # --- Horizon Line ---
    if not plane.failed:
        horizon_3d = get_horizon(plane)
        if (horizon_3d[:, 2] > 0.1).all():
            horizon_2d = pointcloud_to_pixel(K, horizon_3d)
            ax2.plot(horizon_2d[:, 0], horizon_2d[:, 1], color='salmon', alpha=0.8, linewidth=2)

    # Visualize Lanes in 2D
    for i, (lane, exists) in enumerate(zip(lane_2d, lane_exists)):
        if exists:
            # lane is (N, 2)
            lx, ly = lane.T
            ax2.plot(lx, ly, color=lane_colors[i % len(lane_colors)], linewidth=2)

    h, w = img.shape[:2]
    ax2.set_xlim(0, w)
    ax2.set_ylim(h, 0)
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()

if __name__ == "__main__":
    # RESTORE USER PATHS
    bag_path = "ap1_perception/bags/test3.bag"
    reader = BagReader(bag_path)
    
    # Initial estimate: normal along Y axis (down in camera coords), point at origin
    initial_plane = Plane(normal=[0, 1, 0], point=[0, 0, 0])
    ransac = GroundRANSAC(estimate=initial_plane)
    
    # Get image size from first frame
    for img, points, colors in reader:
        h, w = img.shape[:2]
        break
    reader.restart()
    
    K = reader.get_K()
    ufld = UFLD(ori_size=(w, h), K=K)
    
    output_dir = "ap1_perception/ufld_frames"
    os.makedirs(output_dir, exist_ok=True)
    
    model_time = 0
    start = time()
    print("Starting UFLD inference...")
    for i, (img, points, colors) in tqdm(enumerate(reader), total=len(reader), desc="Processing", unit="frames"):
        
        # 1. Estimate ground plane
        plane = ransac(points)
        
        # Calculate horizon slope (m) and intercept (b)
        m, b = 0, 0
        if not plane.failed:
            horizon_3d = get_horizon(plane)
            if (horizon_3d[:, 2] > 0.1).all():
                horizon_2d = pointcloud_to_pixel(K, horizon_3d)
                x1, y1 = horizon_2d[0]
                x2, y2 = horizon_2d[1]
                if abs(x2 - x1) > 1e-6:
                    m = (y2 - y1) / (x2 - x1)
                    b = y1 - m * x1
                else:
                    m = 0
                    b = y1

        # 2. Run UFLD (Timing model inference specifically)
        model_start = time()
        # We need the 2D coordinates for the 2D plot
        lane_2d, lane_exists = ufld.ufld_onnx(img, m=m, b=b)
        # We need the 3D points from the projection
        lane_points, _ = ufld(img, plane)
        model_time += time() - model_start
        
        # 3. Visualize
        file_name = os.path.join(output_dir, f"frame{i:04d}.png")
        visualize_results(img, points, colors, lane_points, lane_exists, lane_2d, plane, K, file_name, 10, 16)
        
    end = time()
    print(f"Visualized {i + 1} frames in {end - start:.2f} seconds.")
    print(f"Processed {i + 1} frames in {model_time:.2f} seconds.")
