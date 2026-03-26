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

def visualize_2d_results(img, lane_2d, lane_exists, plane, K, file_name):
    h, w = img.shape[:2]
    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
    
    # Use fixed margins
    plt.subplots_adjust(left=0.1, right=0.85, top=0.9, bottom=0.1)
    
    ax.imshow(img)
    
    # --- Ground Plane Grid Visualization ---
    if not plane.failed:
        n = plane.normal
        p = plane.point
        d = np.dot(n, p)
        
        # Project origin onto plane to find anchor point p0
        p0 = (d / np.dot(n, n)) * n
        
        # Create a grid relative to p0 (16m forward, 16m left/right)
        grid_x = np.linspace(p0[0] - 16, p0[0] + 16, 11)
        grid_z = np.linspace(p0[2], p0[2] + 16, 11)
        
        # Pre-calculate grid points to make filling squares easier
        grid_points_3d = np.zeros((len(grid_x), len(grid_z), 3))
        for i, x in enumerate(grid_x):
            for j, z in enumerate(grid_z):
                if abs(n[1]) > 1e-6:
                    y = (d - n[0]*x - n[2]*z) / n[1]
                    grid_points_3d[i, j] = [x, y, z]

        # Fill squares first (background)
        for i in range(len(grid_x) - 1):
            for j in range(len(grid_z) - 1):
                # Get the 4 corners of the square
                corners_3d = np.array([
                    grid_points_3d[i, j],
                    grid_points_3d[i+1, j],
                    grid_points_3d[i+1, j+1],
                    grid_points_3d[i, j+1]
                ])
                
                if (corners_3d[:, 2] > 0.1).all():
                    corners_2d = pointcloud_to_pixel(K, corners_3d)
                    ax.fill(corners_2d[:, 0], corners_2d[:, 1], color='skyblue', alpha=0.25)

        # Plot grid lines on top (foreground)
        for i in range(len(grid_x)):
            pts_3d = grid_points_3d[i, :]
            if (pts_3d[:, 2] > 0.1).all():
                pts_2d = pointcloud_to_pixel(K, pts_3d)
                ax.plot(pts_2d[:, 0], pts_2d[:, 1], color='royalblue', alpha=0.5, linewidth=1)

        for j in range(len(grid_z)):
            pts_3d = grid_points_3d[:, j]
            if (pts_3d[:, 2] > 0.1).all():
                pts_2d = pointcloud_to_pixel(K, pts_3d)
                ax.plot(pts_2d[:, 0], pts_2d[:, 1], color='royalblue', alpha=0.5, linewidth=1)

        # --- Horizon Line ---
        horizon_3d = get_horizon(plane)
        if (horizon_3d[:, 2] > 0.1).all():
            horizon_2d = pointcloud_to_pixel(K, horizon_3d)
            ax.plot(horizon_2d[:, 0], horizon_2d[:, 1], color='salmon', alpha=0.8, linewidth=2, label='Horizon')

    # --- Lane Visualization ---
    lane_colors = ['red', 'green', 'blue', 'yellow']
    for i, (lane, exists) in enumerate(zip(lane_2d, lane_exists)):
        if exists:
            lx, ly = lane.T
            ax.plot(lx, ly, color=lane_colors[i % len(lane_colors)], linewidth=3, label=f'Lane {i}')
        else:
            ax.plot([], [], color=lane_colors[i % len(lane_colors)], label=f'Lane {i} (N/A)')

    ax.set_title('UFLD 2D Lane Detection & Ground Plane', fontsize=14, pad=20)
    ax.set_xlabel('Pixel X')
    ax.set_ylabel('Pixel Y')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)

    # Hard-set limits to prevent any auto-scaling shifts
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)

    plt.savefig(file_name)
    plt.close(fig)

if __name__ == "__main__":
    bag_path = "ap1_perception/bags/test3.bag"
    reader = BagReader(bag_path)
    
    # Initial estimate: normal along Y axis (down in camera coords)
    initial_plane = Plane(normal=[0, 1, 0], point=[0, 0, 0])
    ransac = GroundRANSAC(estimate=initial_plane)
    
    # Get image size from first frame
    for img, points, colors in reader:
        h, w = img.shape[:2]
        break
    reader.restart()
    
    K = reader.get_K()
    ufld = UFLD(ori_size=(w, h), K=K)
    
    output_dir = "ap1_perception/ufld_2d_frames"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting UFLD 2D inference with Ground Plane...")
    start = time()
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
        
        # 2. Run UFLD 2D detection with horizon capping
        lane_2d, lane_exists = ufld.ufld_onnx(img, m=m, b=b)
        
        # 3. Visualize
        file_name = os.path.join(output_dir, f"frame{i:04d}.png")
        visualize_2d_results(img, lane_2d, lane_exists, plane, K, file_name)
        
    end = time()
    print(f"Visualized {i + 1} frames in {end - start:.2f} seconds.")
