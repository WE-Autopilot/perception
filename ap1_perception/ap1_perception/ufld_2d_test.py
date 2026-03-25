import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time

from .bag_reader import BagReader
from .ufld import UFLD

def visualize_2d_results(img, lane_2d, lane_exists, file_name):
    h, w = img.shape[:2]
    # Fixed figure size and DPI
    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
    
    # Use fixed margins so the image area never moves
    plt.subplots_adjust(left=0.1, right=0.85, top=0.9, bottom=0.1)
    
    ax.imshow(img)
    
    lane_colors = ['red', 'green', 'blue', 'yellow']
    
    # Plot lanes
    for i, (lane, exists) in enumerate(zip(lane_2d, lane_exists)):
        if exists:
            lx, ly = lane.T
            ax.plot(lx, ly, color=lane_colors[i % len(lane_colors)], linewidth=3, label=f'Lane {i}')
        else:
            # Ghost plot for legend stability (keeps legend entries fixed)
            ax.plot([], [], color=lane_colors[i % len(lane_colors)], label=f'Lane {i} (N/A)')

    # Static title and axes labels
    ax.set_title('UFLD 2D Lane Detection', fontsize=14, pad=20)
    ax.set_xlabel('Pixel X')
    ax.set_ylabel('Pixel Y')

    # Fixed-position legend outside the plot area
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)

    # Hard-set limits to prevent any auto-scaling shifts
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)

    plt.savefig(file_name)
    plt.close(fig)

if __name__ == "__main__":
    bag_path = "ap1_perception/bags/test3.bag"
    reader = BagReader(bag_path)
    
    # Get image size from first frame
    for img, points, colors in reader:
        h, w = img.shape[:2]
        break
    reader.restart()
    
    ufld = UFLD(ori_size=(w, h), K=reader.get_K())
    
    output_dir = "ap1_perception/ufld_2d_frames"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting UFLD 2D inference...")
    start = time()
    for i, (img, points, colors) in tqdm(enumerate(reader), total=len(reader), desc="Processing", unit="frames"):
        
        # Run UFLD 2D detection
        lane_2d, lane_exists = ufld.ufld_onnx(img)
        
        # Visualize
        file_name = os.path.join(output_dir, f"frame{i:04d}.png")
        visualize_2d_results(img, lane_2d, lane_exists, file_name)
        
    end = time()
    print(f"Visualized {i + 1} frames in {end - start:.2f} seconds.")
