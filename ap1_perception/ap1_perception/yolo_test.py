from .bag_reader import BagReader
from .yolo import YOLO
import matplotlib
import numpy as np
from tqdm import tqdm
from time import time

# Using Agg for headless or problematic environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def visualize_results(img, points, colors, boxes, classes, poses, file_name, stride=20, max_d=16):

    fig = plt.figure(figsize=(18, 8))
    
    ax1 = fig.add_subplot(121, projection='3d')
    
    sampled_points = points[::stride]
    sampled_colors = colors[::stride]
    
    x, y, z = sampled_points.T
    
    # Flip Y for visualization (so "up" is positive)
    sc = ax1.scatter(x, -y, z, s=1, c=sampled_colors, alpha=1)
    
    if len(poses) > 0:
        data_mask = (poses != 0).any(axis=-1)
        px, py, pz = poses[data_mask].T
        
        # Flip Y for visualization (so "up" is positive)
        ax1.scatter(px, -py, pz, marker='+', s=10000, c='red', linewidths=2, 
                    label='Detected Poses', zorder=100)
        
        for i, label_id in enumerate(classes[data_mask]):
            label_text = str(int(label_id))
            # Flip Y for visualization
            ax1.text(px[i], -py[i], pz[i], label_text, color='red', 
                     fontsize=14, fontweight='bold', zorder=101)

    ax1.set_xlabel('X (Horizontal)')
    ax1.set_ylabel('Y (Vertical)')
    ax1.set_zlabel('Z (Depth)')
    ax1.set_title('3D Point Cloud & Object Poses')

    ax1.view_init(elev=15, azim=15, vertical_axis='y')

    # Flip the Z-axis
    ax1.invert_zaxis()

    # Explicitly set your desired axis limits here
    ax1.set_xlim(-max_d, max_d)
    ax1.set_ylim(-max_d, max_d)
    ax1.set_zlim(2*max_d, 0)
    ax1.set_aspect('equal')

    ax2 = fig.add_subplot(122)
    ax2.imshow(img)
    
    for box, label_id in zip(boxes, classes):
        x1, y1, x2, y2 = box
        rect = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
        ax2.add_patch(rect)
        ax2.text(x1, y1 - 5, f"ID: {int(label_id)}", color='red', fontsize=10, 
                 fontweight='bold', backgroundcolor='white')

    ax2.set_title('RGB Detection')
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig(file_name)
    plt.show()
    plt.close()


# Execution logic
reader = BagReader("ap1_perception/bags/test1.bag")
yolo = YOLO(classes=None, K=reader.get_K())

# Grab frames
model_time = 0
start = time()
for i, (img, points, colors) in tqdm(enumerate(reader), total=len(reader), desc="Processing", unit="frames"):

    # Inference
    boxes, _ = yolo.forward(img)
    model_start = time()
    poses, classes = yolo(img, points)
    model_time += time() - model_start
    #print(poses)
    #print(classes)

    #print(f"Detected Classes: {[int(c) for c in classes]}")
    file_name = f"ap1_perception/yolo_frames/frame{i:04}.png"
    visualize_results(img, points, colors, boxes, classes, poses, file_name, 10, 16)
    #print(f"Result saved to {file_name}")
end = time()
print(f"Visualized {i + 1} frames in {end - start:.2f} seconds.")
print(f"Processed {i + 1} frames in {model_time:.2f} seconds.")
