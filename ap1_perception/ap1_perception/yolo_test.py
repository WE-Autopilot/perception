from .bag_reader import BagReader
from .yolo import YOLO
import matplotlib
import numpy as np

# Using Agg for headless or problematic environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def visualize_results(img, points, boxes, classes, poses, stride=20):
    fig = plt.figure(figsize=(18, 8))
    
    # 3D Point Cloud Plot
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Downsample background points for performance
    sampled_points = points[::stride]
    
    # Coordinates: X (right), Y (down), Z (forward)
    # Mapping X to horizontal, -Y to vertical (to make Up positive), and Z to depth
    x, y, z = sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2]
    
    # Background cloud colored by depth (Z)
    # Reduced alpha slightly to ensure markers "pop" through the points
    sc = ax1.scatter(x, -y, z, s=1, c=z, cmap='viridis', alpha=1)
    
    # Add colorbar for depth scale
    cbar = plt.colorbar(sc, ax=ax1, pad=0.1)
    cbar.set_label('Depth (Z) in meters')
    
    # Plot detected poses as 'X' markers
    if len(poses) > 0:
        px, py, pz = poses[:, 0], poses[:, 1], poses[:, 2]
        
        # Increased marker size to 800 and set high zorder
        # This helps ensure the marker is drawn "on top" of the point cloud layer
        ax1.scatter(px, -py, pz, marker='x', s=5000, c='red', linewidths=2, 
                    label='Detected Poses', zorder=100)
        
        for i, label_id in enumerate(classes):
            label_text = str(int(label_id))
            ax1.text(px[i], -py[i], pz[i], label_text, color='red', 
                     fontsize=14, fontweight='bold', zorder=101)

    ax1.set_xlabel('X (Horizontal)')
    ax1.set_ylabel('Y (Vertical)')
    ax1.set_zlabel('Z (Depth)')
    ax1.set_title('3D Point Cloud & Object Poses')

    # Facing view: elev=0 looking straight ahead, azim=-90 looking down Z
    # This ensures X is horizontal and Y is vertical on the viewer
    ax1.view_init(elev=90, azim=-90)

    # Calculate limits for equal aspect ratio
    all_pts = np.vstack([sampled_points, poses]) if len(poses) > 0 else sampled_points
    ax_x, ax_y, ax_z = all_pts[:, 0], -all_pts[:, 1], all_pts[:, 2]
    
    max_range = np.array([ax_x.max()-ax_x.min(), ax_y.max()-ax_y.min(), ax_z.max()-ax_z.min()]).max() / 2.0
    mid_x, mid_y, mid_z = (ax_x.max()+ax_x.min())*0.5, (ax_y.max()+ax_y.min())*0.5, (ax_z.max()+ax_z.min())*0.5
    
    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
    ax1.set_zlim(mid_z - max_range, mid_z + max_range)

    # 2D Image Plot
    ax2 = fig.add_subplot(122)
    ax2.imshow(img)
    
    # Draw Bounding Boxes
    for box, label_id in zip(boxes, classes):
        x1, y1, x2, y2 = box
        rect = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='r', facecolor='none')
        ax2.add_patch(rect)
        ax2.text(x1, y1 - 5, f"ID: {int(label_id)}", color='red', fontsize=10, 
                 fontweight='bold', backgroundcolor='white')

    ax2.set_title('RGB Detection')
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig('detection_result.png')
    print("Result saved to detection_result.png")
    plt.show()

# Execution logic
reader = BagReader("ap1_perception/test.bag")
yolo = YOLO(K=reader.get_K())

# Grab frames
img, points = next(reader)

# Inference
boxes, _ = yolo.forward(img)
poses, classes = yolo(img, points)

print(f"Detected Classes: {[int(c) for c in classes]}")
visualize_results(img, points, boxes, classes, poses)
