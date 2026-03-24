import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from .bag_reader import BagReader
from .projection import pointcloud_to_pixel

def test_alignment():
    # Try multiple possible locations for the bag file
    possible_paths = [
        "ap1_perception/ap1_perception/bags/test.bag",
        "ap1_perception/bags/test.bag",
        "bags/test.bag"
    ]
    bag_path = None
    for p in possible_paths:
        if os.path.exists(p):
            bag_path = p
            break
            
    if bag_path is None:
        print(f"Error: Could not find test.bag in any of: {possible_paths}")
        return

    print(f"Loading bag from: {bag_path}")
    reader = BagReader(bag_path)
    K = reader.get_K()
    
    # Grab the first valid frame
    found_frame = False
    for i, (rgb, cloud, colors) in enumerate(reader):
        print(f"Processing frame {i}")
        if i >= 1: # Let's take the second frame or first non-empty
            found_frame = True
            break
    
    if not found_frame:
        print("Error: Could not find a valid frame in bag.")
        return
            
    # BagReader now returns raw camera coordinates (Y-down).
    # pointcloud_to_pixel expects standard camera coords (Y-down).
    # So we don't need any flips here!
    
    # Reproject using projection.py function
    coords = pointcloud_to_pixel(K, cloud)
    
    u = coords[:, 0].astype(int)
    v = coords[:, 1].astype(int)
    
    h, w = rgb.shape[:2]
    
    # Mask valid coordinates within image bounds
    mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u_valid = u[mask]
    v_valid = v[mask]
    colors_valid = colors[mask]
    
    # Create reprojected image (initialized to black)
    reproj_img = np.zeros((h, w, 3))
    # Fill the pixels. Since cloud might have multiple points per pixel, 
    # we just let the last one win for simplicity.
    # Note: colors in BagReader are normalized to [0, 1] for matplotlib in some places, 
    # but let's check bag_reader.py again.
    # In bag_reader.py: colors = rgb_image[v_idx, u] / 255.0
    # So we should multiply back by 255 for imshow if it's float, or keep as is.
    # plt.imshow handles [0, 1] for floats.
    reproj_img[v_valid, u_valid] = colors_valid
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    axes[0].imshow(rgb)
    axes[0].set_title("Original RGB Frame")
    axes[0].axis("off")
    
    axes[1].imshow(reproj_img)
    axes[1].set_title("Reprojected Colored Point Cloud")
    axes[1].axis("off")
    
    plt.tight_layout()
    output_path = "ap1_perception/ap1_perception/yolo_frames/alignment_check.png"
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Alignment check saved to {output_path}")

if __name__ == "__main__":
    test_alignment()
