import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from ._yolo import YOLO
from ..projection import pointcloud_to_pixel, check_box


frames_dir = Path(__file__).resolve().parent / "stop_sign_frames"
model = YOLO()

for frame_dir in sorted(frames_dir.iterdir()):
    img = np.load(frame_dir / "color.npy")         # (H, W, 3) uint8 RGB
    points = np.load(frame_dir / "pointcloud.npy")  # (N, 6) float32 XYZ+RGB
    K = np.load(frame_dir / "camera_info.npy")      # (3, 3)

    boxes, img_cls = model.forward(img)

    # image with bounding boxes and corner markers
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(img_bgr, (x1, y1), 5, (0, 0, 255), -1)
        cv2.circle(img_bgr, (x2, y2), 5, (255, 0, 0), -1)

    cv2.imwrite(str(frame_dir / "output_boxes.jpeg"), img_bgr)

    # point cloud projection with 3D centroid overlaid
    xyz = points[:, :3]
    pixel_coords = pointcloud_to_pixel(K, xyz)  # (N, 2)
    depths = xyz[:, 2]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(img)
    sc = ax.scatter(
        pixel_coords[:, 0], pixel_coords[:, 1],
        c=depths, cmap='plasma', s=0.3, alpha=0.4,
        vmin=depths.min(), vmax=np.percentile(depths, 95),
    )
    plt.colorbar(sc, ax=ax, label='Depth (m)')

    for box in boxes:
        mask = check_box(pixel_coords, box)
        box_xyz = xyz[mask]
        if len(box_xyz) == 0:
            continue

        centroid_3d = np.median(box_xyz, axis=0)  # (3,)
        centroid_px = pointcloud_to_pixel(K, centroid_3d.reshape(1, 3))[0]

        ax.plot(*centroid_px, 'r*', markersize=14,
                label=f"centroid ({centroid_3d[0]:.2f}, {centroid_3d[1]:.2f}, {centroid_3d[2]:.2f}) m")

    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.set_title(frame_dir.name)
    if boxes.size:
        ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(str(frame_dir / "output_centroid.jpeg"), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"{frame_dir.name}: {len(boxes)} detection(s)")
