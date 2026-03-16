import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt
import time

pipeline = rs.pipeline()
config = rs.config()
rs.config.enable_device_from_file(config, "test.bag")

profile = pipeline.start(config)
device = profile.get_device()
playback = device.as_playback()
playback.set_real_time(False) # Process every frame available

# Get intrinsics for the depth stream
depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
intrinsics = depth_stream.get_intrinsics()
print(f"Intrinsics: {intrinsics}")

try:
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()

    pc = rs.pointcloud()
    points = pc.calculate(depth_frame)
    
    v = points.get_vertices()
    verts = np.asanyarray(v).view(np.float32).reshape(-1, 3) 
    
    mask = (verts[:, 2] > 0.1) & (verts[:, 2] < 5.0)
    final_points = verts[mask]

    # Downsample
    stride = 15 
    final_points = final_points[::stride]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    x = final_points[:, 0]
    y = final_points[:, 1]
    z = final_points[:, 2]

    # Invert Y for standard 'Up' orientation
    img = ax.scatter(x, -y, z, s=1, c=z, cmap='viridis')
    
    ax.set_xlabel('X (Right)')
    ax.set_ylabel('Y (Up)')
    ax.set_zlabel('Z (Forward)')
    
    # Normalize axes to prevent stretching
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (-y.max()-y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.colorbar(img, label='Depth (m)')
    plt.show()

finally:
    pipeline.stop()

