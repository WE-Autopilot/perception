import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt

pipeline = rs.pipeline()
config = rs.config()
rs.config.enable_device_from_file(config, "test.bag")

profile = pipeline.start(config)
device = profile.get_device()
playback = device.as_playback()
playback.set_real_time(False)

# Align depth to color to use RGB origin
align_to = rs.stream.color
align = rs.align(align_to)

# Get intrinsics for the color stream
color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
intrinsics = color_stream.get_intrinsics()

# Format and print the K matrix
K = np.array([
    [intrinsics.fx, 0,             intrinsics.ppx],
    [0,             intrinsics.fy, intrinsics.ppy],
    [0,             0,             1]
])
print("RGB Camera K Matrix:")
print(K)

try:
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    if not depth_frame or not color_frame:
        print("Frames not available. Nya.")
    else:
        # Generate point cloud
        pc = rs.pointcloud()
        points = pc.calculate(depth_frame)
        
        v = points.get_vertices()
        verts = np.asanyarray(v).view(np.float32).reshape(-1, 3) 
        
        # Filter for validity and distance
        mask = (verts[:, 2] > 0.1) & (verts[:, 2] < 5.0)
        
        # Downsample for matplotlib
        stride = 20
        final_points = verts[mask][::stride]

        # Setup figure with two subplots
        fig = plt.figure(figsize=(15, 7))
        
        # Plot 1: 3D Point Cloud (RGB origin)
        ax1 = fig.add_subplot(121, projection='3d')
        x, y, z = final_points[:, 0], final_points[:, 1], final_points[:, 2]
        
        # Using depth for color mapping instead of RGB values
        sc = ax1.scatter(x, -y, z, s=1, c=z, cmap='viridis')
        ax1.set_xlabel('X (Right)')
        ax1.set_ylabel('Y (Up)')
        ax1.set_zlabel('Z (Forward)')
        ax1.set_title('Point Cloud (RGB Origin)')
        
        # Aspect ratio normalization
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
        mid_x = (x.max()+x.min()) * 0.5
        mid_y = (-y.max()-y.min()) * 0.5
        mid_z = (z.max()+z.min()) * 0.5
        ax1.set_xlim(mid_x - max_range, mid_x + max_range)
        ax1.set_ylim(mid_y - max_range, mid_y + max_range)
        ax1.set_zlim(mid_z - max_range, mid_z + max_range)

        # Plot 2: Normal RGB Image
        ax2 = fig.add_subplot(122)
        color_img = np.asanyarray(color_frame.get_data())
        ax2.imshow(color_img)
        ax2.set_title('RGB Frame')
        ax2.axis('off')

        plt.tight_layout()
        plt.show()

finally:
    pipeline.stop()
