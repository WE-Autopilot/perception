import pyrealsense2 as rs
import numpy as np

# Setup
pipeline = rs.pipeline()
config = rs.config()
config.enable_device_from_file("your_file.bag")

# Start and set to non-real-time to ensure no frames are dropped
profile = pipeline.start(config)
playback = profile.get_device().as_playback()
playback.set_real_time(False)

# 1. Create align and pc objects
align_to_color = rs.align(rs.stream.color)
pc = rs.pointcloud()

try:
    frames = pipeline.wait_for_frames()

    # 2. Align depth frame to color frame
    aligned_frames = align_to_color.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    # 3. Generate the point cloud centered on the color camera
    pc.map_to(color_frame)
    points = pc.calculate(aligned_depth_frame)

    # 4. Export to a NumPy array (Vertices are [x, y, z] in meters)
    v = points.get_vertices()
    verts = np.asanyarray(v).view(np.float32).reshape(-1, 3) 

    print(f"Point cloud generated with {len(verts)} points.")
    
finally:
    pipeline.stop()
