import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt
import datetime


class BagReader:
    def __init__(self, bag_path, z_min=0.1, z_max=16):
        self.bag_path = bag_path
        self.z_min = z_min
        self.z_max = z_max
        self.is_running = False
        self.pipeline = None
        self.pc = rs.pointcloud()
        self._setup_pipeline()

    def _setup_pipeline(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device_from_file(self.bag_path, repeat_playback=False)
        
        self.profile = self.pipeline.start(self.config)
        self.device = self.profile.get_device()
        self.playback = self.device.as_playback()
        self.playback.set_real_time(False)
        
        self.playback.set_status_changed_callback(self._on_status_change)
        
        self.align = rs.align(rs.stream.color)
        
        color_stream = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.intrinsics = color_stream.get_intrinsics()
        self.fps = color_stream.fps()
        
        self.K = np.array([
            [self.intrinsics.fx, 0,                  self.intrinsics.ppx],
            [0,                  self.intrinsics.fy, self.intrinsics.ppy],
            [0,                  0,                  1]
        ])
        self.is_running = True

    def _on_status_change(self, status):
        if status == rs.playback_status.stopped:
            self.is_running = False

    def __len__(self):
        duration = self.playback.get_duration()
        # duration is a datetime.timedelta object
        duration_sec = duration.total_seconds()
        return int(duration_sec * self.fps)

    def get_K(self):
        return self.K

    def restart(self):
        if not self.is_running:
            self._setup_pipeline()
        else:
            self.playback.seek(datetime.timedelta(0))

    def __iter__(self):
        return self

    def __next__(self):
        if not self.is_running:
            raise StopIteration

        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            aligned_frames = self.align.process(frames)
            
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                self.stop()
                raise StopIteration
            
            rgb_image = np.asanyarray(color_frame.get_data())
            
            # Map pointcloud to color frame for textures
            self.pc.map_to(color_frame)
            points = self.pc.calculate(depth_frame)
            
            # Get vertices
            v = points.get_vertices()
            verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)
            
            # Get texture coordinates
            t = points.get_texture_coordinates()
            tex_coords = np.asanyarray(t).view(np.float32).reshape(-1, 2)
            
            # Convert texture coordinates to pixel indices
            h, w = rgb_image.shape[:2]
            u = (tex_coords[:, 0] * w).astype(int)
            v_idx = (tex_coords[:, 1] * h).astype(int)
            
            # Clip and extract colors
            u = np.clip(u, 0, w - 1)
            v_idx = np.clip(v_idx, 0, h - 1)
            colors = rgb_image[v_idx, u] / 255.0 # Normalize for matplotlib
            
            mask = (verts[:, 2] > self.z_min) & (verts[:, 2] < self.z_max)
            return rgb_image, verts[mask], colors[mask]
                
        except RuntimeError:
            self.stop()
            raise StopIteration

    def stop(self):
        if self.is_running:
            self.pipeline.stop()
            self.is_running = False

    def __del__(self):
        self.stop()

def visualize(rgb_image, verts, colors, stride=20):
    final_points = verts[::stride]
    final_colors = colors[::stride]

    fig = plt.figure(figsize=(15, 7))
    
    ax1 = fig.add_subplot(121, projection='3d')
    x, y, z = final_points.T
    
    # Use final_colors for the 'c' argument
    # Flip Y for visualization (so "up" is positive)
    ax1.scatter(x, -y, z, s=1, c=final_colors)
    ax1.set_xlabel('X (Right)')
    ax1.set_ylabel('Y (Up)')
    ax1.set_zlabel('Z (Forward)')
    ax1.set_title('Point Cloud (RGB Origin)')
    
    # Keep the aspect ratio equal
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    
    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
    ax1.set_zlim(mid_z - max_range, mid_z + max_range)

    ax2 = fig.add_subplot(122)
    ax2.imshow(rgb_image)
    ax2.set_title('RGB Frame')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    reader = BagReader("ap1_perception/bags/test.bag")
    print("RGB Camera K Matrix:")
    print(reader.get_K())
    
    for i, (rgb, cloud, colors) in enumerate(reader):
        print(f"frame {i}")
        visualize(rgb, cloud, colors)
