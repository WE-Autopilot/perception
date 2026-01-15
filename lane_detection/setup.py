from setuptools import setup
from glob import glob
import os

package_name = 'lane_detection'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        # Resource index
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),

        # package.xml
        ('share/' + package_name, ['package.xml']),

        # Install ONNX models
        (os.path.join('share', package_name, 'models'),
            glob('models/*.onnx')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zayan',
    description='ONNX-based UFLD lane detection',
    entry_points={
        'console_scripts': [
            'ufld_onnx_node = lane_detection.ufld_onnx_node:main',
            'camera_image_publisher = lane_detection.camera_image_publisher:main',
            'ufld_visualizer = lane_detection.ufld_visualizer:main',
        ],
    },
)
