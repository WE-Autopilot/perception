from setuptools import setup, find_packages

package_name = 'perception_onnx'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(),
    package_dir={'': '.'},
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zayan',
    maintainer_email='',
    description='ONNX UFLD lane detection package',
    entry_points={
        'console_scripts': [
            'ufld_onnx_node = perception_onnx.nodes.ufld_onnx_node:main',
            'image_publisher = perception_onnx.nodes.camera_image_publisher:main',
        ],
    },
)



