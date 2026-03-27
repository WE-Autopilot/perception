from setuptools import find_packages, setup

package_name = 'ap1_perception'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/yolo.launch.py',
            'launch/ufld_ground.launch.py',
        ]),
    ],
    # ufld onnx stuff
    package_data={
        package_name: ['ufld/model.onnx', 'ufld/config.py'],
    },
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tygo',
    maintainer_email='tcrawley@uwo.ca',
    description='AP1 Perception Launch Pkg',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'yolo_node = ap1_perception.yolo_node:main',
            'ufld_ground_node = ap1_perception.ufld_ground_node:main',
        ],
    },
)
