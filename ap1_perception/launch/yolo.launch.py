import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable
from launch_ros.actions import Node

# Walk up from this file: launch/ -> ap1_perception/ -> workspace root
_WORKSPACE = Path(__file__).resolve().parent.parent.parent

# Resolve the site-packages dir without hardcoding the Python version
_SITE_PACKAGES = next((_WORKSPACE / '.venv' / 'lib').glob('python3*/site-packages'))


def generate_launch_description():
    return LaunchDescription([
        SetEnvironmentVariable(
            'PYTHONPATH',
            str(_SITE_PACKAGES) + ':' + os.environ.get('PYTHONPATH', ''),
        ),
        Node(
            package='ap1_perception',
            executable='yolo_node',
            name='yolo_node',
            output='screen',
        ),
    ])
