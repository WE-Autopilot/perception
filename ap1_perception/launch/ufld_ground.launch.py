import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable
from launch_ros.actions import Node


def _find_venv_site_packages() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / '.venv' / 'lib'
        if candidate.exists():
            return next(candidate.glob('python3*/site-packages'))
    raise RuntimeError('Could not find .venv — is the workspace set up correctly?')

_SITE_PACKAGES = _find_venv_site_packages()


def generate_launch_description():
    return LaunchDescription([
        SetEnvironmentVariable(
            'PYTHONPATH',
            str(_SITE_PACKAGES) + ':' + os.environ.get('PYTHONPATH', ''),
        ),
        Node(
            package='ap1_perception',
            executable='ufld_ground_node',
            name='ufld_ground_node',
            output='screen',
        ),
    ])
