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


def generate_launch_description():
    try:
        site_packages = str(_find_venv_site_packages())
    except RuntimeError:
        site_packages = None

    env_actions = (
        [SetEnvironmentVariable(
            'PYTHONPATH',
            site_packages + ':' + os.environ.get('PYTHONPATH', ''),
        )]
        if site_packages else []
    )

    return LaunchDescription(env_actions + [
        Node(
            package='ap1_perception',
            executable='yolo_node',
            name='yolo_node',
            output='screen',
        ),
    ])
