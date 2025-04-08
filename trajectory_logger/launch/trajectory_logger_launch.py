from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'param_file',
            default_value=PathJoinSubstitution([
                FindPackageShare('trajectory_logger'),
                'config',
                'params.yaml'
            ]),
            description='YAML file with node parameters'
        ),
        Node(
            package='trajectory_logger',
            executable='trajectory_logger_node',
            name='trajectory_logger',
            output='screen',
            parameters=[LaunchConfiguration('param_file')]
        )
    ])
