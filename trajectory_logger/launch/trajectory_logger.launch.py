from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    file_name_arg = DeclareLaunchArgument(
                "file_name",
                default_value="trajectory_log.csv",
                description="Name of the output file to save the trajectory data.",
            )
    file_name = LaunchConfiguration("file_name")
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "param_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("trajectory_logger"), "config", "params.yaml"]
                ),
                description="YAML file with node parameters",
            ),
            file_name_arg,
            Node(
                package="trajectory_logger",
                executable="trajectory_logger_node",
                name="trajectory_logger",
                output="screen",
                parameters=[LaunchConfiguration("param_file")],
                arguments=[
                    "--file_name",
                    file_name,
                ],
            ),
        ]
    )
