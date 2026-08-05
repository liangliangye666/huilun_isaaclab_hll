import launch
import os
from datetime import datetime
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, ExecuteProcess
from launch_ros.actions import Node


def generate_launch_description():
    project_root_path = os.environ.get("PROJECT_ROOT_DIR")
    sim_l2c_path = project_root_path + "/double_wheel_app_v3"

    now = datetime.now()
    date_time_str = now.strftime("-%Y-%m-%d %H:%M:%S")
    rosbag_record_cmd = [
        "ros2",
        "bag",
        "record",
        "-a",  # -a means all topics
        "-o",
        project_root_path + "/data/ros2_bag" + date_time_str,
    ]
    return LaunchDescription(
        [
            ExecuteProcess(
                cmd=rosbag_record_cmd,
                output="screen",
            ),
            ExecuteProcess(
                cmd=[sim_l2c_path],
                output="screen",
            ),
            LogInfo(msg="ROS 2 launch file executed successfully!"),
        ]
    )
