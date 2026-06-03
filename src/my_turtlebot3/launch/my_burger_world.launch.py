#!/usr/bin/env python3
#
# my_burger_world.launch.py
# Camera + Pole이 부착된 버거를 Gazebo에서 실행
#

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # ============================================================
    # 패키지 경로
    # ============================================================
    pkg_my_turtlebot3 = get_package_share_directory('my_turtlebot3')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')

    # ============================================================
    # 환경 변수: Gazebo가 우리 모델/메시를 찾을 수 있게
    # ============================================================
    set_gz_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=os.path.join(pkg_my_turtlebot3, 'models')
    )

    # ============================================================
    # Launch 인자
    # ============================================================
    world = LaunchConfiguration('world')
    x_pose = LaunchConfiguration('x_pose')
    y_pose = LaunchConfiguration('y_pose')

    declare_world = DeclareLaunchArgument(
        'world',
        default_value=os.path.join(
            pkg_my_turtlebot3, 'worlds', 'bell_world.sdf'
        ),
        description='Gazebo world file'
    )
    declare_x = DeclareLaunchArgument('x_pose', default_value='-1.2')
    declare_y = DeclareLaunchArgument('y_pose', default_value='-1.2')

    # ============================================================
    # 1. Gazebo 실행
    # ============================================================
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': ['-r ', world]}.items()
    )

    # ============================================================
    # 2. 로봇 spawn (우리 SDF 모델)
    # ============================================================
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'burger_with_pole',
            '-file', os.path.join(
                pkg_my_turtlebot3, 'models', 'burger_with_pole', 'model.sdf'
            ),
            '-x', x_pose,
            '-y', y_pose,
            '-z', '0.01',
        ],
        output='screen',
    )

    # ============================================================
    # 3. ros_gz_bridge (Gazebo <-> ROS2 토픽 연결)
    # ============================================================
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        parameters=[{
            'config_file': os.path.join(
                pkg_my_turtlebot3, 'params', 'burger_bridge.yaml'
            ),
        }],
        output='screen',
    )

    return LaunchDescription([
        set_gz_resource_path,
        declare_world,
        declare_x,
        declare_y,
        gz_sim,
        spawn_robot,
        bridge,
    ])