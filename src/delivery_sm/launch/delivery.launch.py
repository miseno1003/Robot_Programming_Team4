#!/usr/bin/env python3
"""세 노드를 한 번에 실행하는 런치 파일.

실행:
    # 같은 터미널에서 API 키를 먼저 export (따옴표 없이)
    export ANTHROPIC_API_KEY=sk-ant-...
    ros2 launch delivery_sm delivery.launch.py

주의:
- 이 런치는 vlm_node / nav_node / state_machine_node 만 띄운다.
- Nav2(navigate_to_pose 액션) 와 카메라(/camera/image_raw), LiDAR(/scan) 는
  로봇 bringup + Nav2 가 별도로 실행되어 있어야 한다.
- ANTHROPIC_API_KEY 는 이 런치를 실행한 터미널 환경을 그대로 상속한다.
- state_machine_node 는 OpenCV 창을 띄우므로 output='screen' 으로 둔다.
"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    vlm_node = Node(
        package='delivery_vlm',
        executable='vlm_node',
        name='vlm_node',
        output='screen',
        emulate_tty=True,
    )

    nav_node = Node(
        package='delivery_nav',
        executable='nav_node',
        name='nav_node',
        output='screen',
        emulate_tty=True,
    )

    state_machine_node = Node(
        package='delivery_sm',
        executable='state_machine_node',
        name='state_machine_node',
        output='screen',
        emulate_tty=True,
    )

    return LaunchDescription([
        vlm_node,
        nav_node,
        state_machine_node,
    ])
