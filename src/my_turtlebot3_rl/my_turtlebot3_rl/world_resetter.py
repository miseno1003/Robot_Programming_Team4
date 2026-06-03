#!/usr/bin/env python3
"""
world_resetter.py

매 에피소드 시작 시 환경을 리셋한다:
- 기존 벨 모델 제거
- 유효 영역에서 벨 위치 랜덤 샘플링
- 새 벨 모델 spawn
- /bell_pose 토픽으로 벨 위치 발행
- 로봇을 시작 위치(1.2, -1.2, yaw=90°)로 리셋

서비스:
  /reset_world (std_srvs/Empty): 호출 시 환경 리셋

토픽:
  /bell_pose (geometry_msgs/PoseStamped): 랜덤 생성된 벨 위치 발행
"""

import math
import random
import subprocess
import time

import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty

from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy


WORLD_NAME = 'bell_world'
BELL_MODEL_NAME = 'wall_with_bell'
ROBOT_MODEL_NAME = 'burger_with_pole'

# 영역 경계
WALL_HALF = 1.5
EDGE_MARGIN = 0.15
BELL_Z = 0.110

# 로봇 시작 자세
ROBOT_START_X = 1.2
ROBOT_START_Y = -1.2
ROBOT_START_Z = 0.01
ROBOT_START_YAW = math.pi / 2.0   # 90°


BELL_SDF_TEMPLATE = """<?xml version="1.0" ?>
<sdf version="1.8">
  <model name="{model_name}">
    <static>true</static>
    <pose>{x} {y} 0 0 0 {yaw}</pose>

    <!-- 검은색 벨 베이스 -->
    <link name="bell_base">
      <pose>{base_offset_x} 0 {bell_z} 0 1.5708 0</pose>
      <collision name="base_collision">
        <geometry>
          <cylinder>
            <radius>0.052</radius>
            <length>0.035</length>
          </cylinder>
        </geometry>
      </collision>
      <visual name="base_visual">
        <geometry>
          <cylinder>
            <radius>0.052</radius>
            <length>0.035</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0.01 0.01 0.01 1</ambient>
          <diffuse>0.01 0.01 0.01 1</diffuse>
          <specular>0.1 0.1 0.1 1</specular>
        </material>
      </visual>
    </link>

    <!-- 빨간 버튼 -->
    <link name="red_button">
      <pose>{button_offset_x} 0 {bell_z} 0 1.5708 0</pose>
      <collision name="button_collision">
        <geometry>
          <cylinder>
            <radius>0.044</radius>
            <length>0.042</length>
          </cylinder>
        </geometry>
      </collision>
      <visual name="button_visual">
        <geometry>
          <cylinder>
            <radius>0.044</radius>
            <length>0.042</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>1.0 0.0 0.0 1</ambient>
          <diffuse>1.0 0.0 0.0 1</diffuse>
          <specular>0.9 0.2 0.2 1</specular>
        </material>
      </visual>

      <sensor name="button_contact" type="contact">
        <always_on>true</always_on>
        <update_rate>30</update_rate>
        <contact>
          <collision>button_collision</collision>
        </contact>
      </sensor>
    </link>

  </model>
</sdf>
"""


def sample_bell_pose():
    """
    유효 영역에서 벨의 (x, y, yaw)를 랜덤 샘플링.
    yaw는 벨 모델이 벽 안쪽을 바라보는 방향.
    """
    valid_lengths = [
        ('N', 3.0),
        ('E_top', 1.5),
        ('S_left', 1.5),
        ('W', 3.0),
    ]

    total_length = sum(seg[1] for seg in valid_lengths)
    s = random.uniform(0, total_length)

    cumulative = 0.0

    for name, length in valid_lengths:
        if s < cumulative + length:
            local_s = s - cumulative

            if name == 'N':
                # 북쪽 벽: y = +1.5
                x = -WALL_HALF + EDGE_MARGIN + local_s * (
                    (WALL_HALF * 2.0 - 2.0 * EDGE_MARGIN) / length
                )
                y = WALL_HALF
                yaw = -math.pi / 2.0

            elif name == 'E_top':
                # 동쪽 벽 위쪽: x = +1.5
                x = WALL_HALF
                y = WALL_HALF - EDGE_MARGIN - local_s * (
                    (WALL_HALF - EDGE_MARGIN) / length
                )
                yaw = math.pi

            elif name == 'S_left':
                # 남쪽 벽 왼쪽: y = -1.5
                x = -local_s * (
                    (WALL_HALF - EDGE_MARGIN) / length
                )
                y = -WALL_HALF
                yaw = math.pi / 2.0

            elif name == 'W':
                # 서쪽 벽: x = -1.5
                x = -WALL_HALF
                y = -WALL_HALF + EDGE_MARGIN + local_s * (
                    (WALL_HALF * 2.0 - 2.0 * EDGE_MARGIN) / length
                )
                yaw = 0.0

            return x, y, yaw

        cumulative += length

    return 0.0, WALL_HALF, -math.pi / 2.0


def build_bell_sdf(x, y, yaw):
    """
    벨 SDF 문자열 생성.
    모델 pose는 벽 위치에 두고, link들은 모델 기준 +x 방향으로 튀어나오게 함.
    """
    base_offset_x = 0.025 + 0.0175
    button_offset_x = base_offset_x + 0.0385

    return BELL_SDF_TEMPLATE.format(
        model_name=BELL_MODEL_NAME,
        x=x,
        y=y,
        yaw=yaw,
        bell_z=BELL_Z,
        base_offset_x=base_offset_x,
        button_offset_x=button_offset_x,
    )


def gz_service_call(service_name, req_type, rep_type, req):
    cmd = [
        'gz', 'service',
        '-s', service_name,
        '--reqtype', req_type,
        '--reptype', rep_type,
        '--timeout', '2000',
        '--req', req,
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=5
    )

    return result.returncode == 0, result.stdout, result.stderr


class WorldResetter(Node):
    def __init__(self):
        super().__init__('world_resetter')

        # ------------------------------------------------------------
        # /reset_world 서비스
        # ------------------------------------------------------------
        self.srv = self.create_service(
            Empty,
            '/reset_world',
            self.reset_callback
        )

        # ------------------------------------------------------------
        # /bell_pose publisher
        # TRANSIENT_LOCAL:
        # bell_approach 노드를 나중에 켜도 마지막 벨 좌표를 받을 수 있게 함
        # ------------------------------------------------------------
        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self.bell_pose_pub = self.create_publisher(
            PoseStamped,
            '/bell_pose',
            qos
        )

        self.last_bell_pose_msg = None

        self.get_logger().info('World resetter started.')

        # 시작 시 한 번 자동 리셋
        self.startup_timer = self.create_timer(2.0, self.startup_reset)
        self.startup_done = False

    def startup_reset(self):
        if self.startup_done:
            return

        self.startup_done = True
        self.startup_timer.cancel()

        self.get_logger().info('Performing initial reset...')
        self.reset_episode()

    def reset_callback(self, request, response):
        self.get_logger().info('Reset requested via service.')
        self.reset_episode()
        return response

    # ============================================================
    # Pose publish 관련 함수
    # ============================================================
    def yaw_to_quaternion(self, yaw):
        qz = math.sin(yaw * 0.5)
        qw = math.cos(yaw * 0.5)
        return qz, qw

    def publish_bell_pose(self, x, y, yaw):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'

        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = float(BELL_Z)

        qz, qw = self.yaw_to_quaternion(yaw)
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw

        self.bell_pose_pub.publish(msg)
        self.last_bell_pose_msg = msg

        self.get_logger().info(
            f'Published /bell_pose: '
            f'x={x:.2f}, y={y:.2f}, yaw={math.degrees(yaw):.1f}°'
        )

    # ============================================================
    # Reset sequence
    # ============================================================
    def reset_episode(self):
        # 1. 기존 벨 제거
        self.remove_bell()

        time.sleep(0.2)

        # 2. 새 벨 위치 샘플링
        x, y, yaw = sample_bell_pose()

        self.get_logger().info(
            f'New bell pose: x={x:.2f}, y={y:.2f}, '
            f'yaw={math.degrees(yaw):.1f}°'
        )

        # 3. 벨 spawn
        spawn_ok = self.spawn_bell(x, y, yaw)

        # 4. spawn 성공 시 /bell_pose 발행
        if spawn_ok:
            self.publish_bell_pose(x, y, yaw)

        # 5. 로봇 위치 리셋
        self.reset_robot()

    def remove_bell(self):
        req = f'name: "{BELL_MODEL_NAME}" type: MODEL'

        ok, stdout, stderr = gz_service_call(
            service_name=f'/world/{WORLD_NAME}/remove',
            req_type='gz.msgs.Entity',
            rep_type='gz.msgs.Boolean',
            req=req,
        )

        if ok:
            self.get_logger().debug('Removed existing bell.')
        else:
            self.get_logger().debug('No existing bell to remove.')

    def spawn_bell(self, x, y, yaw):
        sdf = build_bell_sdf(x, y, yaw)

        sdf_escaped = sdf.replace('"', '\\"').replace('\n', '\\n')
        req = f'sdf: "{sdf_escaped}"'

        ok, stdout, stderr = gz_service_call(
            service_name=f'/world/{WORLD_NAME}/create',
            req_type='gz.msgs.EntityFactory',
            rep_type='gz.msgs.Boolean',
            req=req,
        )

        if ok:
            self.get_logger().info('Bell spawned.')
            return True

        self.get_logger().error(f'Failed to spawn bell: {stderr}')
        return False

    def reset_robot(self):
        qz = math.sin(ROBOT_START_YAW / 2.0)
        qw = math.cos(ROBOT_START_YAW / 2.0)

        req = (
            f'name: "{ROBOT_MODEL_NAME}" '
            f'position: {{ x: {ROBOT_START_X}, y: {ROBOT_START_Y}, z: {ROBOT_START_Z} }} '
            f'orientation: {{ x: 0, y: 0, z: {qz}, w: {qw} }}'
        )

        ok, stdout, stderr = gz_service_call(
            service_name=f'/world/{WORLD_NAME}/set_pose',
            req_type='gz.msgs.Pose',
            rep_type='gz.msgs.Boolean',
            req=req,
        )

        if ok:
            self.get_logger().info(
                f'Robot reset to ({ROBOT_START_X}, {ROBOT_START_Y}, '
                f'yaw={math.degrees(ROBOT_START_YAW):.0f}°).'
            )
        else:
            self.get_logger().error(f'Failed to reset robot: {stderr}')


def main():
    rclpy.init()
    node = WorldResetter()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()