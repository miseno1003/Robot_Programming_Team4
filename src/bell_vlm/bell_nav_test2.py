#!/usr/bin/env python3
"""벨 정면 이동 + Nav2 성공 확인 + VLM 정렬 + LiDAR 거리 기반 전진"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus

from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import PoseStamped, TwistStamped
from lifecycle_msgs.srv import ChangeState
from lifecycle_msgs.msg import Transition

import math
import time
import anthropic
import base64
import cv2
import numpy as np


# ===== 벨 설정 =====
BELL_FRONT_X = 0.5915663550652934
BELL_FRONT_Y = 0.13834688301921683
BELL_ORIENT_Z = -0.3913776770869043
BELL_ORIENT_W = 0.9202301418004405

YAW = 2.0 * math.atan2(BELL_ORIENT_Z, BELL_ORIENT_W)

# RViz에서 확인한 목표 위치가 벨 앞이 맞다고 했으므로 기존 계산 유지
GOAL_X = BELL_FRONT_X - 0.3 * math.cos(YAW)
GOAL_Y = BELL_FRONT_Y - 0.3 * math.sin(YAW)

TARGET_DISTANCE = 0.15  # 벨에서 15cm 앞까지 접근
STOP_TOLERANCE = 0.02   # 2cm 오차 허용

print(f"벨 앞 좌표: ({BELL_FRONT_X}, {BELL_FRONT_Y})")
print(f"로봇 yaw: {math.degrees(YAW):.1f}도")
print(f"목표 지점 (정면 30cm): ({GOAL_X:.3f}, {GOAL_Y:.3f})")
print(f"최종 접근 목표: 벨에서 {TARGET_DISTANCE * 100:.0f}cm")


class BellNavigator(Node):
    def __init__(self):
        super().__init__('bell_navigator')

        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)

        self.camera_msg = None
        self.scan_msg = None

        self.cam_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self._cam_cb,
            1
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self._scan_cb,
            1
        )

        self.vlm_client = anthropic.Anthropic()

    def _cam_cb(self, msg):
        self.camera_msg = msg

    def _scan_cb(self, msg):
        self.scan_msg = msg

    # ===== cmd_vel 헬퍼 =====
    def _pub_cmd(self, linear_x=0.0, angular_z=0.0):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x = linear_x
        msg.twist.angular.z = angular_z
        self.cmd_pub.publish(msg)

    def _stop(self):
        self._pub_cmd(0.0, 0.0)

    def _rotate(self, angular_z, duration):
        end_time = time.time() + duration

        while time.time() < end_time:
            self._pub_cmd(0.0, angular_z)
            time.sleep(0.05)

        self._stop()
        time.sleep(0.3)

    # ===== LiDAR 거리 측정 =====
    def _get_front_distance(self):
    """LiDAR 정면 기준 ±15도 범위의 최소 거리 반환"""

    self.scan_msg = None

    for _ in range(20):
        rclpy.spin_once(self, timeout_sec=0.1)
        if self.scan_msg is not None:
            break

    if self.scan_msg is None:
        self.get_logger().warn('scan_msg 자체를 수신하지 못함')
        return None

    scan = self.scan_msg

    if len(scan.ranges) == 0:
        self.get_logger().warn('scan.ranges가 비어 있음')
        return None

    if scan.angle_increment == 0.0:
        self.get_logger().warn('scan.angle_increment가 0임')
        return None

    # 디버그용 출력
    self.get_logger().info(
        f'/scan 수신: ranges={len(scan.ranges)}, '
        f'angle_min={scan.angle_min:.3f}, '
        f'angle_max={scan.angle_max:.3f}, '
        f'angle_increment={scan.angle_increment:.5f}, '
        f'range_min={scan.range_min:.3f}, '
        f'range_max={scan.range_max:.3f}'
    )

    # 정면 각도 0 rad에 해당하는 index 계산
    center_index = int(round((0.0 - scan.angle_min) / scan.angle_increment))

    # 정면 ±15도 범위 사용
    angle_range = math.radians(15.0)
    index_range = int(abs(angle_range / scan.angle_increment))

    valid = []

    for i in range(center_index - index_range, center_index + index_range + 1):
        if i < 0 or i >= len(scan.ranges):
            continue

        r = scan.ranges[i]

        if math.isfinite(r) and scan.range_min < r < scan.range_max:
            valid.append(r)

    if not valid:
        self.get_logger().warn(
            f'정면 ±15도 범위에서 유효한 거리 없음. center_index={center_index}'
        )
        return None

    return min(valid)

    def _wait_lidar_distance_after_alignment(self):
        """VLM 정렬 성공 후 LiDAR 전방 거리 확인"""
        self.get_logger().info('===== 정렬 후 LiDAR 거리 확인 =====')

        while True:
            dist = self._get_front_distance()

            if dist is None:
                self.get_logger().warn('LiDAR 전방 거리 측정 실패, 다시 확인 중...')
                time.sleep(0.5)
                continue

            self.get_logger().info(
                f'정렬 후 LiDAR 전방 거리 확인 완료: {dist:.3f}m ({dist * 100:.1f}cm)'
            )

            return dist

    # ===== 카메라 + VLM =====
    def _get_image_b64(self):
        self.camera_msg = None

        for _ in range(50):
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.camera_msg is not None:
                break

        if self.camera_msg is None:
            return None

        msg = self.camera_msg

        raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height,
            msg.width,
            -1
        )

        if msg.encoding == 'rgb8':
            bgr = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
        elif msg.encoding == 'bgr8':
            bgr = raw
        else:
            bgr = raw

        success, jpeg = cv2.imencode(
            '.jpg',
            bgr,
            [cv2.IMWRITE_JPEG_QUALITY, 85]
        )

        if not success:
            return None

        return base64.b64encode(jpeg.tobytes()).decode('utf-8')

    def _ask_vlm(self, b64, question):
        response = self.vlm_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=256,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": question
                        },
                    ],
                }
            ],
        )

        return response.content[0].text

    # ===== Nav2 컨트롤러 lifecycle =====
    def _deactivate_controller(self):
        self.get_logger().info('Nav2 controller_server 비활성화 중...')

        client = self.create_client(
            ChangeState,
            '/controller_server/change_state'
        )

        if not client.wait_for_service(timeout_sec=3.0):
            self.get_logger().warn('controller_server change_state 서비스 없음, 계속 진행')
            return

        req = ChangeState.Request()
        req.transition = Transition()
        req.transition.id = Transition.TRANSITION_DEACTIVATE

        future = client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)

        if future.result() is not None and future.result().success:
            self.get_logger().info('controller_server 비활성화 성공')
        else:
            self.get_logger().warn('controller_server 비활성화 실패 또는 응답 없음')

        time.sleep(0.5)

    def _reactivate_controller(self):
        self.get_logger().info('Nav2 controller_server 다시 활성화 시도...')

        client = self.create_client(
            ChangeState,
            '/controller_server/change_state'
        )

        if not client.wait_for_service(timeout_sec=3.0):
            self.get_logger().warn('controller_server change_state 서비스 없음')
            return

        req = ChangeState.Request()
        req.transition = Transition()
        req.transition.id = Transition.TRANSITION_ACTIVATE

        future = client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)

        if future.result() is not None and future.result().success:
            self.get_logger().info('controller_server 다시 활성화 성공')
        else:
            self.get_logger().warn('controller_server 다시 활성화 실패 또는 응답 없음')

    # ===== 1단계: Nav2 이동 =====
    def go_to_bell(self):
        self.get_logger().info('===== 1단계: Nav2로 벨 앞 목표 지점 이동 =====')
        self.get_logger().info('Nav2 navigate_to_pose action server 대기 중...')

        if not self.nav_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('Nav2 navigate_to_pose action server 연결 실패')
            return False

        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()

        goal.pose.pose.position.x = GOAL_X
        goal.pose.pose.position.y = GOAL_Y
        goal.pose.pose.position.z = 0.0

        goal.pose.pose.orientation.z = BELL_ORIENT_Z
        goal.pose.pose.orientation.w = BELL_ORIENT_W

        self.get_logger().info(
            f'Nav2 목표 전송: x={GOAL_X:.3f}, y={GOAL_Y:.3f}, yaw={math.degrees(YAW):.1f}도'
        )

        send_goal_future = self.nav_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_goal_future)

        goal_handle = send_goal_future.result()

        if goal_handle is None:
            self.get_logger().error('Nav2 goal handle 수신 실패')
            return False

        if not goal_handle.accepted:
            self.get_logger().error('Nav2 목표가 거부됨')
            return False

        self.get_logger().info('Nav2 목표 수락됨. 이동 시작...')

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result = result_future.result()

        if result is None:
            self.get_logger().error('Nav2 결과 수신 실패')
            return False

        status = result.status

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Nav2 이동 성공. 벨 앞 목표 지점 도착 확인 완료.')
            self._stop()

            # Nav2가 성공했을 때만 컨트롤러 비활성화
            # 이후 직접 cmd_vel로 VLM 정렬/접근 수행
            self._deactivate_controller()
            return True

        self.get_logger().error(f'Nav2 이동 실패. status={status}')

        if status == GoalStatus.STATUS_CANCELED:
            self.get_logger().error('상태: CANCELED')
        elif status == GoalStatus.STATUS_ABORTED:
            self.get_logger().error('상태: ABORTED')
        elif status == GoalStatus.STATUS_UNKNOWN:
            self.get_logger().error('상태: UNKNOWN')
        else:
            self.get_logger().error('상태: SUCCEEDED가 아니므로 VLM 정렬을 시작하지 않음')

        self._stop()
        return False

    # ===== 2단계: VLM 정렬 =====
    def align_to_bell(self):
        self.get_logger().info('===== 2단계: VLM으로 벨 찾기 및 정렬 =====')
        self.get_logger().info('벨 탐색 횟수 제한 없음. 중앙에 올 때까지 계속 탐색합니다.')

        attempt = 0

        while rclpy.ok():
            attempt += 1
            self.get_logger().info(f'정렬 시도 {attempt}')

            dist = self._get_front_distance()
            if dist is not None:
                self.get_logger().info(
                    f'현재 LiDAR 전방 거리: {dist:.3f}m ({dist * 100:.1f}cm)'
                )
            else:
                self.get_logger().warn('현재 LiDAR 전방 거리 측정 실패')

            b64 = self._get_image_b64()

            if b64 is None:
                self.get_logger().error('카메라 이미지 수신 실패. 재시도...')
                time.sleep(1.0)
                continue

            answer = self._ask_vlm(
                b64,
                "이 이미지에 빨간색 벨 또는 둥근 빨간 버튼이 보이나요?\n"
                "반드시 다음 형식 중 하나로만 답하세요:\n"
                "1) 안 보임\n"
                "2) 보임-왼쪽\n"
                "3) 보임-중앙\n"
                "4) 보임-오른쪽\n"
                "번호와 해당 텍스트만 답하세요."
            )

            self.get_logger().info(f'VLM 응답: {answer}')

            if '중앙' in answer or '3)' in answer or '3' == answer.strip():
                self._stop()
                self.get_logger().info('벨이 화면 중앙에 있음. VLM 정렬 완료.')
                return True

            elif '왼쪽' in answer or '2)' in answer or '2' == answer.strip():
                self.get_logger().info('벨이 왼쪽에 있음 → 반시계 방향으로 회전')
                self._rotate(0.3, 0.5)

            elif '오른쪽' in answer or '4)' in answer or '4' == answer.strip():
                self.get_logger().info('벨이 오른쪽에 있음 → 시계 방향으로 회전')
                self._rotate(-0.3, 0.5)

            else:
                self.get_logger().info('벨이 안 보임 → 반시계 방향으로 탐색 회전')
                self._rotate(0.4, 0.8)

            time.sleep(0.5)

        self._stop()
        return False

    # ===== 3단계: LiDAR 기반 정밀 접근 =====
    def approach_with_lidar(self):
        self.get_logger().info('===== 3단계: LiDAR 기반 정밀 접근 =====')

        while rclpy.ok():
            dist = self._get_front_distance()

            if dist is None:
                self.get_logger().warn('LiDAR 거리 측정 실패. 재시도...')
                self._stop()
                time.sleep(0.5)
                continue

            remaining = dist - TARGET_DISTANCE

            self.get_logger().info(
                f'전방 거리: {dist * 100:.1f}cm | '
                f'목표 거리: {TARGET_DISTANCE * 100:.0f}cm | '
                f'남은 거리: {remaining * 100:.1f}cm'
            )

            # 이미 충분히 가까운 경우
            if remaining <= STOP_TOLERANCE:
                self._stop()
                self.get_logger().info(
                    f'목표 거리 도달. 현재 전방 거리: {dist * 100:.1f}cm'
                )
                break

            # 남은 거리에 따라 속도 조절
            if remaining > 0.20:
                speed = 0.08
            elif remaining > 0.10:
                speed = 0.05
            else:
                speed = 0.03

            self._pub_cmd(speed, 0.0)
            time.sleep(0.1)

        self._stop()

        # 최종 VLM 확인
        b64 = self._get_image_b64()

        if b64:
            answer = self._ask_vlm(
                b64,
                "빨간색 벨 또는 둥근 빨간 버튼이 가까이 보이나요? 간단히 답해주세요."
            )

            final_dist = self._get_front_distance()

            print("\n=== 최종 상태 ===")
            if final_dist is not None:
                print(f"LiDAR 실측 거리: {final_dist * 100:.1f}cm")
            else:
                print("LiDAR 실측 거리: 측정 실패")

            print(f"VLM 판단: {answer}")


def main():
    rclpy.init()
    node = BellNavigator()

    nav2_success = False

    try:
        # 1단계: Nav2 성공 확인
        nav2_success = node.go_to_bell()

        if not nav2_success:
            node.get_logger().error('Nav2가 성공하지 않았으므로 벨 탐색을 시작하지 않습니다.')
            return

        # 2단계: VLM으로 벨 찾기/정렬
        aligned = node.align_to_bell()

        if not aligned:
            node.get_logger().error('VLM 정렬 실패 또는 중단')
            return

        # 정렬 후 LiDAR 거리 확인
        node._wait_lidar_distance_after_alignment()

        # 3단계: LiDAR 거리 기반 정밀 접근
        node.approach_with_lidar()

        print("\n=== 전체 미션 완료 ===")

    except KeyboardInterrupt:
        node._stop()
        print('\n긴급 정지')

    finally:
        node._stop()

        if nav2_success:
            node._reactivate_controller()

        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
