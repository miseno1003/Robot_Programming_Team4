#!/usr/bin/env python3
"""벨 정면 이동 + VLM 정렬 + 10cm 전진 — 통합 버전"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from sensor_msgs.msg import Image
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
BELL_FRONT_X = -0.218
BELL_FRONT_Y = -0.157
BELL_ORIENT_Z = -0.9986528693935666
BELL_ORIENT_W = 0.05188878927086355

YAW = 2.0 * math.atan2(BELL_ORIENT_Z, BELL_ORIENT_W)

GOAL_X = BELL_FRONT_X - 0.3 * math.cos(YAW)
GOAL_Y = BELL_FRONT_Y - 0.3 * math.sin(YAW)

print(f"벨 앞 좌표: ({BELL_FRONT_X}, {BELL_FRONT_Y})")
print(f"로봇 yaw: {math.degrees(YAW):.1f}도")
print(f"목표 지점 (정면 30cm): ({GOAL_X:.3f}, {GOAL_Y:.3f})")


class BellNavigator(Node):
    def __init__(self):
        super().__init__('bell_navigator')
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.camera_msg = None
        self.cam_sub = self.create_subscription(
            Image, '/camera/image_raw', self._cam_cb, 1)
        self.vlm_client = anthropic.Anthropic()

    def _cam_cb(self, msg):
        self.camera_msg = msg

    # ===== cmd_vel 헬퍼 (TwistStamped) =====
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

    def _move_forward(self, distance, speed=0.05):
        self.get_logger().info(f'벨을 향해 {distance*100:.0f}cm 전진')
        end_time = time.time() + (distance / speed)
        while time.time() < end_time:
            self._pub_cmd(speed, 0.0)
            time.sleep(0.05)
        self._stop()
        time.sleep(0.5)
        self.get_logger().info('전진 완료!')

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
            msg.height, msg.width, -1)
        if msg.encoding == 'rgb8':
            bgr = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
        else:
            bgr = raw
        _, jpeg = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(jpeg.tobytes()).decode('utf-8')

    def _ask_vlm(self, b64, question):
        response = self.vlm_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=256,
            messages=[{
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
                    {"type": "text", "text": question},
                ],
            }],
        )
        return response.content[0].text

    # ===== Nav2 컨트롤러 lifecycle =====
    def _deactivate_controller(self):
        self.get_logger().info('Nav2 컨트롤러 비활성화 중...')
        client = self.create_client(
            ChangeState, '/controller_server/change_state')

        if not client.wait_for_service(timeout_sec=3.0):
            self.get_logger().warn('controller_server 서비스 없음, 계속 진행')
            return

        req = ChangeState.Request()
        req.transition = Transition()
        req.transition.id = Transition.TRANSITION_DEACTIVATE

        future = client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)

        if future.result() is not None and future.result().success:
            self.get_logger().info('컨트롤러 비활성화 성공!')
        else:
            self.get_logger().warn('컨트롤러 비활성화 실패, 계속 진행')
        time.sleep(0.5)

    def _reactivate_controller(self):
        client = self.create_client(
            ChangeState, '/controller_server/change_state')
        if not client.wait_for_service(timeout_sec=3.0):
            return
        req = ChangeState.Request()
        req.transition = Transition()
        req.transition.id = Transition.TRANSITION_ACTIVATE
        future = client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
        self.get_logger().info('컨트롤러 다시 활성화')

    # ===== 1단계: Nav2 이동 =====
    def go_to_bell(self):
        self.get_logger().info('===== 1단계: Nav2로 벨 근처 이동 =====')
        self.get_logger().info('Nav2 서버 대기 중...')
        self.nav_client.wait_for_server()

        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = GOAL_X
        goal.pose.pose.position.y = GOAL_Y
        goal.pose.pose.position.z = 0.0
        goal.pose.pose.orientation.z = BELL_ORIENT_Z
        goal.pose.pose.orientation.w = BELL_ORIENT_W

        self.get_logger().info(f'목표 전송: ({GOAL_X:.3f}, {GOAL_Y:.3f})')

        future = self.nav_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error('목표가 거부됨!')
            return False

        self.get_logger().info('목표 수락됨, 이동 중...')
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        status = result_future.result().status
        if status == 4:
            self.get_logger().info('Nav2 도착 완료!')
        else:
            self.get_logger().warn(
                f'Nav2 status={status}, 근처 도달로 간주하고 계속 진행')

        # Nav2에서 cmd_vel 제어권 가져오기
        self._deactivate_controller()
        return True

    # ===== 2단계: VLM 정렬 =====
    def align_to_bell(self):
        self.get_logger().info('===== 2단계: VLM으로 벨 정렬 =====')
        attempt = 0

        while True:
            attempt += 1
            self.get_logger().info(f'정렬 시도 {attempt}')

            b64 = self._get_image_b64()
            if b64 is None:
                self.get_logger().error('카메라 이미지 수신 실패, 재시도...')
                time.sleep(1.0)
                continue

            answer = self._ask_vlm(
                b64,
                "이 이미지에 빨간색 벨(둥근 버튼)이 보이나요? "
                "반드시 다음 형식으로만 답하세요:\n"
                "1) 안 보임\n"
                "2) 보임-왼쪽\n"
                "3) 보임-중앙\n"
                "4) 보임-오른쪽\n"
                "번호와 해당 텍스트만 답하세요."
            )
            self.get_logger().info(f'VLM 응답: {answer}')

            if '중앙' in answer or '3)' in answer:
                self.get_logger().info('벨이 화면 중앙! 정렬 완료!')
                return True
            elif '왼쪽' in answer or '2)' in answer:
                self.get_logger().info('왼쪽 → 반시계 회전')
                self._rotate(0.3, 0.5)
            elif '오른쪽' in answer or '4)' in answer:
                self.get_logger().info('오른쪽 → 시계 회전')
                self._rotate(-0.3, 0.5)
            else:
                self.get_logger().info('안 보임 → 반시계 탐색')
                self._rotate(0.4, 0.8)

            time.sleep(0.5)

    # ===== 3단계: 전진 + 최종 확인 =====
    def approach_and_confirm(self):
        self.get_logger().info('===== 3단계: 10cm 전진 + 최종 확인 =====')
        self._move_forward(0.10)

        b64 = self._get_image_b64()
        if b64:
            answer = self._ask_vlm(
                b64,
                "빨간색 벨이 가까이 보이나요? "
                "로봇이 벨을 누를 수 있을 만큼 가까운지 판단해주세요. "
                "간단히 답해주세요."
            )
            print(f"\n=== VLM 최종 확인 ===")
            print(answer)


def main():
    rclpy.init()
    node = BellNavigator()

    try:
        # 1단계: Nav2로 벨 근처 이동
        reached = node.go_to_bell()

        if reached:
            # 2단계: VLM으로 벨 정렬 (찾을 때까지)
            aligned = node.align_to_bell()

            if aligned:
                # 3단계: 10cm 전진 + 최종 확인
                node.approach_and_confirm()
                print("\n=== 전체 미션 완료! ===")

    except KeyboardInterrupt:
        node._stop()
        print('\n긴급 정지!')
    finally:
        node._reactivate_controller()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
