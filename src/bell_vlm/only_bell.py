#!/usr/bin/env python3
"""only_bell.py - VLM 정렬 + LiDAR 단계적 접근 (Nav2 없음)

로봇이 이미 벨 근처(약 50cm)에 있다고 가정하고, 그 지점부터 시작합니다.

작동 흐름:
1) 현재 위치(약 50cm)에서 VLM 정렬
2) 15cm 지점까지 LiDAR 기반 전진 후 다시 VLM 정렬
3) 10cm 지점까지 LiDAR 기반 전진 후 다시 VLM 정렬
4) 버튼 누르기용 짧은 전진(10cm)

실행:
    python3 only_bell.py

환경변수:
    ANTHROPIC_API_KEY 필요 (터미널마다 export, 따옴표 없이)

LiDAR 방향 보정:
    실제 로봇 정면 거리는 LaserScan 기준 LIDAR_FRONT_ANGLE 방향에서 측정합니다.
"""

import math
import time
import base64
from enum import Enum

import cv2
import numpy as np
import anthropic

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import TwistStamped
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy


# =====================================================================
# VLM 설정
# =====================================================================
VLM_MODEL = "claude-sonnet-4-6"
VLM_MAX_TOKENS = 256
JPEG_QUALITY = 85

BELL_DETECT_PROMPT = (
    "이 이미지에 빨간색 벨 또는 둥근 빨간 버튼이 보이나요?\n"
    "반드시 다음 형식 중 하나로만 답하세요:\n"
    "1) 안 보임\n"
    "2) 보임-왼쪽\n"
    "3) 보임-중앙\n"
    "4) 보임-오른쪽\n"
    "번호와 해당 텍스트만 답하세요."
)

BELL_NEAR_PROMPT = (
    "빨간색 벨 또는 둥근 빨간 버튼이 가까이 보이나요? 간단히 답해주세요."
)


class BellPosition(Enum):
    """벨의 화면상 위치 판단 결과"""
    NOT_FOUND = "not_found"
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


class BellVLM:
    """순수 VLM 로직 (ROS 노드에 의존하지 않음)"""

    def __init__(self, model=VLM_MODEL, logger=None):
        self.client = anthropic.Anthropic()
        self.model = model
        self._logger = logger

    def _log(self, msg):
        if self._logger is not None:
            self._logger.info(msg)
        else:
            print(msg)

    # ----- 이미지 변환 -----
    @staticmethod
    def ros_image_to_bgr(msg):
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

        return bgr

    @staticmethod
    def bgr_to_b64(bgr):
        success, jpeg = cv2.imencode(
            '.jpg',
            bgr,
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        )

        if not success:
            return None

        return base64.b64encode(jpeg.tobytes()).decode('utf-8')

    def ros_image_to_b64(self, msg):
        bgr = self.ros_image_to_bgr(msg)
        return self.bgr_to_b64(bgr)

    # ----- VLM 호출 -----
    def ask(self, b64, question):
        response = self.client.messages.create(
            model=self.model,
            max_tokens=VLM_MAX_TOKENS,
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
                            "text": question,
                        },
                    ],
                }
            ],
        )

        return response.content[0].text

    # ----- 벨 위치 판단 -----
    @staticmethod
    def parse_bell_position(answer):
        stripped = answer.strip()

        if '중앙' in answer or '3)' in answer or stripped == '3':
            return BellPosition.CENTER

        if '왼쪽' in answer or '2)' in answer or stripped == '2':
            return BellPosition.LEFT

        if '오른쪽' in answer or '4)' in answer or stripped == '4':
            return BellPosition.RIGHT

        return BellPosition.NOT_FOUND

    def detect_bell(self, b64):
        if b64 is None:
            return None, None

        answer = self.ask(b64, BELL_DETECT_PROMPT)
        self._log(f'VLM 응답: {answer}')

        position = self.parse_bell_position(answer)
        return position, answer

    def confirm_bell_near(self, b64):
        if b64 is None:
            return None
        return self.ask(b64, BELL_NEAR_PROMPT)


# =====================================================================
# 접근 설정
# =====================================================================
# 단계적 접근 거리: (목표 LiDAR 거리[m], 단계 이름)
# 첫 단계는 현재 위치(약 50cm)에서 전진 없이 정렬만 합니다.
APPROACH_STAGES = [
    (0.50, '50cm 정렬'),   # 정렬만
    (0.35, '35cm 접근'),   # 35cm까지 전진 후 정렬
    (0.25, '25cm 접근'),   # 25cm까지 전진 후 정렬
]

STOP_TOLERANCE = 0.02   # 2cm 오차 허용

# 마지막 버튼 누르기용 짧은 전진
FINAL_PUSH_DISTANCE = 0.10  # 10cm
FINAL_PUSH_SPEED = 0.02     # m/s
FINAL_PUSH_TIME = FINAL_PUSH_DISTANCE / FINAL_PUSH_SPEED

# LiDAR 방향 보정
LIDAR_FRONT_ANGLE = 0.0
LIDAR_FRONT_WINDOW_DEG = 15.0


print("===== VLM 정렬 + LiDAR 단계적 접근 (Nav2 없음) =====")
print("단계적 접근:")
for _d, _n in APPROACH_STAGES:
    print(f"  - {_n}: 목표 거리 {_d * 100:.0f}cm")
print(f"최종 버튼 누르기 전진: {FINAL_PUSH_DISTANCE * 100:.0f}cm "
      f"({FINAL_PUSH_SPEED}m/s × {FINAL_PUSH_TIME:.1f}s)")
print(f"LiDAR 정면 기준 각도: {math.degrees(LIDAR_FRONT_ANGLE):.1f}도")
print(f"LiDAR 정면 판정 범위: ±{LIDAR_FRONT_WINDOW_DEG:.1f}도")


# =====================================================================
# 메인 제어 노드
# =====================================================================
class BellController(Node):
    def __init__(self):
        super().__init__('bell_controller')

        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)

        self.camera_msg = None
        self.scan_msg = None

        self.cam_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self._cam_cb,
            1
        )

        scan_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self._scan_cb,
            scan_qos
        )

        self.vlm = BellVLM(logger=self.get_logger())

    def _cam_cb(self, msg):
        self.camera_msg = msg

    def _scan_cb(self, msg):
        self.scan_msg = msg

    # ----- cmd_vel 헬퍼 -----
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

        while time.time() < end_time and rclpy.ok():
            self._pub_cmd(0.0, angular_z)
            time.sleep(0.05)

        self._stop()
        time.sleep(0.3)

    # ----- 각도 처리 헬퍼 -----
    def _normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def _angle_diff(self, a, b):
        return abs(self._normalize_angle(a - b))

    # ----- LiDAR 거리 측정 -----
    def _get_front_distance(self):
        """실제 로봇 정면 기준 ±15도 범위의 최소 거리 반환"""
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

        target_angle = LIDAR_FRONT_ANGLE
        angle_window = math.radians(LIDAR_FRONT_WINDOW_DEG)

        valid = []

        for i, r in enumerate(scan.ranges):
            if not math.isfinite(r):
                continue

            if not (scan.range_min < r < scan.range_max):
                continue

            angle = scan.angle_min + i * scan.angle_increment

            if self._angle_diff(angle, target_angle) <= angle_window:
                valid.append(r)

        if not valid:
            self.get_logger().warn(
                f'정면 ±{LIDAR_FRONT_WINDOW_DEG:.1f}도 범위에서 유효한 거리 없음'
            )
            return None

        front_dist = min(valid)

        self.get_logger().info(
            f'[LiDAR] 정면 거리: {front_dist:.3f}m ({front_dist * 100:.1f}cm)'
        )

        return front_dist

    # ----- 카메라 → base64 -----
    def _get_image_b64(self):
        self.camera_msg = None

        for _ in range(50):
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.camera_msg is not None:
                break

        if self.camera_msg is None:
            return None

        return self.vlm.ros_image_to_b64(self.camera_msg)

    # ----- VLM 정렬 (단계용) -----
    def align_to_bell(self, stage_name=''):
        """벨이 화면 중앙에 올 때까지 VLM으로 정렬"""
        self.get_logger().info(f'----- VLM 정렬 시작 ({stage_name}) -----')

        attempt = 0

        while rclpy.ok():
            attempt += 1
            self.get_logger().info(f'[{stage_name}] 정렬 시도 {attempt}')

            dist = self._get_front_distance()
            if dist is not None:
                self.get_logger().info(
                    f'[{stage_name}] 현재 정면 거리: {dist * 100:.1f}cm'
                )

            b64 = self._get_image_b64()

            if b64 is None:
                self.get_logger().error('카메라 이미지 수신 실패. 재시도...')
                time.sleep(1.0)
                continue

            position, answer = self.vlm.detect_bell(b64)

            if position == BellPosition.CENTER:
                self._stop()
                self.get_logger().info(
                    f'[{stage_name}] 벨이 화면 중앙에 있음. 정렬 완료.'
                )
                return True

            elif position == BellPosition.LEFT:
                self.get_logger().info(
                    f'[{stage_name}] 벨이 왼쪽 → 반시계 방향 회전'
                )
                self._rotate(0.3, 0.5)

            elif position == BellPosition.RIGHT:
                self.get_logger().info(
                    f'[{stage_name}] 벨이 오른쪽 → 시계 방향 회전'
                )
                self._rotate(-0.3, 0.5)

            else:
                self.get_logger().info(
                    f'[{stage_name}] 벨이 안 보임 → 반시계 방향 탐색 회전'
                )
                self._rotate(0.4, 0.8)

            time.sleep(0.5)

        self._stop()
        return False

    # ----- LiDAR 기반 목표 거리까지 전진 -----
    def approach_to_distance(self, target_distance, stage_name=''):
        """LiDAR 정면 거리가 target_distance가 될 때까지 전진"""
        self.get_logger().info(
            f'----- LiDAR 접근 시작 ({stage_name}): '
            f'목표 {target_distance * 100:.0f}cm -----'
        )

        while rclpy.ok():
            dist = self._get_front_distance()

            if dist is None:
                self.get_logger().warn('LiDAR 거리 측정 실패. 재시도...')
                self._stop()
                time.sleep(0.5)
                continue

            remaining = dist - target_distance

            self.get_logger().info(
                f'[{stage_name}] 정면 거리: {dist * 100:.1f}cm | '
                f'목표: {target_distance * 100:.0f}cm | '
                f'남음: {remaining * 100:.1f}cm'
            )

            if remaining <= STOP_TOLERANCE:
                self._stop()
                self.get_logger().info(
                    f'[{stage_name}] 목표 거리 도달. 현재: {dist * 100:.1f}cm'
                )
                return True

            if remaining > 0.20:
                speed = 0.06
            elif remaining > 0.10:
                speed = 0.03
            else:
                speed = 0.02

            self._pub_cmd(speed, 0.0)
            time.sleep(0.1)

        self._stop()
        return False

    # ----- 버튼 누르기용 짧은 전진 -----
    def final_push(self):
        dist_before = self._get_front_distance()
        if dist_before is not None:
            self.get_logger().info(
                f'버튼 누르기 전진 직전 거리: {dist_before * 100:.1f}cm'
            )

        self.get_logger().info(
            f'----- 버튼 누르기 전진: '
            f'{FINAL_PUSH_DISTANCE * 100:.0f}cm '
            f'({FINAL_PUSH_SPEED}m/s × {FINAL_PUSH_TIME:.1f}s) -----'
        )

        push_end_time = time.time() + FINAL_PUSH_TIME
        next_log_time = time.time()

        while time.time() < push_end_time and rclpy.ok():
            self._pub_cmd(FINAL_PUSH_SPEED, 0.0)

            if time.time() >= next_log_time:
                remain_t = push_end_time - time.time()
                self.get_logger().info(f'전진 중... 남은 시간 {remain_t:.1f}s')
                next_log_time = time.time() + 0.5

            time.sleep(0.05)

        self._stop()

        dist_after = self._get_front_distance()
        if dist_after is not None:
            self.get_logger().info(
                f'버튼 누르기 전진 직후 거리: {dist_after * 100:.1f}cm'
            )
            if dist_before is not None:
                moved = (dist_before - dist_after) * 100
                self.get_logger().info(f'실제 전진 거리(추정): {moved:.1f}cm')

        self.get_logger().info('짧은 전진 완료. 정지.')

    # ----- 단계적 정렬 + 접근 전체 -----
    def run_staged_approach(self):
        """50cm 정렬 → 15cm 접근/정렬 → 10cm 접근/정렬 → 버튼 누르기"""
        self.get_logger().info('===== 단계적 VLM 정렬 + LiDAR 접근 시작 =====')

        for idx, (target_dist, stage_name) in enumerate(APPROACH_STAGES):
            self.get_logger().info(
                f'### 단계 {idx + 1}/{len(APPROACH_STAGES)}: {stage_name} ###'
            )

            # 첫 단계(50cm)는 현재 위치이므로 전진 없이 정렬만.
            if idx > 0:
                ok = self.approach_to_distance(target_dist, stage_name)
                if not ok:
                    self.get_logger().error(f'[{stage_name}] 접근 중단')
                    return False

            aligned = self.align_to_bell(stage_name)
            if not aligned:
                self.get_logger().error(f'[{stage_name}] 정렬 실패 또는 중단')
                return False

        self.final_push()

        # 최종 VLM 확인
        b64 = self._get_image_b64()

        if b64:
            answer = self.vlm.confirm_bell_near(b64)
            final_dist = self._get_front_distance()

            print("\n=== 최종 상태 ===")
            if final_dist is not None:
                print(f"LiDAR 정면 거리: {final_dist * 100:.1f}cm")
            else:
                print("LiDAR 정면 거리: 측정 실패")

            print(f"VLM 판단: {answer}")

        return True


def main():
    rclpy.init()
    node = BellController()

    try:
        success = node.run_staged_approach()

        if success:
            print("\n=== 전체 미션 완료 ===")
        else:
            print("\n=== 미션 중단 ===")

    except KeyboardInterrupt:
        node._stop()
        print('\n긴급 정지')

    finally:
        node._stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
