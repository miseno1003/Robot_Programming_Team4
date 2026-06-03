#!/usr/bin/env python3
"""only_bell.py - 빨간색 검출 기반 정렬 + LiDAR 단계적 접근 (Nav2 없음)

로봇이 이미 벨 근처(약 50cm)에 있다고 가정합니다.

정렬 방식:
- 주력: OpenCV 빨간색 HSV 마스킹으로 벨 중심 x좌표를 픽셀 단위로 검출,
  화면 중앙과의 오차에 비례해 회전 (비례 제어)
- 보조: VLM은 빨강 미검출 시 "빨간 벨이 화면에 있는지" 한 번 확인하는 용도

단계별 정렬 강도:
- 50cm, 35cm 단계: 느슨한 정렬 (대략 중앙에 오면 다음 단계로)
- 25cm 단계: 정밀 정렬 (오차 작게 + 연속 N회 중앙 확인)

작동 흐름:
1) 현재 위치(약 50cm)에서 정렬
2) 35cm 지점까지 LiDAR 기반 전진 후 다시 정렬
3) 25cm 지점까지 LiDAR 기반 전진 후 정밀 정렬
4) 버튼 누르기용 짧은 전진(10cm)

실행:
    python3 only_bell.py

환경변수:
    ANTHROPIC_API_KEY 필요 (VLM 보조 확인용, 터미널마다 export)
"""

import math
import time
import base64

import cv2
import numpy as np
import anthropic

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import TwistStamped
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy


# =====================================================================
# 빨간색 검출 설정 (HSV)
# =====================================================================
# 빨간색은 HSV 색상환에서 0도 근처와 180도 근처 양쪽에 걸쳐 있어서
# 두 범위를 따로 마스킹한 뒤 합칩니다.
# H: 0~179, S: 0~255, V: 0~255 (OpenCV 기준)
RED_LOWER1 = np.array([0, 100, 80])      # 빨강 하단 (0도 쪽)
RED_UPPER1 = np.array([10, 255, 255])
RED_LOWER2 = np.array([160, 100, 80])    # 빨강 상단 (180도 쪽)
RED_UPPER2 = np.array([179, 255, 255])

# 노이즈로 무시할 최소 빨간 영역 픽셀 수
MIN_RED_AREA = 150

# 비례 회전 제어 게인 (error에 곱해서 angular_z 생성)
ROTATE_GAIN = 0.5
# 회전 속도 상/하한 (rad/s)
ROTATE_MAX = 0.4
ROTATE_MIN = 0.08

# 정밀 정렬 모드의 더 낮은 회전 속도 (느리고 정확하게)
ROTATE_MAX_FINE = 0.20
ROTATE_MIN_FINE = 0.05


# =====================================================================
# VLM 설정 (보조 확인용)
# =====================================================================
VLM_MODEL = "claude-sonnet-4-6"
VLM_MAX_TOKENS = 128
JPEG_QUALITY = 85

BELL_EXIST_PROMPT = (
    "이 이미지에 빨간색 벨 또는 둥근 빨간 버튼이 보이나요? "
    "'예' 또는 '아니오'로만 답하세요."
)

BELL_NEAR_PROMPT = (
    "빨간색 벨 또는 둥근 빨간 버튼이 가까이 보이나요? 간단히 답해주세요."
)


class BellVLM:
    """VLM 보조 로직 (벨 존재 확인, 근접 확인)"""

    def __init__(self, model=VLM_MODEL, logger=None):
        self.client = anthropic.Anthropic()
        self.model = model
        self._logger = logger

    def _log(self, msg):
        if self._logger is not None:
            self._logger.info(msg)
        else:
            print(msg)

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

    def bell_exists(self, bgr):
        """이미지에 빨간 벨이 있는지 VLM으로 확인 (True/False)"""
        b64 = self.bgr_to_b64(bgr)
        if b64 is None:
            return False

        answer = self.ask(b64, BELL_EXIST_PROMPT)
        self._log(f'VLM 벨 존재 확인: {answer}')
        return '예' in answer or 'yes' in answer.lower()

    def confirm_bell_near(self, bgr):
        b64 = self.bgr_to_b64(bgr)
        if b64 is None:
            return None
        return self.ask(b64, BELL_NEAR_PROMPT)


# =====================================================================
# 빨간색 검출기
# =====================================================================
class RedBellDetector:
    """이미지에서 빨간색 벨의 중심 위치를 검출"""

    def __init__(self, logger=None):
        self._logger = logger

    def _log(self, msg):
        if self._logger is not None:
            self._logger.info(msg)
        else:
            print(msg)

    def detect(self, bgr):
        """빨간색 영역 검출

        반환:
            (found, error, area, debug)
            found: 빨간 벨 검출 여부 (bool)
            error: 화면 중앙 기준 수평 오차 (-1.0=완전 왼쪽, +1.0=완전 오른쪽)
            area:  검출된 빨간 영역 픽셀 수
            debug: (cx, cy, width) 디버그 정보
        """
        h, w = bgr.shape[:2]

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(hsv, RED_LOWER1, RED_UPPER1)
        mask2 = cv2.inRange(hsv, RED_LOWER2, RED_UPPER2)
        mask = cv2.bitwise_or(mask1, mask2)

        # 노이즈 제거 (모폴로지 열기/닫기)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 가장 큰 빨간 덩어리 찾기
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return False, 0.0, 0, None

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area < MIN_RED_AREA:
            return False, 0.0, int(area), None

        # 무게중심 계산
        M = cv2.moments(largest)
        if M['m00'] == 0:
            return False, 0.0, int(area), None

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        # 화면 중앙 기준 정규화 오차 (-1.0 ~ 1.0)
        center_x = w / 2.0
        error = (cx - center_x) / (w / 2.0)

        return True, error, int(area), (cx, cy, w)


# =====================================================================
# 접근 설정
# =====================================================================
# 단계 설정: dict 리스트
#   target   : 목표 LiDAR 거리[m] (첫 단계는 현재 위치이므로 전진 안 함)
#   name     : 단계 이름
#   tol      : 중앙 정렬 허용 오차 (정규화 -1.0~1.0, 작을수록 엄격)
#   confirm  : 연속으로 중앙 판정을 몇 번 받아야 정렬 완료로 볼지
#   fine     : 정밀 모드 여부 (회전 속도를 더 낮춤)
APPROACH_STAGES = [
    {
        'target': 0.50, 'name': '50cm 정렬',
        'tol': 0.10, 'confirm': 1, 'fine': False,
    },
    {
        'target': 0.35, 'name': '35cm 접근',
        'tol': 0.08, 'confirm': 1, 'fine': False,
    },
    {
        'target': 0.25, 'name': '25cm 정밀정렬',
        'tol': 0.03, 'confirm': 3, 'fine': True,   # 가장 엄격
    },
]

STOP_TOLERANCE = 0.02   # LiDAR 접근 거리 2cm 오차 허용

FINAL_PUSH_DISTANCE = 0.10  # 10cm
FINAL_PUSH_SPEED = 0.02     # m/s
FINAL_PUSH_TIME = FINAL_PUSH_DISTANCE / FINAL_PUSH_SPEED

# LiDAR 방향 보정
LIDAR_FRONT_ANGLE = 0.0
LIDAR_FRONT_WINDOW_DEG = 15.0

# 정렬 시 한 번 회전 명령을 유지하는 시간(초)
ALIGN_STEP_TIME = 0.15
ALIGN_STEP_TIME_FINE = 0.10   # 정밀 모드는 짧게 끊어서 미세 조정


print("===== 빨간 벨 검출 정렬 + LiDAR 단계적 접근 (Nav2 없음) =====")
print("단계적 접근:")
for _s in APPROACH_STAGES:
    print(f"  - {_s['name']}: 목표 {_s['target'] * 100:.0f}cm, "
          f"허용오차 ±{_s['tol'] * 100:.0f}%, "
          f"확인 {_s['confirm']}회, "
          f"{'정밀' if _s['fine'] else '일반'} 모드")
print(f"최종 버튼 누르기 전진: {FINAL_PUSH_DISTANCE * 100:.0f}cm "
      f"({FINAL_PUSH_SPEED}m/s × {FINAL_PUSH_TIME:.1f}s)")
print(f"빨간색 검출 최소 영역: {MIN_RED_AREA}px")
print(f"LiDAR 정면 기준 각도: {math.degrees(LIDAR_FRONT_ANGLE):.1f}도")


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

        self.detector = RedBellDetector(logger=self.get_logger())
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
        time.sleep(0.2)

    # ----- 각도 처리 헬퍼 -----
    def _normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def _angle_diff(self, a, b):
        return abs(self._normalize_angle(a - b))

    # ----- LiDAR 거리 측정 -----
    def _get_front_distance(self):
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

    # ----- 카메라 프레임 받기 (BGR) -----
    def _get_bgr(self):
        """최신 카메라 프레임을 BGR ndarray로 반환"""
        self.camera_msg = None

        for _ in range(50):
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.camera_msg is not None:
                break

        if self.camera_msg is None:
            return None

        return self.vlm.ros_image_to_bgr(self.camera_msg)

    # ----- 빨간색 검출 기반 정렬 -----
    def align_to_bell(self, stage):
        """빨간 벨의 무게중심을 화면 중앙에 맞추도록 비례 회전 정렬

        stage: dict (target, name, tol, confirm, fine)
        성공 시 True, 중단 시 False
        """
        name = stage['name']
        tol = stage['tol']
        need_confirm = stage['confirm']
        fine = stage['fine']

        rot_max = ROTATE_MAX_FINE if fine else ROTATE_MAX
        rot_min = ROTATE_MIN_FINE if fine else ROTATE_MIN
        step_time = ALIGN_STEP_TIME_FINE if fine else ALIGN_STEP_TIME

        self.get_logger().info(
            f'----- 정렬 시작 ({name}) | '
            f'허용오차 ±{tol * 100:.0f}%, 확인 {need_confirm}회, '
            f'{"정밀" if fine else "일반"} 모드 -----'
        )

        attempt = 0
        not_found_count = 0
        center_streak = 0   # 연속 중앙 판정 횟수
        vlm_checked = False

        while rclpy.ok():
            attempt += 1

            bgr = self._get_bgr()

            if bgr is None:
                self.get_logger().error('카메라 이미지 수신 실패. 재시도...')
                time.sleep(0.5)
                continue

            found, error, area, debug = self.detector.detect(bgr)

            if found:
                not_found_count = 0
                cx = debug[0] if debug else -1

                # 중앙 허용 오차 안에 들어오는지
                if abs(error) <= tol:
                    center_streak += 1
                    self.get_logger().info(
                        f'[{name}] 중앙 근처: 중심x={cx}, 오차={error:+.3f} '
                        f'(연속 {center_streak}/{need_confirm})'
                    )

                    # 필요한 횟수만큼 연속으로 중앙이면 정렬 완료
                    if center_streak >= need_confirm:
                        self._stop()
                        self.get_logger().info(
                            f'[{name}] 정렬 완료. 최종 오차 {error:+.3f}'
                        )
                        return True

                    # 아직 확인 횟수 부족 → 잠깐 멈추고 다시 측정
                    self._stop()
                    time.sleep(0.15)
                    continue

                # 중앙 밖이면 연속 카운트 초기화
                center_streak = 0

                self.get_logger().info(
                    f'[{name}] 빨강 검출: 중심x={cx}, '
                    f'오차={error:+.3f}, 영역={area}px'
                )

                # 오차에 비례한 회전
                # error > 0 : 벨이 오른쪽 → 시계방향(angular_z < 0)
                # error < 0 : 벨이 왼쪽   → 반시계방향(angular_z > 0)
                angular = -ROTATE_GAIN * error

                sign = 1.0 if angular >= 0 else -1.0
                mag = min(abs(angular), rot_max)
                mag = max(mag, rot_min)
                angular = sign * mag

                direction = '반시계' if angular > 0 else '시계'
                self.get_logger().info(
                    f'[{name}] {direction} 회전 (angular_z={angular:+.3f})'
                )

                self._rotate(angular, step_time)

            else:
                not_found_count += 1
                center_streak = 0
                self.get_logger().info(
                    f'[{name}] 빨강 미검출 (영역={area}px, '
                    f'연속 {not_found_count}회)'
                )

                # 처음 미검출 시 VLM으로 한 번 확인
                if not vlm_checked:
                    vlm_checked = True
                    exists = self.vlm.bell_exists(bgr)
                    if exists:
                        self.get_logger().info(
                            f'[{name}] VLM: 벨 존재함. '
                            f'색 검출 임계값 확인 필요. 탐색 회전 진행.'
                        )
                    else:
                        self.get_logger().info(
                            f'[{name}] VLM: 벨 안 보임. 탐색 회전.'
                        )

                # 벨을 찾기 위해 천천히 탐색 회전
                self._rotate(0.35, 0.4)

            time.sleep(0.15)

        self._stop()
        return False

    # ----- LiDAR 기반 목표 거리까지 전진 -----
    def approach_to_distance(self, target_distance, name=''):
        self.get_logger().info(
            f'----- LiDAR 접근 시작 ({name}): '
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
                f'[{name}] 정면 거리: {dist * 100:.1f}cm | '
                f'목표: {target_distance * 100:.0f}cm | '
                f'남음: {remaining * 100:.1f}cm'
            )

            if remaining <= STOP_TOLERANCE:
                self._stop()
                self.get_logger().info(
                    f'[{name}] 목표 거리 도달. 현재: {dist * 100:.1f}cm'
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
        self.get_logger().info('===== 단계적 정렬 + LiDAR 접근 시작 =====')

        for idx, stage in enumerate(APPROACH_STAGES):
            name = stage['name']
            self.get_logger().info(
                f'### 단계 {idx + 1}/{len(APPROACH_STAGES)}: {name} ###'
            )

            # 첫 단계는 현재 위치이므로 전진 없이 정렬만.
            if idx > 0:
                ok = self.approach_to_distance(stage['target'], name)
                if not ok:
                    self.get_logger().error(f'[{name}] 접근 중단')
                    return False

            aligned = self.align_to_bell(stage)
            if not aligned:
                self.get_logger().error(f'[{name}] 정렬 실패 또는 중단')
                return False

        self.final_push()

        # 최종 VLM 근접 확인
        bgr = self._get_bgr()
        if bgr is not None:
            answer = self.vlm.confirm_bell_near(bgr)
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
