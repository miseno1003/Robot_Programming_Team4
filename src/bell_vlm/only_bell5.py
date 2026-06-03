#!/usr/bin/env python3
"""only_bell.py - VLM 벨 검증 게이트 + 픽셀 정렬 + LiDAR 단계 접근 (Nav2 없음)

로봇이 이미 벨 근처(약 50cm)에 있다고 가정합니다.

전체 흐름:
1) VLM 실행 → 벨이 보이는지/위치(왼쪽·중앙·오른쪽)를 판별 (로그 출력)
2) 화면의 빨간색을 픽셀로 검출하고, 그 빨간색이 '벨이 맞는지' VLM에게 검증
   2-1) VLM이 벨이라고 하면 → 정렬
   2-2) 벨이 아니라고 하면 → 왼쪽으로 회전하며 다른 빨간색 탐색
        → 찾을 때마다 VLM에게 다시 검증, 벨이라고 할 때까지 반복
3) 벽(벨)과 35cm까지 이동
4) 다시 VLM 검증: 벨이 맞으면 정렬, 아니면 탐색 후 VLM 재검증
5) 25cm까지 이동
6) 4단계와 동일하되 더 엄격하게 중앙 정렬
7) 버튼 누르기: 8cm 전진

로그:
- VLM 벨 탐지/미탐지 및 위치(왼쪽/중앙/오른쪽)
- 벽(벨)과의 남은 거리

실행:
    python3 only_bell.py

환경변수:
    ANTHROPIC_API_KEY 필요 (터미널마다 export, 따옴표 없이)
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
# 두 범위를 따로 마스킹한 뒤 합칩니다. (H:0~179, S:0~255, V:0~255)
RED_LOWER1 = np.array([0, 100, 80])
RED_UPPER1 = np.array([10, 255, 255])
RED_LOWER2 = np.array([160, 100, 80])
RED_UPPER2 = np.array([179, 255, 255])

# 노이즈로 무시할 최소 빨간 영역 픽셀 수
MIN_RED_AREA = 150

# 위치 라벨(왼쪽/중앙/오른쪽) 판정 임계값 (정규화 오차 기준)
POSITION_CENTER_BAND = 0.10

# 비례 회전 제어 게인 및 속도 상/하한 (rad/s)
ROTATE_GAIN = 0.5
ROTATE_MAX = 0.4
ROTATE_MIN = 0.08
# 정밀 모드 (느리고 정확하게)
ROTATE_MAX_FINE = 0.20
ROTATE_MIN_FINE = 0.05


# =====================================================================
# VLM 설정
# =====================================================================
VLM_MODEL = "claude-sonnet-4-6"
VLM_MAX_TOKENS = 128
JPEG_QUALITY = 85

# 벨 위치(보임 여부 + 좌우중앙) 판별용
BELL_POSITION_PROMPT = (
    "이 이미지에 빨간색 벨 또는 둥근 빨간 버튼이 보이나요?\n"
    "반드시 다음 중 하나로만 답하세요:\n"
    "없음 / 왼쪽 / 중앙 / 오른쪽"
)

# 화면에 보이는 빨간 물체가 '벨이 맞는지' 검증용
BELL_VERIFY_PROMPT = (
    "이 이미지 중앙 근처에 보이는 빨간색 물체가 "
    "눌러야 하는 빨간색 벨(또는 둥근 빨간 버튼)이 맞나요?\n"
    "'예' 또는 '아니오'로만 답하세요."
)

# 최종 근접 확인용
BELL_NEAR_PROMPT = (
    "빨간색 벨 또는 둥근 빨간 버튼이 가까이 보이나요? 간단히 답해주세요."
)


class BellVLM:
    """VLM 로직: 위치 판별, 벨 여부 검증, 근접 확인"""

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
            msg.height, msg.width, -1
        )
        if msg.encoding == 'rgb8':
            return cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
        return raw

    @staticmethod
    def bgr_to_b64(bgr):
        success, jpeg = cv2.imencode(
            '.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
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
                        {"type": "text", "text": question},
                    ],
                }
            ],
        )
        return response.content[0].text

    def bell_position(self, bgr):
        """벨 위치 판별 → '없음' / '왼쪽' / '중앙' / '오른쪽'"""
        b64 = self.bgr_to_b64(bgr)
        if b64 is None:
            return '없음'

        answer = self.ask(b64, BELL_POSITION_PROMPT)
        if '중앙' in answer:
            return '중앙'
        elif '왼쪽' in answer:
            return '왼쪽'
        elif '오른쪽' in answer:
            return '오른쪽'
        else:
            return '없음'

    def is_bell(self, bgr):
        """화면의 빨간 물체가 벨이 맞는지 검증 → True/False"""
        b64 = self.bgr_to_b64(bgr)
        if b64 is None:
            return False

        answer = self.ask(b64, BELL_VERIFY_PROMPT)
        self._log(f'VLM 벨 검증 응답: {answer}')
        return '예' in answer or 'yes' in answer.lower()

    def confirm_bell_near(self, bgr):
        b64 = self.bgr_to_b64(bgr)
        if b64 is None:
            return None
        return self.ask(b64, BELL_NEAR_PROMPT)


# =====================================================================
# 빨간색 검출기 (픽셀 기반 정밀 위치)
# =====================================================================
class RedBellDetector:
    def __init__(self, logger=None):
        self._logger = logger

    def detect(self, bgr):
        """빨간색 영역 검출

        반환: (found, error, area, debug)
            found: 검출 여부
            error: 화면 중앙 기준 수평 오차 (-1.0=왼쪽 ~ +1.0=오른쪽)
            area:  빨간 영역 픽셀 수
            debug: (cx, cy, width)
        """
        h, w = bgr.shape[:2]
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(hsv, RED_LOWER1, RED_UPPER1)
        mask2 = cv2.inRange(hsv, RED_LOWER2, RED_UPPER2)
        mask = cv2.bitwise_or(mask1, mask2)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return False, 0.0, 0, None

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area < MIN_RED_AREA:
            return False, 0.0, int(area), None

        M = cv2.moments(largest)
        if M['m00'] == 0:
            return False, 0.0, int(area), None

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        center_x = w / 2.0
        error = (cx - center_x) / (w / 2.0)

        return True, error, int(area), (cx, cy, w)

    @staticmethod
    def position_label(found, error):
        """검출 결과 → '없음'/'있음-왼쪽'/'있음-중앙'/'있음-오른쪽'"""
        if not found:
            return '없음'
        if abs(error) <= POSITION_CENTER_BAND:
            return '있음-중앙'
        elif error < 0:
            return '있음-왼쪽'
        else:
            return '있음-오른쪽'


# =====================================================================
# 접근 설정
# =====================================================================
# 단계 설정:
#   target  : 목표 LiDAR 거리[m] (첫 단계는 현재 위치이므로 전진 안 함)
#   name    : 단계 이름
#   tol     : 중앙 정렬 허용 오차 (작을수록 엄격)
#   confirm : 연속 중앙 판정 횟수
#   fine    : 정밀 모드 여부
APPROACH_STAGES = [
    {'target': 0.50, 'name': '50cm 정렬', 'tol': 0.10, 'confirm': 1, 'fine': False},
    {'target': 0.35, 'name': '35cm 접근', 'tol': 0.08, 'confirm': 1, 'fine': False},
    {'target': 0.25, 'name': '25cm 정밀정렬', 'tol': 0.03, 'confirm': 3, 'fine': True},
]

STOP_TOLERANCE = 0.02   # LiDAR 접근 거리 2cm 오차 허용

# 버튼 누르기 전진: 8cm
FINAL_PUSH_DISTANCE = 0.08  # 8cm
FINAL_PUSH_SPEED = 0.02     # m/s
FINAL_PUSH_TIME = FINAL_PUSH_DISTANCE / FINAL_PUSH_SPEED

# LiDAR 방향 보정
LIDAR_FRONT_ANGLE = 0.0
LIDAR_FRONT_WINDOW_DEG = 15.0


# 정렬 회전 명령 1회 지속 시간(초)
ALIGN_STEP_TIME = 0.15
ALIGN_STEP_TIME_FINE = 0.10

# 벨 탐색(왼쪽 회전) 설정 — angular_z 양수 = 반시계 = 왼쪽
SEARCH_ANGULAR = 0.35       # 탐색 회전 속도 (rad/s)
SEARCH_STEP_TIME = 0.4      # 한 번 탐색 회전 지속 시간 (s)
SEARCH_FULL_TURN = 2.0 * math.pi   # 한 바퀴(rad)

# VLM 벨 검증을 위한 탐색 최대 시도 횟수 (무한루프 방지)
MAX_VERIFY_ATTEMPTS = 60


print("===== VLM 벨 검증 게이트 + 픽셀 정렬 + LiDAR 단계 접근 =====")
for _s in APPROACH_STAGES:
    print(f"  - {_s['name']}: 목표 {_s['target']*100:.0f}cm, "
          f"허용오차 ±{_s['tol']*100:.0f}%, 확인 {_s['confirm']}회, "
          f"{'정밀' if _s['fine'] else '일반'} 모드")
print(f"버튼 누르기 전진: {FINAL_PUSH_DISTANCE*100:.0f}cm "
      f"({FINAL_PUSH_SPEED}m/s × {FINAL_PUSH_TIME:.1f}s)")


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
            Image, '/camera/image_raw', self._cam_cb, 1
        )

        scan_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self._scan_cb, scan_qos
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

    # ----- 각도 헬퍼 -----
    def _normalize_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def _angle_diff(self, a, b):
        return abs(self._normalize_angle(a - b))

    # ----- LiDAR 정면 거리 -----
    def _get_front_distance(self):
        self.scan_msg = None
        for _ in range(20):
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.scan_msg is not None:
                break

        if self.scan_msg is None:
            self.get_logger().warn('scan_msg 수신 실패')
            return None

        scan = self.scan_msg
        if len(scan.ranges) == 0 or scan.angle_increment == 0.0:
            self.get_logger().warn('scan 데이터 이상')
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
                f'정면 ±{LIDAR_FRONT_WINDOW_DEG:.0f}도 범위 유효 거리 없음'
            )
            return None

        return min(valid)

    def _log_wall_distance(self, prefix=''):
        """벽(벨)과의 남은 거리를 로그로 출력하고 값 반환"""
        dist = self._get_front_distance()
        if dist is not None:
            self.get_logger().info(
                f'{prefix}벽(벨)과의 남은 거리: {dist*100:.1f}cm'
            )
        else:
            self.get_logger().warn(f'{prefix}거리 측정 실패')
        return dist

    # ----- 카메라 프레임 -----
    def _get_bgr(self):
        self.camera_msg = None
        for _ in range(50):
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.camera_msg is not None:
                break
        if self.camera_msg is None:
            return None
        return self.vlm.ros_image_to_bgr(self.camera_msg)

    # ----- VLM 위치 판별 로그 -----
    def _vlm_report_position(self, name):
        """VLM으로 벨 위치를 판별해 로그 출력. 결과 문자열 반환."""
        bgr = self._get_bgr()
        if bgr is None:
            self.get_logger().warn(f'[{name}] VLM 위치 확인용 이미지 없음')
            return '없음'

        pos = self.vlm.bell_position(bgr)
        if pos == '없음':
            self.get_logger().info(f'[{name}] VLM: 벨 미탐지')
        else:
            self.get_logger().info(f'[{name}] VLM: 벨 탐지 — 위치 {pos}')
        return pos

    # =================================================================
    # 핵심 1: 빨간색을 찾고 'VLM이 벨이라고 인정할 때까지' 탐색
    # =================================================================
    def find_and_verify_bell(self, name):
        """화면에서 빨간색을 찾고, VLM에게 벨이 맞는지 검증.

        - 빨강 검출됨 + VLM '예' → True (정렬로 진행)
        - 빨강 없음 또는 VLM '아니오' → 왼쪽으로 회전하며 다른 빨강 탐색
        - 벨로 인정될 때까지 반복 (최대 MAX_VERIFY_ATTEMPTS)
        """
        self.get_logger().info(f'----- [{name}] 빨간색 탐색 + VLM 벨 검증 -----')

        search_accum = 0.0

        for attempt in range(1, MAX_VERIFY_ATTEMPTS + 1):
            bgr = self._get_bgr()
            if bgr is None:
                self.get_logger().error(f'[{name}] 이미지 수신 실패. 재시도...')
                time.sleep(0.5)
                continue

            found, error, area, debug = self.detector.detect(bgr)
            label = self.detector.position_label(found, error)

            if found:
                # 픽셀로 빨강을 찾았으니 VLM에게 이게 벨인지 검증
                self.get_logger().info(
                    f'[{name}] 빨간색 검출 ({label}, 오차={error:+.3f}, '
                    f'{area}px) → VLM 검증 요청'
                )

                if self.vlm.is_bell(bgr):
                    # VLM 위치 판별도 함께 로그
                    pos = self.vlm.bell_position(bgr)
                    self.get_logger().info(
                        f'[{name}] VLM: 벨 맞음 — 위치 {pos}. 정렬 진행.'
                    )
                    return True
                else:
                    self.get_logger().info(
                        f'[{name}] VLM: 이 빨간색은 벨 아님. 왼쪽으로 탐색.'
                    )
                    self._rotate(SEARCH_ANGULAR, SEARCH_STEP_TIME)
                    search_accum += SEARCH_ANGULAR * SEARCH_STEP_TIME

            else:
                # 빨강 자체가 안 보임 → 왼쪽 탐색 회전
                self.get_logger().info(
                    f'[{name}] VLM: 벨 미탐지 (빨강 없음). 왼쪽으로 탐색 '
                    f'(누적 {math.degrees(search_accum):.0f}도)'
                )
                self._rotate(SEARCH_ANGULAR, SEARCH_STEP_TIME)
                search_accum += SEARCH_ANGULAR * SEARCH_STEP_TIME

            # 한 바퀴 돌면 경고 후 누적 초기화
            if search_accum >= SEARCH_FULL_TURN:
                self.get_logger().warn(
                    f'[{name}] 한 바퀴 돌았지만 벨 검증 실패. 다시 탐색.'
                )
                search_accum = 0.0

            time.sleep(0.2)

        self.get_logger().error(
            f'[{name}] 최대 {MAX_VERIFY_ATTEMPTS}회 탐색했지만 벨 검증 실패'
        )
        return False

    # =================================================================
    # 핵심 2: 픽셀 기반 정밀 정렬 (벨이라고 검증된 뒤 호출)
    # =================================================================
    def align_to_bell(self, stage):
        """빨간 벨의 무게중심을 화면 중앙에 맞추도록 비례 회전"""
        name = stage['name']
        tol = stage['tol']
        need_confirm = stage['confirm']
        fine = stage['fine']

        rot_max = ROTATE_MAX_FINE if fine else ROTATE_MAX
        rot_min = ROTATE_MIN_FINE if fine else ROTATE_MIN
        step_time = ALIGN_STEP_TIME_FINE if fine else ALIGN_STEP_TIME

        self.get_logger().info(
            f'----- [{name}] 정렬 시작 | 허용오차 ±{tol*100:.0f}%, '
            f'확인 {need_confirm}회, {"정밀" if fine else "일반"} 모드 -----'
        )

        center_streak = 0

        while rclpy.ok():
            bgr = self._get_bgr()
            if bgr is None:
                self.get_logger().error('카메라 이미지 수신 실패. 재시도...')
                time.sleep(0.5)
                continue

            found, error, area, debug = self.detector.detect(bgr)
            label = self.detector.position_label(found, error)

            # 남은 거리 로그
            self._log_wall_distance(prefix=f'[{name}] ')

            if not found:
                # 정렬 중 벨을 놓치면 다시 검증 단계로 돌아가도록 False 반환
                self.get_logger().warn(
                    f'[{name}] 정렬 중 빨강 놓침. 재탐색 필요.'
                )
                self._stop()
                return False

            cx = debug[0] if debug else -1

            if abs(error) <= tol:
                center_streak += 1
                self.get_logger().info(
                    f'[{name}] 중앙 근처: {label}, 오차={error:+.3f} '
                    f'(연속 {center_streak}/{need_confirm})'
                )
                if center_streak >= need_confirm:
                    self._stop()
                    self.get_logger().info(
                        f'[{name}] 정렬 완료. 최종 오차 {error:+.3f}'
                    )
                    return True
                self._stop()
                time.sleep(0.15)
                continue

            center_streak = 0
            self.get_logger().info(
                f'[{name}] 빨강 위치: {label}, 오차={error:+.3f}, {area}px'
            )

            # error>0: 오른쪽 → 시계(angular<0) / error<0: 왼쪽 → 반시계(angular>0)
            angular = -ROTATE_GAIN * error
            sign = 1.0 if angular >= 0 else -1.0
            mag = max(min(abs(angular), rot_max), rot_min)
            angular = sign * mag

            direction = '반시계(왼쪽)' if angular > 0 else '시계(오른쪽)'
            self.get_logger().info(
                f'[{name}] {direction} 회전 (angular_z={angular:+.3f})'
            )
            self._rotate(angular, step_time)
            time.sleep(0.1)

        self._stop()
        return False

    # =================================================================
    # 검증 + 정렬을 묶은 한 단계 (놓치면 재검증 반복)
    # =================================================================
    def verify_and_align(self, stage):
        """벨 검증 → 정렬. 정렬 중 놓치면 다시 검증부터 반복."""
        name = stage['name']

        while rclpy.ok():
            # 빨강 찾고 VLM이 벨이라 인정할 때까지
            verified = self.find_and_verify_bell(name)
            if not verified:
                self.get_logger().error(f'[{name}] 벨 검증 실패. 단계 중단.')
                return False

            # 정렬 시도 (놓치면 False → 다시 검증부터)
            aligned = self.align_to_bell(stage)
            if aligned:
                return True

            self.get_logger().info(f'[{name}] 정렬 실패/놓침 → 검증부터 다시.')
            time.sleep(0.3)

        return False

    # =================================================================
    # LiDAR 기반 목표 거리까지 전진
    # =================================================================
    def approach_to_distance(self, target_distance, name=''):
        self.get_logger().info(
            f'----- [{name}] {target_distance*100:.0f}cm까지 이동 -----'
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
                f'[{name}] 벽(벨)과의 남은 거리: {dist*100:.1f}cm '
                f'(목표 {target_distance*100:.0f}cm, 더 갈 거리 {remaining*100:.1f}cm)'
            )

            if remaining <= STOP_TOLERANCE:
                self._stop()
                self.get_logger().info(
                    f'[{name}] 목표 도달. 현재 {dist*100:.1f}cm'
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

    # =================================================================
    # 버튼 누르기 전진 (8cm)
    # =================================================================
    def final_push(self):
        dist_before = self._log_wall_distance(prefix='[버튼누르기 전] ')

        self.get_logger().info(
            f'----- 버튼 누르기 전진: {FINAL_PUSH_DISTANCE*100:.0f}cm '
            f'({FINAL_PUSH_SPEED}m/s × {FINAL_PUSH_TIME:.1f}s) -----'
        )

        push_end_time = time.time() + FINAL_PUSH_TIME
        next_log = time.time()

        while time.time() < push_end_time and rclpy.ok():
            self._pub_cmd(FINAL_PUSH_SPEED, 0.0)
            if time.time() >= next_log:
                self.get_logger().info(
                    f'전진 중... 남은 시간 {push_end_time - time.time():.1f}s'
                )
                next_log = time.time() + 0.5
            time.sleep(0.05)

        self._stop()

        dist_after = self._log_wall_distance(prefix='[버튼누르기 후] ')
        if dist_before is not None and dist_after is not None:
            self.get_logger().info(
                f'실제 전진 거리(추정): {(dist_before - dist_after)*100:.1f}cm'
            )
        self.get_logger().info('버튼 누르기 전진 완료. 정지.')

    # =================================================================
    # 전체 미션
    # =================================================================
    def run(self):
        self.get_logger().info('===== 미션 시작 =====')

        for idx, stage in enumerate(APPROACH_STAGES):
            name = stage['name']
            self.get_logger().info(
                f'### 단계 {idx+1}/{len(APPROACH_STAGES)}: {name} ###'
            )

            # 1) 먼저 VLM으로 벨 위치 보고 (로그)
            self._vlm_report_position(name)

            # 2) 둘째 단계부터는 먼저 목표 거리까지 이동
            if idx > 0:
                if not self.approach_to_distance(stage['target'], name):
                    self.get_logger().error(f'[{name}] 이동 중단')
                    return False

            # 3) 벨 검증 → 정렬 (놓치면 재검증 반복)
            if not self.verify_and_align(stage):
                self.get_logger().error(f'[{name}] 검증/정렬 실패')
                return False

        # 4) 버튼 누르기 전진
        self.final_push()

        # 5) 최종 근접 확인
        bgr = self._get_bgr()
        if bgr is not None:
            answer = self.vlm.confirm_bell_near(bgr)
            final_dist = self._get_front_distance()
            print("\n=== 최종 상태 ===")
            if final_dist is not None:
                print(f"벽(벨)과의 거리: {final_dist*100:.1f}cm")
            print(f"VLM 판단: {answer}")

        return True


def main():
    rclpy.init()
    node = BellController()

    try:
        if node.run():
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